"""Nested recording-group CV. Outer labels never choose hyperparameters."""

import hashlib
import itertools
import json
import platform
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits

from .contract import FEATURE_NAMES, LABEL_NAMES
from .data import load_dataset, write_json
from .inference import export_mlp, predict_proba

# Freeze before looking at held-out predictions. MLP is the deployment family.
CANDIDATES = {
    "majority": [{}],
    "logistic": [{"C": 0.1}, {"C": 1.0}],
    "mlp": [{"hidden_layer_sizes": [32], "alpha": 0.01},
            {"hidden_layer_sizes": [64, 32], "alpha": 0.01}],
}


def assert_disjoint(groups, train, test):
    if set(train) & set(test):
        raise ValueError("Window leakage between partitions")
    if set(groups[train]) & set(groups[test]):
        raise ValueError("Recording leakage between partitions")
    if len(train) == 0 or len(test) == 0:
        raise ValueError("Empty partition")


def model_for(family, params, seed, max_iter):
    if family == "majority":
        estimator = DummyClassifier(strategy="most_frequent")
    elif family == "logistic":
        estimator = LogisticRegression(C=params["C"], max_iter=1000, random_state=seed)
    else:
        estimator = MLPClassifier(hidden_layer_sizes=tuple(params["hidden_layer_sizes"]),
                                  alpha=params["alpha"], max_iter=max_iter,
                                  batch_size=256, early_stopping=False,
                                  n_iter_no_change=20, random_state=seed)
    return Pipeline([("scale", StandardScaler()), ("model", estimator)])


def fit_model(family, params, seed, max_iter, x, y):
    model = model_for(family, params, seed, max_iter)
    fit_kwargs = {} if family == "majority" else {"model__sample_weight": compute_sample_weight("balanced", y)}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x, y, **fit_kwargs)
    return model, sum(issubclass(w.category, ConvergenceWarning) for w in caught)


def metrics(y, pred):
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=range(4), average="macro", zero_division=0)),
        "per_class": classification_report(y, pred, labels=range(4), target_names=LABEL_NAMES,
                                             output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y, pred, labels=range(4)).tolist(),
    }


def select_candidate(family, x, y, groups, seed, inner_folds, max_iter):
    if family == "majority":
        return {}, [], 0
    cv = GroupKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    splits = list(cv.split(x, y, groups))
    records, convergence_count = [], 0
    for candidate_id, params in enumerate(CANDIDATES[family]):
        scores = []
        for fold, (train, val) in enumerate(splits):
            assert_disjoint(groups, train, val)
            if set(np.unique(y[train])) != set(range(4)):
                raise ValueError("Inner training partition lacks a class")
            model, count = fit_model(family, params, seed, max_iter, x[train], y[train])
            convergence_count += count
            score = metrics(y[val], model.predict(x[val]))["macro_f1"]
            scores.append(score)
            records.append({"candidate": candidate_id, "params": params, "inner_fold": fold,
                            "macro_f1": score, "train_recordings": sorted(set(groups[train])),
                            "validation_recordings": sorted(set(groups[val])), "convergence_warnings": count})
        records.append({"candidate": candidate_id, "params": params, "mean_macro_f1": float(np.mean(scores))})
    means = [r for r in records if "mean_macro_f1" in r]
    # Fixed candidate order breaks ties; outer held-out scores are not consulted.
    best = max(means, key=lambda r: r["mean_macro_f1"])
    return best["params"], records, convergence_count


def grouped_bootstrap(y, pred, groups, seed, repeats=1000):
    cms = np.array([confusion_matrix(y[groups == g], pred[groups == g], labels=range(4))
                    for g in sorted(set(groups))])
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(repeats):
        cm = cms[rng.integers(0, len(cms), size=len(cms))].sum(axis=0)
        denominator = cm.sum(axis=0) + cm.sum(axis=1)
        scores.append(np.divide(2 * np.diag(cm), denominator,
                                 out=np.zeros(4, dtype=float), where=denominator != 0).mean())
    return {"macro_f1_95_percentile_interval": np.quantile(scores, [0.025, 0.975]).tolist(),
            "resampling_unit": "recording", "replicates": repeats,
            "note": "Conditional on these OOF predictions; does not capture model-refit or rider uncertainty"}


def evaluate(data_dir, output_dir, outer_folds=5, inner_folds=3, seed=42, max_iter=200):
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Choose an empty output directory to preserve previous experiment results")
    output_dir.mkdir(parents=True, exist_ok=True)
    x, y, groups, ids, manifest = load_dataset(data_dir)
    if len(set(groups)) < outer_folds:
        raise ValueError("Not enough independent recordings")
    config = {"schema_version": 1, "seed": seed, "outer_folds": outer_folds,
              "inner_folds": inner_folds, "max_iter": max_iter, "candidates": CANDIDATES,
              "group_unit": "recording", "deployment_family": "mlp",
              "primary_metric": "pooled OOF four-class macro-F1", "augmentation": "none",
              "class_weighting": "balanced; recomputed on each training partition",
              "preprocessing": "StandardScaler fitted only on each training partition",
              "mlp_early_stopping": False, "blas_threads": 1,
              "data_sha256": manifest["derived_file_sha256"]}
    write_json(output_dir / "experiment.json", config)  # Written before any fitting.
    write_json(output_dir / "data_manifest.json", manifest)
    start = time.perf_counter()
    predictions = {name: np.full(len(y), -1, dtype=int) for name in CANDIDATES}
    seen = np.zeros(len(y), dtype=int)
    folds = np.full(len(y), -1, dtype=int)
    fold_records, split_rows, searches = [], [], []
    total_warnings = 0
    outer = GroupKFold(n_splits=outer_folds, shuffle=True, random_state=seed)
    with threadpool_limits(limits=1):
        for fold, (train, test) in enumerate(outer.split(x, y, groups)):
            assert_disjoint(groups, train, test)
            if set(np.unique(y[train])) != set(range(4)):
                raise ValueError("Outer training partition lacks a class")
            seen[test] += 1
            folds[test] = fold
            for partition, indices in (("train", train), ("test", test)):
                for group in sorted(set(groups[indices])):
                    rows = indices[groups[indices] == group]
                    split_rows.append({"outer_fold": fold, "partition": partition,
                                       "recording_id": group, "windows": len(rows),
                                       **{name: int(np.sum(y[rows] == i)) for i, name in enumerate(LABEL_NAMES)}})
            for family in CANDIDATES:
                params, search, count = select_candidate(family, x[train], y[train], groups[train],
                                                         seed + fold, inner_folds, max_iter)
                total_warnings += count
                searches.append({"outer_fold": fold, "family": family, "search": search, "selected": params})
                model, count = fit_model(family, params, seed, max_iter, x[train], y[train])
                total_warnings += count
                pred = model.predict(x[test])
                predictions[family][test] = pred
                record = {"outer_fold": fold, "family": family, "params": params,
                          "train_windows": len(train), "test_windows": len(test),
                          "convergence_warnings": count, **metrics(y[test], pred)}
                fold_records.append(record)
                print(f"fold={fold + 1}/{outer_folds} model={family} macro_f1={record['macro_f1']:.4f}", flush=True)
            # Keep a recovery/audit record even if a later fit is interrupted.
            write_json(output_dir / "inner_search.json", searches)
            write_json(output_dir / "fold_metrics.json", fold_records)
        if not np.all(seen == 1) or any(np.any(p < 0) for p in predictions.values()):
            raise AssertionError("Every original window must have exactly one held-out prediction per family")
        summary = {}
        for family, pred in predictions.items():
            summary[family] = {**metrics(y, pred), **grouped_bootstrap(y, pred, groups, seed),
                               "outer_fold_macro_f1": [r["macro_f1"] for r in fold_records if r["family"] == family]}
        pd.DataFrame(split_rows).to_csv(output_dir / "outer_splits.csv", index=False)
        oof = pd.DataFrame({"window_id": ids, "recording_id": groups, "outer_fold": folds, "label": y,
                            **{f"pred_{family}": p for family, p in predictions.items()}})
        oof.to_csv(output_dir / "oof_predictions.csv", index=False)
        per_record = []
        for family, pred in predictions.items():
            for group in sorted(set(groups)):
                mask = groups == group
                per_record.append({"family": family, "recording_id": group, "windows": int(mask.sum()),
                                   "accuracy": float(accuracy_score(y[mask], pred[mask])),
                                   "macro_f1_fixed_four_classes": metrics(y[mask], pred[mask])["macro_f1"]})
        pd.DataFrame(per_record).to_csv(output_dir / "per_recording.csv", index=False)
        # A deployment refit is distinct from OOF evaluation; no new test claim.
        params, search, count = select_candidate("mlp", x, y, groups, seed, inner_folds, max_iter)
        total_warnings += count
        write_json(output_dir / "deployment_selection.json", {"family": "mlp", "search": search, "selected": params})
        deployed, count = fit_model("mlp", params, seed, max_iter, x, y)
        total_warnings += count
        export = export_mlp(deployed, manifest["derived_file_sha256"], len(x), len(set(groups)))
        np.testing.assert_allclose(predict_proba(export, x), deployed.predict_proba(x), rtol=1e-9, atol=1e-10)
        write_json(output_dir / "model.json", export)
    environment = {"python": platform.python_version(), "platform": platform.platform(),
                   "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
                   "seconds": time.perf_counter() - start, "convergence_warnings": total_warnings}
    write_json(output_dir / "environment.json", environment)
    write_json(output_dir / "metrics.json", summary)
    plot_results(summary, output_dir)
    write_report(summary, environment, output_dir, manifest, outer_folds, inner_folds)
    write_json(output_dir / "complete.json", {"status": "completed", "model_sha256": hashlib.sha256(
        (output_dir / "model.json").read_bytes()).hexdigest(), "windows_evaluated": len(x)})
    return summary


def plot_results(summary, output_dir):
    plt.rcParams.update({"font.size": 11, "figure.dpi": 140})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    names = list(summary)
    scores = [summary[k]["macro_f1"] for k in names]
    axes[0].bar(names, scores, color=["#94a3b8", "#0d9488", "#2563eb"])
    axes[0].set(ylim=(0, 1), ylabel="Pooled out-of-fold macro-F1", title="Unseen recording IDs · grouped evaluation")
    for i, score in enumerate(scores):
        axes[0].text(i, score + 0.025, f"{score:.3f}", ha="center")
    cm = np.array(summary["mlp"]["confusion_matrix"])
    normalized = cm / cm.sum(axis=1, keepdims=True)
    image = axes[1].imshow(normalized, vmin=0, vmax=1, cmap="Blues")
    short = ["background", "pothole", "evasive", "braking"]
    axes[1].set(xticks=range(4), yticks=range(4), xticklabels=short, yticklabels=short,
                xlabel="Predicted", ylabel="Archived label", title="MLP · row-normalized confusion matrix")
    axes[1].tick_params(axis="x", rotation=25)
    for i, j in itertools.product(range(4), repeat=2):
        axes[1].text(j, i, f"{normalized[i,j]:.0%}\n({cm[i,j]})", ha="center", va="center",
                     color="white" if normalized[i, j] > 0.5 else "black", fontsize=9)
    fig.colorbar(image, ax=axes[1], fraction=0.04)
    fig.savefig(output_dir / "evaluation.png")
    plt.close(fig)


def write_report(summary, environment, output_dir, manifest, outer_folds, inner_folds):
    text = ["# Recording-disjoint evaluation", "",
            f"{manifest['windows']:,} nonaugmented windows from {manifest['recordings']} annotated recordings. "
            f"{outer_folds} outer recording-group folds; {inner_folds} inner group folds select hyperparameters. "
            "All 72 input features are per-window statistics. Every window is tested once per model family.", "",
            "| Model family | OOF accuracy | OOF macro-F1 | Recording-bootstrap 95% interval |", "|---|---:|---:|---:|"]
    for name, result in summary.items():
        lo, hi = result["macro_f1_95_percentile_interval"]
        text.append(f"| {name} | {result['accuracy']:.2%} | {result['macro_f1']:.4f} | [{lo:.4f}, {hi:.4f}] |")
    text += ["", "![Evaluation](evaluation.png)", "", "## MLP per-class results", "",
             "| Archived label | Precision | Recall | F1 | Support (windows) |", "|---|---:|---:|---:|---:|"]
    for label in LABEL_NAMES:
        r = summary["mlp"]["per_class"][label]
        text.append(f"| {label} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1-score']:.4f} | {int(r['support'])} |")
    text += ["", "## Interpretation", "",
             "The bootstrap resamples complete recordings, conditional on the saved OOF predictions. "
             "It is not a model-refit confidence interval. Only 11 recordings are available; fold variability is saved in metrics.json.", "",
             "Background means no recognized archived annotation, not independently verified normal riding. "
             "Rider identity and verified event-instance boundaries are unavailable. Recording-disjoint results do not establish "
             "generalization to unseen riders, unseen cities, or independent incidents. Overlapping windows are not independent events.", "",
             "The original scores used a different split and augmentation procedure. A before/after performance ranking would be misleading. "
             "The data manifest documents whether inputs were rebuilt from raw sensors or imported as archived features. "
             "Annotation press times are not verified physical event boundaries. ESP32 inference has not been tested here. "
             "A separately executed CNN experiment, when present, has its own report.", "",
             "The deployment JSON is an MLP refitted on all 11 recordings after grouped hyperparameter selection. "
             "The table evaluates the training procedure using OOF models, not this final refit on an independent test set.", "",
             f"Training reached its iteration budget in {environment['convergence_warnings']} fits (see saved search records). "
             "Convergence warnings are disclosed rather than hidden.", "",
             "This is a research prototype, not a safety alert or validated road-risk assessment system.", ""]
    (output_dir / "REPORT.md").write_text("\n".join(text), encoding="utf-8")
