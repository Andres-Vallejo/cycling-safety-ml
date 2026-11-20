"""A fixed, small 1D CNN baseline evaluated on the same recording-group folds.

Architecture and epoch budget are fixed before held-out predictions are read.
No architecture search, checkpoint choice or threshold tuning uses test folds.
"""

import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .contract import LABEL_NAMES
from .data import load_dataset, write_json
from .evaluation import assert_disjoint, grouped_bootstrap, metrics


def network():
    return nn.Sequential(nn.Conv1d(6, 16, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
                         nn.Conv1d(16, 32, 3, padding=1), nn.ReLU(),
                         nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(32, 4))


def fit_cnn(windows, labels, seed=42, epochs=15):
    torch.manual_seed(seed)
    np.random.seed(seed)
    mean = windows.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    scale = windows.std(axis=(0, 1), dtype=np.float64).astype(np.float32)
    scale[scale < 1e-8] = 1
    normalized = ((windows - mean) / scale).transpose(0, 2, 1).copy()
    dataset = TensorDataset(torch.from_numpy(normalized), torch.tensor(labels, dtype=torch.long))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=256, shuffle=True, generator=generator, num_workers=0)
    model = network()
    counts = np.bincount(labels, minlength=4)
    if np.any(counts == 0):
        raise ValueError("CNN training partition needs all four classes")
    weights = torch.tensor(len(labels) / (4 * counts), dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    losses = []
    for _ in range(epochs):
        model.train()
        total, batches = 0.0, 0
        for features, y in loader:
            optimizer.zero_grad()
            loss = criterion(model(features), y)
            if not torch.isfinite(loss):
                raise ValueError("CNN training produced a nonfinite loss")
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            batches += 1
        losses.append(total / batches)
    return model.eval(), mean, scale, losses


def probabilities(model, windows, mean, scale):
    outputs = []
    with torch.no_grad():
        for start in range(0, len(windows), 512):
            values = ((windows[start:start+512] - mean) / scale).transpose(0, 2, 1).copy()
            outputs.append(torch.softmax(model(torch.from_numpy(values)), dim=1).numpy())
    return np.concatenate(outputs)


def evaluate_cnn(data_dir, output_dir, folds=5, seed=42, epochs=15):
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Choose an empty output directory")
    _, y, groups, ids, manifest = load_dataset(data_dir)
    with np.load(Path(data_dir) / "controlled_features.npz", allow_pickle=False) as archive:
        if "windows" not in archive:
            raise ValueError("CNN needs rebuilt raw signal windows, not feature-only archives")
        windows = archive["windows"]
    if windows.shape != (len(y), 128, 6) or not np.isfinite(windows).all():
        raise ValueError("Invalid raw window tensor")
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {"seed": seed, "outer_folds": folds, "epochs": epochs, "batch_size": 256,
              "optimizer": "AdamW", "learning_rate": 0.001, "weight_decay": 0.0001,
              "architecture": "Conv1d(6,16,5)-ReLU-MaxPool2-Conv1d(16,32,3)-ReLU-GlobalAvgPool-Linear4",
              "hyperparameter_selection": "none; fixed before held-out evaluation",
              "normalization": "per-channel mean/std fitted only on the outer training recordings",
              "class_weights": "balanced, recomputed only from outer training labels",
              "augmentation": "none", "input": [128, 6], "device": "cpu", "cpu_threads": 2,
              "data_sha256": manifest["derived_file_sha256"]}
    write_json(output_dir / "experiment.json", config)
    write_json(output_dir / "data_manifest.json", manifest)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    start = time.perf_counter()
    pred = np.full(len(y), -1, dtype=int)
    oof_probs = np.full((len(y), 4), np.nan)
    assignment = np.full(len(y), -1, dtype=int)
    seen = np.zeros(len(y), dtype=int)
    records, splits = [], []
    for fold, (train, test) in enumerate(GroupKFold(folds, shuffle=True, random_state=seed).split(windows, y, groups)):
        assert_disjoint(groups, train, test)
        model, mean, scale, losses = fit_cnn(windows[train], y[train], seed, epochs)
        p = probabilities(model, windows[test], mean, scale)
        pred[test], oof_probs[test], assignment[test] = p.argmax(axis=1), p, fold
        seen[test] += 1
        result = {"outer_fold": fold, "train_recordings": sorted(set(groups[train])),
                  "test_recordings": sorted(set(groups[test])), "training_loss": losses,
                  **metrics(y[test], pred[test])}
        records.append(result)
        for index in test:
            splits.append({"window_id": int(ids[index]), "recording_id": groups[index], "outer_fold": fold})
        print(f"CNN fold={fold + 1}/{folds} macro_f1={result['macro_f1']:.4f}", flush=True)
        write_json(output_dir / "fold_metrics.json", records)
    if not np.all(seen == 1) or not np.isfinite(oof_probs).all():
        raise AssertionError("Incomplete CNN out-of-fold predictions")
    summary = {**metrics(y, pred), **grouped_bootstrap(y, pred, groups, seed),
               "outer_fold_macro_f1": [r["macro_f1"] for r in records]}
    write_json(output_dir / "metrics.json", summary)
    pd.DataFrame({"window_id": ids, "recording_id": groups, "outer_fold": assignment, "label": y, "pred_cnn": pred,
                  **{f"prob_{name}": oof_probs[:, i] for i, name in enumerate(LABEL_NAMES)}}).to_csv(output_dir / "oof_predictions.csv", index=False)
    # Refit for future experiments; OOF metrics are not an independent test of this refit.
    model, mean, scale, losses = fit_cnn(windows, y, seed, epochs)
    torch.save(model.state_dict(), output_dir / "cnn_weights.pt")
    write_json(output_dir / "cnn_scaler.json", {"mean": mean.tolist(), "scale": scale.tolist(), "labels": list(LABEL_NAMES)})
    reloaded = network()
    reloaded.load_state_dict(torch.load(output_dir / "cnn_weights.pt", weights_only=True))
    reloaded.eval()
    np.testing.assert_allclose(probabilities(reloaded, windows[:100], mean, scale),
                               probabilities(model, windows[:100], mean, scale), atol=1e-7)
    write_json(output_dir / "environment.json", {"torch": torch.__version__, "device": "cpu",
               "seconds": time.perf_counter() - start, "parameters": sum(p.numel() for p in model.parameters())})
    fig, ax = plt.subplots(figsize=(6.5, 5), constrained_layout=True)
    cm = np.array(summary["confusion_matrix"])
    normalized = cm / cm.sum(axis=1, keepdims=True)
    image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    short = ["background", "pothole", "evasive", "braking"]
    ax.set(xticks=range(4), yticks=range(4), xticklabels=short, yticklabels=short,
           xlabel="Predicted", ylabel="Annotation-derived label", title="Fixed 1D CNN · recording-disjoint OOF")
    ax.tick_params(axis="x", rotation=25)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{normalized[i,j]:.0%}\n({cm[i,j]})", ha="center", va="center",
                    color="white" if normalized[i,j] > .5 else "black")
    fig.colorbar(image, ax=ax)
    fig.savefig(output_dir / "confusion.png", dpi=140)
    plt.close(fig)
    report = ["# Fixed 1D CNN recording-disjoint evaluation", "",
              f"{len(y):,} windows from {len(set(groups))} recordings. {folds} held-out group folds, "
              f"{epochs} fixed training epochs per fold. No architecture or epoch selection uses held-out labels.", "",
              f"Pooled OOF macro-F1: **{summary['macro_f1']:.4f}**. Accuracy: **{summary['accuracy']:.2%}**.", "",
              "![CNN confusion matrix](confusion.png)", "",
              "| Label | Precision | Recall | F1 | Windows |", "|---|---:|---:|---:|---:|"]
    for label in LABEL_NAMES:
        r = summary["per_class"][label]
        report.append(f"| {label} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1-score']:.4f} | {int(r['support'])} |")
    report += ["", "These labels use recorded button-press times, not verified physical event intervals. "
               "Background is unannotated background after a guard period, not verified safe riding. "
               "Overlapping windows do not count independent incidents. Recording separation does not prove unseen-rider, session or device performance.", "",
               "cnn_weights.pt is an all-data refit. OOF metrics characterize the fixed training procedure, not an independently tested final artifact. "
               "No ESP32, TFLite quantization, production latency or field safety validation was performed.", ""]
    (output_dir / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    write_json(output_dir / "complete.json", {"status": "completed", "windows_evaluated": len(y)})
    return summary
