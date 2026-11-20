"""Audit saved evidence without retraining or reading GPS data."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify(root, academic_source=None):
    root = Path(root)
    feature_dir, cnn_dir = root / "reports/grouped-mlp", root / "reports/grouped-cnn"
    left, right = pd.read_csv(feature_dir / "oof_predictions.csv"), pd.read_csv(cnn_dir / "oof_predictions.csv")
    columns = ["window_id", "recording_id", "outer_fold", "label"]
    pd.testing.assert_frame_equal(left[columns], right[columns])
    assert not left.window_id.duplicated().any()
    assert left.groupby("recording_id").outer_fold.nunique().eq(1).all()
    assert len(left) == 9453 and left.recording_id.nunique() == 11
    assert read(feature_dir / "complete.json")["status"] == "completed"
    assert read(cnn_dir / "complete.json")["status"] == "completed"
    assert read(feature_dir / "data_manifest.json")["derived_file_sha256"] == read(cnn_dir / "data_manifest.json")["derived_file_sha256"]
    assert read(feature_dir / "experiment.json")["seed"] == read(cnn_dir / "experiment.json")["seed"]
    feature_metrics, cnn_metrics = read(feature_dir / "metrics.json"), read(cnn_dir / "metrics.json")
    for family, predictions, stored in [(name, left[f"pred_{name}"], value) for name, value in feature_metrics.items()] + [("cnn", right.pred_cnn, cnn_metrics)]:
        assert np.isclose(f1_score(left.label, predictions, labels=range(4), average="macro", zero_division=0), stored["macro_f1"])
        assert np.isclose(accuracy_score(left.label, predictions), stored["accuracy"])
        assert confusion_matrix(left.label, predictions, labels=range(4)).tolist() == stored["confusion_matrix"]
    split_table = pd.read_csv(feature_dir / "outer_splits.csv")
    for fold in range(5):
        rows = split_table[split_table.outer_fold.eq(fold)]
        training = set(rows.loc[rows.partition.eq("train"), "recording_id"])
        test = set(rows.loc[rows.partition.eq("test"), "recording_id"])
        assert not training & test
        assert test == set(left.loc[left.outer_fold.eq(fold), "recording_id"])
        for search in read(feature_dir / "inner_search.json"):
            if search["outer_fold"] != fold:
                continue
            for record in search["search"]:
                if "inner_fold" not in record:
                    continue
                train = set(record["train_recordings"])
                validation = set(record["validation_recordings"])
                assert not train & validation and not test & (train | validation)
                assert train | validation == training
    # The service artifact must be the newly refitted MLP, not a historic model.
    assert (root / "artifacts/model.json").read_bytes() == (feature_dir / "model.json").read_bytes()
    assert hashlib.sha256((root / "artifacts/model.json").read_bytes()).hexdigest() == read(feature_dir / "complete.json")["model_sha256"]
    original_files_checked = 0
    if academic_source is not None:
        for name, expected in read(root / "provenance/academic_snapshot.json")["files"].items():
            assert hashlib.sha256((Path(academic_source) / name).read_bytes()).hexdigest() == expected, name
            original_files_checked += 1
    return {"status": "passed", "retained_windows": len(left), "recordings": int(left.recording_id.nunique()),
            "outer_folds": 5, "same_held_out_assignments_mlp_and_cnn": True,
            "inner_search_disjoint_from_outer_test": True, "metrics_recomputed_from_oof": True,
            "service_artifact_matches_new_refit": True, "unchanged_academic_files_checked": original_files_checked}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--academic-source", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(Path(__file__).resolve().parents[1], args.academic_source)
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text)
