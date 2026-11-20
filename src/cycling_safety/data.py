"""Import only nonaugmented annotated-recording features, with provenance."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .contract import ANNOTATION_LABELS, FEATURE_NAMES, LABEL_NAMES, validate_matrix

SOURCE_COMMIT = "546d11993ffdab3170a02749f67309ae469454f3"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def import_features(source_dir, output_dir, expected_manifest):
    source_dir, output_dir = Path(source_dir), Path(output_dir)
    expected = json.loads(Path(expected_manifest).read_text(encoding="utf-8"))
    hashes = {}
    for filename in ("meta_mlp.csv", "window_features_mlp.csv"):
        hashes[filename] = sha256(source_dir / "features" / filename)
        if hashes[filename] != expected["files"][filename]:
            raise ValueError(f"Source checksum mismatch: {filename}. Use the documented source snapshot.")
    meta = pd.read_csv(source_dir / "features/meta_mlp.csv")
    features = pd.read_csv(source_dir / "features/window_features_mlp.csv")
    if len(meta) != len(features):
        raise ValueError("Metadata and features are not row-aligned")
    if list(features.columns) != list(FEATURE_NAMES) + ["severity_score", "bump_score"]:
        raise ValueError("Unexpected source feature schema")
    selected = meta.source.eq("artificial")
    labels = []
    for annotation in meta.loc[selected, "annotation"]:
        if pd.isna(annotation):
            labels.append(0)
        elif annotation in ANNOTATION_LABELS:
            labels.append(ANNOTATION_LABELS[annotation])
        else:
            raise ValueError(f"Unknown annotation: {annotation}")
    y = np.asarray(labels, dtype=np.int64)
    if not np.array_equal(y, meta.loc[selected, "label_supervised"].to_numpy()):
        raise ValueError("Saved labels disagree with annotation-derived labels")
    if meta.loc[selected, ["route", "lap"]].isna().any().any():
        raise ValueError("Every selected window needs recording/lap metadata")
    # Group by recording (route ID), including ALL laps of each recording.
    route_ids = meta.loc[selected, "route"].astype(str)
    groups = np.array(["rec_" + hashlib.sha256(r.encode()).hexdigest()[:12] for r in route_ids])
    if len(set(groups)) != route_ids.nunique():
        raise ValueError("Recording ID hash collision")
    x = validate_matrix(features.loc[selected, list(FEATURE_NAMES)].to_numpy())
    ids = meta.index[selected].to_numpy(dtype=np.int64)
    if len(x) != 10840 or len(set(groups)) != 11:
        raise ValueError("Unexpected source snapshot cardinality")
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "controlled_features.npz"
    np.savez_compressed(target, X=x, y=y, groups=groups, window_ids=ids)
    manifest = {
        "schema_version": 1, "source_commit": SOURCE_COMMIT,
        "source_repository": "https://github.com/Andres-Vallejo/CyclingSafety",
        "upstream_repository": "https://github.com/w1sec0d/CyclingSafety",
        "source_files_sha256": hashes, "derived_file_sha256": sha256(target),
        "windows": len(x), "recordings": len(set(groups)),
        "features": list(FEATURE_NAMES), "labels": list(LABEL_NAMES),
        "class_counts": {name: int(np.sum(y == i)) for i, name in enumerate(LABEL_NAMES)},
        "excluded": {"natural_unannotated_windows": int(meta.source.eq("natural").sum()),
                     "historic_augmented_windows": int(meta.source.eq("artificial_aug").sum()),
                     "features": ["severity_score", "bump_score"],
                     "metadata": ["GPS", "recording_names", "historic_split_columns"]},
        "background_definition": "No recognized annotation in the archived window; not verified event-free riding",
        "known_limits": ["Raw signals and window boundaries unavailable", "Rider/session/device identities unavailable",
                         "Grouping proves recording-ID separation only", "Sparse annotation and upstream alignment cannot be rechecked"],
    }
    write_json(output_dir / "manifest.json", manifest)
    example = pd.DataFrame(x[[int(np.flatnonzero(y == label)[0]) for label in range(4)]], columns=FEATURE_NAMES)
    example.to_csv(output_dir / "example_features.csv", index=False)
    return manifest


def load_dataset(data_dir):
    data_dir = Path(data_dir)
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    archive_path = data_dir / "controlled_features.npz"
    if sha256(archive_path) != manifest["derived_file_sha256"]:
        raise ValueError("Dataset checksum mismatch")
    with np.load(archive_path, allow_pickle=False) as archive:
        x = validate_matrix(archive["X"])
        y, groups, ids = (archive[key].copy() for key in ("y", "groups", "window_ids"))
    if len(x) != len(y) or len(y) != len(groups) or len(y) != len(ids):
        raise ValueError("Dataset arrays are not aligned")
    if y.dtype.kind not in "iu" or set(np.unique(y)) != set(range(4)):
        raise ValueError("Expected four supervised labels")
    if len(np.unique(ids)) != len(ids) or np.any(groups == ""):
        raise ValueError("Window IDs must be unique and groups nonempty")
    if manifest["features"] != list(FEATURE_NAMES):
        raise ValueError("Feature order mismatch")
    return x, y, groups, ids, manifest
