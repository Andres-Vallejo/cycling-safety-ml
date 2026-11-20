"""Rebuild supervised windows directly from Sensor Logger ZIP archives.

Only accelerometer, gyroscope and annotation CSVs are read. No GPS is loaded.
Archives are streamed, not extracted, so archive paths cannot write files.
"""

import hashlib
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from .contract import ANNOTATION_LABELS, FEATURE_NAMES, LABEL_NAMES, features_from_window
from .data import sha256, write_json

RAW_CONFIG = {"sampling_hz": 50, "window_samples": 128, "stride_samples": 64,
              "max_interpolation_gap_samples": 3, "background_guard_seconds": 1.28,
              "annotation_timing": "recorded button-press time, no peak realignment",
              "mixed_class_windows": "excluded", "augmentation": "none",
              "input_sensors": ["Accelerometer.csv", "Gyroscope.csv", "Annotation.csv"]}


def resample_sensor(frame, prefix, max_gap=3):
    required = ["seconds_elapsed", "x", "y", "z"]
    if not set(required) <= set(frame):
        raise ValueError("Sensor CSV is missing timestamp or axis columns")
    frame = frame[required].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(frame.to_numpy()).all() or len(frame) < 128:
        raise ValueError("Sensor CSV has invalid values or too few samples")
    frame["tick"] = np.rint(frame.seconds_elapsed.to_numpy() * 50).astype(np.int64)
    binned = frame.groupby("tick")[["x", "y", "z"]].mean()
    binned = binned.reindex(np.arange(binned.index.min(), binned.index.max() + 1))
    for column in ("x", "y", "z"):
        missing = binned[column].isna()
        runs = missing.ne(missing.shift()).cumsum()
        gap_length = missing.groupby(runs).transform("sum")
        interpolated = binned[column].interpolate(method="linear", limit_area="inside")
        # Never partially fill a long gap from a distant future observation.
        interpolated[missing & (gap_length > max_gap)] = np.nan
        binned[column] = interpolated
    return binned.rename(columns={axis: prefix + axis for axis in ("x", "y", "z")})


def window_recording(accelerometer, gyroscope, annotations, config=None):
    config = config or RAW_CONFIG
    acc = resample_sensor(accelerometer, "a", config["max_interpolation_gap_samples"])
    gyro = resample_sensor(gyroscope, "g", config["max_interpolation_gap_samples"])
    lower, upper = max(acc.index.min(), gyro.index.min()), min(acc.index.max(), gyro.index.max())
    if upper - lower + 1 < config["window_samples"]:
        raise ValueError("Sensors do not have enough overlapping time")
    aligned = acc.join(gyro, how="outer").reindex(np.arange(lower, upper + 1))
    values = aligned[["ax", "ay", "az", "gx", "gy", "gz"]].to_numpy(dtype=np.float32)
    if not {"seconds_elapsed", "text"} <= set(annotations):
        raise ValueError("Annotation CSV is missing time/text columns")
    unknown = set(annotations.text.dropna().astype(str)) - set(ANNOTATION_LABELS)
    if unknown or annotations.text.isna().any():
        raise ValueError(f"Unrecognized or empty event annotations: {sorted(unknown)}")
    annotations = annotations.sort_values("seconds_elapsed")
    event_times = pd.to_numeric(annotations.seconds_elapsed, errors="raise").to_numpy(dtype=float)
    if len(event_times) == 0 or not np.isfinite(event_times).all():
        raise ValueError("Each controlled recording needs finite event annotations")
    event_labels = annotations.text.map(ANNOTATION_LABELS).to_numpy(dtype=np.int64)
    if np.any(event_times < lower / 50 - .1) or np.any(event_times > (upper + 1) / 50 + .1):
        raise ValueError("Annotations fall outside the overlapping sensor time range")
    windows, labels, starts = [], [], []
    decisions = {"candidates": 0, "missing_sensor_samples": 0, "mixed_class": 0, "background_guard": 0}
    size, stride = config["window_samples"], config["stride_samples"]
    for start in range(0, len(values) - size + 1, stride):
        decisions["candidates"] += 1
        window = values[start:start + size]
        if not np.isfinite(window).all():
            decisions["missing_sensor_samples"] += 1
            continue
        t0, t1 = (lower + start) / 50, (lower + start + size) / 50
        event_slice = (event_times >= t0) & (event_times < t1)
        present = np.unique(event_labels[event_slice])
        if len(present) > 1:
            decisions["mixed_class"] += 1
            continue
        if len(present) == 0:
            guard = config["background_guard_seconds"]
            if np.any((event_times >= t0 - guard) & (event_times < t1 + guard)):
                decisions["background_guard"] += 1
                continue
        windows.append(window)
        labels.append(int(present[0]) if len(present) else 0)
        starts.append(int(lower + start))
    if not windows:
        raise ValueError("No usable windows after quality checks")
    return np.stack(windows), np.array(labels), np.array(starts), decisions


def prepare_archives(archives, output_dir):
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Choose an empty derived-data directory")
    archives = [Path(p) for p in archives]
    with ExitStack() as stack:
        zips = [stack.enter_context(ZipFile(path)) for path in archives]
        recordings = defaultdict(dict)
        for archive in zips:
            for entry in archive.infolist():
                parts = entry.filename.replace("\\", "/").split("/")
                if len(parts) < 3 or "/raw/artificial_events/" not in "/" + "/".join(parts):
                    continue
                filename, recording = parts[-1], parts[-2]
                if filename not in RAW_CONFIG["input_sensors"]:
                    continue
                if filename in recordings[recording]:
                    raise ValueError("Duplicate sensor input: supply one complete ZIP export only")
                recordings[recording][filename] = (archive, entry)
        if len(recordings) < 5:
            raise ValueError("Need at least five complete independent recordings for this protocol")
        windows_all, features_all, labels_all, groups_all, starts_all = [], [], [], [], []
        source_hashes, recording_reports = {}, []
        for name in sorted(recordings):
            entries = recordings[name]
            if set(entries) != set(RAW_CONFIG["input_sensors"]):
                raise ValueError("Incomplete recording: a ZIP part or a sensor/annotation file is missing")
            group_id = "rec_" + hashlib.sha256(name.encode()).hexdigest()[:12]
            frames = {}
            for filename, (archive, entry) in entries.items():
                with archive.open(entry) as handle:
                    content = handle.read()
                source_hashes[f"{group_id}/{filename}"] = hashlib.sha256(content).hexdigest()
                from io import BytesIO
                frames[filename] = pd.read_csv(BytesIO(content))
            windows, labels, starts, decisions = window_recording(
                frames["Accelerometer.csv"], frames["Gyroscope.csv"], frames["Annotation.csv"])
            features = np.array([list(features_from_window(w).values()) for w in windows], dtype=np.float64)
            windows_all.append(windows)
            features_all.append(features)
            labels_all.append(labels)
            groups_all.extend([group_id] * len(windows))
            starts_all.append(starts)
            recording_reports.append({"recording_id": group_id, "windows": len(windows),
                                      "annotation_presses": len(frames["Annotation.csv"]), "decisions": decisions,
                                      "class_counts": {label: int(np.sum(labels == i)) for i, label in enumerate(LABEL_NAMES)}})
            print(f"prepared {group_id}: {len(windows)} windows", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    x, raw, y = np.concatenate(features_all), np.concatenate(windows_all), np.concatenate(labels_all)
    target = output_dir / "controlled_features.npz"
    np.savez_compressed(target, X=x, windows=raw, y=y, groups=np.array(groups_all),
                        window_ids=np.arange(len(y), dtype=np.int64), start_ticks=np.concatenate(starts_all))
    manifest = {
        "schema_version": 1, "data_kind": "rebuilt_from_raw_sensor_csvs", "windows": len(x),
        "recordings": len(recordings), "derived_file_sha256": sha256(target),
        "features": list(FEATURE_NAMES), "labels": list(LABEL_NAMES), "preprocessing": RAW_CONFIG,
        "class_counts": {label: int(np.sum(y == i)) for i, label in enumerate(LABEL_NAMES)},
        "raw_sensor_files_sha256": source_hashes,
        "archive_sha256": {p.name: sha256(p) for p in archives},
        "recording_reports": recording_reports,
        "provenance": "New windowing/quality/label policy over the original team's Sensor Logger recordings",
        "excluded": {"GPS": True, "natural_rides": True, "historic_features_models_and_augmentations": True},
        "background_definition": "No annotation in window or 1.28-second guard; not independently verified normal riding",
        "known_limits": ["Button-press time is a proxy for physical event timing; no verified event intervals",
                         "Recording-disjoint does not imply rider/session/device-disjoint",
                         "Offline centered binning/interpolation is not a validated causal embedded pipeline"],
    }
    write_json(output_dir / "manifest.json", manifest)
    examples = [int(np.flatnonzero(y == i)[0]) for i in range(4)]
    pd.DataFrame(x[examples], columns=FEATURE_NAMES).to_csv(output_dir / "example_features.csv", index=False)
    write_json(output_dir / "example_request.json", {"records": pd.DataFrame(x[examples], columns=FEATURE_NAMES).to_dict(orient="records")})
    return manifest
