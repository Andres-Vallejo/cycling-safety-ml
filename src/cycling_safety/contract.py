"""Explicit feature and label contract; metadata can never become model input."""

import numpy as np
from scipy import stats

CHANNELS = ("ax", "ay", "az", "gx", "gy", "gz")
STATS = ("mean", "std", "min", "max", "range", "rms", "skew", "kurt", "p05", "p95")
MAG_STATS = ("mean", "std", "max", "range", "rms", "p99")
FEATURE_NAMES = tuple(f"{c}_{s}" for c in CHANNELS for s in STATS) + tuple(
    f"{c}_{s}" for c in ("a_mag", "g_mag") for s in MAG_STATS
)
LABEL_NAMES = ("background", "pothole", "evasive_maneuver", "emergency_braking")
ANNOTATION_LABELS = {"Bache": 1, "Esquivada": 2, "Freno de emergencia": 3}
WINDOW_SIZE = 128
SAMPLING_HZ = 50


def validate_matrix(values):
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES) or x.shape[0] == 0:
        raise ValueError("Expected a nonempty matrix with exactly 72 feature columns")
    if not np.isfinite(x).all():
        raise ValueError("Features must be finite numbers")
    return x


def features_from_window(samples):
    """Adapt the upstream 72 per-window statistics; no global learned transform.

    Input must already be 128 contiguous samples at 50 Hz, linear acceleration
    in m/s² and angular velocity in rad/s in the upstream device axis frame.
    This function does not resample, rotate axes or remove gravity.
    """
    x = np.asarray(samples, dtype=np.float32)
    if x.shape != (WINDOW_SIZE, len(CHANNELS)) or not np.isfinite(x).all():
        raise ValueError("Expected 128 x 6 finite, preprocessed IMU samples")
    out = {}
    for i, channel in enumerate(CHANNELS):
        v = x[:, i]
        constant = np.ptp(v) == 0
        for name, value in zip(STATS, (
            np.mean(v), np.std(v), np.min(v), np.max(v), np.ptp(v),
            np.sqrt(np.mean(v ** 2)), 0.0 if constant else stats.skew(v),
            0.0 if constant else stats.kurtosis(v),
            np.percentile(v, 5), np.percentile(v, 95),
        )):
            out[f"{channel}_{name}"] = float(value) if np.isfinite(value) else 0.0
    for name, v in (("a_mag", np.sqrt(np.sum(x[:, :3] ** 2, axis=1))),
                    ("g_mag", np.sqrt(np.sum(x[:, 3:] ** 2, axis=1)))):
        for suffix, value in zip(MAG_STATS, (
            np.mean(v), np.std(v), np.max(v), np.ptp(v),
            np.sqrt(np.mean(v ** 2)), np.percentile(v, 99),
        )):
            out[f"{name}_{suffix}"] = float(value)
    validate_matrix([[out[key] for key in FEATURE_NAMES]])
    return out
