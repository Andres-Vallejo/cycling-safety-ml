import numpy as np
import pandas as pd
import pytest

from cycling_safety.preprocessing import resample_sensor, window_recording


def recording():
    t = np.arange(700) / 50
    sensor = pd.DataFrame({"seconds_elapsed": t, "x": np.sin(t), "y": np.cos(t), "z": t})
    return sensor


def test_long_sensor_gaps_are_not_filled_or_bridged():
    sensor = recording().drop(index=range(150, 160))
    sampled = resample_sensor(sensor, "a")
    assert sampled.loc[150:159, "ax"].isna().all()
    annotations = pd.DataFrame({"seconds_elapsed": [9.0], "text": ["Bache"]})
    windows, _, starts, decisions = window_recording(sensor, recording(), annotations)
    assert decisions["missing_sensor_samples"] > 0
    assert np.isfinite(windows).all()
    assert all(not (start <= 159 and start + 128 > 150) for start in starts)


def test_mixed_event_windows_are_excluded_instead_of_last_label_winning():
    annotations = pd.DataFrame({"seconds_elapsed": [3.0, 3.2], "text": ["Bache", "Esquivada"]})
    _, _, _, decisions = window_recording(recording(), recording(), annotations)
    assert decisions["mixed_class"] > 0


def test_unknown_annotations_fail_instead_of_becoming_background():
    annotations = pd.DataFrame({"seconds_elapsed": [3.0], "text": ["unknown event"]})
    with pytest.raises(ValueError, match="Unrecognized"):
        window_recording(recording(), recording(), annotations)
