import numpy as np
import pytest
from sklearn.model_selection import GroupKFold

from cycling_safety.evaluation import assert_disjoint, evaluate, fit_model, select_candidate


def test_overlapping_recordings_are_rejected_even_when_row_indices_differ():
    groups = np.array(["ride-a", "ride-a", "ride-b", "ride-c"])
    with pytest.raises(ValueError, match="Recording leakage"):
        assert_disjoint(groups, np.array([0, 2]), np.array([1, 3]))


def test_outer_and_inner_folds_keep_all_windows_of_recording_together():
    groups = np.repeat([f"ride-{i}" for i in range(11)], 20)
    y = np.tile([0, 0, 1, 2, 3], 44)
    x = np.arange(len(y))[:, None]
    counts = np.zeros(len(y), dtype=int)
    for train, test in GroupKFold(5, shuffle=True, random_state=42).split(x, y, groups):
        assert_disjoint(groups, train, test)
        counts[test] += 1
        for inner_train, val in GroupKFold(3, shuffle=True, random_state=42).split(x[train], y[train], groups[train]):
            assert_disjoint(groups[train], inner_train, val)
            assert not set(groups[test]) & set(groups[train][val])
    assert np.all(counts == 1)


def test_normalization_is_fitted_on_training_only():
    x = np.tile(np.arange(80).reshape(-1, 1), (1, 72)).astype(float)
    y = np.tile(np.arange(4), 20)
    model, _ = fit_model("logistic", {"C": 0.1}, 7, 20, x[:40], y[:40])
    np.testing.assert_allclose(model.named_steps["scale"].mean_, x[:40].mean(axis=0))
    assert not np.allclose(model.named_steps["scale"].mean_, x.mean(axis=0))


def test_nested_tuning_has_no_access_to_outer_test(monkeypatch):
    import cycling_safety.evaluation as evaluation
    rng = np.random.default_rng(5)
    x = rng.normal(size=(120, 72))
    y = np.tile(np.arange(4), 30)
    groups = np.repeat(np.arange(6), 20).astype(str)
    x[:, 0] = np.repeat(np.arange(6), 20)
    original_fit = evaluation.fit_model
    observed = []

    def observed_fit(family, params, seed, max_iter, features, labels):
        observed.extend(features[:, 0].astype(int).tolist())
        return original_fit(family, params, seed, max_iter, features, labels)

    monkeypatch.setattr(evaluation, "fit_model", observed_fit)
    mask = groups != "5"
    select_candidate("logistic", x[mask], y[mask], groups[mask], 42, 3, 10)
    assert observed and 5 not in observed


def test_experiment_never_overwrites_previous_results(tmp_path):
    (tmp_path / "metrics.json").write_text("previous result")
    with pytest.raises(ValueError, match="empty output"):
        evaluate(tmp_path / "missing-data", tmp_path)
    assert (tmp_path / "metrics.json").read_text() == "previous result"
