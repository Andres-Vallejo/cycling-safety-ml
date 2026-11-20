import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cycling_safety.cnn import fit_cnn, network, probabilities


def test_cnn_scaling_uses_only_supplied_training_windows(tmp_path):
    torch.set_num_threads(1)
    train = np.random.default_rng(5).normal(size=(32, 128, 6)).astype(np.float32)
    labels = np.tile(np.arange(4), 8)
    model, mean, scale, _ = fit_cnn(train, labels, seed=42, epochs=1)
    np.testing.assert_allclose(mean, train.mean(axis=(0, 1), dtype=np.float64), atol=1e-7)
    p = probabilities(model, train[:2] + 10, mean, scale)
    assert p.shape == (2, 4)
    np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-6)
    torch.save(model.state_dict(), tmp_path / "weights.pt")
    restored = network()
    restored.load_state_dict(torch.load(tmp_path / "weights.pt", weights_only=True))
    restored.eval()
    np.testing.assert_allclose(probabilities(restored, train[:2], mean, scale), probabilities(model, train[:2], mean, scale))
