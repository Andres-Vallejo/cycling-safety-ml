# Validation record

Validated locally on Windows 11 with Python 3.12.10.

- **18 automated tests passed**, including the rebuilt sensor pipeline, recording/nested split guards, training-only normalization, numeric model parity, real deployment model request, CLI and API validation.
- One dependency deprecation warning was emitted by Starlette's test client using AnyIO. It did not fail the tests.
- Five outer folds produced exactly one held-out prediction per retained window and model family. All 9,453 windows and 11 recording IDs are covered.
- MLP and CNN have identical held-out window/recording assignments. Their data hashes match.
- Every inner training/validation recording set is disjoint from its outer test recordings.
- Accuracy, macro-F1 and confusion matrices were recomputed from saved OOF CSVs and match the reported values.
- Portable JSON MLP outputs were checked against scikit-learn for the full dataset. The refitted CNN checkpoint was reloaded using `weights_only=True` and checked for prediction parity.
- All **166 tracked academic files** match the preserved snapshot hashes. The original repository's working tree remains clean.
- A Python wheel was built successfully from the new package.
- Docker was unavailable on this host. The supplied Docker image was not built; the service itself was tested through FastAPI's test client.
- [GitHub Actions passed on Ubuntu and Windows](https://github.com/Andres-Vallejo/cycling-safety-ml/actions/runs/44801308623) on November 20, 2026: 17 core tests passed and one optional CNN test was skipped on each runner. The CNN-specific test was executed locally as part of the 18-test suite.
- No public service deployment or physical ESP32 test was performed.

See [machine-readable result audit](validation.json), [feature-model training environment](grouped-mlp/environment.json) and [CNN training environment](grouped-cnn/environment.json).

Re-run evidence checks from the repository root:

```bash
python scripts/verify_results.py
python -m pytest -q
```

The optional `--academic-source /path/to/CyclingSafety` flag also checks the original snapshot file hashes. Testing the new software does not validate physical event labels or establish generalization to new riders/devices.
