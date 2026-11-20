# Independent continuation and validation record

## Original preserved

The academic repository at reference commit `546d11993ffdab3170a02749f67309ae469454f3` is preserved separately. This workspace records the hashes of all 166 tracked files and keeps copies of its overview, credits and license in `provenance/`. The downloaded ZIP files are read only.

## New implementation

- Modular Python package with an explicit 72-feature and four-label contract.
- Direct raw Sensor Logger archive reader, named axes, explicit 50 Hz synchronization and short-gap interpolation.
- Rebuilt window labels with mixed-event exclusion and a background annotation guard.
- Recording-disjoint outer evaluation with grouped inner selection for feature models.
- Fresh logistic/MLP training and a fixed fresh 1D CNN baseline.
- Training-only normalization and class weights; no historic synthetic copies.
- Out-of-fold prediction files, fold manifests, environment records and dataset hashes.
- Numeric JSON MLP inference, feature-CSV CLI and validated local API.
- Tests, CI workflow, Docker packaging, example inputs and data/model cards.

These changes are the scope of Juan Andrés's independent continuation. They do not change the authorship of the original recordings, idea, notebook or maps.

## Validation status

The executed experiment reports and validation summary document actual results. Python tests, model export parity, source preservation and local inference are checked in this workspace. The CI workflow is prepared for a future repository publication; it has not run on GitHub yet.

Docker was not installed on the host, so the image was not built or tested. The service itself was exercised through FastAPI's test client. No physical ESP32 validation, public hosting or external independent dataset evaluation is claimed.

## Next development

1. Review labeling with true event intervals and rider reaction delay.
2. Add a new collection held out by rider and session, including verified negative segments.
3. Test a causal preprocessing implementation before streaming inference.
4. Measure latency and memory on the intended hardware using the new model.
5. Validate probability calibration and alert policies only if a field-use case is pursued.
