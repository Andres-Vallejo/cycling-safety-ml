# Model card

## Intended use

Research and portfolio demonstration of reproducible IMU-window classification. The project compares feature-based baselines and a small temporal CNN using held-out recording IDs. It is not intended to trigger emergency actions, measure crash incidence or rank road safety.

## Models

- Majority-class predictor: a reference for the heavily imbalanced label distribution.
- Logistic regression: standardized 72-feature baseline with balanced training weights.
- MLP: standardized 72-feature neural classifier, with grouped internal selection between two small architectures.
- 1D CNN: fixed convolutional architecture on synchronized 128 × 6 raw windows, with training-only channel standardization.

All new models are trained from scratch in this continuation. The old Keras, TFLite and ESP32 model files are not loaded. The MLP JSON model is the demo service artifact; the CNN refit is provided for research and future extension.

## Evaluation

See [feature models](../reports/grouped-mlp/REPORT.md) and [CNN](../reports/grouped-cnn/REPORT.md). The outer group assignments and raw rebuilt data checksum must agree across experiments. F1 and accuracy are window-level OOF metrics, not event detection or incident statistics.

Probabilities have not been calibrated. No confidence threshold has been validated for deployment. A high output score is not evidence of a verified real-world hazard.

The final artifacts are refits using all 11 recordings, following evaluation of their training procedures. They do not have independent held-out final-model scores. The examples supplied with the API come from the training collection and test interface behavior only.

## Operational limits

The API accepts exactly the 72 named features and supports batches of up to 1,000 windows. It does not perform live sensor synchronization, orientation correction, device calibration or geographic analysis. The feature extractor expects already synchronized 50 Hz linear acceleration and angular velocity in the logged axis convention.

There is no verified ESP32 port for the new weights, no new TFLite quantization and no on-device latency or energy benchmark. An all-data CNN checkpoint can be loaded using PyTorch `weights_only=True` with the documented architecture. The MLP service reads numeric JSON rather than executable pickle.

New phone mounts, riders, surfaces and sampling conditions can produce distribution shift. Future field evaluation should establish recording/session/rider boundaries, true event intervals, independent negative labels and performance on new collections.
