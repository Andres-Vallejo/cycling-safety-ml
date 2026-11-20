# Evaluation methodology

## Research question

How well can a fixed family of classifiers recognize the team's annotated cycling events on different recording IDs from the available collection?

The evaluation unit is a recording. Windows overlap by 50% within recordings; individual windows are not independent events. No claim of rider-independent or device-independent performance follows from the split.

## Raw processing

1. Read `Accelerometer.csv`, `Gyroscope.csv` and `Annotation.csv` for each controlled recording from all parts of one ZIP export. Hash every consumed file and the archive parts.
2. Use named x/y/z columns, regardless of their CSV order. The accelerometer source is the upstream linear-acceleration stream. Expected units are m/s² and rad/s in the logged device frame; no new axis calibration or rotation is inferred.
3. Average observations in nearest 20 ms bins. Restrict both sensors to their overlapping time range. Interpolate only complete internal gaps of at most three samples (60 ms). Do not extrapolate or partially bridge longer gaps.
4. Create 128-sample windows at stride 64. Drop any candidate containing remaining missing sensor values.
5. Use the recorded annotation press times unchanged. A window with annotations of different event types is excluded. If it contains one event type, assign that label. Multiple presses of the same type do not create extra output classes.
6. A window with no annotation is retained as `background` only if no annotation falls inside the window or within a 1.28-second guard around it. This reduces ambiguity but does not verify the absence of an event.
7. Extract the 72 upstream statistical descriptors independently per window. Use the synchronized raw windows directly for the CNN. Exclude severity scores, GPS, recording identities and labels from model features.

The binning and interpolation are offline operations. They may use a nearby future sensor observation within a recording; the resulting pipeline is not advertised as a causal real-time ESP32 implementation.

## Splits and model selection

The outer protocol is `GroupKFold(n_splits=5, shuffle=True, random_state=42)`. All windows of the same complete recording name map to one stable opaque group ID. Recording IDs are never truncated before grouping. The MLP and CNN use the same dataset row order and outer seed, allowing their held-out assignments to be compared directly.

Inside each outer training set, logistic regression and MLP hyperparameters are selected by three recording-group folds. The score is the unweighted mean of inner-fold four-class macro-F1 values. A fixed candidate order resolves ties. Each fit recalculates balanced class weights from its training labels and fits its scaler only on its training features.

The logistic candidates use `C` 0.1 and 1.0. The MLP candidates use hidden layers `[32]` and `[64, 32]`, alpha 0.01 and an iteration budget of 200. MLP early stopping against a random internal window split is disabled. Iteration-budget warnings are saved.

The CNN is a separately specified fixed baseline: two 1D convolution layers with 16 and 32 channels, ReLU, temporal max-pooling, global average pooling and a four-class linear head. It uses 15 fixed epochs, AdamW, learning rate 0.001, balanced cross-entropy and no augmentation. There is no search or epoch selection on outer held-out predictions.

Natural rides with no validated event labels are excluded from the supervised benchmark. All historical augmented copies are excluded. No reversal augmentation is assumed to preserve the physical meaning of braking.

## Metrics and artifacts

The primary metric is pooled out-of-fold macro-F1 with all four labels specified, including zero scores when a class has no predicted support. Accuracy, per-class precision/recall/F1, confusion matrices, fold results and recording-level results are also saved. All original retained windows are evaluated once per model family.

The feature-model report includes a 1,000-replicate recording bootstrap of the pooled OOF macro-F1. Its interval is conditional on these predictions and does not include uncertainty from model refitting, future riders or a new data collection. Eleven recordings give limited statistical precision.

The split is recording-disjoint and addresses the verified historical leakage. It is not an untouched external benchmark: the data and source artifacts were already inspected, and the protocol was designed with knowledge of the project. Future development should use a new collection of rides for an external evaluation.

## Deployment artifacts

After evaluation, the MLP family is tuned again using grouped folds over all available recordings and refitted on all retained windows. It was designated as the demo family before reading OOF results. Its scaler and numeric weights are exported to JSON, and predictions are checked against scikit-learn. No Python pickle is required for serving.

The CNN is also refitted on all windows after its OOF evaluation. Its numeric state dictionary is loaded with `weights_only=True` for the parity check. Neither all-data refit has an independent final-model test score.

## References

- [scikit-learn grouped cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data).
- [scikit-learn leakage and pipelines](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage).
- [Original project and team](../AUTHORS.md).
