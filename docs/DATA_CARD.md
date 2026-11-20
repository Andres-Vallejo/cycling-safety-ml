# Data card

## Origin and scope

The data comes from the original CyclingSafety academic team's controlled Sensor Logger recordings, supplied locally by Juan Andrés as a two-part ZIP export. This continuation uses all 11 controlled recordings with complete accelerometer, gyroscope and annotation streams. It does not ingest GPS or combine the collection with a third-party Kaggle dataset.

The original recordings and labels retain the original team's authorship. They are not newly collected measurements by this continuation.

## Retained label counts

| Label | Meaning in this benchmark | Windows |
|---|---|---:|
| background | No annotation in or near the retained window | 8,829 |
| pothole | Window contains a `Bache` press | 288 |
| evasive_maneuver | Window contains an `Esquivada` press | 169 |
| emergency_braking | Window contains a `Freno de emergencia` press | 167 |

348 annotation presses are present before windowing. Several windows may refer to one physical maneuver, and several presses may also refer to the same event. Window counts must not be reported as independent incidents.

## Quality decisions

10,840 windows are candidates before filtering. The pipeline rejects 750 for remaining sensor gaps, 20 for conflicting event classes and 617 unannotated windows within the annotation guard. The retained total is 9,453.

These decisions are fixed before model evaluation. Per-recording counts and consumed-file hashes appear in each experiment's `data_manifest.json`. Natural recordings and historical synthetic augmentations are excluded entirely.

## Limitations

- Button presses have reaction delay; the true start/end of a physical event is not available.
- The guard interval is a modeling assumption and does not establish clean negative labels.
- Sensor axes follow the recorded phone frame. Mounting, drift, calibration and phone placement have not been independently verified.
- The collection is small and imbalanced. Windows from a single ride are correlated. Metadata from all 11 recordings lists **one unique logged device ID and two collection days**, summarized without identifiers in [collection_summary.json](../provenance/collection_summary.json).
- Different recording IDs can share a rider, day, device or location. Grouping by recording cannot establish generalization across those factors.
- Interpolation and nearest-bin synchronization are offline processing choices, not validated embedded acquisition.

## Reproduction and privacy

The derived NPZ contains numeric features, raw IMU windows, opaque recording IDs, relative window starts and labels. It excludes GPS, device identifiers and original folder names. Local data is ignored by Git. The source file hashes allow verification without redistributing the recordings.

The software's MIT license is retained; a software license alone is not evidence of data redistribution permission. Raw archives and full derived tensors are excluded from the prepared publication bundle. Reports and tiny example feature vectors document the experiment without including complete routes.

The historical feature-only importer is retained as a provenance/audit utility. Its imported dataset differs from this rebuilt benchmark and should not be mixed with the new raw-signal results.
