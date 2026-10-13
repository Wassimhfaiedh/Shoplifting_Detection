# Pose dataset output

This folder is populated by the project pipeline.

Expected generated files include:

- `X.npy` — temporal pose sequences.
- `y.npy` — class labels (`0 = Normal`, `1 = Shoplifting`).
- `shoplifting_lstm.pt` — trained LSTM checkpoint.
- `training_results/` — plots and metric summaries produced during training.

Large generated files are ignored by Git by default.
