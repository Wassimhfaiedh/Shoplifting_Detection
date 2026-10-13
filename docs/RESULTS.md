# Results

Training artifacts are generated automatically by `src/03_train_lstm.py`.

Expected outputs in `pose_dataset/training_results/`:

- `loss_curve.png`
- `accuracy_curve.png`
- `precision_recall_f1_curve.png`
- `confusion_matrix.png`
- `roc_curve.png`
- `precision_recall_curve.png`
- `training_history.csv`
- `final_metrics.csv`
- `metrics_summary.txt`

> Numerical results are intentionally not hard-coded here because no completed training log/metric file was supplied with the repository package. After training, copy the measured values from `metrics_summary.txt` into the main README if you want the GitHub page to show fixed numbers.
