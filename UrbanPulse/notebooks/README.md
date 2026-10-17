# UrbanPulse Notebooks

The exploratory work in `Untitled0.ipynb` has been modularized into `src/` to ensure reproducible pipelines:

- `01_umndata_inspection`: Covered by `src/annotations.py` and `data/umn_annotations.csv`.
- `02_extract_tracking_features`: Covered by `src/video_features.py` (YOLO11n + ByteTrack -> crowd features).
- `03_label_and_validate`: Covered by `src/annotations.py` and `src/sequences.py` (leakage-safe split).
- `04_train_evaluate_lstm`: Covered by `src/train.py` and `src/evaluate.py`.
- `05_demo_inference`: Covered by `src/inference.py` and `app/app.py`.

Use the original `../Untitled0.ipynb` on Google Colab if GPU execution is needed for YOLO tracking and LSTM training.
