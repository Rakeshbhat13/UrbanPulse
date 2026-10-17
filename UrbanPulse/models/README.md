# Model Artifacts

This directory stores the trained model artifacts and configuration.

Expected files:
- `urbanpulse_lstm.keras`: Trained TensorFlow/Keras LSTM binary classification model.
- `urbanpulse_scaler.joblib`: Fitted `StandardScaler` on training clips (to avoid data leakage).
- `urbanpulse_config.json`: Feature list, sequence length, stride, operating threshold, and annotation version.

## Training
To train the model on GPU (e.g. Google Colab) or locally with TensorFlow:
```bash
python -m src.train
```

Once trained, artifacts saved here are automatically loaded by `src.inference` and the Streamlit dashboard in `app/app.py`.
