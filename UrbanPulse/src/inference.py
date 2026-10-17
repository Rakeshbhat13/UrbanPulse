"""Inference: load saved model/scaler/config and run on new feature data."""

import json
import os

import joblib
import numpy as np
import pandas as pd


def load_artifacts(model_dir: str = "models") -> dict:
    """Load model, scaler, and config from the given directory."""
    import tensorflow as tf

    model = tf.keras.models.load_model(
        os.path.join(model_dir, "urbanpulse_lstm.keras")
    )
    scaler = joblib.load(os.path.join(model_dir, "urbanpulse_scaler.joblib"))
    with open(os.path.join(model_dir, "urbanpulse_config.json")) as f:
        config = json.load(f)

    return {"model": model, "scaler": scaler, "config": config}


def predict_windows(
    features: pd.DataFrame,
    artifacts: dict,
    threshold: float | None = None,
) -> pd.DataFrame:
    """Run sliding-window inference on a feature DataFrame.

    Returns a DataFrame with columns: window_end_frame, probability, predicted.
    """
    config = artifacts["config"]
    scaler = artifacts["scaler"]
    model = artifacts["model"]

    feature_cols = config["feature_columns"]
    seq_len = config["sequence_length"]
    stride = config["stride"]
    if threshold is None:
        threshold = config.get("probability_threshold", 0.5)

    X_raw = features[feature_cols].fillna(0).values.astype(np.float32)
    X_scaled = scaler.transform(X_raw).astype(np.float32)

    windows, end_frames = [], []
    for ws in range(0, len(X_scaled) - seq_len + 1, stride):
        windows.append(X_scaled[ws : ws + seq_len])
        end_frames.append(ws + seq_len - 1)

    if not windows:
        return pd.DataFrame(columns=["window_end_frame", "probability", "predicted"])

    X = np.array(windows)
    probs = model.predict(X, verbose=0).ravel()
    preds = (probs >= threshold).astype(int)

    return pd.DataFrame({
        "window_end_frame": end_frames,
        "probability": probs,
        "predicted": preds,
    })
