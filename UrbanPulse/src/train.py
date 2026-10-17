"""LSTM training — replicates notebook cell 47 as a reusable function."""

import json
import os

import joblib
import numpy as np


def build_model(sequence_length: int, n_features: int):
    """Compact LSTM binary classifier matching the notebook architecture."""
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
    from tensorflow.keras.metrics import AUC, Precision, Recall

    tf.keras.utils.set_random_seed(42)

    model = Sequential([
        Input(shape=(sequence_length, n_features)),
        LSTM(32, dropout=0.2),
        Dense(16, activation="relu"),
        Dropout(0.2),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            Precision(name="precision"),
            Recall(name="recall"),
            AUC(name="roc_auc"),
            AUC(curve="PR", name="pr_auc"),
        ],
    )
    return model


def train(
    splits: dict,
    output_dir: str = "models",
    epochs: int = 50,
    batch_size: int = 32,
    patience: int = 8,
) -> dict:
    """Train the LSTM and save model/scaler/config. Returns training history."""
    from tensorflow.keras.callbacks import EarlyStopping

    os.makedirs(output_dir, exist_ok=True)

    X_train, y_train = splits["X_train"], splits["y_train"]
    X_val, y_val = splits["X_val"], splits["y_val"]
    n_features = X_train.shape[2]

    # Class weighting for imbalanced labels
    counts = np.bincount(y_train, minlength=2)
    class_weight = {
        0: len(y_train) / (2 * counts[0]),
        1: len(y_train) / (2 * counts[1]),
    }

    model = build_model(splits["sequence_length"], n_features)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=[
            EarlyStopping(
                monitor="val_pr_auc", mode="max",
                patience=patience, restore_best_weights=True,
            ),
        ],
        verbose=1,
    )

    # Save artifacts
    model_path = os.path.join(output_dir, "urbanpulse_lstm.keras")
    scaler_path = os.path.join(output_dir, "urbanpulse_scaler.joblib")
    config_path = os.path.join(output_dir, "urbanpulse_config.json")

    model.save(model_path)
    joblib.dump(splits["scaler"], scaler_path)

    from . import annotations
    config = {
        "feature_columns": splits["feature_cols"],
        "sequence_length": splits["sequence_length"],
        "stride": splits["stride"],
        "annotation_version": annotations.ANNOTATION_VERSION,
        "fps": 30,
        "clip_starts": annotations.get_clip_starts(),
        "class_weight": {str(k): v for k, v in class_weight.items()},
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Saved: {model_path}, {scaler_path}, {config_path}")
    return {"model": model, "history": history.history, "config": config}
