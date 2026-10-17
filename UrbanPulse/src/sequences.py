"""Windowed sequence creation for temporal classification.

Replicates notebook cells 45-46 with leakage-safe clip-based splitting.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from . import annotations
from .video_features import FEATURE_COLUMNS


DEFAULT_SEQUENCE_LENGTH = 30  # 1 second at 30 FPS
DEFAULT_STRIDE = 5

# Split by clip to prevent leakage; one clip per scene held out for test.
DEFAULT_TRAIN_CLIPS = [1, 3, 5, 7, 8, 11]
DEFAULT_VAL_CLIPS = [4, 9]
DEFAULT_TEST_CLIPS = [2, 6, 10]


def make_sequences(
    X_scaled: np.ndarray,
    labels: np.ndarray,
    clip_ids: np.ndarray,
    selected_clips: list[int],
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    stride: int = DEFAULT_STRIDE,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Build fixed-length windows from selected clips.

    Returns (X, y, metadata) where metadata has clip_id and end_frame per window.
    Skips frames with label == -1 (unresolved clips when excluded).
    """
    clip_starts = annotations.get_clip_starts()
    total = len(X_scaled)
    X_list, y_list, meta = [], [], []

    for clip_id in selected_clips:
        start = clip_starts[clip_id - 1]
        end = clip_starts[clip_id] if clip_id < len(clip_starts) else total

        for ws in range(start, end - sequence_length + 1, stride):
            we = ws + sequence_length
            window_labels = labels[ws:we]
            # Skip windows touching unresolved frames
            if np.any(window_labels == -1):
                continue
            X_list.append(X_scaled[ws:we])
            y_list.append(int(labels[we - 1]))  # label of last frame
            meta.append({"clip_id": clip_id, "end_frame": we - 1})

    return np.array(X_list), np.array(y_list), meta


def prepare_splits(
    features: pd.DataFrame,
    feature_cols: list[str] = FEATURE_COLUMNS,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    stride: int = DEFAULT_STRIDE,
    exclude_unresolved: bool = True,
    train_clips: list[int] = DEFAULT_TRAIN_CLIPS,
    val_clips: list[int] = DEFAULT_VAL_CLIPS,
    test_clips: list[int] = DEFAULT_TEST_CLIPS,
) -> dict:
    """Full pipeline: scale, window, split. Returns dict with arrays and scaler."""
    clip_ids = annotations.assign_clip_ids(len(features))
    labels = annotations.build_frame_labels(len(features), exclude_unresolved)

    # Fill NaN features with 0 for frames with no detections
    X_raw = features[feature_cols].fillna(0).values.astype(np.float32)

    # Fit scaler on training clips only
    train_mask = np.isin(clip_ids, train_clips)
    scaler = StandardScaler()
    scaler.fit(X_raw[train_mask])
    X_scaled = scaler.transform(X_raw).astype(np.float32)

    X_train, y_train, meta_train = make_sequences(
        X_scaled, labels, clip_ids, train_clips, sequence_length, stride
    )
    X_val, y_val, meta_val = make_sequences(
        X_scaled, labels, clip_ids, val_clips, sequence_length, stride
    )
    X_test, y_test, meta_test = make_sequences(
        X_scaled, labels, clip_ids, test_clips, sequence_length, stride
    )

    return {
        "X_train": X_train, "y_train": y_train, "meta_train": meta_train,
        "X_val": X_val, "y_val": y_val, "meta_val": meta_val,
        "X_test": X_test, "y_test": y_test, "meta_test": meta_test,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "sequence_length": sequence_length,
        "stride": stride,
    }
