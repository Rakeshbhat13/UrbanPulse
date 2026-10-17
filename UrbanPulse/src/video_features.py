"""Extract crowd-level features from YOLO+ByteTrack person-tracking output.

Replicates the notebook's feature pipeline (cells 25-32) as importable
functions, with fixes for cross-clip track differencing and direction
averaging.
"""

import numpy as np
import pandas as pd

from . import annotations


def compute_person_movement(df: pd.DataFrame, fps: float) -> pd.DataFrame:
    """Add displacement, speed (px/s), and direction columns per person per clip.

    Expects columns: frame, person_id, center_x, center_y, clip_id.
    Only differences consecutive frames within the same (clip_id, person_id)
    to avoid cross-clip artifacts.
    """
    df = df.sort_values(["clip_id", "person_id", "frame"]).copy()
    grp = df.groupby(["clip_id", "person_id"])
    df["dx"] = grp["center_x"].diff()
    df["dy"] = grp["center_y"].diff()
    df["frame_gap"] = grp["frame"].diff()

    # Only compute speed for consecutive-frame observations (gap == 1)
    valid = df["frame_gap"] == 1
    df["distance"] = np.where(valid, np.sqrt(df["dx"] ** 2 + df["dy"] ** 2), np.nan)
    time_per_frame = 1.0 / fps
    df["speed"] = df["distance"] / time_per_frame

    # Direction: keep as sin/cos components to avoid circular-mean issues
    direction_rad = np.arctan2(df["dy"], df["dx"])
    df["dir_sin"] = np.where(valid, np.sin(direction_rad), np.nan)
    df["dir_cos"] = np.where(valid, np.cos(direction_rad), np.nan)

    return df


def aggregate_frame_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate person-level data to one row per frame.

    Input must have columns from compute_person_movement plus 'clip_id'.
    """
    # Only rows with valid speed for movement stats
    movement = df.dropna(subset=["speed"])

    speed_agg = movement.groupby("frame").agg(
        median_speed=("speed", "median"),
        speed_iqr=("speed", lambda s: s.quantile(0.75) - s.quantile(0.25)),
        speed_std=("speed", "std"),
        max_speed=("speed", "max"),
    )
    speed_agg["speed_std"] = speed_agg["speed_std"].fillna(0)

    # Circular-safe direction: mean sin/cos → resultant length as flow consistency
    dir_agg = movement.groupby("frame").agg(
        mean_dir_sin=("dir_sin", "mean"),
        mean_dir_cos=("dir_cos", "mean"),
    )
    dir_agg["flow_consistency"] = np.sqrt(
        dir_agg["mean_dir_sin"] ** 2 + dir_agg["mean_dir_cos"] ** 2
    )

    # Count: all detections, not just those with valid speed
    count_agg = df.groupby("frame").agg(
        person_count=("person_id", "nunique"),
    )

    features = count_agg.join(speed_agg, how="left").join(dir_agg, how="left")
    features = features.reset_index()
    return features


def reindex_to_all_frames(
    features: pd.DataFrame,
    total_frames: int = annotations.TOTAL_FRAMES,
) -> pd.DataFrame:
    """Ensure one row per frame, filling gaps with zero count and NaN movement."""
    full = pd.DataFrame({"frame": range(total_frames)})
    merged = full.merge(features, on="frame", how="left")
    merged["person_count"] = merged["person_count"].fillna(0).astype(int)
    return merged


def add_count_features(features: pd.DataFrame) -> pd.DataFrame:
    """Add count_delta and rolling count change."""
    features = features.copy()
    features["count_delta"] = features["person_count"].diff().fillna(0)
    features["count_delta_abs_3"] = (
        features["count_delta"].abs().rolling(3, min_periods=1).mean()
    )
    return features


FEATURE_COLUMNS = [
    "person_count",
    "median_speed",
    "speed_iqr",
    "speed_std",
    "max_speed",
    "flow_consistency",
    "count_delta",
    "count_delta_abs_3",
]
