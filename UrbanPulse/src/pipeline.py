"""End-to-end video processing: YOLO11n + ByteTrack → features → LSTM scoring.

Connects the existing modules (video_features, inference, alert) into a single
pipeline callable from the Streamlit dashboard.
"""

import os
import tempfile
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class ProcessingResult:
    """Container for all outputs of a video analysis run."""
    annotated_video_path: str = ""
    telemetry: pd.DataFrame = field(default_factory=pd.DataFrame)
    events: list = field(default_factory=list)
    statuses: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


def process_video(
    video_path: str,
    output_dir: str = "outputs",
    threshold: float = 0.50,
    raise_after: int = 3,
    clear_after: int = 3,
    frame_skip: int = 1,
    progress_callback=None,
) -> ProcessingResult:
    """Run the full YOLO → ByteTrack → features → LSTM → alerts pipeline.

    Args:
        video_path: Path to input video file.
        output_dir: Where to write the annotated output video.
        threshold: Anomaly probability threshold for alert system.
        raise_after: Consecutive windows above threshold to raise alert.
        clear_after: Consecutive windows below threshold to clear alert.
        frame_skip: Process every Nth frame (1 = all frames).
        progress_callback: Optional callable(fraction: float) for progress updates.

    Returns:
        ProcessingResult with annotated video path, telemetry DataFrame, and events.
    """
    import cv2
    from ultralytics import YOLO

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Output video writer
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join(output_dir, f"{base_name}_analyzed.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    # Load YOLO model — uses yolo11n, person class only (class 0)
    model = YOLO("yolo11n.pt")

    # Per-frame tracking data
    frame_records = []      # raw per-person detections
    frame_features = []     # aggregated per-frame features
    prev_centers = {}       # track_id -> (cx, cy) for speed computation

    frame_idx = 0
    processed_count = 0
    import time
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip != 0:
            writer.write(frame)
            frame_idx += 1
            continue

        # YOLO + ByteTrack inference (person class = 0)
        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0],
            verbose=False,
            conf=0.3,
        )

        boxes = results[0].boxes
        person_count = 0
        speeds = []
        dir_sins = []
        dir_coss = []
        current_centers = {}
        annotated_frame = frame.copy()

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                # Extract bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                conf = float(box.conf[0])
                track_id = int(box.id[0]) if box.id is not None else -1

                person_count += 1
                current_centers[track_id] = (cx, cy)

                # Compute speed if we have previous position for this track
                if track_id in prev_centers and track_id >= 0:
                    pcx, pcy = prev_centers[track_id]
                    dx = cx - pcx
                    dy = cy - pcy
                    dist = np.sqrt(dx**2 + dy**2)
                    speed = dist * fps / frame_skip  # pixels/sec
                    speeds.append(speed)
                    direction = np.arctan2(dy, dx)
                    dir_sins.append(np.sin(direction))
                    dir_coss.append(np.cos(direction))

                # Draw bounding box and label
                color = (0, 255, 0)  # green by default
                label = f"ID:{track_id}" if track_id >= 0 else "Person"
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    annotated_frame, label,
                    (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
                )

        prev_centers = current_centers

        # Compute frame-level aggregated features
        avg_speed = float(np.mean(speeds)) if speeds else 0.0
        max_speed = float(np.max(speeds)) if speeds else 0.0
        speed_std = float(np.std(speeds)) if len(speeds) > 1 else 0.0
        median_speed = float(np.median(speeds)) if speeds else 0.0
        speed_iqr = float(np.percentile(speeds, 75) - np.percentile(speeds, 25)) if len(speeds) > 1 else 0.0
        flow_consistency = float(np.sqrt(np.mean(dir_sins)**2 + np.mean(dir_coss)**2)) if dir_sins else 0.0

        frame_features.append({
            "frame": frame_idx,
            "timestamp_sec": frame_idx / fps,
            "person_count": person_count,
            "median_speed": round(median_speed, 2),
            "avg_speed": round(avg_speed, 2),
            "max_speed": round(max_speed, 2),
            "speed_std": round(speed_std, 2),
            "speed_iqr": round(speed_iqr, 2),
            "flow_consistency": round(flow_consistency, 4),
        })

        # Overlay frame info
        cv2.putText(
            annotated_frame,
            f"Frame: {frame_idx} | People: {person_count} | Avg Speed: {avg_speed:.1f} px/s",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
        )

        writer.write(annotated_frame)
        processed_count += 1
        frame_idx += 1

        if progress_callback and total_frames > 0:
            progress_callback(frame_idx / total_frames)

    cap.release()
    writer.release()

    elapsed = time.time() - start_time
    features_df = pd.DataFrame(frame_features)

    # Add count_delta features
    features_df["count_delta"] = features_df["person_count"].diff().fillna(0)
    features_df["count_delta_abs_3"] = (
        features_df["count_delta"].abs().rolling(3, min_periods=1).mean()
    )

    # Run LSTM inference if model artifacts exist
    anomaly_scores = _run_lstm_scoring(features_df, threshold)
    if anomaly_scores is not None:
        features_df["anomaly_probability"] = anomaly_scores
    else:
        # Fallback: heuristic anomaly score based on speed deviation
        features_df["anomaly_probability"] = _heuristic_anomaly_score(features_df)

    # Run alert pipeline
    from . import alert
    statuses, events = alert.run_alert_pipeline(
        frames=features_df["frame"].tolist(),
        probabilities=features_df["anomaly_probability"].tolist(),
        threshold=threshold,
        raise_after=raise_after,
        clear_after=clear_after,
    )
    features_df["alert_status"] = statuses

    # Re-annotate video with alert status overlays
    _overlay_alerts_on_video(out_path, features_df, fps, width, height, frame_skip)

    result = ProcessingResult(
        annotated_video_path=out_path,
        telemetry=features_df,
        events=events,
        statuses=statuses,
        metadata={
            "fps": fps,
            "total_frames": total_frames,
            "processed_frames": processed_count,
            "width": width,
            "height": height,
            "duration_sec": total_frames / fps,
            "inference_time_sec": round(elapsed, 2),
            "frame_skip": frame_skip,
            "events_flagged": len(events),
        },
    )
    return result


def _run_lstm_scoring(features_df: pd.DataFrame, threshold: float):
    """Attempt LSTM inference using saved model artifacts. Returns None if unavailable."""
    try:
        from . import inference
        model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
        model_path = os.path.join(model_dir, "urbanpulse_lstm.keras")
        if not os.path.exists(model_path):
            return None

        artifacts = inference.load_artifacts(model_dir)
        config = artifacts["config"]
        feature_cols = config["feature_columns"]

        # Ensure all required columns exist
        for col in feature_cols:
            if col not in features_df.columns:
                features_df[col] = 0.0

        result_df = inference.predict_windows(features_df, artifacts, threshold)
        if result_df.empty:
            return None

        # Map window-level probabilities back to frames
        scores = np.zeros(len(features_df))
        for _, row in result_df.iterrows():
            end_frame = int(row["window_end_frame"])
            if end_frame < len(scores):
                scores[end_frame] = row["probability"]

        # Forward-fill: propagate last known score to unscored frames
        for i in range(1, len(scores)):
            if scores[i] == 0.0 and scores[i - 1] > 0.0:
                scores[i] = scores[i - 1] * 0.95  # slight decay

        return scores
    except Exception:
        return None


def _heuristic_anomaly_score(features_df: pd.DataFrame) -> np.ndarray:
    """Simple rule-based anomaly score when LSTM model is unavailable.

    Uses z-score of speed and count changes as a proxy. Not a substitute for
    the trained LSTM but provides meaningful demo output.
    """
    scores = np.zeros(len(features_df))
    if features_df.empty:
        return scores

    speed = features_df["median_speed"].values.astype(float)
    count = features_df["person_count"].values.astype(float)

    # Z-score based scoring
    speed_mean = np.mean(speed) if np.mean(speed) > 0 else 1.0
    speed_std = np.std(speed) if np.std(speed) > 0 else 1.0
    speed_z = np.abs(speed - speed_mean) / speed_std

    count_mean = np.mean(count) if np.mean(count) > 0 else 1.0
    count_std = np.std(count) if np.std(count) > 0 else 1.0
    count_delta = np.abs(np.diff(count, prepend=count[0]))
    count_z = count_delta / (count_std + 1e-6)

    # Combined score → sigmoid-like mapping to [0, 1]
    combined = 0.6 * speed_z + 0.4 * count_z
    scores = 1.0 / (1.0 + np.exp(-combined + 2))  # shifted sigmoid

    return np.round(scores, 4)


def _overlay_alerts_on_video(
    video_path: str,
    telemetry: pd.DataFrame,
    fps: float,
    width: int,
    height: int,
    frame_skip: int,
):
    """Re-read the annotated video and add alert status banners.

    Overwrites the video in-place with alert overlays.
    """
    import cv2

    temp_path = video_path + ".tmp.mp4"
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))

    frame_idx = 0
    alert_lookup = dict(zip(telemetry["frame"].values, telemetry["alert_status"].values))
    prob_lookup = dict(zip(telemetry["frame"].values, telemetry["anomaly_probability"].values))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        status = alert_lookup.get(frame_idx, "NORMAL")
        prob = prob_lookup.get(frame_idx, 0.0)

        if status == "ABNORMAL_MOVEMENT":
            # Red translucent overlay at top
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (width, 50), (0, 0, 200), -1)
            frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
            cv2.putText(
                frame, f"UNUSUAL MOVEMENT DETECTED | Score: {prob:.2f}",
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
            )
            # Red border
            cv2.rectangle(frame, (0, 0), (width - 1, height - 1), (0, 0, 255), 3)
        else:
            # Green status bar
            cv2.putText(
                frame, f"NORMAL | Score: {prob:.2f}",
                (width - 280, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2,
            )

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    # Replace original with overlaid version
    os.replace(temp_path, video_path)
