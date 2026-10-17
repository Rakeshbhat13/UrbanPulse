"""UMN clip annotations: boundaries, abnormal onsets, and frame-level labels.

This is the single source of truth for clip annotations. The data comes from
the notebook's visual review (cells 43, 66, 69, 79) and is frozen here for
reproducibility. Change the ANNOTATION_VERSION if you update any row.
"""

import csv
import io
import numpy as np
import pandas as pd

ANNOTATION_VERSION = "1.0"

# Each tuple: (clip_id, start_frame, end_frame, scene,
#               abnormal_onset_frame, onset_type, confidence, notes)
# onset_frame is int or None (unresolved).
_RAW_ANNOTATIONS = [
    (1,  0,    624,  "Lawn-1",   496,  "exact",       "high",   "Panic/run starts at 496; red overlay at 553."),
    (2,  625,  1452, "Lawn-1",   1330, "provisional", "medium", "Onset matches red overlay; needs independent visual confirmation."),
    (3,  1453, 1985, "Indoor-2", 1830, "provisional", "medium", "Abnormal begins before full red overlay."),
    (4,  1986, 2684, "Indoor-2", 2610, "provisional", "medium", "Panic onset at 2610."),
    (5,  2685, 3428, "Indoor-2", None, "unresolved",  "low",    "Frames 3150-3230 reviewed; no clear crowd-wide onset isolated."),
    (6,  3429, 4034, "Indoor-2", None, "unresolved",  "low",    "Frames 3900-3970 reviewed; onset not isolated reliably."),
    (7,  4035, 4928, "Indoor-2", 4780, "provisional", "medium", "Onset range 4775-4805; using 4780 as midpoint."),
    (8,  4929, 5596, "Indoor-2", 5408, "provisional", "medium", "Onset range 5390-5425; using 5408 as midpoint. End 5596."),
    (9,  5597, 6234, "Plaza-3",  6168, "provisional", "medium", "Onset range 6160-6175; using 6168 as midpoint."),
    (10, 6235, 6912, "Plaza-3",  6848, "provisional", "medium", "Dispersal around 6845-6850; using 6848."),
    (11, 6913, 7738, "Plaza-3",  7663, "provisional", "medium", "Dispersal around 7660-7665; using 7663."),
]

COLUMNS = [
    "clip_id", "start_frame", "end_frame", "scene",
    "abnormal_onset_frame", "onset_type", "confidence", "notes",
]

TOTAL_FRAMES = 7739  # 0-indexed: frames 0..7738


def get_annotations() -> pd.DataFrame:
    """Return the annotation table as a DataFrame."""
    return pd.DataFrame(_RAW_ANNOTATIONS, columns=COLUMNS)


def save_annotations_csv(path: str) -> None:
    """Write the annotation CSV for version-controlled archival."""
    df = get_annotations()
    df.to_csv(path, index=False)


def get_clip_starts() -> list[int]:
    """Return the list of clip start frames (length 11)."""
    return [row[1] for row in _RAW_ANNOTATIONS]


def assign_clip_ids(total_frames: int = TOTAL_FRAMES) -> np.ndarray:
    """Return an array mapping each frame index to its clip_id (1-based)."""
    clip_ids = np.zeros(total_frames, dtype=int)
    starts = get_clip_starts()
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else total_frames
        clip_ids[start:end] = i + 1
    return clip_ids


def build_frame_labels(
    total_frames: int = TOTAL_FRAMES,
    exclude_unresolved: bool = False,
) -> np.ndarray:
    """Return per-frame binary labels (0=normal, 1=abnormal).

    Unresolved clips (onset_frame is None) get label 0 everywhere unless
    exclude_unresolved is True, in which case those frames get label -1
    so callers can mask them out.
    """
    labels = np.zeros(total_frames, dtype=np.int64)
    for row in _RAW_ANNOTATIONS:
        clip_id, start, end, _, onset, onset_type, _, _ = row
        if onset is None:
            if exclude_unresolved:
                labels[start : end + 1] = -1
            # else leave as 0
        else:
            labels[onset : end + 1] = 1
    return labels


def validate_annotations() -> list[str]:
    """Check annotation consistency. Returns list of problems (empty = OK)."""
    problems = []
    starts = get_clip_starts()
    df = get_annotations()

    # No gaps or overlaps
    for i in range(len(starts) - 1):
        end_i = int(df.iloc[i]["end_frame"])
        start_next = int(df.iloc[i + 1]["start_frame"])
        if end_i + 1 != start_next:
            problems.append(
                f"Gap/overlap between clip {i+1} end={end_i} and clip {i+2} start={start_next}"
            )

    # Last clip covers remaining frames
    last_end = int(df.iloc[-1]["end_frame"])
    if last_end != TOTAL_FRAMES - 1:
        problems.append(f"Last clip end_frame={last_end}, expected {TOTAL_FRAMES - 1}")

    # Onsets within clip bounds
    for _, row in df.iterrows():
        onset = row["abnormal_onset_frame"]
        if pd.notna(onset) and onset is not None:
            onset = int(onset)
            if not (int(row["start_frame"]) <= onset <= int(row["end_frame"])):
                problems.append(
                    f"Clip {row['clip_id']}: onset {onset} outside [{row['start_frame']}, {row['end_frame']}]"
                )
    return problems
