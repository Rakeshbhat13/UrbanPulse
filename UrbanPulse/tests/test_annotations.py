"""Tests for annotation parsing and validation."""

import numpy as np
import pytest

from src import annotations


def test_annotations_coverage_and_continuity():
    """Ensure all 11 clips have no gaps, overlaps, or boundary issues."""
    problems = annotations.validate_annotations()
    assert problems == [], f"Annotation validation failed: {problems}"


def test_clip_count_and_total_frames():
    """Verify there are exactly 11 clips spanning 7739 frames."""
    df = annotations.get_annotations()
    assert len(df) == 11
    assert df.iloc[0]["start_frame"] == 0
    assert df.iloc[-1]["end_frame"] == annotations.TOTAL_FRAMES - 1


def test_clip_ids_assignment():
    """Verify each frame is mapped to the right 1-based clip ID."""
    clip_ids = annotations.assign_clip_ids()
    assert len(clip_ids) == annotations.TOTAL_FRAMES
    assert clip_ids[0] == 1
    assert clip_ids[624] == 1
    assert clip_ids[625] == 2
    assert clip_ids[-1] == 11


def test_frame_labels_resolved_and_unresolved():
    """Verify normal vs abnormal labels and unresolved masking."""
    # When unresolved excluded: clips 5 and 6 should be marked -1
    labels = annotations.build_frame_labels(exclude_unresolved=True)
    df = annotations.get_annotations()
    
    # Clip 1 has onset at 496
    assert labels[0] == 0
    assert labels[495] == 0
    assert labels[496] == 1
    assert labels[624] == 1
    
    # Clip 5 is unresolved (frames 2685 to 3428)
    row_clip5 = df[df["clip_id"] == 5].iloc[0]
    assert row_clip5["onset_type"] == "unresolved"
    assert np.all(labels[row_clip5["start_frame"] : row_clip5["end_frame"] + 1] == -1)
