"""Tests for sequence windowing and leakage-free dataset splitting."""

import numpy as np
import pandas as pd
import pytest

from src import annotations, sequences
from src.video_features import FEATURE_COLUMNS


def test_make_sequences_no_cross_clip_leakage():
    """Verify that no window crosses a clip boundary and shapes match."""
    total_frames = annotations.TOTAL_FRAMES
    n_features = len(FEATURE_COLUMNS)
    X_dummy = np.random.randn(total_frames, n_features).astype(np.float32)
    labels = annotations.build_frame_labels(exclude_unresolved=True)
    clip_ids = annotations.assign_clip_ids(total_frames)

    # Test with clips 1 and 2
    X_seq, y_seq, meta = sequences.make_sequences(
        X_dummy, labels, clip_ids, selected_clips=[1, 2], sequence_length=30, stride=5
    )

    assert len(X_seq) == len(y_seq) == len(meta)
    assert X_seq.shape[1:] == (30, n_features)

    clip_starts = annotations.get_clip_starts()
    clip_ends = [clip_starts[i] - 1 for i in range(1, len(clip_starts))] + [total_frames - 1]

    for item in meta:
        cid = item["clip_id"]
        end_f = item["end_frame"]
        start_f = end_f - 30 + 1
        # Window must be strictly within the clip boundary
        assert start_f >= clip_starts[cid - 1], f"Window start {start_f} before clip start"
        assert end_f <= clip_ends[cid - 1], f"Window end {end_f} after clip end"


def test_prepare_splits_disjoint_clips():
    """Verify train, val, and test splits have mutually exclusive clips."""
    df_features = pd.DataFrame(
        np.zeros((annotations.TOTAL_FRAMES, len(FEATURE_COLUMNS))),
        columns=FEATURE_COLUMNS,
    )
    splits = sequences.prepare_splits(df_features)

    train_cids = {m["clip_id"] for m in splits["meta_train"]}
    val_cids = {m["clip_id"] for m in splits["meta_val"]}
    test_cids = {m["clip_id"] for m in splits["meta_test"]}

    assert train_cids.isdisjoint(val_cids), "Train and Val share clips"
    assert train_cids.isdisjoint(test_cids), "Train and Test share clips"
    assert val_cids.isdisjoint(test_cids), "Val and Test share clips"
