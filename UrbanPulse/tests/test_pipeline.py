"""Tests for the video processing pipeline module."""

import numpy as np
import pandas as pd
import pytest


def test_heuristic_anomaly_score_returns_bounded_values():
    """Heuristic scorer should return values in [0, 1]."""
    from src.pipeline import _heuristic_anomaly_score

    df = pd.DataFrame({
        "median_speed": [10, 12, 15, 80, 90, 100, 12, 11],
        "person_count": [10, 10, 10, 5, 3, 1, 10, 10],
    })
    scores = _heuristic_anomaly_score(df)

    assert len(scores) == len(df)
    assert np.all(scores >= 0)
    assert np.all(scores <= 1)
    # High-speed frames should score higher than normal ones
    assert scores[4] > scores[0]


def test_heuristic_anomaly_score_empty():
    """Empty DataFrame should return empty array."""
    from src.pipeline import _heuristic_anomaly_score

    scores = _heuristic_anomaly_score(pd.DataFrame({"median_speed": [], "person_count": []}))
    assert len(scores) == 0


def test_processing_result_defaults():
    """ProcessingResult should initialize with sensible defaults."""
    from src.pipeline import ProcessingResult

    result = ProcessingResult()
    assert result.annotated_video_path == ""
    assert result.telemetry.empty
    assert result.events == []
    assert result.statuses == []
    assert result.metadata == {}
