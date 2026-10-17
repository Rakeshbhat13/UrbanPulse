"""Tests for temporal persistence and debounce alert logic."""

import pytest

from src.alert import AlertState, update_alert, finalize, run_alert_pipeline


def test_isolated_spike_does_not_alert():
    """A single high-probability window must NOT raise an alert."""
    state = AlertState()
    status1 = update_alert(state, frame=10, probability=0.9, threshold=0.5, raise_after=3)
    status2 = update_alert(state, frame=15, probability=0.2, threshold=0.5, raise_after=3)

    assert status1 == "NORMAL"
    assert status2 == "NORMAL"
    assert not state.active
    assert len(state.events) == 0


def test_persistent_positive_triggers_and_clears_alert():
    """Three consecutive positives raise alert; three consecutive negatives clear it."""
    state = AlertState()
    # 3 consecutive positives:
    s1 = update_alert(state, frame=10, probability=0.8, threshold=0.5, raise_after=3, clear_after=3)
    s2 = update_alert(state, frame=15, probability=0.85, threshold=0.5, raise_after=3, clear_after=3)
    s3 = update_alert(state, frame=20, probability=0.95, threshold=0.5, raise_after=3, clear_after=3)

    assert s1 == "NORMAL"
    assert s2 == "NORMAL"
    assert s3 == "ABNORMAL_MOVEMENT"
    assert state.active
    assert state.alert_start_frame == 20
    assert state.peak_score == 0.95

    # 2 negatives: still active
    s4 = update_alert(state, frame=25, probability=0.1, threshold=0.5, raise_after=3, clear_after=3)
    s5 = update_alert(state, frame=30, probability=0.1, threshold=0.5, raise_after=3, clear_after=3)
    assert s4 == "ABNORMAL_MOVEMENT"
    assert s5 == "ABNORMAL_MOVEMENT"
    assert state.active

    # 3rd negative: cleared
    s6 = update_alert(state, frame=35, probability=0.1, threshold=0.5, raise_after=3, clear_after=3)
    assert s6 == "NORMAL"
    assert not state.active
    assert len(state.events) == 1
    assert state.events[0].start_frame == 20
    assert state.events[0].end_frame == 35


def test_run_alert_pipeline():
    """Test full alert pipeline helper function."""
    frames = [10, 15, 20, 25, 30, 35, 40]
    probs = [0.1, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1]
    statuses, events = run_alert_pipeline(frames, probs, threshold=0.5, raise_after=3, clear_after=2)

    assert statuses[0] == "NORMAL"
    assert statuses[1] == "NORMAL"
    assert statuses[2] == "NORMAL"
    assert statuses[3] == "ABNORMAL_MOVEMENT"
    assert len(events) >= 1
