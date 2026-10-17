"""Alert logic: temporal persistence/debounce for abnormal movement warnings.

Replicates notebook cell 52 as a tested, reusable function.
"""

from dataclasses import dataclass, field


@dataclass
class AlertEvent:
    """A single alert interval."""
    start_frame: int
    end_frame: int
    peak_score: float = 0.0


@dataclass
class AlertState:
    """Mutable state for the debounced alert tracker."""
    active: bool = False
    positive_run: int = 0
    negative_run: int = 0
    alert_start_frame: int | None = None
    peak_score: float = 0.0
    events: list[AlertEvent] = field(default_factory=list)


def update_alert(
    state: AlertState,
    frame: int,
    probability: float,
    threshold: float,
    raise_after: int = 3,
    clear_after: int = 3,
) -> str:
    """Process one prediction window and return current status.

    Returns one of: "NORMAL", "ABNORMAL_MOVEMENT".
    """
    is_positive = probability >= threshold

    if not state.active:
        state.positive_run = state.positive_run + 1 if is_positive else 0
        if state.positive_run >= raise_after:
            state.active = True
            state.alert_start_frame = frame
            state.negative_run = 0
            state.peak_score = probability
    else:
        if is_positive:
            state.negative_run = 0
            state.peak_score = max(state.peak_score, probability)
        else:
            state.negative_run += 1

        if state.negative_run >= clear_after:
            state.events.append(AlertEvent(
                start_frame=state.alert_start_frame,
                end_frame=frame,
                peak_score=state.peak_score,
            ))
            state.active = False
            state.alert_start_frame = None
            state.positive_run = 0
            state.negative_run = 0
            state.peak_score = 0.0

    return "ABNORMAL_MOVEMENT" if state.active else "NORMAL"


def finalize(state: AlertState, last_frame: int) -> list[AlertEvent]:
    """Close any open alert and return all events."""
    if state.active and state.alert_start_frame is not None:
        state.events.append(AlertEvent(
            start_frame=state.alert_start_frame,
            end_frame=last_frame,
            peak_score=state.peak_score,
        ))
        state.active = False
    return state.events


def run_alert_pipeline(
    frames: list[int],
    probabilities: list[float],
    threshold: float,
    raise_after: int = 3,
    clear_after: int = 3,
) -> tuple[list[str], list[AlertEvent]]:
    """Run the full alert pipeline over a sequence of predictions.

    Returns (statuses, events).
    """
    state = AlertState()
    statuses = []
    for frame, prob in zip(frames, probabilities):
        status = update_alert(state, frame, prob, threshold, raise_after, clear_after)
        statuses.append(status)
    events = finalize(state, frames[-1] if frames else 0)
    return statuses, events
