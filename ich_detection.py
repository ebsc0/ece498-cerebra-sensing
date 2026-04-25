"""Demo-oriented ICH detection based on Delta Hb drift from baseline."""

from __future__ import annotations

from collections import deque
from typing import Dict, Tuple

import numpy as np

from config import ICH_DEMO_DELTA_HB_THRESHOLD, SAMPLE_RATE_HZ


# Demo detector behavior:
# - collect a short per-optode baseline after preprocessing warmup
# - compare the same HbO/HbR values shown on the UI graph against that baseline
# - alert when either Delta Hb trace drifts beyond threshold
# - clear automatically once the signal returns near baseline
BASELINE_WINDOW = max(5, int(round(1.0 * SAMPLE_RATE_HZ)))
SMOOTHING_WINDOW = max(1, int(round(0.25 * SAMPLE_RATE_HZ)))


class OptodeState:
    def __init__(self) -> None:
        self.baseline_ready = False

        self.hbo_history = deque(maxlen=SMOOTHING_WINDOW)
        self.hbr_history = deque(maxlen=SMOOTHING_WINDOW)

        self.baseline_hbo_samples = deque(maxlen=BASELINE_WINDOW)
        self.baseline_hbr_samples = deque(maxlen=BASELINE_WINDOW)

        self.baseline_hbo = 0.0
        self.baseline_hbr = 0.0


state: Dict[int, OptodeState] = {}


def get_state(optode_id: int) -> OptodeState:
    if optode_id not in state:
        state[optode_id] = OptodeState()
    return state[optode_id]


def reset_history() -> None:
    global state
    state = {}


def _mean(values: deque[float]) -> float:
    return float(np.mean(np.array(values, dtype=float))) if values else 0.0


def _capture_baseline(optode_state: OptodeState, *, hbo: float, hbr: float) -> None:
    optode_state.baseline_hbo_samples.append(hbo)
    optode_state.baseline_hbr_samples.append(hbr)

    if len(optode_state.baseline_hbo_samples) < BASELINE_WINDOW:
        return

    optode_state.baseline_hbo = _mean(optode_state.baseline_hbo_samples)
    optode_state.baseline_hbr = _mean(optode_state.baseline_hbr_samples)
    optode_state.baseline_ready = True


def detect_ich(
    optode_data: Dict[int, dict],
    active_optodes: list[int],
) -> Tuple[Dict[int, bool], Dict[int, int]]:
    final_flags: Dict[int, bool] = {}
    flag_counts: Dict[int, int] = {}

    for optode_id in active_optodes:
        if optode_id not in optode_data:
            continue

        optode_state = get_state(optode_id)
        hbo = float(optode_data[optode_id].get("HbO", 0.0))
        hbr = float(optode_data[optode_id].get("HbR", 0.0))

        optode_state.hbo_history.append(hbo)
        optode_state.hbr_history.append(hbr)

        if not optode_state.baseline_ready:
            _capture_baseline(
                optode_state,
                hbo=hbo,
                hbr=hbr,
            )
            continue

        hbo_delta = abs(_mean(optode_state.hbo_history) - optode_state.baseline_hbo)
        hbr_delta = abs(_mean(optode_state.hbr_history) - optode_state.baseline_hbr)

        metric_flags = [
            hbo_delta >= ICH_DEMO_DELTA_HB_THRESHOLD,
            hbr_delta >= ICH_DEMO_DELTA_HB_THRESHOLD,
        ]

        flag_count = sum(metric_flags)
        final_flags[optode_id] = flag_count > 0
        flag_counts[optode_id] = flag_count

    return final_flags, flag_counts
