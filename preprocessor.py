"""Preprocessor for converting logical fNIRS frames to hemoglobin concentrations.

Public contract:
- process_frame(frame, sample_ids) -> Dict[int, PreprocessedResult]

Behavior:
- Dark subtraction with floor clamp
- Per-optode baseline collection
- Optical density via -log(I / I0)
- Short-channel regression cleanup (beta fit on raw long OD)
- Low-pass filter on cleaned long OD
- MBLL conversion for short (raw short OD) and long (filtered long OD)
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, NamedTuple, Optional

import numpy as np
from scipy.signal import butter, lfilter, lfilter_zi

from config import DPF_LONG, DPF_SHORT, DISTANCE_LONG, DISTANCE_SHORT, SAMPLE_RATE_HZ
from packets import AssembledFrame

# Extinction coefficients (cm^-1 / M)
# Rows: [740nm, 860nm], Columns: [HbO, HbR]
EXTINCTION_MATRIX = np.array(
    [
        [1486.0, 3843.0],
        [2526.0, 1798.0],
    ],
    dtype=float,
)
_EXT_INV = np.linalg.pinv(EXTINCTION_MATRIX)

# Processing constants
REGRESSION_WINDOW = 5
BASELINE_SECONDS = 3.0
LOWPASS_CUTOFF_HZ = 0.7
ALPHA_860 = 0.05
ALPHA_740 = 0.10
BETA_INIT = 0.1
EPS = 1e-6

K_HAMPEL = 3
HAMPEL_WINDOW = 10


class PreprocessedResult(NamedTuple):
    sample_id: int
    optode_id: int
    frame_number: int
    timestamp_ms: int
    od_nm740_short: float
    od_nm740_long: float
    od_nm860_short: float
    od_nm860_long: float
    hbo_short: float
    hbr_short: float
    hbo_long: float
    hbr_long: float


@dataclass
class _OptodeState:
    baseline_count: int = 0
    baseline_sum_long_740: float = 0.0
    baseline_sum_long_860: float = 0.0
    baseline_sum_short_740: float = 0.0
    baseline_sum_short_860: float = 0.0
    baseline_ready: bool = False
    i0_long_740: float = 1.0
    i0_long_860: float = 1.0
    i0_short_740: float = 1.0
    i0_short_860: float = 1.0
    beta_860: float = BETA_INIT
    beta_740: float = BETA_INIT
    short_od_860: Deque[float] = field(default_factory=lambda: deque(maxlen=REGRESSION_WINDOW))
    short_od_740: Deque[float] = field(default_factory=lambda: deque(maxlen=REGRESSION_WINDOW))
    long_raw_od_860: Deque[float] = field(default_factory=lambda: deque(maxlen=REGRESSION_WINDOW))
    long_raw_od_740: Deque[float] = field(default_factory=lambda: deque(maxlen=REGRESSION_WINDOW))
    clean_od_860_buffer: Deque[float] = field(default_factory=lambda: deque(maxlen=HAMPEL_WINDOW))
    clean_od_740_buffer: Deque[float] = field(default_factory=lambda: deque(maxlen=HAMPEL_WINDOW))
    zi_860: Optional[np.ndarray] = None
    zi_740: Optional[np.ndarray] = None


def _design_filters(sample_rate_hz: float):
    nyquist = sample_rate_hz / 2.0
    if nyquist <= 0.0:
        raise ValueError("sample_rate_hz must be positive")

    low_norm = min(max(LOWPASS_CUTOFF_HZ / nyquist, 1e-4), 0.99)
    low_b, low_a = butter(2, low_norm, btype="low")

    return low_b, low_a


class Preprocessor:
    """Converts reconstructed intensity data to OD and hemoglobin concentrations."""

    def __init__(self, sample_rate_hz: float = SAMPLE_RATE_HZ):
        self.sample_rate_hz = float(sample_rate_hz)
        self.baseline_samples = max(1, int(round(self.sample_rate_hz * BASELINE_SECONDS)))

        self.dpf_short = DPF_SHORT
        self.dpf_long = DPF_LONG
        self.dist_short = DISTANCE_SHORT
        self.dist_long = DISTANCE_LONG

        self.low_b, self.low_a = _design_filters(self.sample_rate_hz)
        self._states: Dict[int, _OptodeState] = {}

    def reset(self) -> None:
        self._states.clear()

    def _get_state(self, optode_id: int) -> _OptodeState:
        state = self._states.get(optode_id)
        if state is None:
            state = _OptodeState()
            self._states[optode_id] = state
        return state

    def _mbll(self, od_740: float, od_860: float, dpf: float, distance: float) -> tuple[float, float]:
        delta_od = np.array([od_740, od_860], dtype=float)
        chromo = _EXT_INV @ delta_od
        pathlength = max(distance * dpf, EPS)
        hbo = chromo[0] / pathlength
        hbr = chromo[1] / pathlength
        return float(hbo), float(hbr)

    def _apply_lowpass(self, value: float, zi: Optional[np.ndarray]) -> tuple[float, np.ndarray]:
        if zi is None:
            zi = lfilter_zi(self.low_b, self.low_a) * value
        out, zi = lfilter(self.low_b, self.low_a, [value], zi=zi)
        return float(out[0]), zi

    def _update_beta(
        self,
        short_samples: Deque[float],
        long_raw_samples: Deque[float],
        beta: float,
        alpha: float,
    ) -> float:
        if len(short_samples) < REGRESSION_WINDOW or len(long_raw_samples) < REGRESSION_WINDOW:
            return beta

        s = np.array(short_samples, dtype=float)
        l = np.array(long_raw_samples, dtype=float)[-REGRESSION_WINDOW:]
        s_centered = s - np.mean(s)
        l_centered = l - np.mean(l)
        var_s = float(np.mean(s_centered ** 2))
        if var_s <= EPS:
            return beta

        cov_sl = float(np.mean(s_centered * l_centered))
        beta_new = cov_sl / var_s
        return float((1.0 - alpha) * beta + alpha * beta_new)
    
    def hampel(self, x_buffer, window, k):
        if len(x_buffer) < window:
            return x_buffer[-1]

        win = np.array(list(x_buffer)[-window:])
        med = np.median(win)
        mad = np.median(np.abs(win - med))
        # Convert MAD → robust std estimate
        sigma = 1.4826 * mad
        # Prevent divide / threshold collapse
        if sigma < 1e-12:
            return x_buffer[-1]

        latest = x_buffer[-1]
        if np.abs(latest - med) > k * sigma:
            return med
        else:
            return latest

    def _process_values(
        self,
        optode_id: int,
        long_740: float,
        long_860: float,
        short_740: float,
        short_860: float,
        dark: float,
    ) -> Optional[dict]:
        state = self._get_state(optode_id)

        long_740 = max(long_740 - dark, EPS)
        long_860 = max(long_860 - dark, EPS)
        short_740 = max(short_740 - dark, EPS)
        short_860 = max(short_860 - dark, EPS)

        if not state.baseline_ready:
            state.baseline_sum_long_740 += long_740
            state.baseline_sum_long_860 += long_860
            state.baseline_sum_short_740 += short_740
            state.baseline_sum_short_860 += short_860
            state.baseline_count += 1

            if state.baseline_count >= self.baseline_samples:
                inv_n = 1.0 / state.baseline_count
                state.i0_long_740 = max(state.baseline_sum_long_740 * inv_n, EPS)
                state.i0_long_860 = max(state.baseline_sum_long_860 * inv_n, EPS)
                state.i0_short_740 = max(state.baseline_sum_short_740 * inv_n, EPS)
                state.i0_short_860 = max(state.baseline_sum_short_860 * inv_n, EPS)
                state.baseline_ready = True
                state.baseline_sum_long_740 = 0.0
                state.baseline_sum_long_860 = 0.0
                state.baseline_sum_short_740 = 0.0
                state.baseline_sum_short_860 = 0.0

            return None

        od_long_740 = -math.log(long_740 / state.i0_long_740)
        od_long_860 = -math.log(long_860 / state.i0_long_860)
        od_short_740 = -math.log(short_740 / state.i0_short_740)
        od_short_860 = -math.log(short_860 / state.i0_short_860)

        state.short_od_740.append(od_short_740)
        state.short_od_860.append(od_short_860)
        state.long_raw_od_740.append(od_long_740)
        state.long_raw_od_860.append(od_long_860)

        state.beta_740 = self._update_beta(state.short_od_740, state.long_raw_od_740, state.beta_740, ALPHA_740)
        state.beta_860 = self._update_beta(state.short_od_860, state.long_raw_od_860, state.beta_860, ALPHA_860)

        clean_od_740 = od_long_740 - state.beta_740 * od_short_740
        clean_od_860 = od_long_860 - state.beta_860 * od_short_860

        state.clean_od_740_buffer.append(clean_od_740)
        state.clean_od_860_buffer.append(clean_od_860)

        clean_OD_740_unmotioned = self.hampel(state.clean_od_740_buffer, HAMPEL_WINDOW, K_HAMPEL)
        clean_OD_860_unmotioned = self.hampel(state.clean_od_860_buffer, HAMPEL_WINDOW, K_HAMPEL)

        filtered_od_740, state.zi_740 = self._apply_lowpass(clean_OD_740_unmotioned, state.zi_740)
        filtered_od_860, state.zi_860 = self._apply_lowpass(clean_OD_860_unmotioned, state.zi_860)

        hbo_short, hbr_short = self._mbll(od_short_740, od_short_860, self.dpf_short, self.dist_short)
        hbo_long, hbr_long = self._mbll(filtered_od_740, filtered_od_860, self.dpf_long, self.dist_long)

        return {
            "od_short_740": od_short_740,
            "od_short_860": od_short_860,
            "od_long_740_raw": od_long_740,
            "od_long_860_raw": od_long_860,
            "od_long_740_clean": clean_od_740,
            "od_long_860_clean": clean_od_860,
            "od_long_740_filtered": filtered_od_740,
            "od_long_860_filtered": filtered_od_860,
            "hbo_short": hbo_short,
            "hbr_short": hbr_short,
            "hbo_long": hbo_long,
            "hbr_long": hbr_long,
        }

    def process_frame(
        self,
        frame: AssembledFrame,
        sample_ids: Dict[int, int],
    ) -> Dict[int, PreprocessedResult]:
        results: Dict[int, PreprocessedResult] = {}

        for optode_id, logical_sample in frame.logical_samples.items():
            sample_id = sample_ids.get(optode_id)
            if sample_id is None:
                continue

            processed = self._process_values(
                optode_id=optode_id,
                long_740=logical_sample.nm740_long,
                long_860=logical_sample.nm860_long,
                short_740=logical_sample.nm740_short,
                short_860=logical_sample.nm860_short,
                dark=logical_sample.dark,
            )
            if processed is None:
                continue

            results[optode_id] = PreprocessedResult(
                sample_id=sample_id,
                optode_id=optode_id,
                frame_number=frame.frame_number,
                timestamp_ms=frame.timestamp_ms,
                od_nm740_short=processed["od_short_740"],
                od_nm740_long=processed["od_long_740_filtered"],
                od_nm860_short=processed["od_short_860"],
                od_nm860_long=processed["od_long_860_filtered"],
                hbo_short=processed["hbo_short"],
                hbr_short=processed["hbr_short"],
                hbo_long=processed["hbo_long"],
                hbr_long=processed["hbr_long"],
            )
        #print(results)

        return results
