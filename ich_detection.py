"""
Advanced ICH (Intracranial Hemorrhage) detection module.
"""

from typing import Dict, List, Tuple, Optional
from collections import deque
import numpy as np

from config import (
    ICH_OPTODE_PAIRS,
    ICH_PERSISTENCE_RATIO,
    ICH_PERSISTENCE_WINDOW_SECONDS,
    ICH_WINDOW_SECONDS,
    SAMPLE_RATE_HZ,
)

# ==============================
# Configuration
# ==============================

ASYMMETRY_THRESHOLD_OD = 0.033 # from IS whitepaper
ASYMMETRY_THRESHOLD_HBT_PERCENT = 0.01
SLOPE_THRESHOLD = 1e-08
SLOPE_DELTA_EPS = 1e-08
SLOPE_TREND_WINDOW = 8
SLOPE_INCREASING_RATIO = 0.5
SLOPE_PLATEAU_RATIO = 0.5

WINDOW = max(2, int(round(ICH_WINDOW_SECONDS * SAMPLE_RATE_HZ)))
PERSISTENCE_WINDOW = max(1, int(round(ICH_PERSISTENCE_WINDOW_SECONDS * SAMPLE_RATE_HZ)))
PERSISTENCE_RATIO = ICH_PERSISTENCE_RATIO

EPS = 1e-6


# ==============================
# Internal State
# ==============================

class OptodeState:
    def __init__(self):
        self.hbt_history = deque(maxlen=WINDOW)
        self.asymmetry_history = deque(maxlen=WINDOW)
        self.slope_history = deque(maxlen=WINDOW)
        self.flag_history = deque(maxlen=PERSISTENCE_WINDOW)


state: Dict[int, OptodeState] = {}


def get_state(optode_id: int) -> OptodeState:
    if optode_id not in state:
        state[optode_id] = OptodeState()
    return state[optode_id]


def reset_history():
    global state
    state = {}


# ==============================
# Utility
# ==============================

def compute_slope(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values))
    slope, _ = np.polyfit(x, values, 1)
    return slope


def get_paired_optode(optode_id: int) -> Optional[int]:
    return ICH_OPTODE_PAIRS.get(optode_id)


# ==============================
# Flag Logic (Now Pure)
# ==============================

def flag_od_asymmetry(od_diff: float) -> bool:
    # if (abs(od_diff) > ASYMMETRY_THRESHOLD_OD):
    #     print("od diff: ", abs(od_diff), ASYMMETRY_THRESHOLD_OD)
    return abs(od_diff) > ASYMMETRY_THRESHOLD_OD


def flag_hbt_percent_asymmetry(percent_diff: float) -> bool:
    # if (percent_diff > ASYMMETRY_THRESHOLD_HBT_PERCENT):
    #     print("hbt percent: ", percent_diff, ASYMMETRY_THRESHOLD_HBT_PERCENT)
    return percent_diff > ASYMMETRY_THRESHOLD_HBT_PERCENT


def flag_dual_wavelength(f_od_860: bool, f_od_740: bool) -> bool:
    # if f_od_860 and f_od_740:
    #     print("860 and 740: ", f_od_860 , f_od_740)
    return f_od_860 and f_od_740


def flag_slope(slope: float) -> bool:
    #print("slope: ", slope)
    return slope > SLOPE_THRESHOLD


def _slope_trend_metrics(slope_history: List[float]) -> Tuple[float, float]:
    if len(slope_history) < 2:
        return 0.0, 0.0
    deltas = np.diff(np.array(slope_history, dtype=float))
    increasing_ratio = float(np.mean(deltas > SLOPE_DELTA_EPS))
    plateau_ratio = float(np.mean(np.abs(deltas) <= SLOPE_DELTA_EPS))
    return increasing_ratio, plateau_ratio


def flag_slope_increasing_or_plateau(slope_history: List[float]) -> bool:
    if len(slope_history) < SLOPE_TREND_WINDOW:
        return False

    recent = slope_history[-SLOPE_TREND_WINDOW:]
    increasing_ratio, plateau_ratio = _slope_trend_metrics(recent)
    last_slope = recent[-1]
    mean_slope = float(np.mean(recent))

    # Concern if slope is staying elevated and still rising.
    increasing_fast = last_slope > SLOPE_THRESHOLD and increasing_ratio >= SLOPE_INCREASING_RATIO
    # Also concern if slope has reached an elevated level and is not recovering.
    plateau_high = (mean_slope > SLOPE_THRESHOLD and last_slope > SLOPE_THRESHOLD) and (
        plateau_ratio >= SLOPE_PLATEAU_RATIO
    )

    return increasing_fast or plateau_high


def flag_persistence(ratio: float, history_len: int) -> bool:
    #print("historical flag count and ratio: ", history_len, ratio)
    if history_len < PERSISTENCE_WINDOW:
        return False
    return ratio >= PERSISTENCE_RATIO


# ==============================
# Main Detection
# ==============================

def detect_ich(
    optode_data: Dict[int, dict],
    active_optodes: List[int]
) -> Tuple[Dict[int, bool], Dict[int, int]]:

    final_flags = {}
    flag_counts = {}

    for optode_id in active_optodes:

        if optode_id not in optode_data:
            continue

        optode_state = get_state(optode_id)
        paired_id = get_paired_optode(optode_id)

        # ----- Compute Shared Quantities -----

        HbO = optode_data[optode_id].get("HbO", 0)
        HbR = optode_data[optode_id].get("HbR", 0)
        HbT = HbO + HbR
        optode_state.hbt_history.append(HbT)

        OD860 = optode_data[optode_id].get("OD_860", 0)
        OD740 = optode_data[optode_id].get("OD_740", 0)

        od860_diff = 0
        od740_diff = 0
        percent_diff = 0

        if paired_id in optode_data:

            HbO_paired = optode_data[paired_id].get("HbO", 0)
            HbR_paired = optode_data[paired_id].get("HbR", 0)
            HbT_paired = HbO_paired + HbR_paired

            OD860_paired = optode_data[paired_id].get("OD_860", 0)
            OD740_paired = optode_data[paired_id].get("OD_740", 0)

            od860_diff = OD860 - OD860_paired
            od740_diff = OD740 - OD740_paired

            mean_val = (HbT + HbT_paired) / 2
            if mean_val > EPS:
                percent_diff = abs(HbT - HbT_paired) / mean_val

            optode_state.asymmetry_history.append(HbT - HbT_paired)

        slope_val = compute_slope(list(optode_state.asymmetry_history))
        optode_state.slope_history.append(slope_val)

        # ----- Evaluate Flags (No Recalculation) -----

        f_od_860 = flag_od_asymmetry(od860_diff)
        f_od_740 = flag_od_asymmetry(od740_diff)
        f_hbt = flag_hbt_percent_asymmetry(percent_diff)
        f_dual = flag_dual_wavelength(f_od_860, f_od_740)
        f_slope = flag_slope_increasing_or_plateau(list(optode_state.slope_history))

        # Persistence should represent asymmetry over time, not one-frame transients.
        prev_raised = any([f_hbt, f_dual, f_slope, f_od_860, f_od_740])
        
        optode_state.flag_history.append(prev_raised)

        persistence_ratio = (
            sum(optode_state.flag_history) /
            len(optode_state.flag_history)
            if len(optode_state.flag_history) > 0 else 0
        )

        f_persist = flag_persistence(persistence_ratio, len(optode_state.flag_history))

        flag_count = sum([
            f_od_860,
            f_od_740,
            f_hbt,
            f_dual,
            f_slope,
            f_persist
        ])

        # if flag_count >= 2:
        #     print("flags:", 
        #         f_od_860,
        #         f_od_740,
        #         f_hbt,
        #         f_dual,
        #         f_slope,
        #         f_persist
        #     )

        # Final decision requires sustained asymmetry evidence.
                # Required ensemble logic
        if flag_count >= 4:
            final_flags[optode_id] = "ICH_RISK"
        elif flag_count >= 1:
            final_flags[optode_id] = "ABNORMALITY"
        else:
            final_flags[optode_id] = "NORMAL"

        flag_counts[optode_id] = flag_count

    return final_flags, flag_counts
