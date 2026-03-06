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

ASYMMETRY_THRESHOLD_OD = 0.05
ASYMMETRY_THRESHOLD_HBT_PERCENT = 0.01
SLOPE_THRESHOLD = 0.5

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
    return abs(od_diff) > ASYMMETRY_THRESHOLD_OD


def flag_hbt_percent_asymmetry(percent_diff: float) -> bool:
    return percent_diff > ASYMMETRY_THRESHOLD_HBT_PERCENT


def flag_dual_wavelength(f_od_860: bool, f_od_740: bool) -> bool:
    return f_od_860 and f_od_740


def flag_slope(slope: float) -> bool:
    return slope > SLOPE_THRESHOLD


def flag_persistence(ratio: float, history_len: int) -> bool:
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
        pair_id = get_paired_optode(optode_id)

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

        if pair_id in optode_data:

            HbO_pair = optode_data[pair_id].get("HbO", 0)
            HbR_pair = optode_data[pair_id].get("HbR", 0)
            HbT_pair = HbO_pair + HbR_pair

            OD860_pair = optode_data[pair_id].get("OD_860", 0)
            OD740_pair = optode_data[pair_id].get("OD_740", 0)

            od860_diff = OD860 - OD860_pair
            od740_diff = OD740 - OD740_pair

            mean_val = (HbT + HbT_pair) / 2
            if mean_val > EPS:
                percent_diff = abs(HbT - HbT_pair) / mean_val

            optode_state.asymmetry_history.append(HbT - HbT_pair)

        slope_val = compute_slope(list(optode_state.asymmetry_history))

        # ----- Evaluate Flags (No Recalculation) -----

        f_od_860 = flag_od_asymmetry(od860_diff)
        f_od_740 = flag_od_asymmetry(od740_diff)
        f_hbt = flag_hbt_percent_asymmetry(percent_diff)
        f_dual = flag_dual_wavelength(f_od_860, f_od_740)
        f_slope = flag_slope(slope_val)

        # Persistence should represent asymmetry over time, not one-frame transients.
        asymmetry_now = any([f_hbt, f_dual, f_slope])

        optode_state.flag_history.append(asymmetry_now)

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

        # Final decision requires sustained asymmetry evidence.
        final_flags[optode_id] = f_persist and asymmetry_now

        flag_counts[optode_id] = flag_count

    return final_flags, flag_counts
