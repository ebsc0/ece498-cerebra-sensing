"""
Advanced ICH (Intracranial Hemorrhage) detection module.
"""

from typing import Dict, List, Tuple, Optional
from collections import deque
import numpy as np


# ==============================
# Configuration
# ==============================

ASYMMETRY_THRESHOLD_OD = 0.05
ASYMMETRY_THRESHOLD_HBT_PERCENT = 0.01
SLOPE_THRESHOLD = 0.5
PERSISTENCE_RATIO = 0.6

WINDOW_SECONDS = 5
FS = 10
WINDOW = WINDOW_SECONDS * FS
PERSISTENCE_WINDOW = 10

LEFT_OPTODES = list(range(0, 8))
RIGHT_OPTODES = list(range(8, 16))

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
    if optode_id in LEFT_OPTODES:
        return optode_id + 8
    elif optode_id in RIGHT_OPTODES:
        return optode_id - 8
    return None


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


def flag_persistence(ratio: float) -> bool:
    return ratio > PERSISTENCE_RATIO


# ==============================
# Main Detection
# ==============================

def detect_ich(
    optode_data: Dict[int, dict],
    active_optodes: List[int]
) -> Tuple[Dict[int, str], Dict[int, int]]:

    final_flags = {}
    flag_counts = {}
    detailed_flags = {}

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

        any_flag_now = any([
            f_od_860,
            f_od_740,
            f_hbt,
            f_dual,
            f_slope
        ])

        optode_state.flag_history.append(any_flag_now)

        persistence_ratio = (
            sum(optode_state.flag_history) /
            len(optode_state.flag_history)
            if len(optode_state.flag_history) > 0 else 0
        )

        f_persist = flag_persistence(persistence_ratio)

        flag_count = sum([
            f_od_860,
            f_od_740,
            f_hbt,
            f_dual,
            f_slope,
            f_persist
        ])

        # Required ensemble logic
        if flag_count >= 4:
            final_flags[optode_id] = "POTENTIAL_ICH"
        elif flag_count >= 2:
            final_flags[optode_id] = "ABNORMALITY"
        else:
            final_flags[optode_id] = "NORMAL"

        flag_counts[optode_id] = flag_count

        detailed_flags[optode_id] = {
            "od_860": f_od_860,
            "od_740": f_od_740,
            "hbt_asym": f_hbt,
            "dual": f_dual,
            "slope": f_slope,
            "persistence": f_persist
        }
        print("flag:", final_flags, "flag counts", flag_counts, "flag info", detailed_flags)

    return final_flags, flag_counts
