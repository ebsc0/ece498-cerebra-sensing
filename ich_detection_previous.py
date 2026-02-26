"""ICH (Intracranial Hemorrhage) detection algorithms."""

from typing import Dict, List, Tuple, Optional
import numpy as np

from config import TOTAL_OPTODES, ACTIVE_OPTODES

ASYMMETRY_THRESHOLD_OD = 0.05
ASYMMETRY_THRESHOLD_HBR = 0.2
Z_SCORE_THRESHOLD = 2.0

# Left hemisphere optodes (0-7) pair with right hemisphere (8-15)
LEFT_OPTODES = list(range(0, 8))
RIGHT_OPTODES = list(range(8, 16))

# Memory for historical flags (basic version - extend as needed)
flag_history: Dict[int, List[bool]] = {i: [] for i in range(TOTAL_OPTODES)}


def detect_ich(
    optode_data: Dict[int, dict],
    active_optodes: Optional[List[int]] = None
) -> Tuple[Dict[int, bool], Dict[int, int]]:
    """Detect ICH based on optode data.

    This function handles partial optode data gracefully. Only active optodes
    are analyzed, and hemisphere comparisons are skipped if the paired optode
    is inactive.

    Args:
        optode_data: {optode_id: {"HbR": float, "OD_860": float}}
            Only needs to contain data for active optodes.
        active_optodes: List of optode IDs that have data.
            Defaults to ACTIVE_OPTODES from config.

    Returns:
        Tuple of:
            - final_flags: {optode_id: bool} - True if ICH detected
            - final_flag_counts: {optode_id: int} - number of criteria met (0-3)
    """
    if active_optodes is None:
        active_optodes = ACTIVE_OPTODES

    # If no active optodes, return empty results
    if not active_optodes or not optode_data:
        return {}, {}

    flags_od: Dict[int, bool] = {}
    flags_z: Dict[int, bool] = {}
    flags_roc: Dict[int, bool] = {}

    # Get HbR values for active optodes only
    active_hbr = {i: optode_data[i]["HbR"] for i in active_optodes if i in optode_data}

    if not active_hbr:
        return {}, {}

    HbR_values = np.array(list(active_hbr.values()))
    hbr_by_id = active_hbr

    # Step 1: OD asymmetry (only if both hemisphere pairs are active)
    for i in active_optodes:
        if i not in optode_data:
            continue

        # Find hemisphere pair
        if i in LEFT_OPTODES:
            j = i + 8
        elif i in RIGHT_OPTODES:
            j = i - 8
        else:
            continue

        # Skip if pair is not active
        if j not in active_optodes or j not in optode_data:
            continue

        OD_i = optode_data[i].get("OD_860", 0)
        OD_j = optode_data[j].get("OD_860", 0)

        if OD_i - OD_j > ASYMMETRY_THRESHOLD_OD:
            flags_od[i] = True
        elif OD_j - OD_i > ASYMMETRY_THRESHOLD_OD:
            flags_od[j] = True

    # Step 2: HbR z-score outliers (among active optodes only)
    if len(HbR_values) > 1:
        mean_HbR = np.mean(HbR_values)
        std_HbR = np.std(HbR_values) + 1e-6

        for i in active_optodes:
            if i not in hbr_by_id:
                continue
            z = (hbr_by_id[i] - mean_HbR) / std_HbR
            if z > Z_SCORE_THRESHOLD:
                flags_z[i] = True

    # Step 3: Historical rate-of-change detection (sustained flag)
    for i in active_optodes:
        is_flagged_now = flags_od.get(i, False) or flags_z.get(i, False)
        flag_history[i].append(is_flagged_now)

        # Keep last 5 frames
        if len(flag_history[i]) > 5:
            flag_history[i] = flag_history[i][-5:]

        # Sustained anomaly in 3 of last 5
        if sum(flag_history[i]) >= 3:
            flags_roc[i] = True

    # Step 4: Ensemble flag (need at least 2 criteria met)
    final_flags: Dict[int, bool] = {}
    final_flag_counts: Dict[int, int] = {}

    for i in active_optodes:
        count = sum([
            flags_od.get(i, False),
            flags_z.get(i, False),
            flags_roc.get(i, False)
        ])
        final_flags[i] = count >= 2
        final_flag_counts[i] = count

    return final_flags, final_flag_counts


def reset_history():
    """Reset flag history. Call when starting a new session."""
    global flag_history
    flag_history = {i: [] for i in range(TOTAL_OPTODES)}
