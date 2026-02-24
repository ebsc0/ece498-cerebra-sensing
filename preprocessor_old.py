"""Preprocessor for converting raw fNIRS data to hemoglobin concentrations.

Current implementation:
- OD = intensity (placeholder, no baseline)
- No dark subtraction
- Short and long channels processed separately
- Uses MBLL to compute HbO/HbR
"""

import struct
from typing import Dict, NamedTuple
import numpy as np

from config import (
    PACKET_FORMAT,
    DPF_SHORT,
    DPF_LONG,
    DISTANCE_SHORT,
    DISTANCE_LONG,
)
from buffer import CompleteFrame

# Extinction coefficients (cm^-1 / M)
# Rows: [740nm, 860nm], Columns: [HbO, HbR]
EXTINCTION_MATRIX = np.array([
    [1486.0, 3843.0],   # 740 nm: [ε_HbO, ε_HbR]
    [2526.0, 1798.0]    # 860 nm: [ε_HbO, ε_HbR]
])

# Precompute pseudo-inverse for MBLL (used for all conversions)
_EXT_INV = np.linalg.pinv(EXTINCTION_MATRIX)


class PreprocessedResult(NamedTuple):
    """Result from preprocessing a single optode."""
    sample_id: int          # FK to raw_samples
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


class Preprocessor:
    """Converts raw intensity data to optical density and hemoglobin concentrations."""

    def __init__(self):
        self.dpf_short = DPF_SHORT
        self.dpf_long = DPF_LONG
        self.dist_short = DISTANCE_SHORT
        self.dist_long = DISTANCE_LONG

    def process_frame(
        self,
        frame: CompleteFrame,
        sample_ids: Dict[int, int]
    ) -> Dict[int, PreprocessedResult]:
        """Process a complete frame and return preprocessed samples.

        Args:
            frame: CompleteFrame from buffer containing all optode packets.
            sample_ids: Mapping of {optode_id: sample_id} from raw inserts.

        Returns:
            Dict mapping optode_id to PreprocessedResult.
        """
        results = {}

        for optode_id, packet in frame.packets.items():
            sample_id = sample_ids.get(optode_id)
            if sample_id is None:
                continue

            # Unpack packet: metadata, nm740_long, nm860_long, nm740_short, nm860_short, dark
            data = struct.unpack(PACKET_FORMAT, packet)
            nm740_long = data[1]
            nm860_long = data[2]
            nm740_short = data[3]
            nm860_short = data[4]
            # dark = data[5]  # Not used currently

            # Compute OD (placeholder: OD = intensity)
            od_740_short = nm740_short
            od_740_long = nm740_long
            od_860_short = nm860_short
            od_860_long = nm860_long

            # Compute HbO/HbR for short channel
            hbo_short, hbr_short = self._mbll(
                od_740_short, od_860_short,
                self.dpf_short, self.dist_short
            )

            # Compute HbO/HbR for long channel
            hbo_long, hbr_long = self._mbll(
                od_740_long, od_860_long,
                self.dpf_long, self.dist_long
            )

            results[optode_id] = PreprocessedResult(
                sample_id=sample_id,
                optode_id=optode_id,
                frame_number=frame.frame_number,
                timestamp_ms=frame.timestamp_ms,
                od_nm740_short=od_740_short,
                od_nm740_long=od_740_long,
                od_nm860_short=od_860_short,
                od_nm860_long=od_860_long,
                hbo_short=hbo_short,
                hbr_short=hbr_short,
                hbo_long=hbo_long,
                hbr_long=hbr_long,
            )

        return results

    def _mbll(
        self,
        od_740: float,
        od_860: float,
        dpf: float,
        distance: float
    ) -> tuple[float, float]:
        """Apply Modified Beer-Lambert Law to compute HbO and HbR.

        The MBLL equation:
            ΔOD = ε · Δc · d · DPF

        Rearranged:
            Δc = (ε^-1 · ΔOD) / (d · DPF)

        Args:
            od_740: Optical density at 740nm.
            od_860: Optical density at 860nm.
            dpf: Differential pathlength factor.
            distance: Source-detector distance in cm.

        Returns:
            Tuple of (HbO, HbR) concentrations.
        """
        delta_od = np.array([od_740, od_860])
        
        # Solve for chromophore concentrations
        # ExtinctionMatrix · [HbO, HbR] = [OD_740, OD_860]
        chromo = _EXT_INV @ delta_od
        
        # Divide by pathlength (d * DPF)
        pathlength = distance * dpf
        hbo = chromo[0] / pathlength
        hbr = chromo[1] / pathlength

        return float(hbo), float(hbr)
