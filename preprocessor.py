import math
import numpy as np
from scipy.signal import butter, lfilter, lfilter_zi

FS = 10
CUTOFF = 0.7
REGRESSION_WINDOW = 5
BASELINE_SAMPLES = FS * 3
EPS = 1e-6

EXTINCTION_MATRIX = np.array([
    [1486, 3843],   # 740 nm
    [2526, 1798]    # 860 nm
])

DPF = 6.0
DISTANCE = 4.0

INV_EXT = np.linalg.pinv(EXTINCTION_MATRIX)
MBLL_SCALE = 1.0 / (DPF * DISTANCE)

b, a = butter(2, CUTOFF / (FS/2), btype="low")
b_sci, a_sci = butter(2, [0.5 / (FS/2), 2.5 / (FS/2)], btype="band")


class Preprocessor:
    def __init__(self):
        # Buffers
        self.OD_buffer_860 = []
        self.OD_buffer_740 = []
        self.short_OD_buffer_860 = []
        self.short_OD_buffer_740 = []

        self.buffer_740 = []
        self.buffer_860 = []

        # Regression state
        self.beta_860 = 0.1
        self.beta_740 = 0.1

        # Baseline storage
        self.baseline_long_860 = []
        self.baseline_long_740 = []
        self.baseline_short_860 = []
        self.baseline_short_740 = []
        self.baseline_ready = False

        # Filter state
        self.zi_860 = None
        self.zi_740 = None

        self.sample_count = 0

    
    # Helpers 
    def mbll_from_od(self, od_740, od_860):
        delta_od = np.array([od_740, od_860])
        chromo = INV_EXT @ delta_od
        HbO = chromo[0] * MBLL_SCALE
        HbR = chromo[1] * MBLL_SCALE
        return HbO, HbR

    def compute_sci(self):
        if len(self.buffer_740) < 10:
            return None

        sig_740 = np.array(self.buffer_740[-10:])
        sig_860 = np.array(self.buffer_860[-10:])

        if np.std(sig_740) < 1e-8 or np.std(sig_860) < 1e-8:
            return 0.0

        filt_740 = lfilter(b_sci, a_sci, sig_740)
        filt_860 = lfilter(b_sci, a_sci, sig_860)

        c = np.corrcoef(filt_740, filt_860)[0, 1]
        return float(c) if np.isfinite(c) else 0.0

    def update_regression(self, short_buf, long_buf, beta, alpha):
        if len(short_buf) >= REGRESSION_WINDOW and len(long_buf) >= REGRESSION_WINDOW:
            s = np.array(short_buf[-REGRESSION_WINDOW:])
            l = np.array(long_buf[-REGRESSION_WINDOW:])
            if np.var(s) > EPS:
                beta_new = np.cov(s, l)[0,1] / np.var(s)
                beta = (1 - alpha) * beta + alpha * beta_new
        return beta

    def apply_filter(self, value, zi):
        if zi is None:
            zi = lfilter_zi(b, a) * value
        out, zi = lfilter(b, a, [value], zi=zi)
        return out[0], zi

    # Main streaming entry point
    def process_sample(self, frame: dict) -> dict | None:
        """
        Args:
            frame = {
                "optode_id": int or str,
                "long_860": float,
                "long_740": float,
                "short_860": float,
                "short_740": float,
                "dark": float
            }

        Returns:
            dict or None (until baseline complete)
        """

        optode_id = frame["optode_id"]

        long_860 = max(frame["long_860"] - frame["dark"], EPS)
        long_740 = max(frame["long_740"] - frame["dark"], EPS)
        short_860 = max(frame["short_860"] - frame["dark"], EPS)
        short_740 = max(frame["short_740"] - frame["dark"], EPS)

        # Baseline collection 
        if not self.baseline_ready:
            self.baseline_long_860.append(long_860)
            self.baseline_long_740.append(long_740)
            self.baseline_short_860.append(short_860)
            self.baseline_short_740.append(short_740)

            self.sample_count += 1

            if self.sample_count >= BASELINE_SAMPLES:
                self.I0_long_860 = np.mean(self.baseline_long_860)
                self.I0_long_740 = np.mean(self.baseline_long_740)
                self.I0_short_860 = np.mean(self.baseline_short_860)
                self.I0_short_740 = np.mean(self.baseline_short_740)
                self.baseline_ready = True

            return None

        # Optical density 
        OD_long_860  = -math.log(long_860 / self.I0_long_860)
        OD_long_740  = -math.log(long_740 / self.I0_long_740)
        OD_short_860 = -math.log(short_860 / self.I0_short_860)
        OD_short_740 = -math.log(short_740 / self.I0_short_740)

        self.short_OD_buffer_860.append(OD_short_860)
        self.short_OD_buffer_740.append(OD_short_740)

        self.buffer_740.append(OD_long_740)
        self.buffer_860.append(OD_long_860)

        if len(self.buffer_740) > 10:
            self.buffer_740.pop(0)
            self.buffer_860.pop(0)

        sci = self.compute_sci()

        # Regression 
        self.beta_860 = self.update_regression(
            self.short_OD_buffer_860, self.OD_buffer_860, self.beta_860, 0.05
        )
        self.beta_740 = self.update_regression(
            self.short_OD_buffer_740, self.OD_buffer_740, self.beta_740, 0.10
        )

        clean_OD_860 = OD_long_860 - self.beta_860 * OD_short_860
        clean_OD_740 = OD_long_740 - self.beta_740 * OD_short_740

        self.OD_buffer_860.append(clean_OD_860)
        self.OD_buffer_740.append(clean_OD_740)

        # Filtering 
        filtered_OD_860, self.zi_860 = self.apply_filter(clean_OD_860, self.zi_860)
        filtered_OD_740, self.zi_740 = self.apply_filter(clean_OD_740, self.zi_740)

        # MBLL 
        HbO, HbR = self.mbll_from_od(filtered_OD_740, filtered_OD_860)

        return {
            "optode_id": optode_id,
            "raw": OD_long_860,
            "clean": clean_OD_860,
            "filtered": filtered_OD_860,
            "HbO": HbO,
            "HbR": HbR,
            "SCI": sci
        }
