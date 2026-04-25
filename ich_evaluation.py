import numpy as np
import matplotlib.pyplot as plt
from ich_detection import detect_ich, reset_history
from preprocessor import Preprocessor

SAMPLES = 1500
FS = 10  # sampling rate

# ------------------ Fake Frame Classes ------------------
class FakeLogicalSample:
    def __init__(self, nm740_long, nm860_long, nm740_short, nm860_short, dark):
        self.nm740_long = nm740_long
        self.nm860_long = nm860_long
        self.nm740_short = nm740_short
        self.nm860_short = nm860_short
        self.dark = dark

class FakeFrame:
    def __init__(self, frame_number, timestamp_ms, samples):
        self.frame_number = frame_number
        self.timestamp_ms = timestamp_ms
        self.logical_samples = samples

# ------------------ Physiological Signal Generator ------------------
def generate_fnirs(duration_sec=SAMPLES/10, fs=10, case="normal", seed=None):
    rng = np.random.default_rng(seed)

    t = np.arange(0, duration_sec, 1 / fs)
    n = len(t)

    # -----------------------------------
    # Physiological components
    # -----------------------------------

    heart_freq = rng.normal(1.15, 0.08)
    resp_freq = rng.normal(0.25, 0.03)
    mayer_freq = rng.normal(0.09, 0.015)
    brain_freq = rng.normal(0.05, 0.01)

    cardiac = 0.010 * np.sin(2 * np.pi * heart_freq * t + rng.uniform(0, 2 * np.pi))
    respiration = 0.005 * np.sin(2 * np.pi * resp_freq * t + rng.uniform(0, 2 * np.pi))
    mayer = 0.008 * np.sin(2 * np.pi * mayer_freq * t + rng.uniform(0, 2 * np.pi))
    brain = 0.006 * np.sin(2 * np.pi * brain_freq * t + rng.uniform(0, 2 * np.pi))

    # Slow bilateral drift and mild L/R mismatch
    global_drift = rng.normal(0, 2e-5, n).cumsum()
    lr_drift = rng.normal(0, 1.2e-5, n).cumsum()

    scalp = cardiac + respiration + mayer

    # -----------------------------------
    # Base intensities (arb. units)
    # -----------------------------------

    long_860_base = rng.normal(2.0, 0.04)
    long_740_base = rng.normal(1.8, 0.04)
    short_860_base = rng.normal(1.5, 0.03)
    short_740_base = rng.normal(1.3, 0.03)

    def make_long(base, side_sign):
        return (
            base
            + 0.85 * scalp
            + 0.60 * brain
            + global_drift
            + side_sign * lr_drift
            + rng.normal(0, 0.0045, n)
        )

    def make_short(base, side_sign):
        return (
            base
            + 0.95 * scalp
            + 0.12 * brain
            + 0.75 * global_drift
            + 0.15 * side_sign * lr_drift
            + rng.normal(0, 0.006, n)
        )

    A_long_740 = make_long(long_740_base, side_sign=1.0)
    A_long_860 = make_long(long_860_base, side_sign=1.0)
    B_long_740 = make_long(long_740_base, side_sign=-1.0)
    B_long_860 = make_long(long_860_base, side_sign=-1.0)

    A_short_740 = make_short(short_740_base, side_sign=1.0)
    A_short_860 = make_short(short_860_base, side_sign=1.0)
    B_short_740 = make_short(short_740_base, side_sign=-1.0)
    B_short_860 = make_short(short_860_base, side_sign=-1.0)

    # Baseline left-right asymmetry is present in both normal and ICH cases.
    baseline_offset_long = rng.normal(0, 0.012)
    baseline_offset_short = 0.35 * baseline_offset_long + rng.normal(0, 0.003)
    B_long_860 += baseline_offset_long
    B_long_740 += 0.90 * baseline_offset_long
    B_short_860 += baseline_offset_short
    B_short_740 += 0.90 * baseline_offset_short

    # -----------------------------------
    # ICH hemodynamics (subtle, delayed, partially overlapping with normal)
    # -----------------------------------

    if case == "ich":
        # Onset after baseline period to avoid trivially obvious shifts.
        start = int(rng.uniform(0.35, 0.60) * n)
        rise = int(rng.uniform(0.12, 0.25) * n)
        plateau = int(rng.uniform(0.10, 0.20) * n)
        end_rise = min(n, start + rise)
        end_plateau = min(n, end_rise + plateau)

        effect = np.zeros(n)
        if end_rise > start:
            s = np.linspace(0, 1, end_rise - start)
            # Sigmoid-like rise rather than linear ramp.
            effect[start:end_rise] = (1 / (1 + np.exp(-7 * (s - 0.5))))
        if end_plateau > end_rise:
            effect[end_rise:end_plateau] = effect[end_rise - 1]
        if n > end_plateau:
            # Partial recovery / washout, not complete return.
            tau = rng.uniform(0.12, 0.22) * n
            decay_idx = np.arange(n - end_plateau)
            effect[end_plateau:] = effect[end_plateau - 1] * np.exp(-decay_idx / max(tau, 1))

        # ICH should reduce detected intensity (higher absorption).
        # Keep magnitude modest so borderline cases exist.
        if rng.random() < 0.6:
            mag = rng.uniform(0.12, 0.24)
        else:
            mag = rng.uniform(0.03, 0.10)
        ich_delta = mag * effect + rng.normal(0, 0.0015, n)
        ich_side = rng.choice(["A", "B"])

        if ich_side == "A":
            A_long_860 -= 1.05 * ich_delta
            A_long_740 -= 0.85 * ich_delta
            # Partial extracerebral contamination creates imperfect short regression.
            A_short_860 -= 0.04 * ich_delta
            A_short_740 -= 0.03 * ich_delta
        else:
            B_long_860 -= 1.05 * ich_delta
            B_long_740 -= 0.85 * ich_delta
            B_short_860 -= 0.04 * ich_delta
            B_short_740 -= 0.03 * ich_delta
    else:
        # Some normal trials include weak unilateral vasomotor drift to induce overlap.
        if rng.random() < 0.45:
            side = rng.choice(["A", "B"])
            pseudo = 0.012 * np.sin(2 * np.pi * rng.uniform(0.01, 0.03) * t + rng.uniform(0, 2 * np.pi))
            if side == "A":
                A_long_860 += pseudo
                A_long_740 += 0.85 * pseudo
            else:
                B_long_860 += pseudo
                B_long_740 += 0.85 * pseudo
        # Rare non-ICH unilateral absorption event (e.g., probe pressure change / vasomotor effect).
        if rng.random() < 0.3:
            side = rng.choice(["A", "B"])
            start = int(rng.uniform(0.30, 0.70) * n)
            span = int(rng.uniform(0.18, 0.32) * n)
            end = min(n, start + span)
            trans = np.zeros(n)
            trans[start:end] = np.hanning(max(end - start, 1))
            mag = rng.uniform(0.05, 0.09)
            if side == "A":
                A_long_860 -= mag * trans
                A_long_740 -= 0.80 * mag * trans
            else:
                B_long_860 -= mag * trans
                B_long_740 -= 0.80 * mag * trans

    # -----------------------------------
    # Motion artifacts
    # -----------------------------------

    n_spikes = max(2, n // 180)
    spike_idx = rng.choice(n, n_spikes, replace=False)

    for idx in spike_idx:
        width = int(rng.integers(1, 5))
        spike = rng.uniform(0.03, 0.11)
        if rng.random() < 0.5:
            target_channels = [A_long_740, A_long_860, A_short_740, A_short_860]
        else:
            target_channels = [B_long_740, B_long_860, B_short_740, B_short_860]
        end = min(n, idx + width)
        span = end - idx
        shape = np.hanning(span) if span > 1 else np.array([1.0])
        for arr in target_channels:
            arr[idx:end] += spike * shape

    # -----------------------------------
    # Dark channel
    # -----------------------------------

    dark = 0.02 + rng.normal(0, 0.0015, n)

    return {

        "A_860_long":A_long_860,
        "A_740_long":A_long_740,

        "A_860_short":A_short_860,
        "A_740_short":A_short_740,

        "B_860_long":B_long_860,
        "B_740_long":B_long_740,

        "B_860_short":B_short_860,
        "B_740_short":B_short_740,

        "dark":dark
    }

# ------------------ Single Trial ------------------
def visualize_trial(rawA, rawB, procA, procB, detection_frame=None, title="Trial"):
    t = np.arange(len(rawA))
    t_proc = np.arange(len(procA))

    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=False)

    # Raw signals
    axs[0].plot(t, rawA, label="Optode A Raw")
    axs[0].plot(t, rawB, label="Optode B Raw")
    axs[0].set_title(f"{title} - Raw Signals")
    axs[0].legend()

    # Processed HbO signals
    axs[1].plot(t_proc, procA, label="Optode A HbO")
    axs[1].plot(t_proc, procB, label="Optode B HbO")
    axs[1].set_title(f"{title} - Processed HbO")
    axs[1].legend()

    # Asymmetry
    axs[2].plot(t_proc, np.array(procA) - np.array(procB), label="A - B")
    if detection_frame is not None:
        axs[2].axvline(detection_frame, linestyle="--", color='r', label="Detection")
    axs[2].set_title(f"{title} - Asymmetry")
    axs[2].set_xlabel("Frame")
    axs[2].legend()

    plt.tight_layout()
    plt.show()


def run_trial(data, visualize=True, trial_type="Trial"):

    reset_history()

    preprocessor = Preprocessor()
    preprocessor.reset()

    detection_frame = None

    sample_ids = {0:1, 1:0}

    rawA, rawB = [], []
    procA, procB = [], []

    for t in range(SAMPLES):

        frame = FakeFrame(
            frame_number=t,
            timestamp_ms=int(t*100),
            samples={
                0: FakeLogicalSample(
                    nm740_long=data["A_740_long"][t],
                    nm860_long=data["A_860_long"][t],
                    nm740_short=data["A_740_short"][t],
                    nm860_short=data["A_860_short"][t],
                    dark=data["dark"][t]
                ),
                1: FakeLogicalSample(
                    nm740_long=data["B_740_long"][t],
                    nm860_long=data["B_860_long"][t],
                    nm740_short=data["B_740_short"][t],
                    nm860_short=data["B_860_short"][t],
                    dark=data["dark"][t]
                )
            }
        )

        #print("optodes 860 long: ", data["A_860_long"][t], data["B_860_long"][t])
        # print("long std:", np.std(data["A_860_long"]))
        # print("short std:", np.std(data["A_860_short"]))
        # print("Correlation check:", np.corrcoef(data["A_860_long"], data["A_860_short"]))

        rawA.append(data["A_860_long"][t])
        rawB.append(data["B_860_long"][t])

        results = preprocessor.process_frame(frame, sample_ids)

        if len(results) == 0:
            continue

        for optode_id, r in results.items():
            if optode_id == 0:
                procA.append(r.hbo_long)
            if optode_id == 1:
                procB.append(r.hbo_long)

        # print("OD optode A: ", results[0].od_nm860_long)
        # print("OD optode B: ", results[1].od_nm860_long)
        # print("OD Diff: ", results[0].od_nm860_long - results[1].od_nm860_long)
        
        flags, _ = detect_ich(
            {
                opt_id: {
                    'HbO': r.hbo_long,
                    'HbR': r.hbr_long,
                    'OD_860': r.od_nm860_long,
                    'OD_740': r.od_nm740_long
                } 
                for opt_id, r in results.items()
            },
            [0, 1]
        )
        if any(flags.values()):
            detection_frame = len(procA)
            break

    if visualize and len(procA) > 0:
        visualize_trial(np.array(rawA), np.array(rawB), np.array(procA), np.array(procB), detection_frame, title=trial_type)

    return detection_frame is not None

# ------------------ Multi-Trial Evaluation ------------------
def evaluate_detector(m_trials):
    TP = TN = FP = FN = 0
    half_trials = m_trials // 2

    for _ in range(half_trials):
        # print("==================================ich========================================")
        data = generate_fnirs(case='ich')
        detected = run_trial(data, visualize=False, trial_type='ICH')
        TP += detected
        FN += not detected

        if _ % 50 == 0:
            print("TP: ", TP)
            print("FN: ", FN)

    for _ in range(half_trials):
        data = generate_fnirs(case='normal')
        detected = run_trial(data, visualize=False, trial_type='Normal')
        FP += detected
        TN += not detected

        if _ % 50 == 0:
            print("TN: ", TN)
            print("FP: ", FP)

    sensitivity = TP / (TP + FN) if (TP + FN) else 0
    specificity = TN / (TN + FP) if (TN + FP) else 0
    return {'TP': TP, 'FN': FN, 'TN': TN, 'FP': FP, 'sensitivity': sensitivity, 'specificity': specificity}

# ------------------ Run Evaluation ------------------
if __name__ == "__main__":
    results = evaluate_detector(5000)
    print('\nDetector Evaluation')
    print('-------------------')
    for k, v in results.items():
        print(f'{k}: {v}')
