import numpy as np
import matplotlib.pyplot as plt
from preprocessor import Preprocessor

# ------------------ Generate Synthetic Data ------------------
def generate_fnirs(duration_sec=60, fs=10, seed=42):
    np.random.seed(seed)
    t = np.arange(0, duration_sec, 1/fs)
    n = len(t)
    # base signals
    long_860 = 2.0 + 0.03*np.sin(2*np.pi*1.2*t) + 0.015*np.sin(2*np.pi*0.1*t) + np.random.normal(0,0.01,n)
    long_740 = 1.8 + 0.03*np.sin(2*np.pi*1.2*t) + 0.015*np.sin(2*np.pi*0.1*t) + np.random.normal(0,0.01,n)
    short_860 = 1.5 + 0.03*np.sin(2*np.pi*1.2*t) + np.random.normal(0,0.01,n)
    short_740 = 1.3 + 0.03*np.sin(2*np.pi*1.2*t) + np.random.normal(0,0.01,n)
    dark = 0.02 + 0.002*np.sin(2*np.pi*0.05*t) + np.random.normal(0,0.002,n)
    return {'long_860': long_860, 'long_740': long_740, 'short_860': short_860, 'short_740': short_740, 'dark': dark}

# ------------------ Fake Frame Classes for Preprocessor ------------------
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

# ------------------ Run Preprocessor ------------------
def run_preprocessor(data):
    preprocessor = Preprocessor()
    preprocessor.reset()

    sample_ids = {1:1}

    hbo = []
    hbr = []
    od740 = []
    od860 = []
    raw740 = []
    raw860 = []

    for i in range(len(data['long_860'])):
        frame = FakeFrame(
            frame_number=i,
            timestamp_ms=i*100,
            samples={
                1: FakeLogicalSample(
                    data['long_740'][i],
                    data['long_860'][i],
                    data['short_740'][i],
                    data['short_860'][i],
                    data['dark'][i]
                )
            }
        )

        raw740.append(data['long_740'][i])
        raw860.append(data['long_860'][i])

        results = preprocessor.process_frame(frame, sample_ids)
        if not results:
            continue

        r = results[1]
        hbo.append(r.hbo_long)
        hbr.append(r.hbr_long)
        od740.append(r.od_nm740_long)
        od860.append(r.od_nm860_long)

    return raw740, raw860, hbo, hbr, od740, od860

# ------------------ Visualization ------------------
def visualize(raw740, raw860, hbo, hbr, od740, od860):
    t_raw = np.arange(len(raw740))
    t_proc = np.arange(len(hbo))

    fig, axs = plt.subplots(3,1, figsize=(10,8), sharex=False)

    axs[0].plot(t_raw, raw740, label='740 raw')
    axs[0].plot(t_raw, raw860, label='860 raw')
    axs[0].set_title('Raw Intensities')
    axs[0].legend()

    axs[1].plot(t_proc, od740, label='OD 740')
    axs[1].plot(t_proc, od860, label='OD 860')
    axs[1].set_title('Optical Density')
    axs[1].legend()

    axs[2].plot(t_proc, hbo, label='HbO')
    axs[2].plot(t_proc, hbr, label='HbR')
    axs[2].set_title('Hemoglobin Concentration')
    axs[2].legend()

    plt.tight_layout()
    plt.show()

# ------------------ Main ------------------
data = generate_fnirs()
raw740, raw860, hbo, hbr, od740, od860 = run_preprocessor(data)
visualize(raw740, raw860, hbo, hbr, od740, od860)