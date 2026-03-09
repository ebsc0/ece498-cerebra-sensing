import numpy as np
import matplotlib.pyplot as plt

def generate_fnirs(duration_sec=60, fs=10, seed=42):
    np.random.seed(seed)
    t = np.arange(0, duration_sec, 1/fs)
    n = len(t)

    # --- Base intensities ---
    long_base = 2.0
    short_base = 1.5

    # --- Physiological signals ---
    heart_freq = 1.2   # Hz
    resp_freq = 0.25   # Hz
    brain_freq = 0.1   # Hz

    scalp_signal = 0.03*np.sin(2*np.pi*heart_freq*t) + 0.02*np.sin(2*np.pi*resp_freq*t)
    brain_signal = 0.03*np.sin(2*np.pi*brain_freq*t)

    # --- Generate channels ---
    long_channel = long_base + scalp_signal + brain_signal + np.random.normal(0, 0.01, n)
    short_channel = short_base + 0.5*scalp_signal + np.random.normal(0, 0.02, n)

    return t, long_channel, short_channel

# ---------------- Generate signal ----------------
t, long_ch, short_ch = generate_fnirs(duration_sec=60, fs=10)

# ---------------- Compute correlation ----------------
corr = np.corrcoef(long_ch, short_ch)[0,1]
print(f"Correlation between short and long channels: {corr:.3f}")

# ---------------- Plot signals ----------------
plt.figure(figsize=(12,6))
plt.plot(t, long_ch, label='Long Channel (Brain + Scalp)')
plt.plot(t, short_ch, label='Short Channel (Scalp Only)')
plt.xlabel('Time (s)')
plt.ylabel('Optical Density / Signal')
plt.title('Synthetic fNIRS Signals')
plt.legend()
plt.show()