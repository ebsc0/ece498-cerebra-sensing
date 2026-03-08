"""Application configuration."""

# Optode configuration
TOTAL_OPTODES = 16        # Hardware capacity (for head map display)
ACTIVE_OPTODES = [0, 1]   # Which of the 16 are connected
SAMPLE_RATE_HZ = 34.0

# Source selection
DATA_SOURCE = "simulator"  # "simulator" or "uart"

# UART source settings (used when DATA_SOURCE == "uart")
UART_PORT = "/dev/cu.usbmodem1103"
UART_BAUDRATE = 115200
UART_TIMEOUT_S = 0.1
UART_READ_CHUNK_SIZE = 256
UART_SYNC_WORD = b"\xA5\x5A"
UART_AUTO_RECONNECT = True
UART_RECONNECT_INITIAL_S = 0.5
UART_RECONNECT_MAX_S = 5.0

# Buffering safeguards for incomplete or delayed frames
BUFFER_STALE_TIMEOUT_MS = 2000
BUFFER_MAX_PENDING_FRAMES = 256

# ICH detection settings
ICH_WINDOW_SECONDS = 5.0
ICH_PERSISTENCE_WINDOW_SECONDS = 1.0
ICH_PERSISTENCE_RATIO = 0.6
ICH_SESSION_WINDOW_SECONDS = 5.0
ICH_SESSION_PERSISTENCE_RATIO = 0.6
ICH_SESSION_MIN_FRAMES = 5
# Pair mapping used for hemisphere comparisons in detection.
# For full hardware, define complete contralateral pairs (e.g. 0<->8, 1<->9, ...).
ICH_OPTODE_PAIRS = {
    0: 1,
    1: 0,
}

# UI settings
MAX_PLOT_POINTS = 50
UI_UPDATE_RATE_HZ = 10.0
GRAPH_Y_MIN = -0.0005
GRAPH_Y_MAX = 0.0005

# Optical parameters for MBLL
DPF_SHORT = 6.0           # Differential pathlength factor (short channel)
DPF_LONG = 6.0            # Differential pathlength factor (long channel)
DISTANCE_SHORT = 1.5      # Source-detector distance in cm (short channel)
DISTANCE_LONG = 3.0       # Source-detector distance in cm (long channel)

# Optode positions for headset map (16-optode layout)
OPTODE_POSITIONS = {
    0: (0.15, 0.75),  1: (0.35, 0.85),  2: (0.55, 0.85),  3: (0.75, 0.75),
    4: (0.15, 0.55),  5: (0.35, 0.65),  6: (0.55, 0.65),  7: (0.75, 0.55),
    8: (0.15, 0.35),  9: (0.35, 0.45), 10: (0.55, 0.45), 11: (0.75, 0.35),
   12: (0.15, 0.15), 13: (0.35, 0.25), 14: (0.55, 0.25), 15: (0.75, 0.15),
}

# Head map background image path (set to None to disable)
HEADMAP_IMAGE_PATH = 'assets/overhead_head_brain.png'

# Colors for optodes (up to 8 distinct colors)
OPTODE_COLORS = [
    '#1f77b4',  # blue
    '#ff7f0e',  # orange
    '#2ca02c',  # green
    '#d62728',  # red
    '#9467bd',  # purple
    '#8c564b',  # brown
    '#e377c2',  # pink
    '#7f7f7f',  # gray
]
