import math
import random
import struct
import threading
import time
from typing import Callable, Optional


class Simulator:
    """Generates structured packet data.

    packet structure:
        24 bytes => [metadata (int8)][740_long (float)][860_long (float)][740_short (float)][860_short (float)][dark (float)]

    metadata structure (32 bits):
        metadata[31:4] = frame # 
        metadata[3:0] = optode #
    """

    PACKET_FORMAT = '<I5f'  # uint32 metadata + 5 floats
    PACKET_SIZE = struct.calcsize(PACKET_FORMAT)  # 24 bytes

    def __init__(self, num_optodes: int = 2, sample_rate_hz: float = 5.0):
        self.num_optodes = num_optodes
        self.sample_rate_hz = sample_rate_hz
        self._callback: Optional[Callable[[bytes], None]] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False
        self._rng = random.Random()
        self._optode_baseline: list[float] = []
        self._optode_phase: list[float] = []
        self._reset_signal_state()

    def _reset_signal_state(self):
        """Initialize per-optode signal parameters."""
        self._optode_baseline = [
            self._rng.uniform(2500.0, 3500.0)
            for _ in range(self.num_optodes)
        ]
        self._optode_phase = [
            self._rng.uniform(0.0, 2.0 * math.pi)
            for _ in range(self.num_optodes)
        ]

    def _generate_packet(self, optode_id: int, frame_number: int) -> bytes:
        """Generate random fNIRS-like raw values.

        Args:
            optode_id: optode ID
            frame_number: frame number
        
        Returns:
            Struct of bytes with packet structure as defined
        """
        t = frame_number / self.sample_rate_hz if self.sample_rate_hz > 0 else float(frame_number)
        phase = (2.0 * math.pi * 0.2 * t) + self._optode_phase[optode_id]
        pulsatile = math.sin(phase)
        baseline = self._optode_baseline[optode_id]

        long_740 = baseline + 30.0 * pulsatile + self._rng.gauss(0.0, 8.0)
        long_860 = (baseline * 1.03) + 26.0 * pulsatile + self._rng.gauss(0.0, 8.0)
        short_740 = (baseline * 0.72) + 12.0 * pulsatile + self._rng.gauss(0.0, 6.0)
        short_860 = (baseline * 0.75) + 10.0 * pulsatile + self._rng.gauss(0.0, 6.0)
        dark = max(0.5, 8.0 + self._rng.gauss(0.0, 0.8))

        # Ensure channels remain above dark so log-domain preprocessing is valid.
        min_signal = dark + 1.0
        long_740 = max(long_740, min_signal)
        long_860 = max(long_860, min_signal)
        short_740 = max(short_740, min_signal)
        short_860 = max(short_860, min_signal)

        metadata = (frame_number << 4) | optode_id  # 28 bits frame, 4 bits optode
        return struct.pack(
            self.PACKET_FORMAT,
            metadata,
            float(long_740),
            float(long_860),
            float(short_740),
            float(short_860),
            float(dark),
        )

    def _run_loop(self):
        """Call _generate_packet() at sample Hz"""
        interval = 1.0 / self.sample_rate_hz
        frame_number = 0
        while not self._stop_event.is_set():
            for optode_id in range(self.num_optodes):
                if self._stop_event.is_set():
                    break
                if self._callback:
                    self._callback(self._generate_packet(optode_id, frame_number))
            frame_number += 1
            time.sleep(interval)

    def start(self, callback: Callable[[bytes], None]):
        """Start simulator

        Args:
            callback: callback function for handling generated data
        """
        if self._running:
            raise RuntimeError("Simulator is already running")
        self._callback = callback
        self._stop_event.clear()
        self._reset_signal_state()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._running = True

    def stop(self):
        """Stop simulator"""
        if not self._running:
            return
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        self._running = False
        self._thread = None

    def is_running(self) -> bool:
        return self._running


if __name__ == '__main__':
    packets = []

    def collect(data: bytes):
        packets.append(data)

    sim = Simulator(num_optodes=2, sample_rate_hz=5.0)
    sim.start(collect)
    time.sleep(0.5)
    sim.stop()

    print(f"Collected {len(packets)} packets ({Simulator.PACKET_SIZE} bytes each)")
    for i, packet in enumerate(packets[:6]):
        metadata = struct.unpack('<I', packet[0:4])[0]
        frame = metadata >> 4
        optode = metadata & 0xF
        print(f"[{i}] frame={frame} optode={optode} | {packet.hex()}")
