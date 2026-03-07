import math
import random
import struct
import threading
import time
from typing import Callable, Optional


class Simulator:
    """
    Realistic fNIRS signal simulator with optional ICH generation.
    """

    PACKET_FORMAT = "<I5f"
    PACKET_SIZE = struct.calcsize(PACKET_FORMAT)

    def __init__(self, num_optodes=8, sample_rate_hz=5.0):

        self.num_optodes = num_optodes
        self.sample_rate_hz = sample_rate_hz

        self._rng = random.Random()

        self._callback: Optional[Callable[[bytes], None]] = None
        self._thread = None
        self._stop_event = threading.Event()

        self._running = False

        # physiology parameters
        self.heart_freq = 1.1
        self.resp_freq = 0.25
        self.mayer_freq = 0.1
        self.drift_freq = 0.01

        # ICH parameters
        self.has_ich = False
        self.ich_start_frame = 0
        self.ich_optodes = set()
        self.ich_growth_rate = 0.001

    # -----------------------------
    # Trial Control
    # -----------------------------

    def inject_ich(self, optodes=None, start_frame=200):

        self.has_ich = True
        self.ich_start_frame = start_frame

        if optodes is None:
            optodes = [self._rng.randrange(self.num_optodes)]

        self.ich_optodes = set(optodes)

    def clear_ich(self):

        self.has_ich = False
        self.ich_optodes = set()

    def randomize_trial(self):

        if self._rng.random() < 0.5:
            self.inject_ich(
                optodes=[self._rng.randrange(self.num_optodes)],
                start_frame=self._rng.randint(100, 300),
            )
        else:
            self.clear_ich()

    # -----------------------------
    # Physiological Signal Model
    # -----------------------------

    def phys_signal(self, base, t):

        cardiac = 0.03 * math.sin(2 * math.pi * self.heart_freq * t)
        respiration = 0.02 * math.sin(2 * math.pi * self.resp_freq * t)
        mayer = 0.02 * math.sin(2 * math.pi * self.mayer_freq * t)

        slow_drift = 0.05 * math.sin(2 * math.pi * self.drift_freq * t)

        return base + cardiac + respiration + mayer + slow_drift

    # -----------------------------
    # ICH Model
    # -----------------------------

    def ich_drift(self, frame_number, optode_id):

        if not self.has_ich:
            return 0.0

        if optode_id not in self.ich_optodes:
            return 0.0

        if frame_number < self.ich_start_frame:
            return 0.0

        frames_since = frame_number - self.ich_start_frame

        # gradual exponential accumulation
        return 0.3 * (1 - math.exp(-self.ich_growth_rate * frames_since))

    # -----------------------------
    # Packet Generation
    # -----------------------------

    def _generate_packet(self, optode_id, frame_number):

        t = frame_number / self.sample_rate_hz

        # baseline intensities
        long_860_base = 2.0 + 0.05 * optode_id
        long_740_base = 1.8 + 0.05 * optode_id

        short_860_base = 1.4 + 0.03 * optode_id
        short_740_base = 1.2 + 0.03 * optode_id

        dark_base = 0.02

        long_860 = self.phys_signal(long_860_base, t)
        long_740 = self.phys_signal(long_740_base, t)

        short_860 = self.phys_signal(short_860_base, t)
        short_740 = self.phys_signal(short_740_base, t)

        dark = dark_base + 0.002 * math.sin(2 * math.pi * 0.05 * t)

        # -----------------------------
        # ICH Effect (long channels only)
        # -----------------------------

        bleed = self.ich_drift(frame_number, optode_id)

        long_860 += bleed
        long_740 += bleed

        # -----------------------------
        # Noise
        # -----------------------------

        noise_std = 0.01

        long_860 += self._rng.gauss(0, noise_std)
        long_740 += self._rng.gauss(0, noise_std)
        short_860 += self._rng.gauss(0, noise_std)
        short_740 += self._rng.gauss(0, noise_std)

        dark += self._rng.gauss(0, 0.002)

        # -----------------------------
        # Motion artifact (rare)
        # -----------------------------

        if self._rng.random() < 0.003:

            spike = self._rng.uniform(0.1, 0.3)

            long_860 += spike
            long_740 += spike
            short_860 += spike
            short_740 += spike

        # -----------------------------
        # Pack metadata
        # -----------------------------

        metadata = (frame_number << 4) | optode_id

        return struct.pack(
            self.PACKET_FORMAT,
            metadata,
            float(long_740),
            float(long_860),
            float(short_740),
            float(short_860),
            float(dark),
        )

    # -----------------------------
    # Streaming Loop
    # -----------------------------

    def _run_loop(self):

        interval = 1.0 / self.sample_rate_hz
        frame_number = 0

        while not self._stop_event.is_set():

            for optode_id in range(self.num_optodes):

                if self._callback:
                    self._callback(
                        self._generate_packet(optode_id, frame_number)
                    )

            frame_number += 1
            time.sleep(interval)

    # -----------------------------
    # Control
    # -----------------------------

    def start(self, callback):

        if self._running:
            raise RuntimeError("Simulator already running")

        self._callback = callback
        self._stop_event.clear()

        self.randomize_trial()

        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
        )

        self._thread.start()
        self._running = True

    def stop(self):

        if not self._running:
            return

        self._stop_event.set()

        if self._thread:
            self._thread.join(timeout=1)

        self._running = False

    def is_running(self):
        return self._running