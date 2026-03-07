import math
import random
import struct
import threading
import time
from typing import Callable, Optional


class Simulator:
    """
    fNIRS simulator producing physiologically plausible signals
    with optional intracranial hemorrhage (ICH).

    Compatible with evaluation pipelines expecting packets:
    metadata | long_740 | long_860 | short_740 | short_860 | dark
    """

    PACKET_FORMAT = "<I5f"
    PACKET_SIZE = struct.calcsize(PACKET_FORMAT)

    # extinction coefficients (approx relative values)
    EXT = {
        740: (0.45, 1.60),   # (HbO, HbR)
        860: (1.20, 0.80),
    }

    def __init__(self, num_optodes=8, sample_rate_hz=5.0):

        self.num_optodes = num_optodes
        self.sample_rate_hz = sample_rate_hz

        self._rng = random.Random()

        self._callback: Optional[Callable[[bytes], None]] = None
        self._thread = None
        self._stop_event = threading.Event()

        self._running = False

        # physiological frequencies
        self.heart_freq = 1.1
        self.resp_freq = 0.25
        self.mayer_freq = 0.1

        # optode phase offsets (more realistic)
        self.phase_offsets = [
            self._rng.uniform(0, 2 * math.pi) for _ in range(num_optodes)
        ]

        # ICH parameters
        self.has_ich = False
        self.ich_start_frame = 0
        self.ich_optodes = set()

        # slow hemorrhage progression
        self.ich_growth_rate = 0.002

    # ------------------------------------------------
    # Trial randomization
    # ------------------------------------------------

    def randomize_trial(self):

        if self._rng.random() < 0.5:

            self.has_ich = True

            self.ich_start_frame = self._rng.randint(120, 300)

            num = self._rng.randint(1, 2)

            self.ich_optodes = set(
                self._rng.sample(range(self.num_optodes), num)
            )

            print("Simulator: ICH on optodes", self.ich_optodes)

        else:

            self.has_ich = False
            self.ich_optodes = set()

            print("Simulator: NORMAL")

    # ------------------------------------------------
    # Physiological Hb signals
    # ------------------------------------------------

    def phys_hb(self, t, optode):

        phase = self.phase_offsets[optode]

        cardiac = 0.4 * math.sin(2 * math.pi * self.heart_freq * t + phase)
        resp = 0.2 * math.sin(2 * math.pi * self.resp_freq * t + phase)
        mayer = 0.3 * math.sin(2 * math.pi * self.mayer_freq * t + phase)

        hbo = cardiac + resp + mayer
        hbr = -0.6 * cardiac - 0.2 * resp

        return hbo, hbr

    # ------------------------------------------------
    # ICH progression model
    # ------------------------------------------------

    def ich_effect(self, frame_number, optode):

        if not self.has_ich:
            return 0.0, 0.0

        if optode not in self.ich_optodes:
            return 0.0, 0.0

        if frame_number < self.ich_start_frame:
            return 0.0, 0.0

        frames = frame_number - self.ich_start_frame

        # gradual accumulation
        hbr = 3.0 * (1 - math.exp(-self.ich_growth_rate * frames))
        hbo = -0.2 * hbr

        return hbo, hbr

    # ------------------------------------------------
    # Hb -> wavelength absorption
    # ------------------------------------------------

    def hb_to_wavelength(self, hbo, hbr, wl):

        e_hbo, e_hbr = self.EXT[wl]

        return e_hbo * hbo + e_hbr * hbr

    # ------------------------------------------------
    # Packet generation
    # ------------------------------------------------

    def _generate_packet(self, optode_id, frame_number):

        t = frame_number / self.sample_rate_hz

        # baseline intensities
        long_740 = 1.8 + 0.05 * optode_id
        long_860 = 2.0 + 0.05 * optode_id

        short_740 = 1.2 + 0.03 * optode_id
        short_860 = 1.4 + 0.03 * optode_id

        dark = 0.02

        # physiology
        hbo, hbr = self.phys_hb(t, optode_id)

        # hemorrhage contribution
        ich_hbo, ich_hbr = self.ich_effect(frame_number, optode_id)

        hbo += ich_hbo
        hbr += ich_hbr

        # convert to wavelength absorption
        abs740 = self.hb_to_wavelength(hbo, hbr, 740)
        abs860 = self.hb_to_wavelength(hbo, hbr, 860)

        long_740 += abs740
        long_860 += abs860

        # short channels: superficial physiology only
        short_hbo, short_hbr = self.phys_hb(t, optode_id)

        short_740 += self.hb_to_wavelength(short_hbo, short_hbr, 740)
        short_860 += self.hb_to_wavelength(short_hbo, short_hbr, 860)

        # noise
        noise = 0.01

        long_740 += self._rng.gauss(0, noise)
        long_860 += self._rng.gauss(0, noise)

        short_740 += self._rng.gauss(0, noise)
        short_860 += self._rng.gauss(0, noise)

        dark += self._rng.gauss(0, 0.002)

        # occasional motion artifact
        if self._rng.random() < 0.002:

            spike = self._rng.uniform(0.1, 0.3)

            long_740 += spike
            long_860 += spike
            short_740 += spike
            short_860 += spike

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

    # ------------------------------------------------
    # Streaming loop
    # ------------------------------------------------

    def _run_loop(self):

        interval = 1.0 / self.sample_rate_hz
        frame = 0

        while not self._stop_event.is_set():

            start = time.time()

            for optode in range(self.num_optodes):

                if self._callback:

                    self._callback(
                        self._generate_packet(optode, frame)
                    )

            frame += 1

            elapsed = time.time() - start
            sleep = max(0, interval - elapsed)

            time.sleep(sleep)

    # ------------------------------------------------
    # Control
    # ------------------------------------------------

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

        # avoid joining from same thread
        if self._thread and threading.current_thread() != self._thread:
            self._thread.join(timeout=1)

        self._running = False

    def is_running(self):

        return self._running