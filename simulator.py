import math
import random
import struct
import threading
import time
from typing import Callable, Optional, Sequence

from data_source import DataSource, PacketCallback, SourceEvent, SourceEventCallback
from packets import PACKET_FORMAT, PHASE_740, PHASE_860, PHASE_DARK, encode_metadata


class SimulatorSource(DataSource):
    """Simulator-backed packet source."""

    def __init__(self, *, active_optodes: Sequence[int], sample_rate_hz: float):
        self._simulator = Simulator(active_optodes=active_optodes, sample_rate_hz=sample_rate_hz)
        self._event_callback: Optional[SourceEventCallback] = None

    def start(
        self,
        packet_callback: PacketCallback,
        event_callback: Optional[SourceEventCallback] = None,
    ) -> None:
        self._event_callback = event_callback
        self._simulator.start(packet_callback)
        self._emit("connected", "Simulator source started.")

    def stop(self) -> None:
        was_running = self._simulator.is_running()
        self._simulator.stop()
        if was_running:
            self._emit("stopped", "Simulator source stopped.")

    def is_running(self) -> bool:
        return self._simulator.is_running()

    def _emit(self, kind: str, message: str) -> None:
        if self._event_callback:
            self._event_callback(SourceEvent(kind=kind, message=message))


class Simulator:
    """Generates structured phase packets using the d0..d3 wire layout.

    Physical meaning is reversed from index order:
    - d0 is farthest
    - d3 is nearest
    """

    def __init__(self, active_optodes: Sequence[int], sample_rate_hz: float = 5.0):
        self.active_optodes = [int(optode_id) for optode_id in active_optodes]
        self.sample_rate_hz = float(sample_rate_hz)
        self._callback: Optional[Callable[[bytes], None]] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False
        self._rng = random.Random()
        self._optode_baseline: dict[int, float] = {}
        self._optode_surface_phase: dict[int, float] = {}
        self._optode_deep_phase: dict[int, float] = {}
        self._optode_drift_phase: dict[int, float] = {}
        self._reset_signal_state()

    def _reset_signal_state(self) -> None:
        self._optode_baseline = {
            optode_id: self._rng.uniform(2500.0, 3500.0)
            for optode_id in self.active_optodes
        }
        self._optode_surface_phase = {
            optode_id: self._rng.uniform(0.0, 2.0 * math.pi)
            for optode_id in self.active_optodes
        }
        self._optode_deep_phase = {
            optode_id: self._rng.uniform(0.0, 2.0 * math.pi)
            for optode_id in self.active_optodes
        }
        self._optode_drift_phase = {
            optode_id: self._rng.uniform(0.0, 2.0 * math.pi)
            for optode_id in self.active_optodes
        }

    def _build_diode_values(
        self,
        *,
        baseline: float,
        near_scale: float,
        far_scale: float,
        surface: float,
        deep: float,
        drift: float,
        near_surface_gain: float,
        far_surface_gain: float,
        near_deep_gain: float,
        far_deep_gain: float,
        near_drift_gain: float,
        far_drift_gain: float,
        dark_value: float,
    ) -> tuple[float, float, float, float]:
        near_value = (
            baseline * near_scale
            + near_surface_gain * surface
            + near_deep_gain * deep
            + near_drift_gain * drift
            + self._rng.gauss(0.0, 6.0)
        )
        far_value = (
            baseline * far_scale
            + far_surface_gain * surface
            + far_deep_gain * deep
            + far_drift_gain * drift
            + self._rng.gauss(0.0, 8.0)
        )
        inner_near_value = near_value + 0.35 * (far_value - near_value) + self._rng.gauss(0.0, 3.0)
        inner_far_value = near_value + 0.70 * (far_value - near_value) + self._rng.gauss(0.0, 3.0)

        min_signal = dark_value + 1.0
        near_value = max(near_value, min_signal)
        inner_near_value = max(inner_near_value, min_signal)
        inner_far_value = max(inner_far_value, min_signal)
        far_value = max(far_value, min_signal)
        return far_value, inner_far_value, inner_near_value, near_value

    def _channel_values(
        self,
        optode_id: int,
        frame_number: int,
    ) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float], float]:
        t = frame_number / self.sample_rate_hz if self.sample_rate_hz > 0 else float(frame_number)
        baseline = self._optode_baseline[optode_id]
        surface = math.sin((2.0 * math.pi * 0.70 * t) + self._optode_surface_phase[optode_id])
        deep = math.sin((2.0 * math.pi * 0.12 * t) + self._optode_deep_phase[optode_id])
        drift = math.sin((2.0 * math.pi * 0.03 * t) + self._optode_drift_phase[optode_id])
        dark_value = max(0.5, 8.0 + self._rng.gauss(0.0, 0.8))

        values_740 = self._build_diode_values(
            baseline=baseline,
            near_scale=0.74,
            far_scale=1.00,
            surface=surface,
            deep=deep,
            drift=drift,
            near_surface_gain=70.0,
            far_surface_gain=52.0,
            near_deep_gain=35.0,
            far_deep_gain=3200.0,
            near_drift_gain=35.0,
            far_drift_gain=220.0,
            dark_value=dark_value,
        )
        values_860 = self._build_diode_values(
            baseline=baseline * 1.03,
            near_scale=0.73,
            far_scale=1.01,
            surface=surface,
            deep=-deep,
            drift=drift,
            near_surface_gain=60.0,
            far_surface_gain=44.0,
            near_deep_gain=28.0,
            far_deep_gain=2800.0,
            near_drift_gain=28.0,
            far_drift_gain=180.0,
            dark_value=dark_value,
        )
        return values_740, values_860, dark_value

    def _pack_phase_packet(
        self,
        optode_id: int,
        frame_number: int,
        phase: int,
        far_value: float,
        inner_far_value: float,
        inner_near_value: float,
        near_value: float,
        dark_value: float,
    ) -> bytes:
        if phase == PHASE_DARK:
            values = (
                max(0.5, dark_value + self._rng.gauss(0.0, 0.15)),
                max(0.5, dark_value + self._rng.gauss(0.0, 0.15)),
                max(0.5, dark_value + self._rng.gauss(0.0, 0.15)),
                max(0.5, dark_value + self._rng.gauss(0.0, 0.15)),
            )
        elif phase in (PHASE_740, PHASE_860):
            values = (far_value, inner_far_value, inner_near_value, near_value)
        else:
            raise ValueError(f"Unsupported phase: {phase}")

        metadata = encode_metadata(frame_number, optode_id, phase)
        return struct.pack(PACKET_FORMAT, metadata, *[float(value) for value in values])

    def _run_loop(self) -> None:
        interval = 1.0 / self.sample_rate_hz if self.sample_rate_hz > 0 else 0.2
        frame_number = 0
        while not self._stop_event.is_set():
            for optode_id in self.active_optodes:
                if self._stop_event.is_set():
                    break
                values_740, values_860, dark_value = self._channel_values(optode_id, frame_number)
                if self._callback:
                    self._callback(
                        self._pack_phase_packet(
                            optode_id,
                            frame_number,
                            PHASE_740,
                            *values_740,
                            dark_value,
                        )
                    )
                    self._callback(
                        self._pack_phase_packet(
                            optode_id,
                            frame_number,
                            PHASE_860,
                            *values_860,
                            dark_value,
                        )
                    )
                    self._callback(
                        self._pack_phase_packet(
                            optode_id,
                            frame_number,
                            PHASE_DARK,
                            *values_740,
                            dark_value,
                        )
                    )
            frame_number += 1
            time.sleep(interval)

    def start(self, callback: Callable[[bytes], None]) -> None:
        if self._running:
            raise RuntimeError("Simulator is already running")
        self._callback = callback
        self._stop_event.clear()
        self._reset_signal_state()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._running = True

    def stop(self) -> None:
        if not self._running:
            return
        self._stop_event.set()
        if self._thread:
            self._thread.join()
        self._running = False
        self._thread = None

    def is_running(self) -> bool:
        return self._running
