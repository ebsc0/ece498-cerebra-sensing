"""Phase-aware packet buffer and frame assembly."""

from __future__ import annotations

import time
from typing import Dict, Iterable, Optional

from config import ACTIVE_OPTODES
from packets import AssembledFrame, PACKET_SIZE, PhasePacket, VALID_PHASES, decode_packet, reconstruct_logical_sample


class PhaseFrameBuffer:
    """Buffers phase packets and emits completed logical frames."""

    def __init__(
        self,
        active_optodes: Optional[Iterable[int]] = None,
        stale_timeout_ms: int = 2000,
        max_pending_frames: int = 256,
    ):
        source_optodes = ACTIVE_OPTODES if active_optodes is None else active_optodes
        self.active_optodes = tuple(int(optode_id) for optode_id in source_optodes)
        self._active_optode_set = set(self.active_optodes)
        self.stale_timeout_ms = int(stale_timeout_ms)
        self.max_pending_frames = int(max_pending_frames)
        self._pending: Dict[int, Dict[int, Dict[int, PhasePacket]]] = {}
        self._start_time_ms: Optional[int] = None
        self._dropped_frames = 0

    def _evict_stale_and_overflow(self, current_timestamp_ms: int) -> None:
        if self._pending:
            stale_frames = []
            for frame_number, per_optode in self._pending.items():
                oldest_ts = min(
                    phase_packet.timestamp_ms
                    for optode_phases in per_optode.values()
                    for phase_packet in optode_phases.values()
                )
                if current_timestamp_ms - oldest_ts > self.stale_timeout_ms:
                    stale_frames.append(frame_number)

            for frame_number in stale_frames:
                self._pending.pop(frame_number, None)
                self._dropped_frames += 1

        if len(self._pending) > self.max_pending_frames:
            overflow = len(self._pending) - self.max_pending_frames
            for frame_number in sorted(self._pending.keys())[:overflow]:
                self._pending.pop(frame_number, None)
                self._dropped_frames += 1

    def add_packet(
        self,
        packet: bytes,
        packet_timestamp_ms: Optional[int] = None,
    ) -> Optional[AssembledFrame]:
        if len(packet) != PACKET_SIZE:
            return None

        current_time_ms = int(time.time() * 1000) if packet_timestamp_ms is None else int(packet_timestamp_ms)
        if self._start_time_ms is None:
            self._start_time_ms = current_time_ms
        relative_timestamp_ms = current_time_ms - self._start_time_ms

        try:
            phase_packet = decode_packet(packet, relative_timestamp_ms)
        except ValueError:
            return None

        if phase_packet.optode_id not in self._active_optode_set:
            return None
        if phase_packet.phase not in VALID_PHASES:
            return None

        self._evict_stale_and_overflow(relative_timestamp_ms)

        frame_packets = self._pending.setdefault(phase_packet.frame_number, {})
        optode_packets = frame_packets.setdefault(phase_packet.optode_id, {})
        optode_packets[phase_packet.phase] = phase_packet

        self._evict_stale_and_overflow(relative_timestamp_ms)
        if phase_packet.frame_number not in self._pending:
            return None

        pending_frame = self._pending[phase_packet.frame_number]
        is_complete = all(
            optode_id in pending_frame and all(phase in pending_frame[optode_id] for phase in VALID_PHASES)
            for optode_id in self.active_optodes
        )
        if not is_complete:
            return None

        completed = self._pending.pop(phase_packet.frame_number)
        frame_timestamp_ms = min(
            packet_record.timestamp_ms
            for optode_phases in completed.values()
            for packet_record in optode_phases.values()
        )
        phase_packets = {
            optode_id: dict(optode_phases)
            for optode_id, optode_phases in completed.items()
        }
        logical_samples = {
            optode_id: reconstruct_logical_sample(optode_phases, frame_timestamp_ms)
            for optode_id, optode_phases in phase_packets.items()
        }
        return AssembledFrame(
            frame_number=phase_packet.frame_number,
            timestamp_ms=frame_timestamp_ms,
            phase_packets=phase_packets,
            logical_samples=logical_samples,
        )

    def dropped_frames(self) -> int:
        return self._dropped_frames
