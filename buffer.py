import struct
import time
from typing import Dict, Optional, Tuple, NamedTuple


def decode_metadata(packet: bytes) -> Tuple[int, int]:
    """Extract frame number and optode ID from packet metadata."""
    metadata = struct.unpack('<I', packet[0:4])[0]
    frame = metadata >> 4       # upper 28 bits
    optode = metadata & 0xF     # lower 4 bits
    return frame, optode


class Packet(NamedTuple):
    """Packet with timestamp added at buffer insertion time."""
    packet: bytes
    timestamp_ms: int


class CompleteFrame(NamedTuple):
    """Complete frame with all optode packets."""
    frame_number: int
    timestamp_ms: int
    packets: Dict[int, bytes]  # {optode_id: packet_bytes}


class Buffer:
    """Buffers packets and groups them by frame number."""

    def __init__(
        self,
        num_optodes: int,
        stale_timeout_ms: int = 2000,
        max_pending_frames: int = 256,
    ):
        self.num_optodes = num_optodes
        self.stale_timeout_ms = stale_timeout_ms
        self.max_pending_frames = max_pending_frames
        self._pending: Dict[int, Dict[int, Packet]] = {}  # frame -> {optode: Packet}
        self._start_time_ms: Optional[int] = None  # Set on first packet
        self._dropped_frames = 0

    def _evict_stale_and_overflow(self, current_timestamp_ms: int):
        """Evict stale or excessive pending frames to prevent unbounded growth."""
        if self._pending:
            stale_frames = []
            for frame_number, packets in self._pending.items():
                oldest_ts = min(tp.timestamp_ms for tp in packets.values())
                if current_timestamp_ms - oldest_ts > self.stale_timeout_ms:
                    stale_frames.append(frame_number)

            for frame_number in stale_frames:
                self._pending.pop(frame_number, None)
                self._dropped_frames += 1

        if len(self._pending) > self.max_pending_frames:
            # Drop oldest frame numbers first.
            overflow = len(self._pending) - self.max_pending_frames
            for frame_number in sorted(self._pending.keys())[:overflow]:
                self._pending.pop(frame_number, None)
                self._dropped_frames += 1

    def add_packet(self, packet: bytes) -> Optional[CompleteFrame]:
        """
        Add a packet to the buffer.
        
        Args:
            packet: Raw packet bytes with uint32 metadata header.
        
        Returns:
            CompleteFrame if frame is complete, None otherwise.
        """
        frame, optode = decode_metadata(packet)
        
        # Auto-initialize start time on first packet (first frame = 0ms)
        current_time_ms = int(time.time() * 1000)
        if self._start_time_ms is None:
            self._start_time_ms = current_time_ms
        
        # Relative timestamp from session start
        timestamp_ms = current_time_ms - self._start_time_ms
        self._evict_stale_and_overflow(timestamp_ms)

        if frame not in self._pending:
            self._pending[frame] = {}

        self._pending[frame][optode] = Packet(packet, timestamp_ms)

        # Check if frame is complete
        if len(self._pending[frame]) == self.num_optodes:
            pending_frame = self._pending.pop(frame)
            # Use timestamp from first packet received for this frame
            first_timestamp = min(tp.timestamp_ms for tp in pending_frame.values())
            packets = {optode_id: tp.packet for optode_id, tp in pending_frame.items()}
            return CompleteFrame(
                frame_number=frame,
                timestamp_ms=first_timestamp,
                packets=packets
            )

        return None

    def pending_frames(self) -> int:
        """Return number of incomplete frames in buffer."""
        return len(self._pending)

    def clear(self):
        """Clear all pending frames and reset start time."""
        self._pending.clear()
        self._start_time_ms = None  # Will be set on first packet
        self._dropped_frames = 0

    def dropped_frames(self) -> int:
        """Return total number of dropped incomplete frames."""
        return self._dropped_frames


if __name__ == '__main__':
    from simulator import Simulator

    frames = []
    buffer = Buffer(num_optodes=2)

    def collect(data: bytes):
        complete_frame = buffer.add_packet(data)
        if complete_frame:
            frames.append(complete_frame)

    sim = Simulator(num_optodes=2, sample_rate_hz=5.0)
    sim.start(collect)
    time.sleep(0.5)
    sim.stop()

    print(f"Collected {len(frames)} complete frames")
    for complete_frame in frames[:3]:
        print(f"\nFrame {complete_frame.frame_number} @ {complete_frame.timestamp_ms}ms:")
        for optode_id, packet in sorted(complete_frame.packets.items()):
            frame_num, optode = decode_metadata(packet)
            print(f"  optode={optode} | frame_num={frame_num} | {packet.hex()}")
