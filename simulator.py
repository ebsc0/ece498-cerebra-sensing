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
        self._counter = 0

    def _generate_packet(self, optode_id: int, frame_number: int) -> bytes:
        """Generate random data (data+1 for every next channel)

        Args:
            optode_id: optode ID
            frame_number: frame number
        
        Returns:
            Struct of bytes with packet structure as defined
        """
        self._counter += 1
        metadata = (frame_number << 4) | optode_id  # 28 bits frame, 4 bits optode
        return struct.pack(
            self.PACKET_FORMAT,
            metadata,
            float(self._counter),
            float(self._counter + 1),
            float(self._counter + 2),
            float(self._counter + 3),
            0.1,
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
        self._counter = 0
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
