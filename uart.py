"""UART-backed DataSource with sync-word framing and reconnect handling."""

import threading
from typing import Callable, Optional, Sequence

from config import UART_SYNC_WORD
from data_source import DataSource, PacketCallback, SourceEvent, SourceEventCallback
from packets import PACKET_SIZE, decode_packet


class _PacketFramer:
    """Incrementally extracts sync-prefixed payload packets from a byte stream."""

    def __init__(
        self,
        *,
        payload_size: int,
        sync_word: bytes,
        payload_validator: Optional[Callable[[bytes], bool]] = None,
    ):
        self.payload_size = max(1, int(payload_size))
        self.sync_word = bytes(sync_word)
        if not self.sync_word:
            raise ValueError("UART sync framing requires a non-empty sync_word")
        self.payload_validator = payload_validator
        self._buffer = bytearray()

    def reset(self) -> None:
        self._buffer.clear()

    def feed(self, data: bytes) -> tuple[list[bytes], int]:
        self._buffer.extend(data)
        packets: list[bytes] = []
        dropped_bytes = 0
        minimum_needed = len(self.sync_word) + self.payload_size

        while True:
            sync_index = self._buffer.find(self.sync_word)
            if sync_index < 0:
                keep = max(0, len(self.sync_word) - 1)
                discard = max(0, len(self._buffer) - keep)
                if discard:
                    dropped_bytes += discard
                    del self._buffer[:discard]
                break

            if sync_index > 0:
                dropped_bytes += sync_index
                del self._buffer[:sync_index]

            if len(self._buffer) < minimum_needed:
                break

            payload_start = len(self.sync_word)
            payload_end = payload_start + self.payload_size
            payload = bytes(self._buffer[payload_start:payload_end])
            if self.payload_validator and not self.payload_validator(payload):
                dropped_bytes += 1
                del self._buffer[0]
                continue

            packets.append(payload)
            del self._buffer[:minimum_needed]

        return packets, dropped_bytes


class UartSource(DataSource):
    """UART data source with optional auto-reconnect."""

    def __init__(
        self,
        *,
        port: str,
        baudrate: int,
        active_optodes: Sequence[int],
        timeout_s: float = 0.1,
        read_chunk_size: int = 256,
        sync_word: bytes = UART_SYNC_WORD,
        auto_reconnect: bool = True,
        reconnect_initial_s: float = 0.5,
        reconnect_max_s: float = 5.0,
    ):
        self.port = port
        self.baudrate = int(baudrate)
        self.packet_size = PACKET_SIZE
        self.active_optodes = set(int(x) for x in active_optodes)
        self.timeout_s = max(0.01, float(timeout_s))
        self.read_chunk_size = max(1, int(read_chunk_size))
        self.auto_reconnect = bool(auto_reconnect)
        self.reconnect_initial_s = max(0.1, float(reconnect_initial_s))
        self.reconnect_max_s = max(self.reconnect_initial_s, float(reconnect_max_s))

        self._framer = _PacketFramer(
            payload_size=self.packet_size,
            sync_word=sync_word,
            payload_validator=self._is_valid_payload,
        )
        self._packet_callback: Optional[PacketCallback] = None
        self._event_callback: Optional[SourceEventCallback] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False
        self._serial = None
        self._serial_lock = threading.Lock()
        self._invalid_packet_count = 0
        self._was_ever_connected = False

    def start(
        self,
        packet_callback: PacketCallback,
        event_callback: Optional[SourceEventCallback] = None,
    ) -> None:
        if self._running:
            raise RuntimeError("UART source is already running")

        self._packet_callback = packet_callback
        self._event_callback = event_callback
        self._stop_event.clear()
        self._framer.reset()
        self._invalid_packet_count = 0
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._running = True

    def stop(self) -> None:
        if not self._running:
            return

        self._stop_event.set()
        self._close_serial()
        if self._thread and threading.current_thread() is not self._thread:
            self._thread.join(timeout=2.0)
        self._thread = None
        self._running = False
        self._emit("stopped", "UART source stopped.")

    def is_running(self) -> bool:
        return self._running

    def _run_loop(self) -> None:
        reconnect_delay = self.reconnect_initial_s
        while not self._stop_event.is_set():
            if self._serial is None:
                try:
                    self._serial = self._open_serial()
                    self._framer.reset()
                    if self._was_ever_connected:
                        self._emit("reconnected", f"UART reconnected on {self.port}.")
                    else:
                        self._emit("connected", f"UART connected on {self.port}.")
                    self._was_ever_connected = True
                    reconnect_delay = self.reconnect_initial_s
                except Exception as exc:
                    self._emit("disconnected", f"UART open failed: {exc}")
                    if not self.auto_reconnect:
                        self._emit("fatal_disconnected", "UART unavailable and auto-reconnect is disabled.")
                        break
                    if self._sleep_or_stopped(reconnect_delay):
                        break
                    reconnect_delay = min(self.reconnect_max_s, reconnect_delay * 2.0)
                    continue

            try:
                chunk = self._serial.read(self.read_chunk_size)
                if not chunk:
                    continue

                packets, dropped_candidates = self._framer.feed(chunk)
                if dropped_candidates > 0:
                    self._emit(
                        "warning",
                        f"Dropped {dropped_candidates} UART byte(s) while re-aligning stream.",
                    )

                for packet in packets:
                    self._handle_packet(packet)
            except Exception as exc:
                self._emit("disconnected", f"UART read failed: {exc}")
                self._close_serial()
                self._framer.reset()
                if not self.auto_reconnect:
                    self._emit("fatal_disconnected", "UART link lost and auto-reconnect is disabled.")
                    break
                if self._sleep_or_stopped(reconnect_delay):
                    break
                reconnect_delay = min(self.reconnect_max_s, reconnect_delay * 2.0)

        self._close_serial()
        self._running = False

    def _handle_packet(self, packet: bytes) -> None:
        if not self._is_valid_payload(packet):
            self._invalid_packet_count += 1
            if self._invalid_packet_count in (1, 10) or self._invalid_packet_count % 100 == 0:
                self._emit(
                    "warning",
                    f"Dropping invalid UART payloads (count={self._invalid_packet_count}).",
                )
            return
        if self._packet_callback:
            self._packet_callback(packet)

    def _open_serial(self):
        try:
            import serial  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError("Missing pyserial dependency. Install requirements and retry.") from exc

        return serial.Serial(
            port=self.port,
            baudrate=self.baudrate,
            timeout=self.timeout_s,
        )

    def _close_serial(self) -> None:
        with self._serial_lock:
            if self._serial is not None:
                try:
                    self._serial.close()
                except Exception:
                    pass
                finally:
                    self._serial = None

    def _sleep_or_stopped(self, delay_s: float) -> bool:
        return self._stop_event.wait(timeout=max(0.05, delay_s))

    def _is_valid_payload(self, packet: bytes) -> bool:
        if len(packet) != self.packet_size:
            return False
        try:
            decoded = decode_packet(packet, timestamp_ms=0)
        except ValueError:
            return False
        if decoded.optode_id not in self.active_optodes:
            return False
        if any(abs(value) > 1e6 for value in (decoded.d0, decoded.d1, decoded.d2, decoded.d3)):
            return False
        return True

    def _emit(self, kind: str, message: str) -> None:
        if self._event_callback:
            self._event_callback(SourceEvent(kind=kind, message=message))
