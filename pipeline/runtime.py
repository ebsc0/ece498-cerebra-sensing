"""Runtime coordinator for threaded acquisition/processing pipeline."""

import queue
import struct
import threading
import time
from typing import Optional, Sequence

from config import (
    ACTIVE_OPTODES,
    BUFFER_MAX_PENDING_FRAMES,
    BUFFER_STALE_TIMEOUT_MS,
    NUM_OPTODES,
    PACKET_FORMAT,
)
from database.database import DatabaseManager
from preprocessor import Preprocessor

from pipeline.types import MatchedFrame, PipelineSummary, RawPacket, UiFrameResult
from pipeline.workers import FrameWorker, PreprocessWorker


class PipelineRuntime:
    """Owns worker threads, queues, and pipeline counters."""

    def __init__(
        self,
        *,
        db: DatabaseManager,
        preprocessor: Preprocessor,
        num_optodes: int = NUM_OPTODES,
        stale_timeout_ms: int = BUFFER_STALE_TIMEOUT_MS,
        max_pending_frames: int = BUFFER_MAX_PENDING_FRAMES,
        packet_format: str = PACKET_FORMAT,
        active_optodes: Sequence[int] = ACTIVE_OPTODES,
        raw_queue_size: int = 2048,
        matched_queue_size: int = 512,
        ui_queue_size: int = 512,
    ):
        self.db = db
        self.preprocessor = preprocessor
        self.num_optodes = num_optodes
        self.stale_timeout_ms = stale_timeout_ms
        self.max_pending_frames = max_pending_frames
        self.packet_format = packet_format
        self.packet_size = struct.calcsize(packet_format)
        self.active_optodes = list(active_optodes)

        self.raw_packet_queue: queue.Queue[Optional[RawPacket]] = queue.Queue(maxsize=raw_queue_size)
        self.matched_frame_queue: queue.Queue[Optional[MatchedFrame]] = queue.Queue(maxsize=matched_queue_size)
        self.preprocessed_queue: queue.Queue[UiFrameResult] = queue.Queue(maxsize=ui_queue_size)
        self.error_queue: queue.Queue[str] = queue.Queue(maxsize=512)

        self._frame_worker: Optional[FrameWorker] = None
        self._preprocess_worker: Optional[PreprocessWorker] = None
        self._frame_worker_thread: Optional[threading.Thread] = None
        self._preprocess_worker_thread: Optional[threading.Thread] = None

        self._lock = threading.Lock()
        self._captured_frames = 0
        self._processed_frames = 0
        self._dropped_incomplete_frames = 0
        self._session_hemorrhage_detected = False
        self._invalid_ingest_packets = 0

    def start(self, session_id: int) -> None:
        """Reset pipeline state and start worker threads for a session.

        Stops existing workers first. Raises RuntimeError if prior workers did
        not fully stop within timeout.
        """

        self.preprocessor.reset()
        self._drain_queue(self.raw_packet_queue)
        self._drain_queue(self.matched_frame_queue)
        self._drain_queue(self.preprocessed_queue)
        self._drain_queue(self.error_queue)

        with self._lock:
            self._captured_frames = 0
            self._processed_frames = 0
            self._dropped_incomplete_frames = 0
            self._session_hemorrhage_detected = False
            self._last_drain_complete = True
            self._invalid_ingest_packets = 0

        self._frame_worker = FrameWorker(
            session_id=session_id,
            db=self.db,
            raw_packet_queue=self.raw_packet_queue,
            matched_frame_queue=self.matched_frame_queue,
            put_drop_oldest=self._put_drop_oldest,
            put_control=self._put_control,
            on_captured_frame=self._on_captured_frame,
            on_dropped_incomplete_frames=self._on_dropped_incomplete_frames,
            on_error=self._on_worker_error,
            num_optodes=self.num_optodes,
            stale_timeout_ms=self.stale_timeout_ms,
            max_pending_frames=self.max_pending_frames,
            packet_format=self.packet_format,
        )
        self._preprocess_worker = PreprocessWorker(
            session_id=session_id,
            db=self.db,
            preprocessor=self.preprocessor,
            matched_frame_queue=self.matched_frame_queue,
            preprocessed_queue=self.preprocessed_queue,
            put_drop_oldest=self._put_drop_oldest,
            on_session_hemorrhage=self._on_session_hemorrhage,
            on_processed_frame=self._on_processed_frame,
            on_error=self._on_worker_error,
            active_optodes=self.active_optodes,
        )

        self._frame_worker_thread = threading.Thread(target=self._frame_worker.run, daemon=True)
        self._preprocess_worker_thread = threading.Thread(target=self._preprocess_worker.run, daemon=True)
        self._frame_worker_thread.start()
        self._preprocess_worker_thread.start()

    def stop(self, timeout_s: float = 10.0) -> PipelineSummary:
        """Stop workers and return current pipeline summary."""
        self._stop_workers(timeout_s=timeout_s)
        return self.get_summary()

    def ingest_packet(self, packet: bytes) -> None:
        """Thread1 API: ingest one raw packet."""
        if not self._is_valid_ingest_packet(packet):
            with self._lock:
                self._invalid_ingest_packets += 1
                invalid_count = self._invalid_ingest_packets
            if invalid_count in (1, 10) or invalid_count % 100 == 0:
                self._put_drop_oldest(
                    self.error_queue,
                    f"Dropping invalid incoming packets (count={invalid_count}).",
                )
            return

        envelope = RawPacket(
            packet=packet,
            ingress_timestamp_ms=time.monotonic_ns() // 1_000_000,
        )
        self._put_drop_oldest(self.raw_packet_queue, envelope)

    def drain_ui_results(self) -> list[UiFrameResult]:
        """Drain preprocessed UI results (non-blocking)."""
        items: list[UiFrameResult] = []
        while not self.preprocessed_queue.empty():
            try:
                items.append(self.preprocessed_queue.get_nowait())
            except queue.Empty:
                break
        return items

    def clear_ui_results(self) -> None:
        """Discard pending UI frame results."""
        self._drain_queue(self.preprocessed_queue)

    def drain_errors(self) -> list[str]:
        """Drain worker/runtime errors (non-blocking)."""
        errors: list[str] = []
        while not self.error_queue.empty():
            try:
                errors.append(self.error_queue.get_nowait())
            except queue.Empty:
                break
        return errors

    def get_summary(self) -> PipelineSummary:
        """Get counters/state snapshot for current or last session."""
        with self._lock:
            return PipelineSummary(
                captured_frames=self._captured_frames,
                processed_frames=self._processed_frames,
                dropped_incomplete_frames=self._dropped_incomplete_frames,
                session_hemorrhage_detected=self._session_hemorrhage_detected,
                drain_complete=self._last_drain_complete,
            )

    def _stop_workers(self, timeout_s: float) -> None:
        frame_thread = self._frame_worker_thread
        preprocess_thread = self._preprocess_worker_thread
        deadline = time.monotonic() + max(0.0, timeout_s)
        drain_complete = True

        frame_was_alive = bool(frame_thread and frame_thread.is_alive())
        if frame_was_alive:
            # Strict drain: enqueue shutdown marker without dropping queued data.
            if not self._put_control(self.raw_packet_queue, None, deadline=deadline):
                drain_complete = False
            remaining = max(0.0, deadline - time.monotonic())
            frame_thread.join(timeout=remaining)
            if frame_thread.is_alive():
                drain_complete = False

        if not (frame_thread and frame_thread.is_alive()):
            self._frame_worker_thread = None
            self._frame_worker = None

        if preprocess_thread and preprocess_thread.is_alive():
            # If frame worker was already down before stop, ensure downstream can exit.
            if not frame_was_alive:
                if not self._put_control(self.matched_frame_queue, None, deadline=deadline):
                    drain_complete = False
            remaining = max(0.0, deadline - time.monotonic())
            preprocess_thread.join(timeout=remaining)
            if preprocess_thread.is_alive():
                drain_complete = False

        if not (preprocess_thread and preprocess_thread.is_alive()):
            self._preprocess_worker_thread = None
            self._preprocess_worker = None

        with self._lock:
            self._last_drain_complete = drain_complete

    def _on_captured_frame(self) -> None:
        with self._lock:
            self._captured_frames += 1

    def _on_processed_frame(self) -> None:
        with self._lock:
            self._processed_frames += 1

    def _on_dropped_incomplete_frames(self, dropped: int) -> None:
        with self._lock:
            self._dropped_incomplete_frames = dropped

    def _on_session_hemorrhage(self, detected: bool) -> None:
        with self._lock:
            self._session_hemorrhage_detected = detected

    def _on_worker_error(self, text: str) -> None:
        self._put_drop_oldest(self.error_queue, text)

    def _is_valid_ingest_packet(self, packet: bytes) -> bool:
        if len(packet) != self.packet_size:
            return False
        try:
            metadata = struct.unpack_from("<I", packet, 0)[0]
        except struct.error:
            return False
        optode = metadata & 0xF
        return optode in self.active_optodes

    @staticmethod
    def _put_drop_oldest(q: queue.Queue, item: object) -> None:
        while True:
            try:
                q.put_nowait(item)
                return
            except queue.Full:
                try:
                    q.get_nowait()
                except queue.Empty:
                    return

    @staticmethod
    def _put_control(
        q: queue.Queue,
        item: object,
        *,
        deadline: Optional[float] = None,
        timeout_s: float = 0.1,
    ) -> bool:
        """Enqueue control marker without dropping existing data."""
        while deadline is None or time.monotonic() < deadline:
            try:
                q.put(item, timeout=timeout_s)
                return True
            except queue.Full:
                continue
        return False

    @staticmethod
    def _drain_queue(q: queue.Queue) -> None:
        while True:
            try:
                q.get_nowait()
            except queue.Empty:
                break
