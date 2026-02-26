"""Runtime coordinator for threaded acquisition/processing pipeline."""

import queue
import threading
from typing import Callable, Optional, Sequence

from config import (
    ACTIVE_OPTODES,
    BUFFER_MAX_PENDING_FRAMES,
    BUFFER_STALE_TIMEOUT_MS,
    NUM_OPTODES,
    PACKET_FORMAT,
)
from database.database import DatabaseManager
from preprocessor import Preprocessor

from pipeline.types import MatchedFrame, PipelineSummary, UiFrameResult
from pipeline.workers import FrameWorker, PreprocessWorker


class PipelineRuntime:
    """Owns worker threads, queues, and pipeline counters."""

    def __init__(
        self,
        *,
        db: DatabaseManager,
        preprocessor: Preprocessor,
        error_logger: Callable[[str], None],
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
        self.error_logger = error_logger
        self.num_optodes = num_optodes
        self.stale_timeout_ms = stale_timeout_ms
        self.max_pending_frames = max_pending_frames
        self.packet_format = packet_format
        self.active_optodes = list(active_optodes)

        self.raw_packet_queue: queue.Queue[Optional[bytes]] = queue.Queue(maxsize=raw_queue_size)
        self.matched_frame_queue: queue.Queue[Optional[MatchedFrame]] = queue.Queue(maxsize=matched_queue_size)
        self.preprocessed_queue: queue.Queue[UiFrameResult] = queue.Queue(maxsize=ui_queue_size)

        self._frame_worker: Optional[FrameWorker] = None
        self._preprocess_worker: Optional[PreprocessWorker] = None
        self._frame_worker_thread: Optional[threading.Thread] = None
        self._preprocess_worker_thread: Optional[threading.Thread] = None

        self._lock = threading.Lock()
        self._captured_frames = 0
        self._processed_frames = 0
        self._dropped_incomplete_frames = 0
        self._last_frame_hemorrhage_detected = False

    def start(self, session_id: int) -> None:
        """Reset pipeline state and start worker threads for a session."""
        self.preprocessor.reset()
        self._drain_queue(self.raw_packet_queue)
        self._drain_queue(self.matched_frame_queue)
        self._drain_queue(self.preprocessed_queue)

        with self._lock:
            self._captured_frames = 0
            self._processed_frames = 0
            self._dropped_incomplete_frames = 0
            self._last_frame_hemorrhage_detected = False

        self._frame_worker = FrameWorker(
            session_id=session_id,
            db=self.db,
            raw_packet_queue=self.raw_packet_queue,
            matched_frame_queue=self.matched_frame_queue,
            put_drop_oldest=self._put_drop_oldest,
            put_control=self._put_control,
            on_captured_frame=self._on_captured_frame,
            on_dropped_incomplete_frames=self._on_dropped_incomplete_frames,
            on_error=self.error_logger,
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
            on_last_frame_hemorrhage=self._on_last_frame_hemorrhage,
            on_processed_frame=self._on_processed_frame,
            on_error=self.error_logger,
            active_optodes=self.active_optodes,
        )

        self._frame_worker_thread = threading.Thread(target=self._frame_worker.run, daemon=True)
        self._preprocess_worker_thread = threading.Thread(target=self._preprocess_worker.run, daemon=True)
        self._frame_worker_thread.start()
        self._preprocess_worker_thread.start()

    def stop(self) -> PipelineSummary:
        """Stop workers and return current pipeline summary."""
        self._stop_workers()
        return self.get_summary()

    def ingest_packet(self, packet: bytes) -> None:
        """Thread1 API: ingest one raw packet."""
        self._put_drop_oldest(self.raw_packet_queue, packet)

    def drain_ui_results(self) -> list[UiFrameResult]:
        """Drain preprocessed UI results (non-blocking)."""
        items: list[UiFrameResult] = []
        while not self.preprocessed_queue.empty():
            try:
                items.append(self.preprocessed_queue.get_nowait())
            except queue.Empty:
                break
        return items

    def get_summary(self) -> PipelineSummary:
        """Get counters/state snapshot for current or last session."""
        with self._lock:
            return PipelineSummary(
                captured_frames=self._captured_frames,
                processed_frames=self._processed_frames,
                dropped_incomplete_frames=self._dropped_incomplete_frames,
                last_frame_hemorrhage_detected=self._last_frame_hemorrhage_detected,
            )

    def _stop_workers(self) -> None:
        frame_thread = self._frame_worker_thread
        preprocess_thread = self._preprocess_worker_thread

        frame_was_alive = bool(frame_thread and frame_thread.is_alive())
        if frame_was_alive:
            # Strict drain: enqueue shutdown marker without dropping queued data.
            self._put_control(self.raw_packet_queue, None)
            frame_thread.join()
        self._frame_worker_thread = None
        self._frame_worker = None

        if preprocess_thread and preprocess_thread.is_alive():
            # If frame worker was already down before stop, ensure downstream can exit.
            if not frame_was_alive:
                self._put_control(self.matched_frame_queue, None)
            preprocess_thread.join()
        self._preprocess_worker_thread = None
        self._preprocess_worker = None

    def _on_captured_frame(self) -> None:
        with self._lock:
            self._captured_frames += 1

    def _on_processed_frame(self) -> None:
        with self._lock:
            self._processed_frames += 1

    def _on_dropped_incomplete_frames(self, dropped: int) -> None:
        with self._lock:
            self._dropped_incomplete_frames = dropped

    def _on_last_frame_hemorrhage(self, detected: bool) -> None:
        with self._lock:
            self._last_frame_hemorrhage_detected = detected

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
    def _put_control(q: queue.Queue, item: object, timeout_s: float = 0.1) -> None:
        """Enqueue control marker without dropping existing data."""
        while True:
            try:
                q.put(item, timeout=timeout_s)
                return
            except queue.Full:
                continue

    @staticmethod
    def _drain_queue(q: queue.Queue) -> None:
        while True:
            try:
                q.get_nowait()
            except queue.Empty:
                break
