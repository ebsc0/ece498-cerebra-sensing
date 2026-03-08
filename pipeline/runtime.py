"""Runtime coordinator for threaded acquisition/processing pipeline."""

from __future__ import annotations

import queue
import threading
import time
from typing import Optional, Sequence

from config import ACTIVE_OPTODES, BUFFER_MAX_PENDING_FRAMES, BUFFER_STALE_TIMEOUT_MS
from database.database import DatabaseManager
from preprocessor import Preprocessor

from pipeline.types import PipelineSummary, RawPacket, StoredLogicalFrame, UiFrameResult
from pipeline.workers import AssemblyWorker, PreprocessWorker


class PipelineRuntime:
    """Owns worker threads, queues, and pipeline counters."""

    def __init__(
        self,
        *,
        db: DatabaseManager,
        preprocessor: Preprocessor,
        stale_timeout_ms: int = BUFFER_STALE_TIMEOUT_MS,
        max_pending_frames: int = BUFFER_MAX_PENDING_FRAMES,
        active_optodes: Sequence[int] = ACTIVE_OPTODES,
        raw_queue_size: int = 2048,
        logical_queue_size: int = 512,
        ui_queue_size: int = 512,
    ):
        self.db = db
        self.preprocessor = preprocessor
        self.stale_timeout_ms = stale_timeout_ms
        self.max_pending_frames = max_pending_frames
        self.active_optodes = [int(optode_id) for optode_id in active_optodes]

        self.raw_packet_queue: queue.Queue[Optional[RawPacket]] = queue.Queue(maxsize=raw_queue_size)
        self.logical_frame_queue: queue.Queue[Optional[StoredLogicalFrame]] = queue.Queue(maxsize=logical_queue_size)
        self.preprocessed_queue: queue.Queue[UiFrameResult] = queue.Queue(maxsize=ui_queue_size)
        self.error_queue: queue.Queue[str] = queue.Queue(maxsize=512)

        self._assembly_worker: Optional[AssemblyWorker] = None
        self._preprocess_worker: Optional[PreprocessWorker] = None
        self._assembly_worker_thread: Optional[threading.Thread] = None
        self._preprocess_worker_thread: Optional[threading.Thread] = None

        self._lock = threading.Lock()
        self._captured_frames = 0
        self._processed_frames = 0
        self._dropped_incomplete_frames = 0
        self._session_hemorrhage_detected = False
        self._last_drain_complete = True

    def start(self, session_id: int) -> None:
        if not self._stop_workers(timeout_s=2.0):
            raise RuntimeError("Previous pipeline workers did not stop cleanly.")
        self.preprocessor.reset()
        self._drain_queue(self.raw_packet_queue)
        self._drain_queue(self.logical_frame_queue)
        self._drain_queue(self.preprocessed_queue)
        self._drain_queue(self.error_queue)

        with self._lock:
            self._captured_frames = 0
            self._processed_frames = 0
            self._dropped_incomplete_frames = 0
            self._session_hemorrhage_detected = False
            self._last_drain_complete = True

        self._assembly_worker = AssemblyWorker(
            session_id=session_id,
            db=self.db,
            raw_packet_queue=self.raw_packet_queue,
            logical_frame_queue=self.logical_frame_queue,
            put_drop_oldest=self._put_drop_oldest,
            put_control=self._put_control,
            on_captured_frame=self._on_captured_frame,
            on_dropped_incomplete_frames=self._on_dropped_incomplete_frames,
            on_error=self._on_worker_error,
            active_optodes=self.active_optodes,
            stale_timeout_ms=self.stale_timeout_ms,
            max_pending_frames=self.max_pending_frames,
        )
        self._preprocess_worker = PreprocessWorker(
            session_id=session_id,
            db=self.db,
            preprocessor=self.preprocessor,
            logical_frame_queue=self.logical_frame_queue,
            preprocessed_queue=self.preprocessed_queue,
            put_drop_oldest=self._put_drop_oldest,
            on_session_hemorrhage=self._on_session_hemorrhage,
            on_processed_frame=self._on_processed_frame,
            on_error=self._on_worker_error,
            active_optodes=self.active_optodes,
        )

        self._assembly_worker_thread = threading.Thread(target=self._assembly_worker.run, daemon=True)
        self._preprocess_worker_thread = threading.Thread(target=self._preprocess_worker.run, daemon=True)
        self._assembly_worker_thread.start()
        self._preprocess_worker_thread.start()

    def stop(self, timeout_s: float = 10.0) -> PipelineSummary:
        self._stop_workers(timeout_s=timeout_s)
        return self.get_summary()

    def ingest_packet(self, packet: bytes) -> None:
        envelope = RawPacket(packet=packet, ingress_timestamp_ms=time.monotonic_ns() // 1_000_000)
        self._put_drop_oldest(self.raw_packet_queue, envelope)

    def drain_ui_results(self) -> list[UiFrameResult]:
        items: list[UiFrameResult] = []
        while not self.preprocessed_queue.empty():
            try:
                items.append(self.preprocessed_queue.get_nowait())
            except queue.Empty:
                break
        return items

    def clear_ui_results(self) -> None:
        self._drain_queue(self.preprocessed_queue)

    def drain_errors(self) -> list[str]:
        errors: list[str] = []
        while not self.error_queue.empty():
            try:
                errors.append(self.error_queue.get_nowait())
            except queue.Empty:
                break
        return errors

    def get_summary(self) -> PipelineSummary:
        with self._lock:
            return PipelineSummary(
                captured_frames=self._captured_frames,
                processed_frames=self._processed_frames,
                dropped_incomplete_frames=self._dropped_incomplete_frames,
                session_hemorrhage_detected=self._session_hemorrhage_detected,
                drain_complete=self._last_drain_complete,
            )

    def _stop_workers(self, timeout_s: float) -> bool:
        assembly_thread = self._assembly_worker_thread
        preprocess_thread = self._preprocess_worker_thread
        deadline = time.monotonic() + max(0.0, timeout_s)
        drain_complete = True

        assembly_was_alive = bool(assembly_thread and assembly_thread.is_alive())
        if assembly_was_alive:
            if not self._put_control(self.raw_packet_queue, None, deadline=deadline):
                drain_complete = False
            remaining = max(0.0, deadline - time.monotonic())
            assembly_thread.join(timeout=remaining)
            if assembly_thread.is_alive():
                drain_complete = False

        if not (assembly_thread and assembly_thread.is_alive()):
            self._assembly_worker_thread = None
            self._assembly_worker = None

        if preprocess_thread and preprocess_thread.is_alive():
            if not self._put_control(self.logical_frame_queue, None, deadline=deadline):
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
        return drain_complete

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
