"""Worker implementations for the streaming acquisition pipeline."""

import queue
from typing import Callable, Dict, Optional, Sequence

from buffer import Buffer
from config import (
    ACTIVE_OPTODES,
    BUFFER_MAX_PENDING_FRAMES,
    BUFFER_STALE_TIMEOUT_MS,
    NUM_OPTODES,
    PACKET_FORMAT,
)
from database.database import DatabaseManager
from ich_detection import detect_ich
from preprocessor import PreprocessedResult, Preprocessor

from pipeline.persistence import store_preprocessed_frame, store_raw_frame
from pipeline.types import MatchedFrame, UiFrameResult


class FrameWorker:
    """Thread2: build complete frames and write raw samples."""

    def __init__(
        self,
        *,
        session_id: int,
        db: DatabaseManager,
        raw_packet_queue: queue.Queue[Optional[bytes]],
        matched_frame_queue: queue.Queue[Optional[MatchedFrame]],
        put_drop_oldest: Callable[[queue.Queue, object], None],
        put_control: Callable[[queue.Queue, object], None],
        on_captured_frame: Callable[[], None],
        on_dropped_incomplete_frames: Callable[[int], None],
        on_error: Callable[[str], None],
        num_optodes: int = NUM_OPTODES,
        stale_timeout_ms: int = BUFFER_STALE_TIMEOUT_MS,
        max_pending_frames: int = BUFFER_MAX_PENDING_FRAMES,
        packet_format: str = PACKET_FORMAT,
    ):
        self.session_id = session_id
        self.db = db
        self.raw_packet_queue = raw_packet_queue
        self.matched_frame_queue = matched_frame_queue
        self.put_drop_oldest = put_drop_oldest
        self.put_control = put_control
        self.on_captured_frame = on_captured_frame
        self.on_dropped_incomplete_frames = on_dropped_incomplete_frames
        self.on_error = on_error
        self.num_optodes = num_optodes
        self.stale_timeout_ms = stale_timeout_ms
        self.max_pending_frames = max_pending_frames
        self.packet_format = packet_format

    def run(self) -> None:
        frame_buffer = Buffer(
            num_optodes=self.num_optodes,
            stale_timeout_ms=self.stale_timeout_ms,
            max_pending_frames=self.max_pending_frames,
        )

        while True:
            packet = self.raw_packet_queue.get()
            if packet is None:
                break

            try:
                complete_frame = frame_buffer.add_packet(packet)
                if not complete_frame:
                    continue

                self.on_captured_frame()
                sample_ids = store_raw_frame(
                    db=self.db,
                    session_id=self.session_id,
                    frame=complete_frame,
                    packet_format=self.packet_format,
                )
                if not sample_ids:
                    continue

                self.put_drop_oldest(
                    self.matched_frame_queue,
                    MatchedFrame(frame=complete_frame, sample_ids=sample_ids),
                )
            except Exception as exc:
                self.on_error(f"Error in frame worker: {exc}")

        self.on_dropped_incomplete_frames(frame_buffer.dropped_frames())
        # Unblock downstream stage after all matched frames are emitted.
        self.put_control(self.matched_frame_queue, None)


class PreprocessWorker:
    """Thread3: preprocess complete frames, run ICH, and write processed rows."""

    def __init__(
        self,
        *,
        session_id: int,
        db: DatabaseManager,
        preprocessor: Preprocessor,
        matched_frame_queue: queue.Queue[Optional[MatchedFrame]],
        preprocessed_queue: queue.Queue[UiFrameResult],
        put_drop_oldest: Callable[[queue.Queue, object], None],
        on_last_frame_hemorrhage: Callable[[bool], None],
        on_processed_frame: Callable[[], None],
        on_error: Callable[[str], None],
        active_optodes: Optional[Sequence[int]] = None,
    ):
        self.session_id = session_id
        self.db = db
        self.preprocessor = preprocessor
        self.matched_frame_queue = matched_frame_queue
        self.preprocessed_queue = preprocessed_queue
        self.put_drop_oldest = put_drop_oldest
        self.on_last_frame_hemorrhage = on_last_frame_hemorrhage
        self.on_processed_frame = on_processed_frame
        self.on_error = on_error
        self.active_optodes = list(active_optodes) if active_optodes is not None else ACTIVE_OPTODES

    @staticmethod
    def _prepare_ich_data(preprocessed: Dict[int, PreprocessedResult]) -> Dict[int, dict]:
        ich_data = {}
        for optode_id, result in preprocessed.items():
            ich_data[optode_id] = {
                "HbR": result.hbr_long,
                "OD_860": result.od_nm860_long,
            }
        return ich_data

    def run(self) -> None:
        while True:
            matched = self.matched_frame_queue.get()
            if matched is None:
                break

            try:
                preprocessed = self.preprocessor.process_frame(matched.frame, matched.sample_ids)
                if not preprocessed:
                    continue

                store_preprocessed_frame(
                    db=self.db,
                    session_id=self.session_id,
                    preprocessed=preprocessed,
                )
                ich_data = self._prepare_ich_data(preprocessed)
                flags, counts = detect_ich(ich_data, self.active_optodes)
                self.on_last_frame_hemorrhage(any(flags.values()) if flags else False)
                self.on_processed_frame()

                self.put_drop_oldest(
                    self.preprocessed_queue,
                    UiFrameResult(
                        frame=matched.frame,
                        preprocessed=preprocessed,
                        ich_flags=flags,
                        ich_counts=counts,
                    ),
                )
            except Exception as exc:
                self.on_error(f"Error in preprocess worker: {exc}")
