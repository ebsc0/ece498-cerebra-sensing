"""Worker implementations for the streaming acquisition pipeline."""

from collections import deque
import queue
from typing import Callable, Deque, Dict, Optional, Sequence

from buffer import Buffer
from config import (
    ACTIVE_OPTODES,
    BUFFER_MAX_PENDING_FRAMES,
    BUFFER_STALE_TIMEOUT_MS,
    ICH_SESSION_MIN_FRAMES,
    ICH_SESSION_PERSISTENCE_RATIO,
    ICH_SESSION_WINDOW_SECONDS,
    NUM_OPTODES,
    PACKET_FORMAT,
    SAMPLE_RATE_HZ,
)
from database.database import DatabaseManager
from ich_detection import detect_ich
from preprocessor import PreprocessedResult, Preprocessor

from pipeline.persistence import store_preprocessed_frame, store_raw_frame
from pipeline.types import MatchedFrame, RawPacket, UiFrameResult


class FrameWorker:
    """Thread2: build complete frames and write raw samples."""

    def __init__(
        self,
        *,
        session_id: int,
        db: DatabaseManager,
        raw_packet_queue: queue.Queue[Optional[RawPacket]],
        matched_frame_queue: queue.Queue[Optional[MatchedFrame]],
        put_drop_oldest: Callable[[queue.Queue, object], None],
        put_control: Callable[[queue.Queue, object], bool],
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
        self._last_emitted_frame_number = -1

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
                complete_frame = frame_buffer.add_packet(packet.packet, packet.ingress_timestamp_ms)
                if not complete_frame:
                    continue

                # Preserve strict ordering for downstream temporal algorithms.
                if complete_frame.frame_number <= self._last_emitted_frame_number:
                    self.on_error(
                        "Dropped late complete frame "
                        f"{complete_frame.frame_number} (last={self._last_emitted_frame_number})."
                    )
                    continue

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
                self._last_emitted_frame_number = complete_frame.frame_number
                self.on_captured_frame()
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
        on_session_hemorrhage: Callable[[bool], None],
        on_processed_frame: Callable[[], None],
        on_error: Callable[[str], None],
        active_optodes: Optional[Sequence[int]] = None,
        session_window_frames: int = max(1, int(round(ICH_SESSION_WINDOW_SECONDS * SAMPLE_RATE_HZ))),
        session_persistence_ratio: float = ICH_SESSION_PERSISTENCE_RATIO,
        session_min_frames: int = ICH_SESSION_MIN_FRAMES,
    ):
        self.session_id = session_id
        self.db = db
        self.preprocessor = preprocessor
        self.matched_frame_queue = matched_frame_queue
        self.preprocessed_queue = preprocessed_queue
        self.put_drop_oldest = put_drop_oldest
        self.on_session_hemorrhage = on_session_hemorrhage
        self.on_processed_frame = on_processed_frame
        self.on_error = on_error
        self.active_optodes = list(active_optodes) if active_optodes is not None else ACTIVE_OPTODES
        self.session_window_frames = max(1, int(session_window_frames))
        self.session_persistence_ratio = float(session_persistence_ratio)
        self.session_min_frames = max(1, min(int(session_min_frames), self.session_window_frames))
        self._session_hemorrhage_detected = False
        self._session_flag_history: Deque[bool] = deque(maxlen=self.session_window_frames)

    @staticmethod
    def _prepare_ich_data(preprocessed: Dict[int, PreprocessedResult]) -> Dict[int, dict]:
        ich_data = {}
        for optode_id, result in preprocessed.items():
            ich_data[optode_id] = {
                "HbO": result.hbo_long,
                "HbR": result.hbr_long,
                "OD_860": result.od_nm860_long,
                "OD_740": result.od_nm740_long,
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
                    preprocessed=preprocessed,
                )
                ich_data = self._prepare_ich_data(preprocessed)
                flags, counts = detect_ich(ich_data, self.active_optodes)
                frame_flagged = any(flags.values()) if flags else False
                self._session_flag_history.append(frame_flagged)
                if not self._session_hemorrhage_detected:
                    history_len = len(self._session_flag_history)
                    if history_len >= self.session_min_frames:
                        ratio = sum(self._session_flag_history) / history_len
                        if ratio >= self.session_persistence_ratio:
                            self._session_hemorrhage_detected = True
                self.on_session_hemorrhage(self._session_hemorrhage_detected)
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
