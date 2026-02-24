"""Application entry point with data collection and real-time UI updates."""

import struct
import queue
import datetime
import os
import time
from typing import Dict, Tuple

# Ensure GUI/cache config paths are writable before importing Kivy/Matplotlib.
_APP_DIR = os.path.dirname(__file__)
if "KIVY_HOME" not in os.environ:
    os.environ["KIVY_HOME"] = os.path.join(_APP_DIR, ".kivy")
    os.makedirs(os.environ["KIVY_HOME"], exist_ok=True)
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = os.path.join(_APP_DIR, ".mplconfig")
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

from kivy.app import App
from kivy.clock import Clock

from config import (
    ACTIVE_OPTODES,
    BUFFER_MAX_PENDING_FRAMES,
    BUFFER_STALE_TIMEOUT_MS,
    NUM_OPTODES,
    PACKET_FORMAT,
    SAMPLE_RATE_HZ,
    UI_UPDATE_RATE_HZ,
)
from simulator import Simulator
from buffer import Buffer, CompleteFrame
from database.database import DatabaseManager, RawSample, PreprocessedSample
from preprocessor import Preprocessor, PreprocessedResult
from ui import MainScreen
from ich_detection import detect_ich, reset_history

# Database configuration
DB_DIR = os.path.join(os.path.dirname(__file__), "db")
DB_FILE = os.path.join(DB_DIR, "fnirs_data.db")


class CerebraApp(App):
    """Kivy app with integrated data collection and real-time visualization."""

    def build(self):
        # Data collection components
        self.buffer = Buffer(
            num_optodes=NUM_OPTODES,
            stale_timeout_ms=BUFFER_STALE_TIMEOUT_MS,
            max_pending_frames=BUFFER_MAX_PENDING_FRAMES,
        )
        self.simulator = Simulator(num_optodes=NUM_OPTODES, sample_rate_hz=SAMPLE_RATE_HZ)

        # Queue for passing preprocessed data to UI thread
        # Items: (CompleteFrame, Dict[int, PreprocessedResult])
        self.preprocessed_queue: queue.Queue[Tuple[CompleteFrame, Dict[int, PreprocessedResult]]] = queue.Queue(
            maxsize=512
        )

        # Database
        self.db = DatabaseManager(db_file=DB_FILE)
        self.db.connect()
        self.session_id = None

        # Session timing
        self.session_start_time = None
        self.captured_frame_count = 0
        self.processed_frame_count = 0
        self._last_error_log_time = 0.0

        # Preprocessor
        self.preprocessor = Preprocessor()

        # UI
        self.screen = MainScreen(
            on_start=self.start_collection,
            on_stop=self.stop_collection,
        )

        # Poll for preprocessed data and update UI
        Clock.schedule_interval(self._update_ui, 1.0 / UI_UPDATE_RATE_HZ)
        return self.screen

    def _timestamp(self) -> str:
        """Get current time as formatted string."""
        return datetime.datetime.now().strftime("%H:%M:%S")

    def _elapsed_time_str(self) -> str:
        """Get elapsed time since session start as HH:MM:SS."""
        if self.session_start_time is None:
            return "--:--:--"
        elapsed = time.time() - self.session_start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def start_collection(self):
        """Start data collection and create new session."""
        self.buffer.clear()
        self.preprocessor.reset()
        while not self.preprocessed_queue.empty():
            try:
                self.preprocessed_queue.get_nowait()
            except queue.Empty:
                break

        # Reset ICH detection history
        reset_history()

        # Reset session tracking
        self.session_start_time = time.time()
        self.captured_frame_count = 0
        self.processed_frame_count = 0

        # Create new session
        start_time = datetime.datetime.now().isoformat()
        self.session_id = self.db.create_session(
            start_time=start_time,
            sample_rate_hz=SAMPLE_RATE_HZ,
            num_optodes=NUM_OPTODES
        )

        # Update session info in UI
        self.screen.update_session_info(
            session_id=self.session_id,
            elapsed_str="00:00:00",
            captured_count=0,
            processed_count=0,
        )

        # Log session start
        self.screen.append_log(f"[{self._timestamp()}] Session started (ID: {self.session_id})\n")

        self.simulator.start(self._on_packet)

    def stop_collection(self):
        """Stop data collection and end session."""
        self.simulator.stop()

        # End session
        if self.session_id:
            end_time = datetime.datetime.now().isoformat()
            self.db.end_session(self.session_id, end_time)

            # Log session end
            elapsed = self._elapsed_time_str()
            self.screen.append_log(
                f"[{self._timestamp()}] Session stopped (ID: {self.session_id}, "
                f"Duration: {elapsed}, Captured: {self.captured_frame_count}, "
                f"Processed: {self.processed_frame_count}, "
                f"Dropped(incomplete): {self.buffer.dropped_frames()})\n"
            )

            self.session_id = None
            self.session_start_time = None

    def _on_packet(self, packet: bytes):
        """Callback from simulator/data source thread.

        Buffers packets, stores to DB, and queues preprocessed data for UI.
        """
        try:
            complete_frame = self.buffer.add_packet(packet)
            if not complete_frame:
                return

            self.captured_frame_count += 1
            preprocessed = self._store_frame(complete_frame)
            if not preprocessed:
                return

            try:
                self.preprocessed_queue.put_nowait((complete_frame, preprocessed))
            except queue.Full:
                # Drop oldest UI item first to keep freshest data visible.
                try:
                    self.preprocessed_queue.get_nowait()
                except queue.Empty:
                    pass
                self.preprocessed_queue.put_nowait((complete_frame, preprocessed))
        except Exception as e:
            self._log_error_throttled(f"[{self._timestamp()}] Error in packet handler: {e}\n")

    def _store_frame(self, frame: CompleteFrame) -> Dict[int, PreprocessedResult]:
        """Store frame data to database (raw + preprocessed).

        Returns:
            Dict of preprocessed results for UI update, or empty dict if no session.
        """
        if not self.session_id:
            return {}

        # Batch-insert each optode's raw sample
        raw_batch = []
        optode_order = []
        for optode_id, packet in frame.packets.items():
            data = struct.unpack(PACKET_FORMAT, packet)
            sample = RawSample(
                optode_id=optode_id,
                nm740_long=data[1],
                nm860_long=data[2],
                nm740_short=data[3],
                nm860_short=data[4],
                dark=data[5]
            )
            raw_batch.append((frame.frame_number, frame.timestamp_ms, sample))
            optode_order.append(optode_id)
        raw_ids = self.db.insert_raw_samples_batch(self.session_id, raw_batch)
        sample_ids = {optode_id: sample_id for optode_id, sample_id in zip(optode_order, raw_ids)}

        # Preprocess the frame
        preprocessed = self.preprocessor.process_frame(frame, sample_ids)

        # Batch-store preprocessed samples
        pre_batch = []
        for optode_id, result in preprocessed.items():
            sample = PreprocessedSample(
                optode_id=result.optode_id,
                od_nm740_short=result.od_nm740_short,
                od_nm740_long=result.od_nm740_long,
                od_nm860_short=result.od_nm860_short,
                od_nm860_long=result.od_nm860_long,
                hbo_short=result.hbo_short,
                hbr_short=result.hbr_short,
                hbo_long=result.hbo_long,
                hbr_long=result.hbr_long,
            )
            pre_batch.append(
                (result.sample_id, result.frame_number, result.timestamp_ms, sample)
            )
        if pre_batch:
            self.db.insert_preprocessed_samples_batch(self.session_id, pre_batch)

        return preprocessed

    def _prepare_ich_data(self, preprocessed: Dict[int, PreprocessedResult]) -> Dict[int, dict]:
        """Convert preprocessed data to format expected by ICH detection.

        Args:
            preprocessed: {optode_id: PreprocessedResult}

        Returns:
            {optode_id: {"HbR": float, "OD_860": float}}
        """
        ich_data = {}
        for optode_id, result in preprocessed.items():
            # Use long channel values for ICH detection
            ich_data[optode_id] = {
                "HbR": result.hbr_long,
                "OD_860": result.od_nm860_long,
            }
        return ich_data

    def _get_pending_data(self) -> list:
        """Drain and return all pending preprocessed data (non-blocking)."""
        items = []
        while not self.preprocessed_queue.empty():
            try:
                items.append(self.preprocessed_queue.get_nowait())
            except queue.Empty:
                break
        return items

    def _update_ui(self, dt):
        """Poll for new preprocessed data and update UI."""
        pending = self._get_pending_data()

        for frame, preprocessed in pending:
            self.processed_frame_count += 1

            # Update graph and head map with preprocessed data
            self.screen.update_graph(preprocessed)

            # Run ICH detection
            ich_data = self._prepare_ich_data(preprocessed)
            flags, counts = detect_ich(ich_data, ACTIVE_OPTODES)

            # Update ICH status in UI
            self.screen.update_ich_status(flags, counts)

            # Log frame (less verbose - only log every 10 frames)
            if self.processed_frame_count % 10 == 0:
                self.screen.append_log(
                    f"[{self._timestamp()}] Frame {frame.frame_number} @ {frame.timestamp_ms}ms\n"
                )

        # Update session info (always, to keep timer running)
        if self.session_id:
            self.screen.update_session_info(
                session_id=self.session_id,
                elapsed_str=self._elapsed_time_str(),
                captured_count=self.captured_frame_count,
                processed_count=self.processed_frame_count,
            )

    def _log_error_throttled(self, text: str, interval_s: float = 2.0):
        """Log repeated runtime errors with basic time-based throttling."""
        now = time.time()
        if now - self._last_error_log_time >= interval_s:
            self._last_error_log_time = now
            self.screen.append_log(text)

    def on_stop(self):
        """Clean up when app closes."""
        if self.simulator.is_running():
            self.simulator.stop()

        # End session if still active
        if self.session_id:
            end_time = datetime.datetime.now().isoformat()
            self.db.end_session(self.session_id, end_time)

        # Close database connection
        self.db.close()


if __name__ == '__main__':
    CerebraApp().run()
