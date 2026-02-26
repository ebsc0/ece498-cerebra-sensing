"""Application entry point with data collection and real-time UI updates."""

import datetime
import os
import time

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
from database.database import DatabaseManager
from ich_detection import reset_history
from pipeline.runtime import PipelineRuntime
from preprocessor import Preprocessor
from simulator import Simulator
from ui import MainScreen

# Database configuration
DB_DIR = os.path.join(os.path.dirname(__file__), "db")
DB_FILE = os.path.join(DB_DIR, "fnirs_data.db")


class CerebraApp(App):
    """Kivy app with integrated data collection and real-time visualization."""

    def build(self):
        # Data source
        self.simulator = Simulator(num_optodes=NUM_OPTODES, sample_rate_hz=SAMPLE_RATE_HZ)

        # Database
        self.db = DatabaseManager(db_file=DB_FILE)
        self.db.connect()
        self.session_id = None

        # Session timing/counters
        self.session_start_time = None
        self.captured_frame_count = 0
        self.processed_frame_count = 0
        self.rendered_frame_count = 0
        self._last_error_log_time = 0.0

        # Processing runtime
        self.preprocessor = Preprocessor()
        self.pipeline = PipelineRuntime(
            db=self.db,
            preprocessor=self.preprocessor,
            error_logger=lambda text: self._log_error_throttled(f"[{self._timestamp()}] {text}\n"),
            num_optodes=NUM_OPTODES,
            stale_timeout_ms=BUFFER_STALE_TIMEOUT_MS,
            max_pending_frames=BUFFER_MAX_PENDING_FRAMES,
            packet_format=PACKET_FORMAT,
            active_optodes=ACTIVE_OPTODES,
        )

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
        # Ensure source is stopped before starting a fresh session.
        if self.simulator.is_running():
            self.simulator.stop()

        # Ensure no stale workers remain before resetting algorithmic state.
        self.pipeline.stop()

        # Reset algorithmic history.
        reset_history()

        # Reset session tracking
        self.session_start_time = time.time()
        self.captured_frame_count = 0
        self.processed_frame_count = 0
        self.rendered_frame_count = 0

        # Create new session
        start_time = datetime.datetime.now().isoformat()
        self.session_id = self.db.create_session(
            start_time=start_time,
            sample_rate_hz=SAMPLE_RATE_HZ,
            num_optodes=NUM_OPTODES,
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

        # Start pipeline workers, then source ingestion.
        self.pipeline.start(self.session_id)
        self.simulator.start(self.pipeline.ingest_packet)

    def stop_collection(self):
        """Stop data collection and end session."""
        self.simulator.stop()

        if not self.session_id:
            return

        summary = self.pipeline.stop()
        self.captured_frame_count = summary.captured_frames
        self.processed_frame_count = summary.processed_frames
        self.db.set_hemorrhage_result(self.session_id, summary.last_frame_hemorrhage_detected)
        end_time = datetime.datetime.now().isoformat()
        self.db.end_session(self.session_id, end_time)

        # Log session end
        elapsed = self._elapsed_time_str()
        self.screen.append_log(
            f"[{self._timestamp()}] Session stopped (ID: {self.session_id}, "
            f"Duration: {elapsed}, Captured: {self.captured_frame_count}, "
            f"Processed: {self.processed_frame_count}, "
            f"Dropped(incomplete): {summary.dropped_incomplete_frames})\n"
        )

        self.session_id = None
        self.session_start_time = None

    def _update_ui(self, dt):
        """Poll pipeline output and update UI."""
        pending = self.pipeline.drain_ui_results()

        for result in pending:
            self.rendered_frame_count += 1

            # Update graph and head map with preprocessed data
            self.screen.update_graph(result.preprocessed)

            # Update ICH status in UI
            self.screen.update_ich_status(result.ich_flags, result.ich_counts)

            # Log frame (less verbose - only log every 10 frames)
            if self.rendered_frame_count % 10 == 0:
                self.screen.append_log(
                    f"[{self._timestamp()}] Frame {result.frame.frame_number} @ "
                    f"{result.frame.timestamp_ms}ms\n"
                )

        # Update session info (always, to keep timer running)
        if self.session_id:
            summary = self.pipeline.get_summary()
            self.captured_frame_count = summary.captured_frames
            self.processed_frame_count = summary.processed_frames
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

        summary = self.pipeline.stop()

        # End session if still active
        if self.session_id:
            self.db.set_hemorrhage_result(self.session_id, summary.last_frame_hemorrhage_detected)
            end_time = datetime.datetime.now().isoformat()
            self.db.end_session(self.session_id, end_time)

        # Close database connection
        self.db.close()


if __name__ == '__main__':
    CerebraApp().run()
