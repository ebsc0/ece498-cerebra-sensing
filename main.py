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
    DATA_SOURCE,
    ICH_OPTODE_PAIRS,
    NUM_OPTODES,
    PACKET_FORMAT,
    SAMPLE_RATE_HZ,
    UART_AUTO_RECONNECT,
    UART_BAUDRATE,
    UART_PORT,
    UART_READ_CHUNK_SIZE,
    UART_RECONNECT_INITIAL_S,
    UART_RECONNECT_MAX_S,
    UART_TIMEOUT_S,
    UI_UPDATE_RATE_HZ,
)
from data_source import DataSource, SourceEvent
from database.database import DatabaseManager
from ich_detection import reset_history
from pipeline.runtime import PipelineRuntime
from simulator import SimulatorSource
from ui import MainScreen
from uart import UartSource
from preprocessor import Preprocessor

# Database configuration
DB_DIR = os.path.join(os.path.dirname(__file__), "db")
DB_FILE = os.path.join(DB_DIR, "fnirs_data.db")


class CerebraApp(App):
    """Kivy app with integrated data collection and real-time visualization."""

    def build(self):
        self.source: DataSource = self._build_data_source()

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

    @staticmethod
    def _build_data_source() -> DataSource:
        if DATA_SOURCE == "simulator":
            return SimulatorSource(num_optodes=NUM_OPTODES, sample_rate_hz=SAMPLE_RATE_HZ)
        if DATA_SOURCE == "uart":
            return UartSource(
                port=UART_PORT,
                baudrate=UART_BAUDRATE,
                packet_format=PACKET_FORMAT,
                active_optodes=ACTIVE_OPTODES,
                timeout_s=UART_TIMEOUT_S,
                read_chunk_size=UART_READ_CHUNK_SIZE,
                auto_reconnect=UART_AUTO_RECONNECT,
                reconnect_initial_s=UART_RECONNECT_INITIAL_S,
                reconnect_max_s=UART_RECONNECT_MAX_S,
            )
        raise RuntimeError(f"Unsupported DATA_SOURCE '{DATA_SOURCE}'.")

    def _on_source_event(self, event: SourceEvent) -> None:
        def _handle(_dt):
            self.screen.append_log(f"[{self._timestamp()}] Source {event.kind}: {event.message}\n")
            if event.kind == "fatal_disconnected" and self.session_id:
                self.screen.append_log(
                    f"[{self._timestamp()}] Source disconnected. Terminating session.\n"
                )
                self.stop_collection()

        Clock.schedule_once(_handle, 0)

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

    def _elapsed_ms(self) -> int:
        """Get elapsed time since session start in milliseconds."""
        if self.session_start_time is None:
            return 0
        return max(0, int((time.time() - self.session_start_time) * 1000))

    def start_collection(self):
        """Start data collection and create new session."""
        # Ensure source is stopped before starting a fresh session.
        if self.source.is_running():
            self.source.stop()

        # Reset algorithmic history.
        reset_history()

        # Reset session tracking
        self.session_start_time = time.time()
        self.captured_frame_count = 0
        self.processed_frame_count = 0
        self.rendered_frame_count = 0

        try:
            # Create new session
            self.session_id = self.db.create_session(
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
            self.source.start(self.pipeline.ingest_packet, self._on_source_event)

        except Exception as exc:
            self._log_error_throttled(f"[{self._timestamp()}] Startup error: {exc}\n")
            if self.source.is_running():
                self.source.stop()

            if self.session_id:
                try:
                    self.db.end_session(self.session_id, self._elapsed_ms())
                except Exception as db_exc:
                    self._log_error_throttled(
                        f"[{self._timestamp()}] Startup rollback DB cleanup error: {db_exc}\n"
                    )

            self.pipeline.clear_ui_results()
            self.session_id = None
            self.session_start_time = None
            self.captured_frame_count = 0
            self.processed_frame_count = 0
            self.rendered_frame_count = 0
            self.screen.update_session_info(
                session_id=None,
                elapsed_str="--:--:--",
                captured_count=0,
                processed_count=0,
            )
            self.screen.start_btn.disabled = False
            self.screen.stop_btn.disabled = True

    def stop_collection(self):
        """Stop data collection and end session."""
        self.source.stop()

        if not self.session_id:
            return

        summary = self.pipeline.stop()
        self.captured_frame_count = summary.captured_frames
        self.processed_frame_count = summary.processed_frames
        self.pipeline.clear_ui_results()
        try:
            self.db.set_hemorrhage_result(self.session_id, summary.session_hemorrhage_detected)
            self.db.end_session(self.session_id, self._elapsed_ms())
        except Exception as exc:
            self._log_error_throttled(f"[{self._timestamp()}] Session finalization error: {exc}\n")

        # Log session end
        elapsed = self._elapsed_time_str()
        self.screen.append_log(
            f"[{self._timestamp()}] Session stopped (ID: {self.session_id}, "
            f"Duration: {elapsed}, Captured: {self.captured_frame_count}, "
            f"Processed: {self.processed_frame_count}, "
            f"Dropped(incomplete): {summary.dropped_incomplete_frames})\n"
        )
        if not summary.drain_complete:
            self.screen.append_log(
                f"[{self._timestamp()}] Warning: pipeline stop timed out; session ended with partial drain.\n"
            )

        self.session_id = None
        self.session_start_time = None
        self.screen.update_session_info(
            session_id=None,
            elapsed_str="--:--:--",
            captured_count=0,
            processed_count=0,
        )

    def _update_ui(self, dt):
        """Poll pipeline output and update UI."""
        for err in self.pipeline.drain_errors():
            self._log_error_throttled(f"[{self._timestamp()}] {err}\n")

        if not self.session_id:
            self.pipeline.clear_ui_results()
            self.screen.update_session_info(
                session_id=None,
                elapsed_str="--:--:--",
                captured_count=0,
                processed_count=0,
            )
            return

        pending = self.pipeline.drain_ui_results()
        if pending:
            # High-rate streams can produce many frames per UI tick. Coalesce to
            # the latest frame so rendering stays responsive.
            latest = pending[-1]
            self.rendered_frame_count += len(pending)

            # Update graph and ICH status using most recent processed frame only.
            self.screen.update_graph(latest.preprocessed)
            self.screen.update_ich_status(latest.ich_flags, latest.ich_counts)

            # Keep logs bounded at higher rates.
            if self.rendered_frame_count % 100 == 0:
                self.screen.append_log(
                    f"[{self._timestamp()}] Frame {latest.frame.frame_number} @ "
                    f"{latest.frame.timestamp_ms}ms\n"
                )

        # Update session info (always, to keep timer running)
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
        if self.source.is_running():
            self.source.stop()

        summary = self.pipeline.stop()
        self.pipeline.clear_ui_results()
        if not summary.drain_complete:
            self.screen.append_log(
                f"[{self._timestamp()}] Warning: pipeline stop timed out during app shutdown.\n"
            )

        # End session if still active
        if self.session_id:
            try:
                self.db.set_hemorrhage_result(self.session_id, summary.session_hemorrhage_detected)
                self.db.end_session(self.session_id, self._elapsed_ms())
            except Exception as exc:
                self._log_error_throttled(f"[{self._timestamp()}] Shutdown finalization error: {exc}\n")

        # Close database connection
        self.db.close()


if __name__ == '__main__':
    CerebraApp().run()
