"""
Terminal-only application entry point.

Runs simulator, preprocessing, and ICH detection.
Prints per-frame decisions to the terminal.
"""

import struct
import time
import datetime
from typing import Dict

from config import (
    ACTIVE_OPTODES,
    BUFFER_MAX_PENDING_FRAMES,
    BUFFER_STALE_TIMEOUT_MS,
    NUM_OPTODES,
    PACKET_FORMAT,
    SAMPLE_RATE_HZ,
)

from simulator import Simulator
from buffer import Buffer, CompleteFrame
from preprocessor import Preprocessor, PreprocessedResult
from ich_detection import detect_ich, reset_history


class CerebraTerminalApp:
    def __init__(self):
        self.buffer = Buffer(
            num_optodes=NUM_OPTODES,
            stale_timeout_ms=BUFFER_STALE_TIMEOUT_MS,
            max_pending_frames=BUFFER_MAX_PENDING_FRAMES,
        )

        self.simulator = Simulator(
            num_optodes=NUM_OPTODES,
            sample_rate_hz=SAMPLE_RATE_HZ,
        )

        self.preprocessor = Preprocessor()

        self.frame_count = 0
        self.start_time = None

    def start(self):
        print("=== Cerebra Terminal Mode ===")
        print("Starting session...\n")

        reset_history()
        self.preprocessor.reset()
        self.buffer.clear()

        self.start_time = time.time()
        self.frame_count = 0

        self.simulator.start(self._on_packet)

    def stop(self):
        print("\nStopping session...")
        self.simulator.stop()

        elapsed = time.time() - self.start_time
        print(f"Session duration: {elapsed:.2f} seconds")
        print(f"Processed frames: {self.frame_count}")

    def _on_packet(self, packet: bytes):
        complete_frame = self.buffer.add_packet(packet)
        if not complete_frame:
            return

        preprocessed = self._process_frame(complete_frame)
        if not preprocessed:
            return

        self.frame_count += 1
        self._run_ich_detection(complete_frame, preprocessed)

    def _process_frame(
        self, frame: CompleteFrame
    ) -> Dict[int, PreprocessedResult]:
        """
        Convert packets into PreprocessedResults (no DB storage).
        """
        sample_ids = {}

        # Fake sample IDs since DB is removed
        for optode_id in frame.packets.keys():
            sample_ids[optode_id] = 0

        return self.preprocessor.process_frame(frame, sample_ids)

    def _prepare_ich_data(
        self, preprocessed: Dict[int, PreprocessedResult]
    ) -> Dict[int, dict]:
        """
        Convert preprocessed data to ICH format.
        """
        ich_data = {}

        for optode_id, result in preprocessed.items():
            ich_data[optode_id] = {
                "HbR": result.hbr_long,
                "OD_860": result.od_nm860_long,
            }

        return ich_data

    def _run_ich_detection(
        self,
        frame: CompleteFrame,
        preprocessed: Dict[int, PreprocessedResult],
    ):
        ich_data = self._prepare_ich_data(preprocessed)

        final_flags, counts = detect_ich(ich_data, ACTIVE_OPTODES)

        timestamp = datetime.datetime.now().strftime("%H:%M:%S")

        print(
            f"\n[{timestamp}] Frame {frame.frame_number} "
            f"@ {frame.timestamp_ms} ms"
        )

        for optode_id in ACTIVE_OPTODES:

            decision = final_flags.get(optode_id, "UNKNOWN")
            vote_count = counts.get(optode_id, 0)

            print(
                f"  Optode {optode_id}: "
                f"{decision} | Votes: {vote_count} | "
            )


if __name__ == "__main__":
    app = CerebraTerminalApp()

    try:
        app.start()

        # Run until interrupted
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        app.stop()