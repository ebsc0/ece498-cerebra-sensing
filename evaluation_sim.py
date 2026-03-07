"""
Minimal terminal application for evaluating ICH detection.

Pipeline:
Simulator -> Buffer -> Preprocessor -> ICH Detection

Prints decisions to the terminal.
Stop with Ctrl+C.
"""

import time
import datetime
from typing import Dict

from config import (
    ACTIVE_OPTODES,
    BUFFER_MAX_PENDING_FRAMES,
    BUFFER_STALE_TIMEOUT_MS,
    NUM_OPTODES,
    SAMPLE_RATE_HZ,
)

from simulator_min import Simulator
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

    def start(self):

        print("=== Cerebra Terminal Evaluation ===")
        print("Press Ctrl+C to stop\n")

        reset_history()
        self.preprocessor.reset()
        self.buffer.clear()

        self.simulator.start(self._on_packet)

    def stop(self):

        print("\nStopping...")
        self.simulator.stop()
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

        sample_ids = {optode_id: 0 for optode_id in frame.packets.keys()}

        return self.preprocessor.process_frame(frame, sample_ids)

    def _prepare_ich_data(
        self, preprocessed: Dict[int, PreprocessedResult]
    ) -> Dict[int, dict]:

        ich_data = {}

        for optode_id, result in preprocessed.items():
            ich_data[optode_id] = {
                "sample_id": result.sample_id,
                "optode_id": result.optode_id,
                "frame_number": result.frame_number,
                "timestamp_ms": result.timestamp_ms,
                "OD_740_SHORT": result.od_nm740_short,
                "OD_740": result.od_nm740_long,
                "OD_860_SHORT": result.od_nm860_short,
                "OD_860": result.od_nm860_long,
                "HBO_SHORT": result.hbo_short,
                "HBR_SHORT": result.hbr_short,
                "HbO": result.hbo_long,
                "HbR": result.hbr_long,
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

        #print(f"\n[{timestamp}] Frame {frame.frame_number}")
        print("Simulator ICH:", self.simulator.has_ich)
        for optode_id in ACTIVE_OPTODES:

            decision = final_flags.get(optode_id, "UNKNOWN")
            votes = counts.get(optode_id, 0)

            print(
                f"Optode {optode_id}: {decision} | Votes: {votes}"
            )


if __name__ == "__main__":

    app = CerebraTerminalApp()

    try:

        app.start()

        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        app.stop()