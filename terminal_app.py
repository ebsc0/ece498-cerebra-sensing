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

        # -------------------------
        # Evaluation parameters
        # -------------------------

        self.frames_per_trial = 20
        self.total_trials = 10

        self.current_trial = 0

        # confusion matrix
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

        self.votes = 17

        # trial detection tracking
        self.trial_detected = False
        self.trial_finished = False

    def _start_trial(self):

        print(f"\n=== Trial {self.current_trial + 1}/{self.total_trials} ===")

        reset_history()
        self.preprocessor.reset()
        self.buffer.clear()

        self.frame_count = 0
        self.trial_detected = False

        self.simulator.start(self._on_packet)

    def start(self):

        print("=== Cerebra Terminal Mode ===")
        print("Starting evaluation...\n")

        self.current_trial = 0
        self.start_time = time.time()

        self._start_trial()

    def _end_trial(self):

        self.simulator.stop()

        ground_truth = self.simulator.has_ich

        if ground_truth and self.trial_detected:
            self.tp += 1
        elif ground_truth and not self.trial_detected:
            self.fn += 1
        elif not ground_truth and self.trial_detected:
            self.fp += 1
        else:
            self.tn += 1

        print(
            f"Trial result | "
            f"ICH={ground_truth} | "
            f"Detected={self.trial_detected}"
        )

        if self.trial_detected == False:
            print("vote count: ", self.votes)
            
        self.current_trial += 1

        if self.current_trial >= self.total_trials:
            self._print_evaluation()
            return

        time.sleep(1)
        self._start_trial()

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

        if self.frame_count >= self.frames_per_trial:
            self.trial_finished = True

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

        # print(
        #     f"\n[{timestamp}] Frame {frame.frame_number} "
        #     f"@ {frame.timestamp_ms} ms"
        # )
        self.votes = 0
        for optode_id in ACTIVE_OPTODES:

            decision = final_flags.get(optode_id, "UNKNOWN")
            self.votes = max(self.votes, counts.get(optode_id, 0))

            if decision != "NORMAL":
                self.trial_detected = True

            # print(
            #     f"  Optode {optode_id}: "
            #     f"{decision} | Votes: {vote_count} | "
            # )

    def _print_evaluation(self):
        print("\n============================")
        print("MODEL EVALUATION RESULTS")
        print("============================")

        print(f"Trials: {self.total_trials}")
        print(f"TP: {self.tp}")
        print(f"TN: {self.tn}")
        print(f"FP: {self.fp}")
        print(f"FN: {self.fn}")

        sensitivity = (
            self.tp / (self.tp + self.fn)
            if (self.tp + self.fn) > 0 else 0
        )

        specificity = (
            self.tn / (self.tn + self.fp)
            if (self.tn + self.fp) > 0 else 0
        )

        print(f"\nSensitivity: {sensitivity:.3f}")
        print(f"Specificity: {specificity:.3f}")


if __name__ == "__main__":

    app = CerebraTerminalApp()

    try:
        app.start()

        while app.current_trial < app.total_trials:

            if app.trial_finished:

                app.simulator.stop()
                app._end_trial()

                app.trial_finished = False

            time.sleep(0.1)

    except KeyboardInterrupt:
        app.simulator.stop()