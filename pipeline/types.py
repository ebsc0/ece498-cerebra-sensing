"""Data types used by the acquisition/processing pipeline."""

from typing import Dict, NamedTuple

from buffer import CompleteFrame
from preprocessor import PreprocessedResult


class MatchedFrame(NamedTuple):
    """A complete frame plus raw-sample DB IDs keyed by optode."""

    frame: CompleteFrame
    sample_ids: Dict[int, int]


class UiFrameResult(NamedTuple):
    """Per-frame data consumed by the UI update loop."""

    frame: CompleteFrame
    preprocessed: Dict[int, PreprocessedResult]
    ich_flags: Dict[int, bool]
    ich_counts: Dict[int, int]


class PipelineSummary(NamedTuple):
    """Session/runtime counters exposed to app orchestrator."""

    captured_frames: int
    processed_frames: int
    dropped_incomplete_frames: int
    last_frame_hemorrhage_detected: bool
