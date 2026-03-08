"""Data types used by the acquisition/processing pipeline."""

from typing import Dict, NamedTuple

from packets import AssembledFrame
from preprocessor import PreprocessedResult


class RawPacket(NamedTuple):
    """Packet envelope with ingress timestamp (monotonic ms) captured at thread1."""

    packet: bytes
    ingress_timestamp_ms: int


class StoredLogicalFrame(NamedTuple):
    """A complete logical frame plus logical-sample DB IDs keyed by optode."""

    frame: AssembledFrame
    sample_ids: Dict[int, int]


class UiFrameResult(NamedTuple):
    """Per-frame data consumed by the UI update loop."""

    frame: AssembledFrame
    preprocessed: Dict[int, PreprocessedResult]
    ich_flags: Dict[int, bool]
    ich_counts: Dict[int, int]


class PipelineSummary(NamedTuple):
    """Session/runtime counters exposed to app orchestrator."""

    captured_frames: int
    processed_frames: int
    dropped_incomplete_frames: int
    session_hemorrhage_detected: bool
    drain_complete: bool
