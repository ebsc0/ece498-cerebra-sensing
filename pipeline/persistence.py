"""Persistence helpers for pipeline workers."""

import struct
from typing import Dict

from buffer import CompleteFrame
from database.database import DatabaseManager, PreprocessedSample, RawSample
from preprocessor import PreprocessedResult


def store_raw_frame(
    db: DatabaseManager,
    session_id: int,
    frame: CompleteFrame,
    packet_format: str,
) -> Dict[int, int]:
    """Insert complete-frame raw samples and return {optode_id: sample_id}."""
    raw_batch = []
    optode_order = []
    for optode_id, packet in frame.packets.items():
        data = struct.unpack(packet_format, packet)
        sample = RawSample(
            optode_id=optode_id,
            nm740_long=data[1],
            nm860_long=data[2],
            nm740_short=data[3],
            nm860_short=data[4],
            dark=data[5],
        )
        raw_batch.append((frame.frame_number, frame.timestamp_ms, sample))
        optode_order.append(optode_id)

    if not raw_batch:
        return {}

    raw_ids = db.insert_raw_samples_batch(session_id, raw_batch)
    return {optode_id: sample_id for optode_id, sample_id in zip(optode_order, raw_ids)}


def store_preprocessed_frame(
    db: DatabaseManager,
    session_id: int,
    preprocessed: Dict[int, PreprocessedResult],
) -> None:
    """Insert per-optode preprocessed samples for one frame."""
    pre_batch = []
    for result in preprocessed.values():
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
        pre_batch.append((result.sample_id, result.frame_number, result.timestamp_ms, sample))

    if pre_batch:
        db.insert_preprocessed_samples_batch(session_id, pre_batch)

