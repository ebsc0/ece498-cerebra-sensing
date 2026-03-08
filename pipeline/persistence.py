"""Persistence helpers for pipeline workers."""

from __future__ import annotations

from typing import Dict

from database.database import (
    DatabaseManager,
    LogicalSampleRecord,
    PreprocessedSample,
    RawPacketRecord,
)
from packets import AssembledFrame, PHASE_740, PHASE_860, PHASE_DARK
from preprocessor import PreprocessedResult


def store_assembled_frame(
    db: DatabaseManager,
    session_id: int,
    frame: AssembledFrame,
) -> Dict[int, int]:
    """Insert raw phase packets and logical samples for one completed frame."""
    raw_batch = []
    raw_keys = []
    for optode_id in sorted(frame.phase_packets.keys()):
        for phase in (PHASE_740, PHASE_860, PHASE_DARK):
            packet = frame.phase_packets[optode_id][phase]
            raw_batch.append(
                (
                    frame.frame_number,
                    packet.timestamp_ms,
                    RawPacketRecord(
                        optode_id=optode_id,
                        phase=phase,
                        d0=packet.d0,
                        d1=packet.d1,
                        d2=packet.d2,
                        d3=packet.d3,
                    ),
                )
            )
            raw_keys.append((optode_id, phase))

    if not raw_batch:
        return {}

    with db.transaction() as cursor:
        packet_ids = db.insert_raw_packets_batch(session_id, raw_batch, cursor=cursor)
        packet_id_map = {key: packet_id for key, packet_id in zip(raw_keys, packet_ids)}

        logical_batch = []
        logical_optodes = []
        for optode_id in sorted(frame.logical_samples.keys()):
            sample = frame.logical_samples[optode_id]
            logical_batch.append(
                (
                    frame.frame_number,
                    sample.timestamp_ms,
                    LogicalSampleRecord(
                        optode_id=optode_id,
                        nm740_long=sample.nm740_long,
                        nm860_long=sample.nm860_long,
                        nm740_short=sample.nm740_short,
                        nm860_short=sample.nm860_short,
                        dark=sample.dark,
                        packet_740_id=packet_id_map[(optode_id, PHASE_740)],
                        packet_860_id=packet_id_map[(optode_id, PHASE_860)],
                        packet_dark_id=packet_id_map[(optode_id, PHASE_DARK)],
                    ),
                )
            )
            logical_optodes.append(optode_id)

        sample_ids = db.insert_logical_samples_batch(session_id, logical_batch, cursor=cursor)

    return {optode_id: sample_id for optode_id, sample_id in zip(logical_optodes, sample_ids)}


def store_preprocessed_frame(
    db: DatabaseManager,
    preprocessed: Dict[int, PreprocessedResult],
) -> None:
    """Insert per-optode preprocessed samples for one logical frame."""
    pre_batch = []
    for result in preprocessed.values():
        sample = PreprocessedSample(
            od_nm740_short=result.od_nm740_short,
            od_nm740_long=result.od_nm740_long,
            od_nm860_short=result.od_nm860_short,
            od_nm860_long=result.od_nm860_long,
            hbo_short=result.hbo_short,
            hbr_short=result.hbr_short,
            hbo_long=result.hbo_long,
            hbr_long=result.hbr_long,
        )
        pre_batch.append((result.sample_id, sample))

    if pre_batch:
        db.insert_preprocessed_samples_batch(pre_batch)
