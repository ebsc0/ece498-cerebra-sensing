"""Shared packet definition, decoding, and logical reconstruction helpers.

Wire order remains ``d0``..``d3``, but physical meaning is:
- ``d0``: farthest diode
- ``d3``: nearest diode
"""

from __future__ import annotations

from dataclasses import dataclass
import struct
from typing import Dict

import math

PACKET_FORMAT = "<I4f"
PACKET_SIZE = struct.calcsize(PACKET_FORMAT)

PHASE_740 = 0
PHASE_860 = 1
PHASE_DARK = 2
VALID_PHASES = (PHASE_740, PHASE_860, PHASE_DARK)

@dataclass(frozen=True)
class PhasePacket:
    frame_number: int
    optode_id: int
    phase: int
    timestamp_ms: int
    d0: float
    d1: float
    d2: float
    d3: float


@dataclass(frozen=True)
class LogicalSample:
    frame_number: int
    optode_id: int
    timestamp_ms: int
    nm740_long: float
    nm860_long: float
    nm740_short: float
    nm860_short: float
    dark: float


@dataclass(frozen=True)
class AssembledFrame:
    frame_number: int
    timestamp_ms: int
    phase_packets: Dict[int, Dict[int, PhasePacket]]
    logical_samples: Dict[int, LogicalSample]


def encode_metadata(frame_number: int, optode_id: int, phase: int) -> int:
    if frame_number < 0 or frame_number >= (1 << 26):
        raise ValueError("frame_number must fit in 26 bits")
    if optode_id < 0 or optode_id >= (1 << 4):
        raise ValueError("optode_id must fit in 4 bits")
    if phase not in VALID_PHASES:
        raise ValueError(f"Invalid phase: {phase}")
    return (frame_number << 6) | (optode_id << 2) | phase


def decode_metadata(metadata: int) -> tuple[int, int, int]:
    if metadata < 0:
        raise ValueError("metadata must be non-negative")
    frame_number = metadata >> 6
    optode_id = (metadata >> 2) & 0xF
    phase = metadata & 0x3
    return frame_number, optode_id, phase


def decode_packet(packet: bytes, timestamp_ms: int) -> PhasePacket:
    if len(packet) != PACKET_SIZE:
        raise ValueError(f"Expected {PACKET_SIZE} bytes, got {len(packet)}")
    metadata, d0, d1, d2, d3 = struct.unpack(PACKET_FORMAT, packet)
    frame_number, optode_id, phase = decode_metadata(metadata)
    if phase not in VALID_PHASES:
        raise ValueError(f"Invalid phase: {phase}")
    values = (float(d0), float(d1), float(d2), float(d3))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Packet contains non-finite diode values")
    return PhasePacket(
        frame_number=frame_number,
        optode_id=optode_id,
        phase=phase,
        timestamp_ms=int(timestamp_ms),
        d0=values[0],
        d1=values[1],
        d2=values[2],
        d3=values[3],
    )


def reconstruct_logical_sample(phases: Dict[int, PhasePacket], frame_timestamp_ms: int) -> LogicalSample:
    """Reconstruct logical short/long channels from phase packets.

    Physical layout is reversed relative to wire index:
    - nearest diode is ``d3`` -> short channel
    - farthest diode is ``d0`` -> long channel
    """
    missing = set(VALID_PHASES).difference(phases)
    if missing:
        raise ValueError(f"Missing phases for logical reconstruction: {sorted(missing)}")

    phase_740 = phases[PHASE_740]
    phase_860 = phases[PHASE_860]
    phase_dark = phases[PHASE_DARK]

    if len({phase_740.frame_number, phase_860.frame_number, phase_dark.frame_number}) != 1:
        raise ValueError("Phase packets must share frame_number")
    if len({phase_740.optode_id, phase_860.optode_id, phase_dark.optode_id}) != 1:
        raise ValueError("Phase packets must share optode_id")

    dark = (phase_dark.d0 + phase_dark.d1 + phase_dark.d2 + phase_dark.d3) / 4.0
    return LogicalSample(
        frame_number=phase_740.frame_number,
        optode_id=phase_740.optode_id,
        timestamp_ms=int(frame_timestamp_ms),
        nm740_long=phase_740.d2,
        nm860_long=phase_860.d2,
        nm740_short=phase_740.d3,
        nm860_short=phase_860.d3,
        dark=float(dark),
    )
