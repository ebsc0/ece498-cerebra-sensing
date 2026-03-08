import logging
import os
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional


@dataclass(frozen=True)
class RawPacketRecord:
    optode_id: int
    phase: int
    d0: float
    d1: float
    d2: float
    d3: float


@dataclass(frozen=True)
class LogicalSampleRecord:
    optode_id: int
    nm740_long: float
    nm860_long: float
    nm740_short: float
    nm860_short: float
    dark: float
    packet_740_id: int
    packet_860_id: int
    packet_dark_id: int


@dataclass(frozen=True)
class PreprocessedSample:
    od_nm740_short: float
    od_nm740_long: float
    od_nm860_short: float
    od_nm860_long: float
    hbo_short: float
    hbr_short: float
    hbo_long: float
    hbr_long: float


class DatabaseManager:
    """Thread-safe SQLite access for sessions, raw packets, logical samples, and preprocessing."""

    def __init__(self, db_file: str):
        self.db_file = db_file
        self.connection: Optional[sqlite3.Connection] = None
        self.lock = threading.Lock()

    def connect(self) -> None:
        db_dir = os.path.dirname(self.db_file)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        self.connection = sqlite3.connect(self.db_file, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.connection.execute("PRAGMA synchronous = NORMAL")
        self._create_tables()
        logging.info("Connected to database: %s", self.db_file)

    def _create_tables(self) -> None:
        if not self.connection:
            raise RuntimeError("Database connection is not open")

        schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
        with open(schema_path, "r", encoding="utf-8") as handle:
            schema_sql = handle.read()
        with self.lock:
            self.connection.executescript(schema_sql)
            self.connection.commit()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Cursor]:
        if not self.connection:
            raise RuntimeError("Database connection is not open")
        with self.lock:
            cursor = self.connection.cursor()
            try:
                yield cursor
                self.connection.commit()
            except Exception:
                self.connection.rollback()
                raise

    def create_session(self, sample_rate_hz: float = 5.0, num_optodes: int = 2) -> int:
        if not self.connection:
            raise RuntimeError("Database connection is not open")
        with self.lock:
            cursor = self.connection.cursor()
            cursor.execute(
                "INSERT INTO sessions (sample_rate_hz, num_optodes) VALUES (?, ?)",
                (sample_rate_hz, num_optodes),
            )
            self.connection.commit()
            session_id = cursor.lastrowid
            if session_id is None:
                raise RuntimeError("Failed to get session ID after insert")
            return int(session_id)

    def end_session(self, session_id: int, elapsed_ms: Optional[int]) -> None:
        if not self.connection:
            raise RuntimeError("Database connection is not open")
        with self.lock:
            self.connection.execute(
                "UPDATE sessions SET elapsed_ms = ? WHERE session_id = ?",
                (elapsed_ms, session_id),
            )
            self.connection.commit()

    def set_hemorrhage_result(self, session_id: int, detected: bool) -> None:
        if not self.connection:
            raise RuntimeError("Database connection is not open")
        with self.lock:
            self.connection.execute(
                "UPDATE sessions SET hemorrhage_detected = ? WHERE session_id = ?",
                (1 if detected else 0, session_id),
            )
            self.connection.commit()

    def get_session(self, session_id: int) -> Optional[Dict[str, Any]]:
        if not self.connection:
            return None
        with self.lock:
            cursor = self.connection.cursor()
            cursor.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def insert_raw_packets_batch(
        self,
        session_id: int,
        packets: List[tuple[int, int, RawPacketRecord]],
        *,
        cursor: Optional[sqlite3.Cursor] = None,
    ) -> List[int]:
        if not self.connection:
            raise RuntimeError("Database connection is not open")
        sql = (
            "INSERT INTO raw_packets ("
            "session_id, frame_number, optode_id, phase, timestamp_ms, d0, d1, d2, d3"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
        )

        owns_transaction = cursor is None
        if owns_transaction:
            with self.transaction() as txn_cursor:
                return self.insert_raw_packets_batch(session_id, packets, cursor=txn_cursor)

        packet_ids: List[int] = []
        for frame_number, timestamp_ms, packet in packets:
            cursor.execute(
                sql,
                (
                    session_id,
                    frame_number,
                    packet.optode_id,
                    packet.phase,
                    timestamp_ms,
                    packet.d0,
                    packet.d1,
                    packet.d2,
                    packet.d3,
                ),
            )
            packet_id = cursor.lastrowid
            if packet_id is None:
                raise RuntimeError("Failed to capture packet_id after raw packet insert")
            packet_ids.append(int(packet_id))
        return packet_ids

    def insert_logical_samples_batch(
        self,
        session_id: int,
        samples: List[tuple[int, int, LogicalSampleRecord]],
        *,
        cursor: Optional[sqlite3.Cursor] = None,
    ) -> List[int]:
        if not self.connection:
            raise RuntimeError("Database connection is not open")
        sql = (
            "INSERT INTO logical_samples ("
            "session_id, frame_number, optode_id, timestamp_ms, nm740_long, nm860_long, "
            "nm740_short, nm860_short, dark, packet_740_id, packet_860_id, packet_dark_id"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        )

        owns_transaction = cursor is None
        if owns_transaction:
            with self.transaction() as txn_cursor:
                return self.insert_logical_samples_batch(session_id, samples, cursor=txn_cursor)

        sample_ids: List[int] = []
        for frame_number, timestamp_ms, sample in samples:
            cursor.execute(
                sql,
                (
                    session_id,
                    frame_number,
                    sample.optode_id,
                    timestamp_ms,
                    sample.nm740_long,
                    sample.nm860_long,
                    sample.nm740_short,
                    sample.nm860_short,
                    sample.dark,
                    sample.packet_740_id,
                    sample.packet_860_id,
                    sample.packet_dark_id,
                ),
            )
            sample_id = cursor.lastrowid
            if sample_id is None:
                raise RuntimeError("Failed to capture sample_id after logical sample insert")
            sample_ids.append(int(sample_id))
        return sample_ids

    def insert_preprocessed_samples_batch(
        self,
        samples: List[tuple[int, PreprocessedSample]],
        *,
        cursor: Optional[sqlite3.Cursor] = None,
    ) -> None:
        if not self.connection:
            raise RuntimeError("Database connection is not open")
        sql = (
            "INSERT INTO preprocessed_samples ("
            "sample_id, od_nm740_short, od_nm740_long, od_nm860_short, od_nm860_long, "
            "hbo_short, hbr_short, hbo_long, hbr_long"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
        )

        owns_transaction = cursor is None
        if owns_transaction:
            with self.transaction() as txn_cursor:
                self.insert_preprocessed_samples_batch(samples, cursor=txn_cursor)
            return

        for sample_id, sample in samples:
            cursor.execute(
                sql,
                (
                    sample_id,
                    sample.od_nm740_short,
                    sample.od_nm740_long,
                    sample.od_nm860_short,
                    sample.od_nm860_long,
                    sample.hbo_short,
                    sample.hbr_short,
                    sample.hbo_long,
                    sample.hbr_long,
                ),
            )

    def query_latest_raw_packets(
        self,
        session_id: int,
        limit: int = 100,
        optode_id: Optional[int] = None,
        phase: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if not self.connection:
            return []

        conditions = ["session_id = ?"]
        params: List[Any] = [session_id]
        if optode_id is not None:
            conditions.append("optode_id = ?")
            params.append(optode_id)
        if phase is not None:
            conditions.append("phase = ?")
            params.append(phase)
        params.append(limit)
        sql = (
            "SELECT * FROM raw_packets WHERE " + " AND ".join(conditions) +
            " ORDER BY timestamp_ms DESC, packet_id DESC LIMIT ?"
        )

        with self.lock:
            cursor = self.connection.cursor()
            cursor.execute(sql, params)
            return [dict(row) for row in cursor.fetchall()]

    def query_latest_logical_samples(
        self,
        session_id: int,
        limit: int = 100,
        optode_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if not self.connection:
            return []

        conditions = ["session_id = ?"]
        params: List[Any] = [session_id]
        if optode_id is not None:
            conditions.append("optode_id = ?")
            params.append(optode_id)
        params.append(limit)
        sql = (
            "SELECT * FROM logical_samples WHERE " + " AND ".join(conditions) +
            " ORDER BY timestamp_ms DESC, sample_id DESC LIMIT ?"
        )

        with self.lock:
            cursor = self.connection.cursor()
            cursor.execute(sql, params)
            return [dict(row) for row in cursor.fetchall()]

    def query_latest_preprocessed_samples(
        self,
        session_id: int,
        limit: int = 100,
        optode_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if not self.connection:
            return []

        conditions = ["l.session_id = ?"]
        params: List[Any] = [session_id]
        if optode_id is not None:
            conditions.append("l.optode_id = ?")
            params.append(optode_id)
        params.append(limit)
        sql = (
            "SELECT p.sample_id, l.session_id, l.optode_id, l.frame_number, l.timestamp_ms, "
            "p.od_nm740_short, p.od_nm740_long, p.od_nm860_short, p.od_nm860_long, "
            "p.hbo_short, p.hbr_short, p.hbo_long, p.hbr_long "
            "FROM preprocessed_samples p "
            "INNER JOIN logical_samples l ON l.sample_id = p.sample_id "
            "WHERE " + " AND ".join(conditions) +
            " ORDER BY l.timestamp_ms DESC, p.sample_id DESC LIMIT ?"
        )

        with self.lock:
            cursor = self.connection.cursor()
            cursor.execute(sql, params)
            return [dict(row) for row in cursor.fetchall()]

    def query_samples_by_session(
        self,
        session_id: int,
        include_raw: bool = True,
        include_logical: bool = True,
        include_preprocessed: bool = True,
    ) -> Dict[str, List[Dict[str, Any]]]:
        result: Dict[str, List[Dict[str, Any]]] = {}
        if include_raw:
            result["raw_packets"] = self.query_latest_raw_packets(session_id, limit=1_000_000)
        if include_logical:
            result["logical_samples"] = self.query_latest_logical_samples(session_id, limit=1_000_000)
        if include_preprocessed:
            result["preprocessed"] = self.query_latest_preprocessed_samples(session_id, limit=1_000_000)
        return result

    def query_samples_by_time_range(
        self,
        session_id: int,
        start_ms: int,
        end_ms: int,
        table: str = "preprocessed",
        optode_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if not self.connection:
            return []

        if table == "raw":
            sql = (
                "SELECT * FROM raw_packets WHERE session_id = ? AND timestamp_ms BETWEEN ? AND ?"
            )
            params: List[Any] = [session_id, start_ms, end_ms]
            if optode_id is not None:
                sql += " AND optode_id = ?"
                params.append(optode_id)
            sql += " ORDER BY frame_number, optode_id, phase"
        elif table == "logical":
            sql = (
                "SELECT * FROM logical_samples WHERE session_id = ? AND timestamp_ms BETWEEN ? AND ?"
            )
            params = [session_id, start_ms, end_ms]
            if optode_id is not None:
                sql += " AND optode_id = ?"
                params.append(optode_id)
            sql += " ORDER BY frame_number, optode_id"
        else:
            sql = (
                "SELECT p.sample_id, l.session_id, l.optode_id, l.frame_number, l.timestamp_ms, "
                "p.od_nm740_short, p.od_nm740_long, p.od_nm860_short, p.od_nm860_long, "
                "p.hbo_short, p.hbr_short, p.hbo_long, p.hbr_long "
                "FROM preprocessed_samples p "
                "INNER JOIN logical_samples l ON l.sample_id = p.sample_id "
                "WHERE l.session_id = ? AND l.timestamp_ms BETWEEN ? AND ?"
            )
            params = [session_id, start_ms, end_ms]
            if optode_id is not None:
                sql += " AND l.optode_id = ?"
                params.append(optode_id)
            sql += " ORDER BY l.frame_number, l.optode_id"

        with self.lock:
            cursor = self.connection.cursor()
            cursor.execute(sql, params)
            return [dict(row) for row in cursor.fetchall()]

    def close(self) -> None:
        if self.connection:
            self.connection.close()
            self.connection = None
            logging.info("Database connection closed")
