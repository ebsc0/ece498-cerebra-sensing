import sqlite3
import threading
import os
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import logging


@dataclass
class RawSample:
    """Raw intensity values from a single optode."""
    optode_id: int
    nm740_long: float
    nm860_long: float
    nm740_short: float
    nm860_short: float
    dark: float


@dataclass
class PreprocessedSample:
    """Optical density and hemoglobin values from a single optode."""
    optode_id: int
    od_nm740_short: float
    od_nm740_long: float
    od_nm860_short: float
    od_nm860_long: float
    hbo_short: float
    hbr_short: float
    hbo_long: float
    hbr_long: float


class DatabaseManager:
    """
    Manages all interactions with the SQLite database in a thread-safe manner.
    Uses the schema defined in schema.sql.
    """

    def __init__(self, db_file: str):
        """
        Initializes the DatabaseManager.

        Args:
            db_file: The path to the SQLite database file.
        """
        self.db_file = db_file
        self.connection: Optional[sqlite3.Connection] = None
        self.lock = threading.Lock()

    def connect(self):
        """
        Establishes a connection to the database.
        Creates the directory and tables if they don't exist.
        """
        try:
            db_dir = os.path.dirname(self.db_file)
            if db_dir:
                os.makedirs(db_dir, exist_ok=True)

            self.connection = sqlite3.connect(self.db_file, check_same_thread=False)
            self.connection.row_factory = sqlite3.Row
            self.connection.execute("PRAGMA foreign_keys = ON")
            logging.info(f"Connected to database: {self.db_file}")
            self._create_tables()
        except (sqlite3.Error, OSError) as e:
            logging.error(f"Error connecting to database: {e}")
            raise

    def _create_tables(self):
        """Creates the database tables from schema.sql if they don't exist."""
        if not self.connection:
            return

        schema_path = os.path.join(os.path.dirname(__file__), 'schema.sql')

        try:
            with open(schema_path, 'r') as f:
                schema_sql = f.read()

            with self.lock:
                cursor = self.connection.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sessions'")
                if cursor.fetchone() is None:
                    self.connection.executescript(schema_sql)
                    logging.info("Database schema initialized from schema.sql")
                else:
                    logging.info("Database tables already exist")
        except FileNotFoundError:
            logging.error(f"Schema file not found: {schema_path}")
            raise
        except sqlite3.Error as e:
            logging.error(f"Error creating tables: {e}")
            raise

    # =========================================================================
    # Session Management
    # =========================================================================

    def create_session(
        self,
        start_time: str,
        sample_rate_hz: float = 5.0,
        num_optodes: int = 2
    ) -> int:
        """
        Creates a new session and returns its ID.

        Args:
            start_time: ISO8601 timestamp for session start.
            sample_rate_hz: Sample rate in Hz (default 5.0).
            num_optodes: Number of optodes (default 2).

        Returns:
            The session_id of the newly created session.
        """
        if not self.connection:
            raise RuntimeError("Database connection is not open")

        sql = """
            INSERT INTO sessions (start_time, sample_rate_hz, num_optodes)
            VALUES (?, ?, ?)
        """

        try:
            with self.lock:
                cursor = self.connection.cursor()
                cursor.execute(sql, (start_time, sample_rate_hz, num_optodes))
                self.connection.commit()
                session_id = cursor.lastrowid
                if session_id is None:
                    raise RuntimeError("Failed to get session ID after insert")
                logging.info(f"Created session {session_id}")
                return session_id
        except sqlite3.Error as e:
            logging.error(f"Error creating session: {e}")
            raise

    def end_session(self, session_id: int, end_time: str):
        """Marks a session as ended."""
        if not self.connection:
            raise RuntimeError("Database connection is not open")

        sql = "UPDATE sessions SET end_time = ? WHERE session_id = ?"

        try:
            with self.lock:
                cursor = self.connection.cursor()
                cursor.execute(sql, (end_time, session_id))
                self.connection.commit()
                logging.info(f"Ended session {session_id}")
        except sqlite3.Error as e:
            logging.error(f"Error ending session: {e}")
            raise

    def set_hemorrhage_result(self, session_id: int, detected: bool):
        """Sets the hemorrhage detection result for a session."""
        if not self.connection:
            raise RuntimeError("Database connection is not open")

        sql = "UPDATE sessions SET hemorrhage_detected = ? WHERE session_id = ?"

        try:
            with self.lock:
                cursor = self.connection.cursor()
                cursor.execute(sql, (1 if detected else 0, session_id))
                self.connection.commit()
                logging.info(f"Set hemorrhage_detected={detected} for session {session_id}")
        except sqlite3.Error as e:
            logging.error(f"Error setting hemorrhage result: {e}")
            raise

    def get_session(self, session_id: int) -> Optional[Dict[str, Any]]:
        """Retrieves session information."""
        if not self.connection:
            return None

        try:
            with self.lock:
                cursor = self.connection.cursor()
                cursor.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
                row = cursor.fetchone()
                return dict(row) if row else None
        except sqlite3.Error as e:
            logging.error(f"Error getting session: {e}")
            return None

    # =========================================================================
    # Raw Sample Operations
    # =========================================================================

    def insert_raw_sample(
        self,
        session_id: int,
        frame_number: int,
        timestamp_ms: int,
        sample: RawSample
    ) -> int:
        """
        Inserts a raw sample into the database.

        Args:
            session_id: The session this sample belongs to.
            frame_number: Frame number within the session.
            timestamp_ms: Milliseconds since epoch.
            sample: RawSample dataclass with intensity values for one optode.

        Returns:
            The sample_id of the inserted sample.
        """
        if not self.connection:
            raise RuntimeError("Database connection is not open")

        sql = """
            INSERT INTO raw_samples (
                session_id, optode_id, frame_number, timestamp_ms,
                nm740_long, nm860_long, nm740_short, nm860_short, dark
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        try:
            with self.lock:
                cursor = self.connection.cursor()
                cursor.execute(sql, (
                    session_id,
                    sample.optode_id,
                    frame_number,
                    timestamp_ms,
                    sample.nm740_long,
                    sample.nm860_long,
                    sample.nm740_short,
                    sample.nm860_short,
                    sample.dark
                ))
                self.connection.commit()
                sample_id = cursor.lastrowid
                if sample_id is None:
                    raise RuntimeError("Failed to get sample ID after insert")
                return sample_id
        except sqlite3.Error as e:
            logging.error(f"Error inserting raw sample: {e}")
            raise

    def insert_raw_samples_batch(
        self,
        session_id: int,
        samples: List[tuple]
    ) -> List[int]:
        """
        Batch insert raw samples for better performance.

        Args:
            session_id: The session these samples belong to.
            samples: List of tuples (frame_number, timestamp_ms, RawSample)

        Returns:
            List of sample_ids of the inserted samples.
        """
        if not self.connection:
            raise RuntimeError("Database connection is not open")

        sql = """
            INSERT INTO raw_samples (
                session_id, optode_id, frame_number, timestamp_ms,
                nm740_long, nm860_long, nm740_short, nm860_short, dark
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        sample_ids = []
        try:
            with self.lock:
                cursor = self.connection.cursor()
                for frame_number, timestamp_ms, sample in samples:
                    cursor.execute(sql, (
                        session_id,
                        sample.optode_id,
                        frame_number,
                        timestamp_ms,
                        sample.nm740_long,
                        sample.nm860_long,
                        sample.nm740_short,
                        sample.nm860_short,
                        sample.dark
                    ))
                    sample_ids.append(cursor.lastrowid)
                self.connection.commit()
                return sample_ids
        except sqlite3.Error as e:
            logging.error(f"Error batch inserting raw samples: {e}")
            raise

    def query_latest_raw_samples(
        self,
        session_id: int,
        limit: int = 100,
        optode_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Queries the most recent raw samples for a session.

        Args:
            session_id: The session to query.
            limit: Maximum number of samples to retrieve.
            optode_id: Filter by specific optode (optional).

        Returns:
            List of raw samples as dictionaries, ordered by timestamp descending.
        """
        if not self.connection:
            return []

        if optode_id is not None:
            sql = """
                SELECT * FROM raw_samples
                WHERE session_id = ? AND optode_id = ?
                ORDER BY timestamp_ms DESC
                LIMIT ?
            """
            params = (session_id, optode_id, limit)
        else:
            sql = """
                SELECT * FROM raw_samples
                WHERE session_id = ?
                ORDER BY timestamp_ms DESC
                LIMIT ?
            """
            params = (session_id, limit)

        try:
            with self.lock:
                cursor = self.connection.cursor()
                cursor.execute(sql, params)
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
        except sqlite3.Error as e:
            logging.error(f"Error querying raw samples: {e}")
            return []

    # =========================================================================
    # Preprocessed Sample Operations
    # =========================================================================

    def insert_preprocessed_sample(
        self,
        sample_id: int,
        session_id: int,
        frame_number: int,
        timestamp_ms: int,
        sample: PreprocessedSample
    ):
        """
        Inserts a preprocessed sample into the database.

        Args:
            sample_id: The corresponding raw sample ID.
            session_id: The session this sample belongs to.
            frame_number: Frame number within the session.
            timestamp_ms: Milliseconds since epoch.
            sample: PreprocessedSample dataclass with OD and Hb values.
        """
        if not self.connection:
            raise RuntimeError("Database connection is not open")

        sql = """
            INSERT INTO preprocessed_samples (
                sample_id, session_id, optode_id, frame_number, timestamp_ms,
                od_nm740_short, od_nm740_long, od_nm860_short, od_nm860_long,
                hbo_short, hbr_short, hbo_long, hbr_long
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        try:
            with self.lock:
                cursor = self.connection.cursor()
                cursor.execute(sql, (
                    sample_id,
                    session_id,
                    sample.optode_id,
                    frame_number,
                    timestamp_ms,
                    sample.od_nm740_short,
                    sample.od_nm740_long,
                    sample.od_nm860_short,
                    sample.od_nm860_long,
                    sample.hbo_short,
                    sample.hbr_short,
                    sample.hbo_long,
                    sample.hbr_long
                ))
                self.connection.commit()
        except sqlite3.Error as e:
            logging.error(f"Error inserting preprocessed sample: {e}")
            raise

    def query_latest_preprocessed_samples(
        self,
        session_id: int,
        limit: int = 100,
        optode_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Queries the most recent preprocessed samples for a session.

        Args:
            session_id: The session to query.
            limit: Maximum number of samples to retrieve.
            optode_id: Filter by specific optode (optional).

        Returns:
            List of preprocessed samples as dictionaries, ordered by timestamp descending.
        """
        if not self.connection:
            return []

        if optode_id is not None:
            sql = """
                SELECT * FROM preprocessed_samples
                WHERE session_id = ? AND optode_id = ?
                ORDER BY timestamp_ms DESC
                LIMIT ?
            """
            params = (session_id, optode_id, limit)
        else:
            sql = """
                SELECT * FROM preprocessed_samples
                WHERE session_id = ?
                ORDER BY timestamp_ms DESC
                LIMIT ?
            """
            params = (session_id, limit)

        try:
            with self.lock:
                cursor = self.connection.cursor()
                cursor.execute(sql, params)
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
        except sqlite3.Error as e:
            logging.error(f"Error querying preprocessed samples: {e}")
            return []

    # =========================================================================
    # Combined Queries
    # =========================================================================

    def query_samples_by_session(
        self,
        session_id: int,
        include_raw: bool = True,
        include_preprocessed: bool = True
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Queries all samples for a session."""
        result = {}

        if include_raw:
            if not self.connection:
                result['raw'] = []
            else:
                try:
                    with self.lock:
                        cursor = self.connection.cursor()
                        cursor.execute(
                            "SELECT * FROM raw_samples WHERE session_id = ? ORDER BY frame_number, optode_id",
                            (session_id,)
                        )
                        result['raw'] = [dict(row) for row in cursor.fetchall()]
                except sqlite3.Error as e:
                    logging.error(f"Error querying raw samples: {e}")
                    result['raw'] = []

        if include_preprocessed:
            if not self.connection:
                result['preprocessed'] = []
            else:
                try:
                    with self.lock:
                        cursor = self.connection.cursor()
                        cursor.execute(
                            "SELECT * FROM preprocessed_samples WHERE session_id = ? ORDER BY frame_number, optode_id",
                            (session_id,)
                        )
                        result['preprocessed'] = [dict(row) for row in cursor.fetchall()]
                except sqlite3.Error as e:
                    logging.error(f"Error querying preprocessed samples: {e}")
                    result['preprocessed'] = []

        return result

    def query_samples_by_time_range(
        self,
        session_id: int,
        start_ms: int,
        end_ms: int,
        table: str = 'preprocessed',
        optode_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Queries samples within a time range for a session."""
        if not self.connection:
            return []

        table_name = 'raw_samples' if table == 'raw' else 'preprocessed_samples'

        if optode_id is not None:
            sql = f"""
                SELECT * FROM {table_name}
                WHERE session_id = ? AND optode_id = ? AND timestamp_ms BETWEEN ? AND ?
                ORDER BY frame_number, optode_id
            """
            params = (session_id, optode_id, start_ms, end_ms)
        else:
            sql = f"""
                SELECT * FROM {table_name}
                WHERE session_id = ? AND timestamp_ms BETWEEN ? AND ?
                ORDER BY frame_number, optode_id
            """
            params = (session_id, start_ms, end_ms)

        try:
            with self.lock:
                cursor = self.connection.cursor()
                cursor.execute(sql, params)
                return [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            logging.error(f"Error querying samples by time range: {e}")
            return []

    def close(self):
        """Closes the database connection."""
        if self.connection:
            self.connection.close()
            self.connection = None
            logging.info("Database connection closed")


if __name__ == '__main__':
    import datetime

    logging.basicConfig(level=logging.INFO)
    print("Running database tests with per-optode schema...")

    APP_DIR = os.path.dirname(__file__) if os.path.dirname(__file__) else '.'
    DB_DIR = os.path.join(APP_DIR, "db")
    DB_FILE = os.path.join(DB_DIR, "test_fnirs.db")

    db = DatabaseManager(db_file=DB_FILE)

    try:
        db.connect()

        # --- Test 1: Create a session ---
        print("\n--- Test 1: Creating a session ---")
        start_time = datetime.datetime.now().isoformat()
        session_id = db.create_session(
            start_time=start_time,
            sample_rate_hz=5.0,
            num_optodes=2
        )
        print(f"Created session with ID: {session_id}")

        # --- Test 2: Insert raw samples (per optode) ---
        print("\n--- Test 2: Inserting raw samples ---")
        for frame_idx in range(5):
            timestamp_ms = frame_idx * 200  # 5Hz = 200ms intervals
            for optode in range(2):
                raw = RawSample(
                    optode_id=optode,
                    nm740_long=100.0 + frame_idx + optode * 10,
                    nm860_long=110.0 + frame_idx + optode * 10,
                    nm740_short=120.0 + frame_idx + optode * 10,
                    nm860_short=130.0 + frame_idx + optode * 10,
                    dark=0.1
                )
                sample_id = db.insert_raw_sample(
                    session_id=session_id,
                    frame_number=frame_idx,
                    timestamp_ms=timestamp_ms,
                    sample=raw
                )

                # Insert corresponding preprocessed sample
                preprocessed = PreprocessedSample(
                    optode_id=optode,
                    od_nm740_short=0.1 + frame_idx * 0.01,
                    od_nm740_long=0.2 + frame_idx * 0.01,
                    od_nm860_short=0.15 + frame_idx * 0.01,
                    od_nm860_long=0.25 + frame_idx * 0.01,
                    hbo_short=50.0 + frame_idx,
                    hbr_short=25.0 + frame_idx,
                    hbo_long=55.0 + frame_idx,
                    hbr_long=27.0 + frame_idx
                )
                db.insert_preprocessed_sample(
                    sample_id=sample_id,
                    session_id=session_id,
                    frame_number=frame_idx,
                    timestamp_ms=timestamp_ms,
                    sample=preprocessed
                )
        print("Inserted 5 frames x 2 optodes = 10 raw and 10 preprocessed samples")

        # --- Test 3: Query latest samples ---
        print("\n--- Test 3: Querying latest samples ---")
        latest_raw = db.query_latest_raw_samples(session_id, limit=5)
        latest_preprocessed = db.query_latest_preprocessed_samples(session_id, limit=5)
        print(f"Retrieved {len(latest_raw)} raw samples, {len(latest_preprocessed)} preprocessed samples")

        # --- Test 4: Query by optode ---
        print("\n--- Test 4: Querying by optode ---")
        optode0_samples = db.query_latest_raw_samples(session_id, limit=10, optode_id=0)
        optode1_samples = db.query_latest_raw_samples(session_id, limit=10, optode_id=1)
        print(f"Optode 0: {len(optode0_samples)} samples, Optode 1: {len(optode1_samples)} samples")

        # --- Test 5: Query by time range ---
        print("\n--- Test 5: Querying by time range ---")
        range_samples = db.query_samples_by_time_range(
            session_id=session_id,
            start_ms=200,
            end_ms=600,
            table='raw'
        )
        print(f"Found {len(range_samples)} samples in range 200-600ms")

        # --- Test 6: End session ---
        print("\n--- Test 6: Ending session ---")
        end_time = datetime.datetime.now().isoformat()
        db.end_session(session_id, end_time)
        db.set_hemorrhage_result(session_id, detected=False)

        session = db.get_session(session_id)
        print(f"Session: {session}")

        print("\nAll tests passed!")

    except Exception as e:
        print(f"Error during testing: {e}")
        raise
    finally:
        db.close()
        if os.path.exists(DB_FILE):
            os.remove(DB_FILE)
            print("Cleaned up test database file")
        if os.path.exists(DB_DIR) and not os.listdir(DB_DIR):
            os.rmdir(DB_DIR)
            print("Cleaned up test database directory")
