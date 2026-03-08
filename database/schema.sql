CREATE TABLE IF NOT EXISTS sessions (
    session_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    elapsed_ms          INTEGER,
    sample_rate_hz      REAL NOT NULL,
    num_optodes         INTEGER NOT NULL,
    hemorrhage_detected INTEGER,
    created_at          TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS raw_packets (
    packet_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    INTEGER NOT NULL,
    frame_number  INTEGER NOT NULL,
    optode_id     INTEGER NOT NULL,
    phase         INTEGER NOT NULL CHECK (phase IN (0, 1, 2)),
    timestamp_ms  INTEGER NOT NULL,
    d0            REAL NOT NULL,
    d1            REAL NOT NULL,
    d2            REAL NOT NULL,
    d3            REAL NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE,
    UNIQUE (session_id, frame_number, optode_id, phase)
);

CREATE TABLE IF NOT EXISTS logical_samples (
    sample_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id     INTEGER NOT NULL,
    frame_number   INTEGER NOT NULL,
    optode_id      INTEGER NOT NULL,
    timestamp_ms   INTEGER NOT NULL,
    nm740_long     REAL NOT NULL,
    nm860_long     REAL NOT NULL,
    nm740_short    REAL NOT NULL,
    nm860_short    REAL NOT NULL,
    dark           REAL NOT NULL,
    packet_740_id  INTEGER NOT NULL,
    packet_860_id  INTEGER NOT NULL,
    packet_dark_id INTEGER NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE,
    FOREIGN KEY (packet_740_id) REFERENCES raw_packets(packet_id),
    FOREIGN KEY (packet_860_id) REFERENCES raw_packets(packet_id),
    FOREIGN KEY (packet_dark_id) REFERENCES raw_packets(packet_id),
    UNIQUE (session_id, frame_number, optode_id)
);

CREATE TABLE IF NOT EXISTS preprocessed_samples (
    sample_id      INTEGER PRIMARY KEY,
    od_nm740_short REAL NOT NULL,
    od_nm740_long  REAL NOT NULL,
    od_nm860_short REAL NOT NULL,
    od_nm860_long  REAL NOT NULL,
    hbo_short      REAL NOT NULL,
    hbr_short      REAL NOT NULL,
    hbo_long       REAL NOT NULL,
    hbr_long       REAL NOT NULL,
    FOREIGN KEY (sample_id) REFERENCES logical_samples(sample_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_raw_packets_session_time
    ON raw_packets(session_id, timestamp_ms);

CREATE INDEX IF NOT EXISTS idx_raw_packets_session_optode
    ON raw_packets(session_id, optode_id, frame_number, phase);

CREATE INDEX IF NOT EXISTS idx_logical_samples_session_time
    ON logical_samples(session_id, timestamp_ms);

CREATE INDEX IF NOT EXISTS idx_logical_samples_session_optode
    ON logical_samples(session_id, optode_id, frame_number);
