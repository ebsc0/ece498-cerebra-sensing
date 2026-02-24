-- fNIRS Data Storage Schema (per-optode)

CREATE TABLE sessions (
    session_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    start_time          TEXT NOT NULL,
    end_time            TEXT,
    sample_rate_hz      REAL NOT NULL DEFAULT 5.0,
    num_optodes         INTEGER NOT NULL DEFAULT 2,
    hemorrhage_detected INTEGER,
    created_at          TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE raw_samples (
    sample_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id     INTEGER NOT NULL,
    optode_id      INTEGER NOT NULL,
    frame_number   INTEGER NOT NULL,
    timestamp_ms   INTEGER NOT NULL,
    nm740_long     REAL NOT NULL,
    nm860_long     REAL NOT NULL,
    nm740_short    REAL NOT NULL,
    nm860_short    REAL NOT NULL,
    dark           REAL NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
);

CREATE TABLE preprocessed_samples (
    sample_id      INTEGER PRIMARY KEY,
    session_id     INTEGER NOT NULL,
    optode_id      INTEGER NOT NULL,
    frame_number   INTEGER NOT NULL,
    timestamp_ms   INTEGER NOT NULL,
    od_nm740_short REAL NOT NULL,
    od_nm740_long  REAL NOT NULL,
    od_nm860_short REAL NOT NULL,
    od_nm860_long  REAL NOT NULL,
    hbo_short      REAL NOT NULL,
    hbr_short      REAL NOT NULL,
    hbo_long       REAL NOT NULL,
    hbr_long       REAL NOT NULL,
    FOREIGN KEY (sample_id) REFERENCES raw_samples(sample_id) ON DELETE CASCADE,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
);

CREATE INDEX idx_raw_session_time ON raw_samples(session_id, timestamp_ms);
CREATE INDEX idx_raw_session_optode ON raw_samples(session_id, optode_id, frame_number);
CREATE INDEX idx_preprocessed_session_time ON preprocessed_samples(session_id, timestamp_ms);
CREATE INDEX idx_preprocessed_session_optode ON preprocessed_samples(session_id, optode_id, frame_number);
