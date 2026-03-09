#!/usr/bin/env python3
"""Export preprocessed sample rows for a session to CSV."""

from __future__ import annotations

import argparse
import csv
import sqlite3
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = APP_DIR / "db" / "fnirs_data.db"

CSV_COLUMNS = [
    "sample_id",
    "session_id",
    "frame_number",
    "optode_id",
    "timestamp_ms",
    "od_nm740_short",
    "od_nm740_long",
    "od_nm860_short",
    "od_nm860_long",
    "hbo_short",
    "hbr_short",
    "hbo_long",
    "hbr_long",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export preprocessed_samples rows for a session to CSV.",
    )
    parser.add_argument("session_id", type=int, help="Session ID to export.")
    parser.add_argument(
        "--db",
        default=str(DEFAULT_DB_PATH),
        help=f"SQLite database path (default: {DEFAULT_DB_PATH}).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (default: preprocessed_session_<session_id>.csv).",
    )
    return parser.parse_args()


def default_output_path(session_id: int) -> Path:
    return APP_DIR / f"preprocessed_session_{session_id}.csv"


def fetch_rows(db_path: Path, session_id: int) -> list[sqlite3.Row]:
    sql = """
        SELECT
            p.sample_id,
            l.session_id,
            l.frame_number,
            l.optode_id,
            l.timestamp_ms,
            p.od_nm740_short,
            p.od_nm740_long,
            p.od_nm860_short,
            p.od_nm860_long,
            p.hbo_short,
            p.hbr_short,
            p.hbo_long,
            p.hbr_long
        FROM preprocessed_samples AS p
        INNER JOIN logical_samples AS l
            ON l.sample_id = p.sample_id
        WHERE l.session_id = ?
        ORDER BY l.timestamp_ms ASC, l.frame_number ASC, l.optode_id ASC, p.sample_id ASC
    """
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        cursor = connection.execute(sql, (session_id,))
        return cursor.fetchall()
    finally:
        connection.close()


def write_csv(rows: list[sqlite3.Row], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row[column] for column in CSV_COLUMNS})


def main() -> int:
    args = parse_args()
    db_path = Path(args.db).expanduser().resolve()
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else default_output_path(args.session_id)
    )

    if not db_path.exists():
        raise SystemExit(f"Database file not found: {db_path}")

    rows = fetch_rows(db_path, args.session_id)
    write_csv(rows, output_path)
    print(f"Wrote {len(rows)} row(s) to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
