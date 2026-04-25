#!/usr/bin/env python3
"""Plot HbO Long and HbR Long for Optode 0 from an exported CSV file."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay HbO Long and HbR Long for Optode 0 from an exported session CSV.",
    )
    parser.add_argument("csv_file", help="Path to the exported CSV file.")
    parser.add_argument(
        "--hbo-color",
        default="tab:red",
        help="Matplotlib color for the HbO Long line. Default: %(default)s",
    )
    parser.add_argument(
        "--hbr-color",
        default="tab:blue",
        help="Matplotlib color for the HbR Long line. Default: %(default)s",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional image path to save the figure instead of only showing it.",
    )
    return parser.parse_args()


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def plot_rows(
    rows: list[dict[str, str]],
    *,
    title: str,
    output_path: Path | None,
    hbo_color: str,
    hbr_color: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError("matplotlib is not installed. Install requirements and retry.") from exc

    if not rows:
        raise RuntimeError("CSV contains no data rows.")

    x_values: list[float] = []
    hbo_long_values: list[float] = []
    hbr_long_values: list[float] = []

    for row in rows:
        if int(row["optode_id"]) != 0:
            continue
        x_values.append(float(row["timestamp_ms"]) / 1000.0)
        hbo_long_values.append(float(row["hbo_long"]))
        hbr_long_values.append(float(row["hbr_long"]))

    if not x_values:
        raise RuntimeError("CSV contains no rows for Optode 0.")

    fig, axis = plt.subplots(figsize=(14, 6))
    axis.plot(x_values, hbo_long_values, color=hbo_color, linewidth=1.8, label="HbO Long")
    axis.plot(x_values, hbr_long_values, color=hbr_color, linewidth=1.8, label="HbR Long")
    axis.set_title(f"{title} - Optode 0")
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Value")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")
    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {output_path}")

    plt.show()


def main() -> int:
    args = parse_args()
    csv_path = Path(args.csv_file).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve() if args.output else None

    if not csv_path.exists():
        raise SystemExit(f"CSV file not found: {csv_path}")

    rows = load_rows(csv_path)
    plot_rows(
        rows,
        title=csv_path.name,
        output_path=output_path,
        hbo_color=args.hbo_color,
        hbr_color=args.hbr_color,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
