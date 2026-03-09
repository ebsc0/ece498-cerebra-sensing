#!/usr/bin/env python3
"""Plot all preprocessed sample columns from an exported CSV file."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from config import OPTODE_COLORS


PLOT_COLUMNS = [
    ("od_nm740_short", "OD 740 Short"),
    ("od_nm740_long", "OD 740 Long"),
    ("od_nm860_short", "OD 860 Short"),
    ("od_nm860_long", "OD 860 Long"),
    ("hbo_short", "HbO Short"),
    ("hbr_short", "HbR Short"),
    ("hbo_long", "HbO Long"),
    ("hbr_long", "HbR Long"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot all preprocessed values from an exported session CSV.",
    )
    parser.add_argument("csv_file", help="Path to the exported CSV file.")
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


def plot_rows(rows: list[dict[str, str]], *, title: str, output_path: Path | None) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError("matplotlib is not installed. Install requirements and retry.") from exc

    if not rows:
        raise RuntimeError("CSV contains no data rows.")

    grouped: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    x_values: dict[int, list[float]] = defaultdict(list)

    for row in rows:
        optode_id = int(row["optode_id"])
        time_s = float(row["timestamp_ms"]) / 1000.0
        x_values[optode_id].append(time_s)
        for column, _label in PLOT_COLUMNS:
            grouped[optode_id][column].append(float(row[column]))

    fig, axes = plt.subplots(4, 2, sharex=True, figsize=(14, 12))
    fig.suptitle(title)
    flat_axes = axes.flatten()

    optodes = sorted(x_values.keys())
    for index, (column, label) in enumerate(PLOT_COLUMNS):
        axis = flat_axes[index]
        for optode_id in optodes:
            color = OPTODE_COLORS[optode_id % len(OPTODE_COLORS)]
            axis.plot(
                x_values[optode_id],
                grouped[optode_id][column],
                color=color,
                linewidth=1.5,
                label=f"Optode {optode_id}",
            )
        axis.set_title(label)
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.3)
        if optodes:
            axis.legend(loc="best", fontsize="small")

    for axis in flat_axes[-2:]:
        axis.set_xlabel("Time (s)")

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
    plot_rows(rows, title=csv_path.name, output_path=output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
