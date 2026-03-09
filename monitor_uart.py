#!/usr/bin/env python3
"""Live UART monitor that decodes phase packets and plots diode values in real time."""

from __future__ import annotations

import argparse
from collections import deque
import glob
import os
import shutil
import struct
import sys
import time
from typing import Optional

try:
    import serial  # type: ignore
except ModuleNotFoundError:
    serial = None

from config import ACTIVE_OPTODES, OPTODE_COLORS, UART_BAUDRATE, UART_SYNC_WORD
from packets import PACKET_SIZE, PHASE_740, PHASE_860, PHASE_DARK, decode_packet
from uart import _PacketFramer


CHANNEL_SPECS = [
    ("d0", "D0"),
    ("d1", "D1"),
    ("d2", "D2"),
    ("d3", "D3"),
]
PHASE_LABELS = {
    PHASE_740: "740",
    PHASE_860: "860",
    PHASE_DARK: "dark",
}
PHASE_LINESTYLES = {
    PHASE_740: "-",
    PHASE_860: "--",
    PHASE_DARK: ":",
}
PHASE_ORDER = (PHASE_740, PHASE_860, PHASE_DARK)
TABLE_HEADER = " time       frame     opt  phase       meta        d0        d1        d2        d3  | raw"
TABLE_RULE = " -----------------------------------------------------------------------------------------------"
D2_Y_MIN = 0.0
D2_Y_MAX = 2.0
D2_Y_PAD = 0.01


class RealTimePlotter:
    def __init__(
        self,
        *,
        optodes: list[int],
        max_points: int,
        y_pad_ratio: float,
        min_span: float,
        x_tick_seconds: float,
    ):
        try:
            import matplotlib.pyplot as plt
            from matplotlib.ticker import MultipleLocator
        except ModuleNotFoundError as exc:
            raise RuntimeError("matplotlib is not installed. Install requirements and retry.") from exc

        self._plt = plt
        self._MultipleLocator = MultipleLocator
        self._plt.ion()
        self._optodes = list(optodes)
        self._max_points = int(max_points)
        self._y_pad_ratio = max(0.0, float(y_pad_ratio))
        self._y_bottom_pad_ratio = min(self._y_pad_ratio * 0.25, self._y_pad_ratio)
        self._min_span = max(1e-6, float(min_span))
        self._x_tick_seconds = max(0.1, float(x_tick_seconds))
        self._series: dict[tuple[int, int], dict[str, deque[float]]] = {}
        self._lines: dict[str, dict[tuple[int, int], object]] = {key: {} for key, _label in CHANNEL_SPECS}

        self.fig, axes = self._plt.subplots(len(CHANNEL_SPECS), 1, sharex=True, figsize=(12, 9))
        self.axes = list(axes)
        self.fig.canvas.manager.set_window_title("Cerebra UART Monitor")

        for axis, (channel_key, label) in zip(self.axes, CHANNEL_SPECS):
            axis.set_ylabel(label)
            axis.grid(True, alpha=0.3)
            axis.xaxis.set_major_locator(self._MultipleLocator(self._x_tick_seconds))
            axis.margins(x=0.0)
            for optode in self._optodes:
                color = OPTODE_COLORS[optode % len(OPTODE_COLORS)]
                for phase in (PHASE_740, PHASE_860, PHASE_DARK):
                    key = (optode, phase)
                    (line,) = axis.plot(
                        [],
                        [],
                        color=color,
                        linestyle=PHASE_LINESTYLES[phase],
                        linewidth=1.4,
                        label=f"Opt {optode} {PHASE_LABELS[phase]}",
                    )
                    self._lines[channel_key][key] = line
            if self._optodes:
                axis.legend(loc="upper right", fontsize="x-small", ncol=max(1, min(6, len(self._optodes) * 3)))

        self.axes[-1].set_xlabel("Time (s)")
        self.fig.tight_layout()
        self.fig.show()

    def _new_series(self) -> dict[str, list[float] | deque[float]]:
        if self._max_points > 0:
            return {
                "time_s": deque(maxlen=self._max_points),
                "d0": deque(maxlen=self._max_points),
                "d1": deque(maxlen=self._max_points),
                "d2": deque(maxlen=self._max_points),
                "d3": deque(maxlen=self._max_points),
            }
        return {
            "time_s": [],
            "d0": [],
            "d1": [],
            "d2": [],
            "d3": [],
        }

    def add_packet(self, *, time_s: float, optode: int, phase: int, d0: float, d1: float, d2: float, d3: float) -> None:
        key = (optode, phase)
        series = self._series.setdefault(key, self._new_series())
        series["time_s"].append(float(time_s))
        series["d0"].append(d0)
        series["d1"].append(d1)
        series["d2"].append(d2)
        series["d3"].append(d3)

    def refresh(self) -> None:
        for axis, (channel_key, _label) in zip(self.axes, CHANNEL_SPECS):
            has_data = False
            channel_frames: list[float] = []
            channel_values: list[float] = []
            for optode in self._optodes:
                for phase in (PHASE_740, PHASE_860, PHASE_DARK):
                    key = (optode, phase)
                    line = self._lines[channel_key][key]
                    series = self._series.get(key)
                    if not series or not series["time_s"]:
                        line.set_data([], [])
                        continue
                    line.set_data(series["time_s"], series[channel_key])
                    channel_frames.extend(series["time_s"])
                    channel_values.extend(series[channel_key])
                    has_data = True
            if has_data:
                x_max = max(max(channel_frames), 1.0)
                axis.set_xlim(0.0, x_max)

                if channel_key == "d2":
                    axis.set_ylim(D2_Y_MIN - D2_Y_PAD, D2_Y_MAX + D2_Y_PAD)
                    continue

                y_min = min(channel_values)
                y_max = max(channel_values)
                span = max(y_max - y_min, 0.0)
                upper_pad = max(span, self._min_span) * self._y_pad_ratio
                lower_pad = max(span, self._min_span) * self._y_bottom_pad_ratio
                y_low = y_min - lower_pad
                y_high = y_max + upper_pad

                if (y_high - y_low) < self._min_span:
                    deficit = self._min_span - (y_high - y_low)
                    y_low -= deficit * 0.2
                    y_high += deficit * 0.8

                axis.set_ylim(y_low, y_high)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        self._plt.pause(0.001)

    def is_closed(self) -> bool:
        return not self._plt.fignum_exists(self.fig.number)


def detect_default_port() -> Optional[str]:
    candidates: list[str] = []
    candidates.extend(sorted(glob.glob("/dev/cu.usbmodem*")))
    candidates.extend(sorted(glob.glob("/dev/cu.usbserial*")))
    return candidates[0] if candidates else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor and decode UART packets.")
    parser.add_argument("--port", default=detect_default_port(), help="Serial port path (default: auto-detect).")
    parser.add_argument("--baud", type=int, default=UART_BAUDRATE, help=f"Baud rate (default: {UART_BAUDRATE}).")
    parser.add_argument(
        "--optode",
        type=int,
        default=None,
        help="Decode, print, and plot only this optode ID (default: use ACTIVE_OPTODES).",
    )
    parser.add_argument("--timeout", type=float, default=0.05, help="Serial read timeout seconds.")
    parser.add_argument("--chunk-size", type=int, default=4096, help="Maximum number of bytes to read per serial read.")
    parser.add_argument(
        "--max-points",
        type=int,
        default=0,
        help="Maximum decoded points to retain per series; 0 keeps full history (default: 0).",
    )
    parser.add_argument("--plot-interval", type=float, default=0.1, help="Minimum interval in seconds between plot refreshes.")
    parser.add_argument(
        "--plot-y-pad-ratio",
        type=float,
        default=0.20,
        help="Fractional vertical padding added above and below plotted data (default: 0.20).",
    )
    parser.add_argument(
        "--plot-min-span",
        type=float,
        default=10.0,
        help="Minimum y-axis span per subplot for readability (default: 1.0).",
    )
    parser.add_argument(
        "--plot-x-tick-seconds",
        type=float,
        default=10.0,
        help="Major x-axis tick spacing in seconds (default: 10.0).",
    )
    parser.add_argument(
        "--raw-hex",
        action="store_true",
        help="Bypass framing/decoding and print raw serial bytes as hex chunks.",
    )
    parser.add_argument("--binary", action="store_true", help="Render the raw packet dump as binary instead of hex.")
    parser.add_argument("--ascii", action="store_true", help="Also show a printable ASCII preview alongside the raw packet dump.")
    parser.add_argument("--show-rate", action="store_true", help="Print running byte rate once per second.")
    parser.add_argument(
        "--sticky-header",
        action="store_true",
        help="Redraw the terminal so the table header stays visible at the top.",
    )
    return parser.parse_args()


def _ascii_preview(data: bytes) -> str:
    return "".join(chr(b) if 32 <= b <= 126 else "." for b in data)


def _format_bytes(data: bytes, *, binary: bool) -> str:
    if binary:
        return " ".join(f"{byte:08b}" for byte in data)
    return data.hex()


def _format_float_fields(values: tuple[float, float, float, float]) -> str:
    return " ".join(f"{value:9.3f}" for value in values)


def _is_valid_payload(packet: bytes, active_optode_set: set[int]) -> bool:
    if len(packet) != PACKET_SIZE:
        return False
    try:
        decoded = decode_packet(packet, timestamp_ms=0)
    except ValueError:
        return False
    if active_optode_set and decoded.optode_id not in active_optode_set:
        return False
    if any(abs(value) > 1e6 for value in (decoded.d0, decoded.d1, decoded.d2, decoded.d3)):
        return False
    return True


def _redraw_terminal(
    *,
    port: str,
    baud: int,
    sync_word: bytes,
    packet_count: int,
    byte_count: int,
    dropped: int,
    active_optodes: list[int],
    row_order: list[tuple[int, int]],
    row_values: dict[tuple[int, int], str],
    recent_warnings: deque[str],
    current_bps: Optional[float],
) -> None:
    terminal_size = shutil.get_terminal_size(fallback=(160, 24))
    active_text = ",".join(str(optode) for optode in active_optodes) if active_optodes else "all"
    status = (
        f"{port} @ {baud} | packets={packet_count} bytes={byte_count} "
        f"dropped={dropped} | sync={sync_word.hex()} | active={active_text}"
    )
    if current_bps is not None:
        status += f" | rate={current_bps:0.1f} B/s"

    warning_lines = list(recent_warnings)[-2:]
    header_lines = [status]
    if warning_lines:
        header_lines.extend(warning_lines)
    else:
        header_lines.append("Press Ctrl+C to stop.")
    header_lines.append(TABLE_HEADER)
    header_lines.append(TABLE_RULE)

    available_rows = max(1, terminal_size.lines - len(header_lines))
    rows = [row_values[key] for key in row_order[:available_rows] if key in row_values]
    frame = "\x1b[H\x1b[2J" + "\n".join(header_lines + rows) + "\x1b[J"
    sys.stdout.write(frame)
    sys.stdout.flush()


def _placeholder_row(optode_id: int, phase: int) -> str:
    phase_label = PHASE_LABELS[phase]
    return (
        f" {'--:--:--':>8s}  {'-':>8s}  {optode_id:6d}  {phase_label:>5s}  "
        f"{'--------':>10s}  {'--':>9s} {'--':>9s} {'--':>9s} {'--':>9s}  | --"
    )


def main() -> int:
    args = parse_args()
    if serial is None:
        print("pyserial is not installed. Activate .venv and run: pip install pyserial", file=sys.stderr)
        return 2
    if not args.port:
        print("No serial port detected. Pass --port /dev/cu.usbmodemXXXX", file=sys.stderr)
        return 2
    if not os.path.exists(args.port):
        print(f"Serial port does not exist: {args.port}", file=sys.stderr)
        return 2

    chunk_size = max(1, int(args.chunk_size))
    packet_size = PACKET_SIZE
    sync_word = bytes(UART_SYNC_WORD)
    wire_size = len(sync_word) + packet_size
    raw_hex_mode = bool(args.raw_hex)
    active_optodes = [int(args.optode)] if args.optode is not None else sorted(int(x) for x in ACTIVE_OPTODES)
    active_optode_set = set(active_optodes)
    framer = _PacketFramer(
        payload_size=packet_size,
        sync_word=sync_word,
        payload_validator=lambda payload: _is_valid_payload(payload, active_optode_set),
    )
    raw_label = "binary" if args.binary else "hex"
    plotter = None if raw_hex_mode else RealTimePlotter(
        optodes=active_optodes,
        max_points=args.max_points,
        y_pad_ratio=args.plot_y_pad_ratio,
        min_span=args.plot_min_span,
        x_tick_seconds=args.plot_x_tick_seconds,
    )
    next_plot_refresh = time.monotonic()
    sticky_header = bool(args.sticky_header and sys.stdout.isatty())
    recent_warnings: deque[str] = deque(maxlen=10)
    current_bps: Optional[float] = None
    if active_optodes:
        row_order = [(optode_id, phase) for optode_id in active_optodes for phase in PHASE_ORDER]
    else:
        row_order = []
    row_values: dict[tuple[int, int], str] = {
        key: _placeholder_row(*key) for key in row_order
    }

    if raw_hex_mode:
        print(
            f"Opening {args.port} @ {args.baud} baud "
            f"(chunk_size={chunk_size}, mode=raw-hex, raw={raw_label})"
        )
        if args.optode is not None:
            print(f"Optode filter {args.optode} is ignored in raw-hex mode.")
        print("Press Ctrl+C to stop.\n")
        print(" time         bytes  | raw")
        print(" ----------------------------")
    elif sticky_header:
        _redraw_terminal(
            port=args.port,
            baud=args.baud,
            sync_word=sync_word,
            packet_count=0,
            byte_count=0,
            dropped=0,
            active_optodes=active_optodes,
            row_order=row_order,
            row_values=row_values,
            recent_warnings=recent_warnings,
            current_bps=current_bps,
        )
    else:
        print(
            f"Opening {args.port} @ {args.baud} baud "
            f"(payload_size={packet_size}, wire_size={wire_size}, chunk_size={chunk_size}, raw={raw_label})"
        )
        if active_optodes:
            print(f"Active optodes filter: {active_optodes}")
        print(f"Sync word: {sync_word.hex()}")
        print("Press Ctrl+C to stop.\n")
        print(TABLE_HEADER)
        print(TABLE_RULE)

    packet_count = 0
    byte_count = 0
    dropped = 0
    rate_window_start = time.time()
    rate_bytes_start = 0
    first_packet_time_s: Optional[float] = None

    try:
        with serial.Serial(port=args.port, baudrate=args.baud, timeout=args.timeout) as ser:
            while True:
                now_mono = time.monotonic()
                redrew_terminal = False
                if args.show_rate and time.time() - rate_window_start >= 1.0:
                    elapsed = max(1e-6, time.time() - rate_window_start)
                    current_bps = (byte_count - rate_bytes_start) / elapsed
                    if sticky_header:
                        _redraw_terminal(
                            port=args.port,
                            baud=args.baud,
                            sync_word=sync_word,
                            packet_count=packet_count,
                            byte_count=byte_count,
                            dropped=dropped,
                            active_optodes=active_optodes,
                            row_order=row_order,
                            row_values=row_values,
                            recent_warnings=recent_warnings,
                            current_bps=current_bps,
                        )
                        redrew_terminal = True
                    else:
                        print(f"[rate] {current_bps:9.1f} B/s  packets={packet_count}  dropped={dropped}")
                    rate_window_start = time.time()
                    rate_bytes_start = byte_count

                if plotter is not None and not plotter.is_closed() and now_mono >= next_plot_refresh:
                    plotter.refresh()
                    next_plot_refresh = now_mono + max(0.01, float(args.plot_interval))

                chunk = ser.read(chunk_size)
                if not chunk:
                    continue

                byte_count += len(chunk)
                if raw_hex_mode:
                    now = time.strftime("%H:%M:%S")
                    row = f" {now}  {len(chunk):8d}  | {_format_bytes(chunk, binary=args.binary)}"
                    if args.ascii:
                        row += f"  | {_ascii_preview(chunk)}"
                    print(row)
                    continue

                packets, dropped_now = framer.feed(chunk)
                if dropped_now > 0:
                    dropped += dropped_now
                    warning_text = (
                        f"[warn] dropped {dropped_now} byte(s) while re-aligning stream "
                        f"(total={dropped})"
                    )
                    if sticky_header:
                        recent_warnings.append(warning_text)
                    else:
                        print(warning_text)

                for packet in packets:
                    decoded = decode_packet(packet, timestamp_ms=0)
                    metadata = struct.unpack_from("<I", packet, 0)[0]
                    packet_count += 1
                    now = time.strftime("%H:%M:%S")
                    row = (
                        f" {now}  {decoded.frame_number:8d}  {decoded.optode_id:6d}  {PHASE_LABELS[decoded.phase]:>5s}  "
                        f"0x{metadata:08X}  "
                        f"{_format_float_fields((decoded.d0, decoded.d1, decoded.d2, decoded.d3))}"
                        f"  | {_format_bytes(packet, binary=args.binary)}"
                    )
                    if args.ascii:
                        row += f"  | {_ascii_preview(packet)}"
                    if sticky_header:
                        key = (decoded.optode_id, decoded.phase)
                        if key not in row_values:
                            row_order.append(key)
                            row_values[key] = _placeholder_row(*key)
                        row_values[key] = row
                    else:
                        print(row)

                    if plotter is not None and not plotter.is_closed():
                        packet_time_s = time.monotonic()
                        if first_packet_time_s is None:
                            first_packet_time_s = packet_time_s
                        elapsed_s = max(0.0, packet_time_s - first_packet_time_s)
                        plotter.add_packet(
                            time_s=elapsed_s,
                            optode=decoded.optode_id,
                            phase=decoded.phase,
                            d0=decoded.d0,
                            d1=decoded.d1,
                            d2=decoded.d2,
                            d3=decoded.d3,
                        )

                if sticky_header and (packets or dropped_now > 0) and not redrew_terminal:
                    _redraw_terminal(
                        port=args.port,
                        baud=args.baud,
                        sync_word=sync_word,
                        packet_count=packet_count,
                        byte_count=byte_count,
                        dropped=dropped,
                        active_optodes=active_optodes,
                        row_order=row_order,
                        row_values=row_values,
                        recent_warnings=recent_warnings,
                        current_bps=current_bps,
                    )
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as exc:
        print(f"\nUART monitor error: {exc}", file=sys.stderr)
        return 1

    if raw_hex_mode:
        print(f"Summary: bytes={byte_count}")
    else:
        print(f"Summary: packets={packet_count}, bytes={byte_count}, dropped={dropped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
