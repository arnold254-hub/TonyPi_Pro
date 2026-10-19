#!/usr/bin/env python3
"""
Convert legacy Hiwonder *.rob action-group files (ACT-40 format) to *.d6a.

The .rob container stores little-endian 16-bit words where 0x0000 and 0x5555
act as padding/sentinels around the real time/servo values. Once those
sentinels are removed, the remaining numbers are arranged as:

    [frame_count_header, ?, time_1, servo1..servo16, time_2, ...]

This script turns every frame into a SQLite-based .d6a ActionGroup table that
TonyPi uses at runtime. Servos 17/18 are padded with a neutral value (default
500) because legacy .rob files only contain 16 servo channels.
"""

from __future__ import annotations

import argparse
import sqlite3
import struct
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

PAD_VALUES = (0, 0x5555)


def parse_rob(path: Path) -> Tuple[Sequence[int], List[Tuple[int, List[int]]]]:
    """Return (header_words, [(time, [servo pulses...]), ...])."""
    raw = path.read_bytes()
    if len(raw) % 2:
        raise ValueError(f"{path} has odd length ({len(raw)} bytes); cannot unpack 16-bit words.")

    word_count = len(raw) // 2
    words = struct.unpack("<" + "H" * word_count, raw)
    filtered = [val for val in words if val not in PAD_VALUES]
    if len(filtered) < 6:
        raise ValueError(f"{path} doesn't look like ACT-40; filtered payload too small.")

    header = filtered[:5]
    payload = filtered[5:]
    if len(payload) % 17 != 0:
        raise ValueError(
            f"{path} payload length {len(payload)} is not divisible by 17 "
            "(expected time + 16 servo channels per frame)."
        )

    frames: List[Tuple[int, List[int]]] = []
    for idx in range(0, len(payload), 17):
        chunk = payload[idx : idx + 17]
        time_ms = chunk[0]
        servo_values = list(chunk[1:])
        frames.append((time_ms, servo_values))

    frame_count_header = header[3]
    if frame_count_header not in (0, len(frames)):
        # Not fatal, but worth surfacing to the caller.
        print(
            f"Warning: header says {frame_count_header} frames, "
            f"but parsed {len(frames)} frames."
        )

    return header, frames


def write_d6a(frames: Iterable[Tuple[int, Sequence[int]]], dest: Path, pad_value: int = 500) -> None:
    """Persist frames into a SQLite ActionGroup table."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(dest)
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS ActionGroup;")
    cur.execute(
        """
        CREATE TABLE ActionGroup(
            [Index] INTEGER PRIMARY KEY AUTOINCREMENT,
            Time INT,
            Servo1 INT,
            Servo2 INT,
            Servo3 INT,
            Servo4 INT,
            Servo5 INT,
            Servo6 INT,
            Servo7 INT,
            Servo8 INT,
            Servo9 INT,
            Servo10 INT,
            Servo11 INT,
            Servo12 INT,
            Servo13 INT,
            Servo14 INT,
            Servo15 INT,
            Servo16 INT,
            Servo17 INT,
            Servo18 INT
        );
        """
    )

    insert_sql = (
        "INSERT INTO ActionGroup("
        "Time, Servo1, Servo2, Servo3, Servo4, Servo5, Servo6, Servo7, Servo8, "
        "Servo9, Servo10, Servo11, Servo12, Servo13, Servo14, Servo15, Servo16, Servo17, Servo18"
        ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
    )

    for idx, (time_ms, servo_values) in enumerate(frames, start=1):
        if len(servo_values) > 18:
            raise ValueError(f"Frame {idx} has {len(servo_values)} servos; expected <= 18.")
        padded = list(servo_values)[:18]
        while len(padded) < 18:
            padded.append(pad_value)
        cur.execute(insert_sql, (time_ms, *padded))

    conn.commit()
    conn.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert ACT-40 .rob files to TonyPi .d6a format.")
    parser.add_argument("source", type=Path, help=".rob file to convert")
    parser.add_argument(
        "destination",
        type=Path,
        nargs="?",
        help="Target .d6a path (defaults to replacing .rob with .d6a in-place)",
    )
    parser.add_argument(
        "--pad",
        type=int,
        default=500,
        help="Pulse value used to fill missing servo17/servo18 entries (default: 500).",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    source: Path = args.source
    dest: Path = args.destination or source.with_suffix(".d6a")
    header, frames = parse_rob(source)
    print(
        f"Parsed {len(frames)} frames from {source.name} "
        f"(header signature: {header[0]:#06x} {header[1]:#06x} {header[2]:#06x})."
    )
    write_d6a(frames, dest, pad_value=args.pad)
    print(f"Wrote {len(frames)} frames → {dest}")


if __name__ == "__main__":
    main()
