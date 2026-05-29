#!/usr/bin/env python3
"""
Convenience launcher for gesture-hand-sensor.

Examples:
  python run.py demo          # simulated pipeline (no webcam)
  python run.py hand          # hand sensor webcam demo
  python run.py inference     # webcam + gesture heuristics
  python run.py setup         # install deps + download models
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MAIN = ROOT / "src" / "main.py"


MODES = {
    "demo": ["--mode", "demo"],
    "inference": ["--mode", "inference"],
    "full": ["--mode", "full"],
    "collect": ["--mode", "collect", "--gesture", "forward", "--samples", "5"],
    "train": ["--mode", "train", "--epochs", "10"],
    "setup": None,
    "hand": None,
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)

    command = sys.argv[1].lower()
    extra = sys.argv[2:]

    if command == "setup":
        subprocess.check_call([sys.executable, str(ROOT / "scripts" / "setup.py")])
        return

    if command == "hand":
        subprocess.check_call([sys.executable, str(ROOT / "src" / "hand_tracker.py")])
        return

    if command not in MODES:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)

    args = [sys.executable, str(MAIN), *MODES[command], *extra]
    subprocess.check_call(args, cwd=ROOT)


if __name__ == "__main__":
    main()
