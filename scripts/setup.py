#!/usr/bin/env python3
"""Install dependencies and download required models for gesture-hand-sensor."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(cmd: list[str]) -> None:
    print(">", " ".join(cmd))
    subprocess.check_call(cmd, cwd=ROOT)


def main() -> None:
    print("Installing Python dependencies...")
    run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

    print("\nDownloading SpaCy English model (optional for NLP)...")
    try:
        run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    except subprocess.CalledProcessError:
        print("SpaCy model download skipped; rule-based NLP still works.")

    print("\nDownloading MediaPipe hand landmarker model...")
    sys.path.insert(0, str(ROOT / "src"))
    from hand_tracker import ensure_hand_model

    ensure_hand_model()

    print("\nSetup complete. Try:")
    print(f"  cd {ROOT}")
    print("  python run.py demo")
    print("  python run.py hand")


if __name__ == "__main__":
    main()
