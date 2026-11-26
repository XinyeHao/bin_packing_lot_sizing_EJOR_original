"""After the current 40101 run exits, solve the other large seeds, then all medium seeds.

Full stdout of each run is saved under result/bp_logs/.
"""
from __future__ import annotations

import ctypes
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "result" / "bp_logs"
RUNNER = Path(__file__).resolve().parent / "run_rf_small.py"
LARGE = (
    40201, 40301, 40401, 40501,
    41101, 41201, 41301, 41401, 41501,
    41601, 41701, 41801, 41901, 42001,
)
MEDIUM = (
    30101, 30201, 30301, 30401, 30501,
    31101, 31201, 31301, 31401, 31501,
    31601, 31701, 31801, 31901, 32001,
)


def _alive(pid: int) -> bool:
    # os.kill(pid, 0) is not a valid existence check on Windows (WinError 87).
    handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
    if not handle:
        return False
    ctypes.windll.kernel32.CloseHandle(handle)
    return True


def _wait(pid: int) -> None:
    while _alive(pid):
        time.sleep(20)


def _run(scale: str, seed: int) -> None:
    log_path = LOG_DIR / f"{scale}_{seed}.log"
    print(f"START {scale} {seed}", flush=True)
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            [sys.executable, "-u", str(RUNNER), "--scale", scale, "--seeds", str(seed)],
            cwd=str(RUNNER.parent),
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    print(f"END {scale} {seed} exit={completed.returncode}", flush=True)


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) > 1:
        pid = int(sys.argv[1])
        print(f"waiting for pid {pid}", flush=True)
        _wait(pid)
        source = sys.argv[2] if len(sys.argv) > 2 else ""
        if source:
            src = Path(source)
            if src.exists():
                shutil.copyfile(src, LOG_DIR / "large_40101.log")
                print("copied 40101 log", flush=True)
    for seed in LARGE:
        _run("large", seed)
    for seed in MEDIUM:
        _run("medium", seed)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
