"""Iterative ML: 1h collect → coarse train → small/med smoke → long collect + periodic retrain.

Phases (default):
  1) Collect ``--first-hours`` into ``--out-dir`` (fresh).
  2) Train coarse Top-5 model → ``ml_branch_model.txt``.
  3) Smoke BPC+ML on a few small + medium seeds (exclusive Gurobi).
  4) Spawn long collect (``--long-hours``, --append --skip-train).
  5) Every ``--retrain-every-hours``, retrain from growing CSV (no Gurobi).

Further BPC rounds: run ``run_bpc_ml_smoke_iter.py --round N`` when ready
(preferably when collect is idle, or accept single-license contention).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

OUT_DIR = "ml_train_data_v7_w1216"
LOG_DIR = ROOT / "supplement_materials"


def _run(cmd: list[str], log_path: Path | None = None) -> int:
    print(f"[exec] {' '.join(cmd)}", flush=True)
    if log_path is None:
        return subprocess.call(cmd)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(f"\n===== {' '.join(cmd)} @ {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
        fh.flush()
        p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)
        return p.wait()


def _spawn(cmd: list[str], log_path: Path) -> subprocess.Popen:
    print(f"[spawn] {' '.join(cmd)} -> {log_path}", flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = log_path.open("a", encoding="utf-8")
    fh.write(f"\n===== SPAWN {' '.join(cmd)} @ {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
    fh.flush()
    return subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str, default=OUT_DIR)
    ap.add_argument("--first-hours", type=float, default=1.0)
    ap.add_argument("--long-hours", type=float, default=48.0)
    ap.add_argument("--retrain-every-hours", type=float, default=8.0)
    ap.add_argument("--seed0", type=int, default=200001)
    ap.add_argument("--skip-smoke", action="store_true")
    ap.add_argument("--min-groups", type=int, default=8)
    args = ap.parse_args()

    py = sys.executable
    out_dir = args.out_dir
    collect_py = str(ROOT / "experiments" / "ml_full_labelA_8h.py")
    retrain_py = str(ROOT / "experiments" / "ml_retrain_top5_coarse.py")
    smoke_py = str(ROOT / "experiments" / "run_bpc_ml_smoke_iter.py")
    model = str(ROOT / "ml_branch_model.txt")

    # --- Phase 1: first-hour collect ---
    rc = _run(
        [
            py, "-u", collect_py,
            "--hours", str(args.first_hours),
            "--out-dir", out_dir,
            "--seed0", str(args.seed0),
            "--skip-train",
            "--no-train-reserve",
        ],
        LOG_DIR / "ml_iter_collect_first.log",
    )
    if rc != 0:
        print(f"[fatal] first collect rc={rc}", flush=True)
        sys.exit(rc)

    # --- Phase 2: train coarse ---
    rc = _run(
        [
            py, "-u", retrain_py,
            "--data-dir", out_dir,
            "--model-out", model,
            "--label", "y_min",
            "--min-groups", str(args.min_groups),
        ],
        LOG_DIR / "ml_iter_train.log",
    )
    if rc != 0:
        print(f"[warn] coarse train rc={rc}; try label-A fallback from collect script", flush=True)
        rc = _run(
            [
                py, "-u", collect_py,
                "--hours", "0.01",
                "--out-dir", out_dir,
                "--append",
                "--seed0", str(args.seed0 + 50000),
                "--no-train-reserve",
            ],
            LOG_DIR / "ml_iter_train.log",
        )
        # above still collects a bit then trains; if still fail, abort
        if not Path(model).exists():
            print("[fatal] no model after train attempts", flush=True)
            sys.exit(1)

    # --- Phase 3: smoke tests (exclusive) ---
    if not args.skip_smoke:
        _run(
            [
                py, "-u", smoke_py,
                "--round", "1",
                "--small-seeds", "7401,7501",
                "--medium-seeds", "8201",
            ],
            LOG_DIR / "ml_iter_smoke_round1.log",
        )

    # --- Phase 4: long collect in background ---
    long_seed = args.seed0 + 100000
    long_proc = _spawn(
        [
            py, "-u", collect_py,
            "--hours", str(args.long_hours),
            "--out-dir", out_dir,
            "--seed0", str(long_seed),
            "--append",
            "--skip-train",
            "--no-train-reserve",
        ],
        LOG_DIR / "ml_iter_collect_long.log",
    )
    print(f"[info] long collect pid={long_proc.pid}", flush=True)

    # --- Phase 5: periodic retrain while collect runs ---
    round_id = 1
    interval = max(0.25, args.retrain_every_hours) * 3600.0
    next_t = time.perf_counter() + interval
    while long_proc.poll() is None:
        rem = next_t - time.perf_counter()
        if rem > 0:
            time.sleep(min(60.0, rem))
            continue
        round_id += 1
        print(f"[retrain] round={round_id} while collect alive", flush=True)
        _run(
            [
                py, "-u", retrain_py,
                "--data-dir", out_dir,
                "--model-out", model,
                "--label", "y_min",
                "--min-groups", str(args.min_groups),
            ],
            LOG_DIR / "ml_iter_train.log",
        )
        # optional extra smoke with new model (1 small) — may contend for Gurobi
        _run(
            [
                py, "-u", smoke_py,
                "--round", str(round_id),
                "--small-seeds", str(7401 + round_id),
                "--medium-seeds", "",
                "--out",
                str(LOG_DIR / f"bpc_ml_iter_round{round_id}.json"),
            ],
            LOG_DIR / f"ml_iter_smoke_round{round_id}.log",
        )
        next_t = time.perf_counter() + interval

    print(f"[done] long collect exited rc={long_proc.returncode}", flush=True)
    _run(
        [
            py, "-u", retrain_py,
            "--data-dir", out_dir,
            "--model-out", model,
            "--label", "y_min",
            "--min-groups", str(args.min_groups),
        ],
        LOG_DIR / "ml_iter_train.log",
    )
    print("[done] final model refreshed", flush=True)


if __name__ == "__main__":
    main()
