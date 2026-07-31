"""Collect SB-labeled listwise groups for ML branch ranker.

Stage A (pretrain): fixed M=U=5, smaller I×T grid.
Stage B (finetune): target small/medium I×T, few seeds, shallow only.
"""
from __future__ import annotations

import argparse
import copy
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ["GRB_CS_QUIET"] = "1"

import configuration as cfg
from configuration import SetConfig
from instance_generator import generator as genmod
from instance_generator.generator import generate_instance
from model_and_algo.branch_utils import BranchDecision, copy_column_pool
from model_and_algo.column_generation import ColumnGenerationSolver
from model_and_algo.ml_branch_selector import MLBranchSelector, sb_label
from preprocessor.preprocess import preprocess

# Mild LT so small T stays feasible
cfg.LEAD_TIME_CHOICES = (3, 4, 5)
genmod.LEAD_TIME_CHOICES = (3, 4, 5)
cfg.WINDOW_GENERATION_BY_SET["small_cg"] = {
    "length_frac": (0.35, 0.55),
    "rel_frac": (0.05, 0.25),
    "min_length": 4,
}

M = 5
U = 5
PRETRAIN_GRID = [
    (12, 12), (12, 16), (14, 16), (14, 20), (16, 20),
    (16, 24), (18, 24), (18, 28), (20, 24), (20, 28),
]
FINETUNE_TARGETS = [
    ("small", 25, 35),
    ("medium", 30, 40),
]

SB_TIME = 0.7
SB_TOP_K = 10
MAX_DEPTH = 5
MAX_NODES_PER_INST = 6
CG_TIME = 25.0
TARGET_ROWS_PRETRAIN = 8_000
TARGET_ROWS_FINETUNE = 1_200


def _cuts_off():
    return {
        "same_mt_setup": False,
        "wip_process_lb": False,
        "tight_deadline": False,
        "setup_pair_static": False,
        "setup_pair_separate": False,
    }


def _run_cg(data, fixed_y=None, fixed_scrap=None, fixed_wip=None, inherited=None, tlim=CG_TIME):
    cg = ColumnGenerationSolver(
        data,
        max_iterations=60,
        pricing_mode="ebrp",
        ebrp_force=True,
        use_gurobi_init=inherited is None,
        skip_cross_check=True,
        pricing_inprocess=True,
        verbose=False,
        time_limit=tlim,
        lrmp_cuts="standard",
        fixed_y=fixed_y or {},
        fixed_scrap=fixed_scrap or {},
        fixed_wip=fixed_wip or {},
        inherited_columns=inherited,
    )
    try:
        cg.run()
    except RuntimeError as e:
        if "duplicate" not in str(e).lower() and "inconsistent" not in str(e).lower():
            raise
    return cg


def _apply_side(dec: BranchDecision, fixed_y, fixed_scrap, fixed_wip, side: str):
    fy = dict(fixed_y)
    fs = dict(fixed_scrap)
    fw = dict(fixed_wip)
    if dec.kind in ("setup_y0", "setup"):
        key = dec.key
        fy[key] = 0 if side == "L" else 1
    elif dec.kind == "scrap":
        fs[int(dec.key)] = 1 if side == "L" else 0
    elif dec.kind == "wip":
        i = int(dec.key)
        k = int(dec.wip_fix[1]) if dec.wip_fix else int(dec.value // 1)
        fw[i] = ("le", k) if side == "L" else ("ge", k + 1)
    return fy, fs, fw


def _sb_one(data, parent_cg, parent_lb, cand, incumbent, deadline):
    rem = deadline - time.perf_counter()
    if rem < 2 * SB_TIME + 1:
        return None
    inherited = copy_column_pool(parent_cg.columns)
    lbs = []
    for side in ("L", "R"):
        fy, fs, fw = _apply_side(
            cand.decision,
            parent_cg.fixed_y,
            parent_cg.fixed_scrap,
            parent_cg.fixed_wip,
            side,
        )
        try:
            child = _run_cg(
                data, fy, fs, fw, inherited=inherited, tlim=min(SB_TIME, rem / 2)
            )
            lb = child.lrmp_obj
            if lb is None:
                return None
            lbs.append(max(float(lb), parent_lb))
        except Exception:
            lbs.append(float("inf"))
        rem = deadline - time.perf_counter()
    return sb_label(parent_lb, lbs[0], lbs[1], incumbent)


def _count_rows(d: Path) -> int:
    n = 0
    if not d.exists():
        return 0
    for f in d.glob("*.csv"):
        with f.open(encoding="utf-8") as fh:
            n += max(0, sum(1 for _ in fh) - 1)
    return n


def collect_instance(data, out_csv: Path, query_prefix: str, deadline: float, sel: MLBranchSelector):
    rows_before = _count_rows(out_csv.parent) if out_csv.parent.exists() else 0
    cg = _run_cg(data)
    parent_lb = float(cg.lrmp_obj or 0.0)
    incumbent = parent_lb * 1.08
    fixed_y, fixed_scrap, fixed_wip = {}, {}, {}
    nodes = 0
    qid = 0
    while nodes < MAX_NODES_PER_INST and time.perf_counter() < deadline:
        cands = sel.collect_candidates(data, cg, fixed_y, fixed_scrap, fixed_wip)
        if len(cands) < 2:
            break
        labels = []
        keep = []
        for c in cands:
            y = _sb_one(data, cg, parent_lb, c, incumbent, deadline)
            if y is None:
                continue
            keep.append(c)
            labels.append(y)
            if time.perf_counter() >= deadline:
                break
        if len(keep) >= 2:
            qid += 1
            sel.append_group_csv(
                out_csv,
                f"{query_prefix}_n{qid}",
                keep,
                labels,
                extra={"T": len(data.periods), "I": len(data.cured_items)},
            )
        # advance along heuristic-best for diversity
        best = max(keep or cands, key=lambda c: c.h_score)
        fy, fs, fw = _apply_side(best.decision, fixed_y, fixed_scrap, fixed_wip, "L")
        depth = nodes + 1
        if depth > MAX_DEPTH:
            break
        try:
            cg = _run_cg(data, fy, fs, fw, inherited=copy_column_pool(cg.columns))
        except Exception:
            break
        if cg.lrmp_obj is None:
            break
        parent_lb = max(parent_lb, float(cg.lrmp_obj))
        fixed_y, fixed_scrap, fixed_wip = fy, fs, fw
        nodes += 1
    return _count_rows(out_csv.parent) - rows_before


def stage_pretrain(out_dir: Path, wall_deadline: float) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    sel = MLBranchSelector(top_k=SB_TOP_K)
    seed = 30000
    total = _count_rows(out_dir)
    print(f"[pretrain] start rows={total} target={TARGET_ROWS_PRETRAIN}", flush=True)
    while total < TARGET_ROWS_PRETRAIN and time.perf_counter() < wall_deadline:
        for T, I in PRETRAIN_GRID:
            if total >= TARGET_ROWS_PRETRAIN or time.perf_counter() >= wall_deadline:
                break
            seed += 1
            scfg = SetConfig("small_cg", T, U, 0, I, M)
            try:
                raw = generate_instance(scfg, instance_id=seed % 1000, seed=seed)
                data = preprocess(raw)
            except Exception as e:
                print(f"  skip gen seed={seed}: {e}", flush=True)
                continue
            csv_path = out_dir / f"batch_s{seed}_{int(time.time())}.csv"
            t0 = time.perf_counter()
            try:
                added = collect_instance(
                    data, csv_path, f"s{seed}_T{T}_I{I}", wall_deadline, sel
                )
            except Exception as e:
                print(f"  skip collect seed={seed}: {e}", flush=True)
                added = 0
            total = _count_rows(out_dir)
            print(
                f"  seed={seed} T={T} I={I} +{added} total={total} "
                f"({time.perf_counter()-t0:.1f}s)",
                flush=True,
            )
            try:
                data.dispose()
            except Exception:
                pass
    print(f"[pretrain] done rows={total}", flush=True)
    return total


def stage_finetune(out_dir: Path, wall_deadline: float) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    sel = MLBranchSelector(top_k=SB_TOP_K)
    # slightly harder LT for target-ish windows
    cfg.LEAD_TIME_CHOICES = (5, 6, 7)
    genmod.LEAD_TIME_CHOICES = (5, 6, 7)
    seed = 40000
    total = _count_rows(out_dir)
    print(f"[finetune] start rows={total} target={TARGET_ROWS_FINETUNE}", flush=True)
    while total < TARGET_ROWS_FINETUNE and time.perf_counter() < wall_deadline:
        for name, T, I in FINETUNE_TARGETS:
            if total >= TARGET_ROWS_FINETUNE or time.perf_counter() >= wall_deadline:
                break
            seed += 1
            scfg = SetConfig("small_cg", T, U, 0, I, M)
            try:
                raw = generate_instance(scfg, instance_id=seed % 1000, seed=seed)
                data = preprocess(raw)
            except Exception as e:
                print(f"  skip gen seed={seed}: {e}", flush=True)
                continue
            csv_path = out_dir / f"ft_{name}_s{seed}_{int(time.time())}.csv"
            try:
                added = collect_instance(
                    data, csv_path, f"ft_{name}_s{seed}", wall_deadline, sel
                )
            except Exception as e:
                print(f"  skip collect seed={seed}: {e}", flush=True)
                added = 0
            total = _count_rows(out_dir)
            print(f"  {name} seed={seed} T={T} I={I} +{added} total={total}", flush=True)
            try:
                data.dispose()
            except Exception:
                pass
    print(f"[finetune] done rows={total}", flush=True)
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("pretrain", "finetune", "both"), default="both")
    ap.add_argument("--hours", type=float, default=2.0)
    args = ap.parse_args()
    t0 = time.perf_counter()
    deadline = t0 + args.hours * 3600.0
    pre_dir = ROOT / "ml_train_data" / "pretrain"
    ft_dir = ROOT / "ml_train_data" / "finetune"
    if args.stage in ("pretrain", "both"):
        # leave ~20% wall for finetune+train if both
        pre_deadline = deadline if args.stage == "pretrain" else t0 + args.hours * 3600.0 * 0.75
        stage_pretrain(pre_dir, pre_deadline)
    if args.stage in ("finetune", "both"):
        stage_finetune(ft_dir, deadline)
    print(f"COLLECT DONE elapsed={time.perf_counter()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
