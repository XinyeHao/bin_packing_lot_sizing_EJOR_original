"""ML branch collect+train schema v5.

- BENCH generation + Deadline Cover
- Pool: frac Top-20; SB labels on dual_share Top-10 within pool
- SB_TIME=3.0s per child CG
- Features v5: + dens_rel
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ["GRB_CS_QUIET"] = "1"

import configuration as cfg
from configuration import ML_BRANCH_MODEL_PATH, SetConfig
from instance_generator.generator import generate_instance
from model_and_algo.branch_utils import BranchDecision, copy_column_pool
from model_and_algo.column_generation import ColumnGenerationSolver
from model_and_algo.ml_branch_selector import MLBranchSelector, sb_label
from preprocessor.preprocess import preprocess

SCALE_GRID: list[tuple[str, int, int]] = [
    ("xs", 18, 24),
    ("s", 20, 28),
    ("s2", 22, 32),
    ("small", 25, 35),
    ("medium", 30, 40),
    ("large", 35, 50),
]
SCALE_CYCLE = (
    ["xs", "s", "s2", "small", "small", "medium", "medium", "large"]
)

SB_TIME = 3.0
TOP_K = 20          # frac pool (match inference)
SB_LABEL_K = 10     # dual_share Top-K within pool for strong branching
MAX_DEPTH = 5
MAX_NODES_PER_INST = 6
CG_TIME = 25.0
TRAIN_RESERVE_SEC = 180.0


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
        fy[dec.key] = 0 if side == "L" else 1
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
        # Within frac pool, only SB-label dual_share Top-K
        to_label = sorted(cands, key=lambda c: c.dual, reverse=True)[:SB_LABEL_K]
        if len(to_label) < 2:
            break
        # Re-assign features relative to the labeled subset
        from model_and_algo.ml_branch_selector import _assign_features

        _assign_features(to_label)
        labels = []
        keep = []
        for c in to_label:
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
        best = max(keep or cands, key=lambda c: c.dual)
        fy, fs, fw = _apply_side(best.decision, fixed_y, fixed_scrap, fixed_wip, "L")
        if nodes + 1 > MAX_DEPTH:
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=2.5)
    ap.add_argument("--out-dir", type=str, default="ml_train_data_v5")
    ap.add_argument("--model-out", type=str, default=str(ML_BRANCH_MODEL_PATH))
    ap.add_argument("--seed0", type=int, default=91001)
    args = ap.parse_args()

    cfg.apply_bench_generation()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "RUN_INFO.txt").write_text(
        f"schema_v5 SB_TIME={SB_TIME} TOP_K={TOP_K} SB_LABEL_K={SB_LABEL_K}\n"
        f"hours={args.hours} dens_rel=on\n",
        encoding="utf-8",
    )

    t0 = time.perf_counter()
    wall = args.hours * 3600.0
    collect_deadline = t0 + wall - TRAIN_RESERVE_SEC
    sel = MLBranchSelector(top_k=TOP_K)
    scale_map = {name: (T, I) for name, T, I in SCALE_GRID}

    seed = args.seed0
    total = _count_rows(out_dir)
    step = 0
    print(
        f"[collect] start rows={total} deadline_collect={collect_deadline - t0:.0f}s "
        f"SB_TIME={SB_TIME} pool={TOP_K} label_k={SB_LABEL_K}",
        flush=True,
    )

    while time.perf_counter() < collect_deadline:
        name = SCALE_CYCLE[step % len(SCALE_CYCLE)]
        step += 1
        T, I = scale_map[name]
        seed += 1
        scfg = SetConfig("small_cg", T, 5, 0, I, 5)
        try:
            raw = generate_instance(scfg, instance_id=seed % 1000, seed=seed)
            data = preprocess(raw)
        except Exception as e:
            print(f"  skip gen {name} seed={seed}: {e}", flush=True)
            continue
        csv_path = out_dir / f"batch_{name}_s{seed}_{int(time.time())}.csv"
        try:
            added = collect_instance(
                data, csv_path, f"{name}_s{seed}", collect_deadline, sel
            )
        except Exception as e:
            print(f"  skip collect {name} seed={seed}: {e}", flush=True)
            added = 0
        total = _count_rows(out_dir)
        print(
            f"  {name} seed={seed} T={T} I={I} +{added} total_rows={total} "
            f"elapsed={time.perf_counter() - t0:.0f}s",
            flush=True,
        )
        try:
            data.dispose()
        except Exception:
            pass

    print(f"[collect] done rows={total} elapsed={time.perf_counter() - t0:.1f}s", flush=True)

    model_out = Path(args.model_out)
    if total < 50:
        print(f"[train] ABORT too few rows={total}", flush=True)
        sys.exit(1)
    print(f"[train] LightGBM from {out_dir} -> {model_out}", flush=True)
    trainer = MLBranchSelector(top_k=TOP_K)
    path = trainer.train_from_dir(out_dir, out_path=model_out, num_boost_round=200)
    print(
        f"[train] OK model={path} schema_v5 rows={total} "
        f"total_elapsed={time.perf_counter() - t0:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
