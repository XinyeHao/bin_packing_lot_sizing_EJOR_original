"""BPC frac_sb on medium 30101/30201 + large 40101/40201; stop at mip_gap=1% or time limit."""
from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ["GRB_CS_QUIET"] = "1"

import gurobipy as gp

import configuration as cfg
from configuration import BENCH_LARGE, BENCH_MEDIUM, FRAC_SB_REFINE_K, SetConfig
from instance_generator.generator import generate_instance
from model_and_algo.branch_and_price import BranchAndPriceSolver
from preprocessor.preprocess import preprocess

JOBS = [
    ("medium", 30101, 1800.0),
    ("medium", 30201, 1800.0),
    ("large", 40101, 3600.0),
    ("large", 40201, 3600.0),
]
MIP_GAP = 0.01
OUT_JSON = ROOT / "supplement_materials" / "bpc_frac_gap1pct_301_302_large.json"
OUT_MD = ROOT / "supplement_materials" / "bpc_frac_gap1pct_301_302_large.md"


def _gap(ub, lb):
    if ub is None or lb is None or ub <= 1e-9:
        return None
    return round(100.0 * max(0.0, (ub - lb) / ub), 4)


def _cleanup(*objs) -> None:
    for o in objs:
        try:
            del o
        except Exception:
            pass
    gc.collect()
    try:
        gp.disposeDefaultEnv()
    except Exception:
        pass
    gc.collect()


def run_one(scale: str, seed: int, limit: float) -> dict:
    T, I = BENCH_MEDIUM if scale == "medium" else BENCH_LARGE
    print(
        f"\n======== BPC frac_sb {scale} seed={seed} T={T} I={I} "
        f"limit={limit:.0f}s mip_gap={100 * MIP_GAP:.0f}% ========",
        flush=True,
    )
    raw = generate_instance(SetConfig("small_cg", T, 5, 0, I, 5), seed % 1000, seed)
    data = preprocess(raw)
    bp = BranchAndPriceSolver(
        data,
        max_nodes=8000,
        time_limit=limit,
        mip_gap=MIP_GAP,
        branch_strategy="frac_sb",
        use_node_im=False,
        root_im=False,
        final_im=False,
        cg_kwargs=dict(
            pricing_mode="ebrp",
            ebrp_force=True,
            use_gurobi_init=True,
            skip_cross_check=True,
            pricing_inprocess=True,
            verbose=False,
            progress_interval_sec=120.0,
            lrmp_cuts="standard",
            max_iterations=120,
            ebrp_time_limit=20.0,
        ),
    )
    t0 = time.perf_counter()
    res = bp.solve()
    elapsed = time.perf_counter() - t0
    ub, lb = res.objective, res.lower_bound
    gap = _gap(ub, lb)
    row = {
        "scale": scale,
        "seed": seed,
        "T": T,
        "I": I,
        "M": 5,
        "time_limit": limit,
        "mip_gap": MIP_GAP,
        "status": res.status,
        "ub": ub,
        "lb": lb,
        "gap_pct": gap,
        "runtime_sec": round(elapsed, 2),
        "nodes_explored": res.nodes_explored,
        "nodes_pruned": res.nodes_pruned,
        "nodes_cg_failed": res.nodes_cg_failed,
        "proved": res.status == "optimal"
        or (ub is not None and lb is not None and abs(ub - lb) <= 1e-4),
    }
    print(
        f"  -> status={row['status']} ub={ub} lb={lb} gap={gap}% "
        f"nodes={row['nodes_explored']} cg_fail={row['nodes_cg_failed']} "
        f"sec={row['runtime_sec']}",
        flush=True,
    )
    _cleanup(bp, res, data, raw)
    return row


def _write_md(meta: dict) -> None:
    lines = [
        "# BPC frac_sb：30101 / 30201 + large 40101 / 40201（gap≤1% 停）",
        "",
        f"- 策略：`frac_sb`（k={FRAC_SB_REFINE_K}），**无 ML**",
        f"- 停止：相对 gap `(UB−tree_LB)/UB ≤ {100 * MIP_GAP:.0f}%`，或达到时限",
        f"- 全局下界：开节点 LB 最小值（`_tree_dual_bound`）",
        f"- medium `(T,I)={BENCH_MEDIUM}` @1800s；large `{BENCH_LARGE}` @3600s；M=U=5",
        "",
        "| scale | seed | T | I | status | UB | LB | gap% | time(s) | nodes | cg_fail |",
        "|-------|------|---|---|--------|----|----|------|---------|-------|---------|",
    ]
    for r in meta["rows"]:
        if "error" in r:
            lines.append(
                f"| {r.get('scale','')} | {r['seed']} | — | — | ERROR | — | — | — | — | — | {r['error'][:40]} |"
            )
            continue
        lines.append(
            f"| {r['scale']} | {r['seed']} | {r['T']} | {r['I']} | {r['status']} | "
            f"{_fmt(r.get('ub'))} | {_fmt(r.get('lb'))} | {_fmt(r.get('gap_pct'))} | "
            f"{_fmt(r.get('runtime_sec'), 1)} | {r.get('nodes_explored')} | {r.get('nodes_cg_failed')} |"
        )
    lines.extend(["", "## 小结", ""])
    ok = [r for r in meta["rows"] if "error" not in r]
    n_gap = sum(1 for r in ok if r.get("status") == "gap_limit")
    n_opt = sum(1 for r in ok if r.get("status") == "optimal")
    n_tl = sum(1 for r in ok if r.get("status") == "time_limit")
    lines.append(
        f"- 完成 {len(ok)}/{len(JOBS)}：`gap_limit`={n_gap}，`optimal`={n_opt}，`time_limit`={n_tl}"
    )
    if ok:
        gaps = [r["gap_pct"] for r in ok if r.get("gap_pct") is not None]
        times = [r["runtime_sec"] for r in ok if r.get("runtime_sec") is not None]
        if gaps:
            lines.append(
                f"- gap%：min={min(gaps):.3f}，max={max(gaps):.3f}，mean={sum(gaps)/len(gaps):.3f}"
            )
        if times:
            lines.append(
                f"- 时间(s)：min={min(times):.1f}，max={max(times):.1f}，mean={sum(times)/len(times):.1f}"
            )
        cg = sum(int(r.get("nodes_cg_failed") or 0) for r in ok)
        lines.append(f"- 合计 `nodes_cg_failed`={cg}")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(v, nd=2):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def main():
    cfg.apply_bench_generation()
    rows = []
    meta = {
        "note": "BPC frac_sb mip_gap=1% on medium 30101/30201 + large 40101/40201",
        "frac_sb_refine_k": FRAC_SB_REFINE_K,
        "mip_gap": MIP_GAP,
        "jobs": [{"scale": s, "seed": seed, "limit": lim} for s, seed, lim in JOBS],
        "rows": rows,
    }
    OUT_JSON.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    for scale, seed, limit in JOBS:
        try:
            rows.append(run_one(scale, seed, limit))
        except Exception as e:
            print(f"  FAIL {scale} seed={seed}: {e}", flush=True)
            rows.append({"scale": scale, "seed": seed, "error": str(e)})
        OUT_JSON.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        _write_md(meta)
        print(f"[checkpoint] {len(rows)}/{len(JOBS)} -> {OUT_JSON}", flush=True)

    print("\n=== DONE ===", flush=True)
    for r in rows:
        if "error" in r:
            print(f"{r['scale']} seed={r['seed']}: ERROR {r['error']}", flush=True)
        else:
            print(
                f"{r['scale']} seed={r['seed']}: status={r['status']} "
                f"gap={r['gap_pct']} sec={r['runtime_sec']} cg_fail={r['nodes_cg_failed']}",
                flush=True,
            )
    print(f"\nJSON: {OUT_JSON}\nMD:   {OUT_MD}", flush=True)


if __name__ == "__main__":
    main()
