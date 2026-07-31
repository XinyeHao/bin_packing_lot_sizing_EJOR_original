"""IM optima + BPC best_bound for seeds 101..505 (5 seeds).

Hard instance config matches recent BPC experiments.
Writes incremental JSON + final markdown under supplement_materials/.
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import os

ROOT = Path(r"C:\Users\haoxi\PycharmProjects\bin_packing_and_scheduling")
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ["GRB_CS_QUIET"] = "1"

import gurobipy as gp
from gurobipy import GRB

import configuration as cfg

cfg.apply_bench_generation()

from configuration import SetConfig
from instance_generator.generator import generate_instance
from model_and_algo.branch_and_price import solve_branch_and_price
from model_and_algo.im_model import build_im_model
from preprocessor.preprocess import preprocess

SEEDS = [101, 202, 303, 404, 505]
SCFG = SetConfig("small_cg", 35, 5, 0, 50, 5)
IM_TIME = 3600.0
IM_GAP = 0.0
BPC_TIME = 600.0
OUT_JSON = ROOT / "supplement_materials" / "im_bpc_seeds_101_505.json"
OUT_MD = ROOT / "supplement_materials" / "test_instance.md"


def _cleanup(*objs) -> None:
    for o in objs:
        try:
            if o is not None and hasattr(o, "dispose"):
                o.dispose()
        except Exception:
            pass
        try:
            del o
        except Exception:
            pass
    gc.collect()
    time.sleep(1.0)


def _fmt(x, nd=2):
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def _gap_pct(ub, lb):
    if ub is None or lb is None or abs(ub) < 1e-9:
        return None
    return abs(ub - lb) / abs(ub) * 100.0


def run_im(seed: int) -> dict:
    print(f"\n{'=' * 60}\nIM seed={seed} limit={IM_TIME:.0f}s gap={IM_GAP}\n{'=' * 60}", flush=True)
    data = preprocess(generate_instance(SCFG, instance_id=seed % 100, seed=seed))
    print(f"T={len(data.periods)} I={len(data.cured_items)} M={len(data.machines)}", flush=True)
    t0 = time.perf_counter()
    model, _vars = build_im_model(data, time_limit=IM_TIME, mip_gap=IM_GAP, verbose=False)
    model.Params.OutputFlag = 0
    model.optimize()
    elapsed = time.perf_counter() - t0
    status_map = {
        GRB.OPTIMAL: "optimal",
        GRB.TIME_LIMIT: "time_limit",
        GRB.INFEASIBLE: "infeasible",
        GRB.INF_OR_UNBD: "infeasible_or_unbounded",
    }
    status = status_map.get(model.Status, f"status_{model.Status}")
    obj = float(model.ObjVal) if model.SolCount > 0 else None
    lb = float(model.ObjBound) if model.Status != GRB.INFEASIBLE else None
    gap = _gap_pct(obj, lb)
    row = {
        "seed": seed,
        "status": status,
        "objective": obj,
        "lower_bound": lb,
        "mip_gap_pct": gap,
        "runtime_sec": elapsed,
        "sol_count": int(model.SolCount),
    }
    print(
        f"IM DONE seed={seed}: status={status} obj={_fmt(obj)} lb={_fmt(lb)} "
        f"gap={_fmt(gap)}% time={elapsed:.1f}s",
        flush=True,
    )
    _cleanup(model, data, _vars)
    return row


def run_bpc(seed: int) -> dict:
    print(f"\n{'=' * 60}\nBPC best_bound seed={seed} limit={BPC_TIME:.0f}s\n{'=' * 60}", flush=True)
    data = preprocess(generate_instance(SCFG, instance_id=seed % 100, seed=seed))
    print(f"T={len(data.periods)} I={len(data.cured_items)} M={len(data.machines)}", flush=True)
    cg_kwargs = {
        "pricing_mode": "ebrp",
        "ebrp_force": True,
        "use_gurobi_init": False,
        "skip_cross_check": True,
        "pricing_inprocess": True,
        "verbose": False,
        "progress_interval_sec": 20.0,
        "time_limit": BPC_TIME,
        "lrmp_cuts": "standard",
    }
    t0 = time.perf_counter()
    r = solve_branch_and_price(
        data,
        max_nodes=5000,
        im_time_limit=90.0,
        time_limit=BPC_TIME,
        use_node_im=False,
        root_im=True,
        final_im=True,
        branch_strategy="best_bound",
        cg_kwargs=cg_kwargs,
    )
    elapsed = time.perf_counter() - t0
    extra = r.extra or {}
    row = {
        "seed": seed,
        "status": r.status,
        "objective": r.objective,
        "lower_bound": r.lower_bound,
        "tree_lb": extra.get("tree_lb"),
        "root_lrmp": extra.get("root_lrmp"),
        "root_valid_lb": extra.get("root_valid_lb"),
        "mip_gap_pct": _gap_pct(r.objective, r.lower_bound),
        "nodes": extra.get("nodes_explored"),
        "max_depth": extra.get("max_depth"),
        "runtime_sec": elapsed,
    }
    print(
        f"BPC DONE seed={seed}: status={r.status} obj={_fmt(r.objective)} "
        f"lb={_fmt(r.lower_bound)} gap={_fmt(row['mip_gap_pct'])}% "
        f"nodes={row['nodes']} time={elapsed:.1f}s",
        flush=True,
    )
    _cleanup(data, r)
    return row


def save_json(payload: dict) -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_md(payload: dict) -> None:
    im_rows = payload.get("im", [])
    bpc_rows = payload.get("bpc", [])
    lines = [
        "# IM optima & BPC best_bound (seeds 101–505)",
        "",
        "## Setup",
        "",
        f"- Seeds: `{SEEDS}`",
        f"- Instance: `SetConfig('small_cg', 35, 5, 0, 50, 5)`, unified bench LT/window",
        f"- IM: time_limit={IM_TIME:.0f}s, mip_gap={IM_GAP} (prove optimal when status=`optimal`)",
        f"- BPC: `branch_strategy=best_bound`, `lrmp_cuts=standard`, EBRP pricing, time_limit={BPC_TIME:.0f}s, root_im/final_im on",
        f"- Code: current branch-and-price (incl. primal UB helpers: CGFO-FO / periodic RMP-IP / stronger rounding)",
        "",
        "## IM (integrated MIP)",
        "",
        "| seed | status | UB (obj) | LB | gap % | time (s) |",
        "|------|--------|----------|----|-------|----------|",
    ]
    for r in im_rows:
        lines.append(
            f"| {r['seed']} | {r['status']} | {_fmt(r.get('objective'))} | "
            f"{_fmt(r.get('lower_bound'))} | {_fmt(r.get('mip_gap_pct'))} | {_fmt(r.get('runtime_sec'), 1)} |"
        )
    lines += [
        "",
        "## BPC best_bound (branch-price-cut)",
        "",
        "| seed | status | UB | LB | root LRMP | gap % | nodes | max_depth | time (s) |",
        "|------|--------|----|----|-----------|-------|-------|-----------|----------|",
    ]
    for r in bpc_rows:
        lines.append(
            f"| {r['seed']} | {r['status']} | {_fmt(r.get('objective'))} | {_fmt(r.get('lower_bound'))} | "
            f"{_fmt(r.get('root_lrmp'))} | {_fmt(r.get('mip_gap_pct'))} | {r.get('nodes', '—')} | "
            f"{r.get('max_depth', '—')} | {_fmt(r.get('runtime_sec'), 1)} |"
        )
    lines += [
        "",
        "## IM vs BPC (same seed)",
        "",
        "| seed | IM status | IM obj | BPC UB | BPC LB | BPC gap % | note |",
        "|------|-----------|--------|--------|--------|-----------|------|",
    ]
    im_by = {r["seed"]: r for r in im_rows}
    bpc_by = {r["seed"]: r for r in bpc_rows}
    for seed in SEEDS:
        im = im_by.get(seed, {})
        bp = bpc_by.get(seed, {})
        note = ""
        if im.get("status") == "optimal" and im.get("objective") is not None and bp.get("objective") is not None:
            diff = bp["objective"] - im["objective"]
            note = f"BPC−IM={diff:.2f}"
        elif im.get("status") != "optimal" and im:
            note = "IM not proved optimal"
        lines.append(
            f"| {seed} | {im.get('status', '—')} | {_fmt(im.get('objective'))} | "
            f"{_fmt(bp.get('objective'))} | {_fmt(bp.get('lower_bound'))} | "
            f"{_fmt(bp.get('mip_gap_pct'))} | {note or '—'} |"
        )
    lines += [
        "",
        f"_Generated at {time.strftime('%Y-%m-%d %H:%M:%S')}_",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT_MD}", flush=True)


def main() -> None:
    payload = {
        "seeds": SEEDS,
        "im_time": IM_TIME,
        "im_gap": IM_GAP,
        "bpc_time": BPC_TIME,
        "im": [],
        "bpc": [],
    }
    if OUT_JSON.exists():
        try:
            payload = json.loads(OUT_JSON.read_text(encoding="utf-8"))
            print(f"Resuming from {OUT_JSON}", flush=True)
        except Exception:
            pass

    done_im = {r["seed"] for r in payload.get("im", [])}
    for seed in SEEDS:
        if seed in done_im:
            print(f"Skip IM seed={seed} (already done)", flush=True)
            continue
        row = run_im(seed)
        payload.setdefault("im", []).append(row)
        save_json(payload)
        write_md(payload)

    done_bpc = {r["seed"] for r in payload.get("bpc", [])}
    for seed in SEEDS:
        if seed in done_bpc:
            print(f"Skip BPC seed={seed} (already done)", flush=True)
            continue
        row = run_bpc(seed)
        payload.setdefault("bpc", []).append(row)
        save_json(payload)
        write_md(payload)

    write_md(payload)
    print("\nALL DONE", flush=True)
    print(f"JSON: {OUT_JSON}")
    print(f"MD:   {OUT_MD}")


if __name__ == "__main__":
    main()
