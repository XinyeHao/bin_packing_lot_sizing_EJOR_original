"""Experiment 3: direct same-dual pricing sensitivity to autoclave capacity.

For each capacity multiplier, extract genuine duals from the initial RMP and
solve the five machine pricing subproblems exactly once with EBRP and once with
Gurobi.  This is deliberately not a full column-generation experiment.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ["GRB_CS_QUIET"] = "1"

import gurobipy as gp
import configuration as cfg
from experiments.paper_experiment_config import MODEL_VERSION, PROCESS_ASSUMPTION, SCALES, make_set_config
from instance_generator.generator import generate_instance
from model_and_algo.column_generation import ColumnGenerationSolver
from model_and_algo.pricing_ebrp import EBRPStats, solve_subproblem_ebrp
from model_and_algo.pricing_im_gurobi import solve_subproblem_im_gurobi
from preprocessor.preprocess import preprocess

VERSION = "direct_same_dual_five_machine_pricing_v1"
RC_TOL = 1e-4


def args_parser() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E3 direct EBRP/Gurobi pricing sensitivity")
    p.add_argument("--scales", nargs="+", choices=tuple(SCALES), default=list(SCALES))
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument("--capacity-multipliers", nargs="+", type=float, default=(0.8, 1.0, 1.2))
    p.add_argument("--subproblem-time-limit", type=float, default=60.0)
    p.add_argument("--output-prefix", default="exp3_direct_pricing_capacity")
    return p.parse_args()


def feasible(raw) -> list[str]:
    return [
        item for item in raw.cured_item_ids
        if not any(raw.machine_capacities[m] + 1e-9 >= raw.tray_lengths[item]
                   for m in (raw.item_machines.get(item) or raw.machine_ids))
    ]


def initial_duals(data):
    """Build just the initial RMP, then read the actual dual vector used for pricing."""
    cg = ColumnGenerationSolver(
        data, max_iterations=1, time_limit=None, pricing_mode="ebrp",
        ebrp_force=True, use_gurobi_init=False, pricing_inprocess=True,
        skip_cross_check=True, lrmp_cuts="none", verbose=False,
    )
    cg.initialize_columns()
    cg._init_master(relaxed=True)
    if cg._model is None or cg._artifacts is None:
        raise RuntimeError("initial RMP was not created")
    cg._model.optimize()
    if cg._model.Status not in (gp.GRB.OPTIMAL, gp.GRB.SUBOPTIMAL):
        raise RuntimeError(f"initial RMP status={cg._model.Status}")
    n_i, n_t = len(data.cured_items), len(data.periods)
    alpha = [[0.0] * n_t for _ in range(n_i)]
    flow = cg._artifacts.get("flow_items") or cg._artifacts.get("flow_cured", {})
    for (i, t), row in flow.items():
        alpha[i][t] = float(row.Pi)
    rho = [float(cg._artifacts["wip_balance"][i].Pi) for i in range(n_i)]
    beta = [float(cg._artifacts["choose_one"][m].Pi) for m in data.machines]
    return cg, alpha, rho, beta


def solve_one(scale: str, seed: int, multiplier: float, time_limit: float) -> list[dict]:
    cfg.apply_bench_generation()
    spec = SCALES[scale]
    raw = generate_instance(make_set_config(spec), seed % 1000, seed)
    raw.machine_capacities = {m: round(q * multiplier, 4) for m, q in raw.machine_capacities.items()}
    oversized = feasible(raw)
    common = {
        "scale": scale, "seed": seed, "T": spec.periods, "I": spec.items,
        "capacity_multiplier": multiplier, "capacity_label": f"{multiplier:g}q",
    }
    if oversized:
        return [{**common, "machine_index": None, "machine": None, "capacity_feasible": False,
                 "oversized_items": oversized, "comparison_valid": False, "error": "capacity_infeasible"}]
    data = preprocess(raw)
    cg = None
    try:
        cg, alpha, rho, beta = initial_duals(data)
        rows = []
        for m_idx, machine in enumerate(data.machines):
            stats = EBRPStats()
            t0 = time.perf_counter()
            e = solve_subproblem_ebrp(
                data, m_idx, alpha, beta[m_idx], rho=rho, ctx=cg._ebrp_contexts[m_idx],
                stats=stats, time_limit=time_limit, progress_interval_sec=0.0, skip_cross_check=True,
            )
            e_time = time.perf_counter() - t0
            t0 = time.perf_counter()
            g = solve_subproblem_im_gurobi(
                data, m_idx, alpha, beta[m_idx], rho=rho,
                time_limit=time_limit, progress_interval_sec=0.0,
            )
            g_time = time.perf_counter() - t0
            rc_delta = float(e.reduced_cost) - float(g.reduced_cost)
            exact = bool(e.is_optimal) and bool(g.is_optimal)
            rows.append({
                **common, "machine_index": m_idx, "machine": machine, "capacity_feasible": True,
                "oversized_items": [], "ebrp_time_sec": e_time, "gurobi_time_sec": g_time,
                "ebrp_reduced_cost": float(e.reduced_cost), "gurobi_reduced_cost": float(g.reduced_cost),
                "reduced_cost_difference": rc_delta, "ebrp_is_optimal": bool(e.is_optimal),
                "gurobi_is_optimal": bool(g.is_optimal), "ebrp_states_expanded": stats.nodes_expanded,
                "comparison_valid": exact and abs(rc_delta) <= RC_TOL,
                "ebrp_speedup_vs_gurobi": g_time / e_time if e_time > 1e-12 else None, "error": None,
            })
        return rows
    finally:
        try:
            if cg is not None:
                cg._model.dispose()
        except Exception:
            pass
        gc.collect()
        try:
            gp.disposeDefaultEnv()
        except Exception:
            pass


def mean(values):
    values = [float(v) for v in values if v is not None]
    return sum(values) / len(values) if values else None


def summaries(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["scale"], row["seed"], row["capacity_multiplier"])].append(row)
    output = []
    for (_, _, _), group in sorted(groups.items()):
        first = group[0]
        valid = [r for r in group if r.get("comparison_valid")]
        output.append({
            "scale": first["scale"], "seed": first["seed"], "T": first["T"], "I": first["I"],
            "capacity_multiplier": first["capacity_multiplier"], "capacity_label": first["capacity_label"],
            "machines": len([r for r in group if r.get("machine_index") is not None]),
            "valid_machine_comparisons": len(valid), "capacity_feasible": first["capacity_feasible"],
            "mean_ebrp_time_sec": mean(r.get("ebrp_time_sec") for r in valid),
            "mean_gurobi_time_sec": mean(r.get("gurobi_time_sec") for r in valid),
            "total_ebrp_time_sec": sum(r["ebrp_time_sec"] for r in valid) if valid else None,
            "total_gurobi_time_sec": sum(r["gurobi_time_sec"] for r in valid) if valid else None,
            "mean_ebrp_speedup_vs_gurobi": mean(r.get("ebrp_speedup_vs_gurobi") for r in valid),
        })
    return output


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def main() -> None:
    args = args_parser()
    if args.subproblem_time_limit <= 0 or any(x <= 0 for x in args.capacity_multipliers):
        raise ValueError("all time limits and multipliers must be positive")
    selected = {}
    for scale in args.scales:
        chosen = tuple(args.seeds) if args.seeds is not None else SCALES[scale].seeds
        bad = sorted(set(chosen) - set(SCALES[scale].seeds))
        if bad:
            raise ValueError(f"seed(s) {bad} are not configured for {scale}")
        selected[scale] = chosen
    result = ROOT / "result" / "exp3"
    result.mkdir(parents=True, exist_ok=True)
    machine_csv = result / f"{args.output_prefix}_machine.csv"
    summary_csv = result / f"{args.output_prefix}_summary.csv"
    json_path = result / f"{args.output_prefix}.json"
    machine_rows = []
    total = sum(len(selected[s]) for s in args.scales) * len(args.capacity_multipliers)
    index = 0
    for scale in args.scales:
        for seed in selected[scale]:
            for multiplier in args.capacity_multipliers:
                index += 1
                print(f"[{index}/{total}] {scale} seed={seed} capacity={multiplier:g}q: 5 direct pricing pairs", flush=True)
                machine_rows.extend(solve_one(scale, seed, multiplier, args.subproblem_time_limit))
                summary_rows = summaries(machine_rows)
                json_path.write_text(json.dumps({
                    "experiment": "E3 direct five-autoclave same-dual pricing sensitivity",
                    "experiment_version": VERSION, "process_assumption": PROCESS_ASSUMPTION,
                    "model_version": MODEL_VERSION, "capacity_multipliers": args.capacity_multipliers,
                    "subproblem_time_limit_sec": args.subproblem_time_limit, "machine_rows": machine_rows,
                    "summary_rows": summary_rows,
                }, ensure_ascii=False, indent=2), encoding="utf-8")
                write_csv(machine_csv, machine_rows, [
                    "scale", "seed", "T", "I", "capacity_multiplier", "capacity_label", "machine_index", "machine",
                    "capacity_feasible", "oversized_items", "ebrp_time_sec", "gurobi_time_sec", "ebrp_reduced_cost",
                    "gurobi_reduced_cost", "reduced_cost_difference", "ebrp_is_optimal", "gurobi_is_optimal",
                    "ebrp_states_expanded", "comparison_valid", "ebrp_speedup_vs_gurobi", "error",
                ])
                write_csv(summary_csv, summary_rows, [
                    "scale", "seed", "T", "I", "capacity_multiplier", "capacity_label", "machines",
                    "valid_machine_comparisons", "capacity_feasible", "mean_ebrp_time_sec", "mean_gurobi_time_sec",
                    "total_ebrp_time_sec", "total_gurobi_time_sec", "mean_ebrp_speedup_vs_gurobi",
                ])
    print(f"Finished. Results: {summary_csv}", flush=True)


if __name__ == "__main__":
    main()
