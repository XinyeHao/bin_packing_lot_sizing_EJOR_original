"""Experiment 4: EBRP pricing validation and autoclave-length sensitivity.

Part ``pricing`` compares root CG using EBRP and exact Gurobi pricing.
Part ``sensitivity`` runs complete BPC after changing only machine capacities.
Both parts are resumable and write raw plus 15-instance summary tables.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.paper_experiment_config import MODEL_VERSION, PROCESS_ASSUMPTION, SCALES

TOL = 1e-5


def args_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E4: EBRP pricing and autoclave-length sensitivity.")
    parser.add_argument("--part", choices=("both", "pricing", "sensitivity"), default="both")
    parser.add_argument("--scales", nargs="+", choices=tuple(SCALES), default=list(SCALES))
    parser.add_argument("--capacity-multipliers", nargs="+", type=float, default=(0.8, 1.0, 1.2))
    parser.add_argument("--output-prefix", default="exp4_ebrp_capacity_sensitivity")
    parser.add_argument("--rerun-incomplete", action="store_true")
    return parser.parse_args()


def multiplier_key(multiplier: float) -> str:
    return f"{multiplier:.4f}".rstrip("0").rstrip(".")


def as_number(value):
    return float(value) if value is not None else None


def run_worker(method: str, scale: str, seed: int, multiplier: float, tmp: Path) -> dict:
    tag = multiplier_key(multiplier).replace(".", "p")
    out = tmp / f"{method}_{scale}_{seed}_{tag}.json"
    log = tmp / f"{method}_{scale}_{seed}_{tag}.log"
    limit = SCALES[scale].time_limit_sec
    cmd = [
        sys.executable, "-B", "experiments/paper_experiment_worker.py", method,
        scale, str(seed), str(out), "--limit", str(limit),
        "--capacity-multiplier", str(multiplier),
    ]
    with log.open("w", encoding="utf-8") as handle:
        try:
            subprocess.run(cmd, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                           timeout=limit + 180.0, check=False)
        except subprocess.TimeoutExpired:
            pass
    if out.exists():
        try:
            return json.loads(out.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            return {"status": "worker_error", "error": f"cannot read worker result: {exc}"}
    return {"method": method, "scale": scale, "seed": seed,
            "capacity_multiplier": multiplier, "status": "wall_time_limit",
            "error": "worker did not return after solver budget plus setup buffer"}


def exact_root_row(scale: str, seed: int, mult: float, ebrp: dict, gurobi: dict) -> dict:
    e_lb, g_lb = as_number(ebrp.get("lb")), as_number(gurobi.get("lb"))
    e_ok = ebrp.get("status") == "converged" and e_lb is not None
    g_ok = gurobi.get("status") == "converged" and g_lb is not None
    diff = e_lb - g_lb if e_ok and g_ok else None
    matched = diff is not None and abs(diff) <= TOL * max(1.0, abs(g_lb))
    g_time, e_time = as_number(gurobi.get("runtime_sec")), as_number(ebrp.get("runtime_sec"))
    g_pr, e_pr = as_number(gurobi.get("pricing_time_sec")), as_number(ebrp.get("pricing_time_sec"))
    return {
        "scale": scale, "seed": seed, "T": SCALES[scale].periods, "I": SCALES[scale].items,
        "capacity_multiplier": mult, "capacity_feasible": bool(ebrp.get("capacity_feasible")) and bool(gurobi.get("capacity_feasible")),
        "ebrp_status": ebrp.get("status"), "ebrp_root_lb": e_lb if e_ok else None,
        "ebrp_cg_time_sec": e_time, "ebrp_pricing_time_sec": e_pr,
        "ebrp_iterations": ebrp.get("iterations"), "ebrp_columns": ebrp.get("columns"),
        "gurobi_status": gurobi.get("status"), "gurobi_root_lb": g_lb if g_ok else None,
        "gurobi_cg_time_sec": g_time, "gurobi_pricing_time_sec": g_pr,
        "gurobi_iterations": gurobi.get("iterations"), "gurobi_columns": gurobi.get("columns"),
        "both_root_cg_converged": e_ok and g_ok,
        "lb_difference_ebrp_minus_gurobi": diff, "root_lb_matches": matched,
        "cg_speedup_gurobi_over_ebrp": g_time / e_time if e_time and g_time else None,
        "pricing_speedup_gurobi_over_ebrp": g_pr / e_pr if e_pr and g_pr else None,
        "ebrp_error": ebrp.get("error"), "gurobi_error": gurobi.get("error"),
        "raw_ebrp": ebrp, "raw_gurobi": gurobi,
    }


def bpc_row(scale: str, seed: int, mult: float, bpc: dict) -> dict:
    certified = bool(bpc.get("lower_bound_certified"))
    return {
        "scale": scale, "seed": seed, "T": SCALES[scale].periods, "I": SCALES[scale].items,
        "capacity_multiplier": mult, "capacity_feasible": bpc.get("capacity_feasible"),
        "oversized_items": bpc.get("oversized_items", []), "bpc_status": bpc.get("status"),
        "bpc_ub": as_number(bpc.get("ub")),
        "bpc_lb": as_number(bpc.get("lb")) if certified else None,
        "bpc_gap_pct": as_number(bpc.get("gap_pct")) if certified else None,
        "bpc_runtime_sec": as_number(bpc.get("runtime_sec")),
        "bpc_nodes": bpc.get("nodes"), "bpc_nodes_pruned": bpc.get("nodes_pruned"),
        "bpc_nodes_cg_failed": bpc.get("nodes_cg_failed"), "bpc_root_lb": as_number(bpc.get("root_lb")),
        "bpc_root_converged": bpc.get("root_converged"), "bpc_error": bpc.get("error"), "raw_bpc": bpc,
    }


def avg(values):
    values = [float(v) for v in values if v is not None]
    return sum(values) / len(values) if values else None


def root_summary(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows: groups[(row["scale"], row["capacity_multiplier"])].append(row)
    result = []
    for (scale, mult), data in sorted(groups.items()):
        result.append({
            "scale": scale, "capacity_multiplier": mult, "instances": len(data),
            "both_root_cg_converged": sum(bool(r["both_root_cg_converged"]) for r in data),
            "root_lb_matches": sum(bool(r["root_lb_matches"]) for r in data),
            "mean_ebrp_cg_time_sec": avg(r["ebrp_cg_time_sec"] for r in data),
            "mean_gurobi_cg_time_sec": avg(r["gurobi_cg_time_sec"] for r in data),
            "mean_ebrp_pricing_time_sec": avg(r["ebrp_pricing_time_sec"] for r in data),
            "mean_gurobi_pricing_time_sec": avg(r["gurobi_pricing_time_sec"] for r in data),
            "mean_cg_speedup_gurobi_over_ebrp": avg(r["cg_speedup_gurobi_over_ebrp"] for r in data),
            "mean_pricing_speedup_gurobi_over_ebrp": avg(r["pricing_speedup_gurobi_over_ebrp"] for r in data),
        })
    return result


def sensitivity_summary(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows: groups[(row["scale"], row["capacity_multiplier"])].append(row)
    result = []
    for (scale, mult), data in sorted(groups.items()):
        result.append({
            "scale": scale, "capacity_multiplier": mult, "instances": len(data),
            "capacity_feasible": sum(bool(r["capacity_feasible"]) for r in data),
            "optimal": sum(r["bpc_status"] == "optimal" for r in data),
            "root_cg_converged": sum(bool(r["bpc_root_converged"]) for r in data),
            "mean_bpc_ub": avg(r["bpc_ub"] for r in data), "mean_bpc_lb": avg(r["bpc_lb"] for r in data),
            "mean_bpc_gap_pct": avg(r["bpc_gap_pct"] for r in data),
            "mean_bpc_runtime_sec": avg(r["bpc_runtime_sec"] for r in data),
            "mean_bpc_nodes": avg(r["bpc_nodes"] for r in data),
            "mean_bpc_nodes_pruned": avg(r["bpc_nodes_pruned"] for r in data),
        })
    return result


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in rows: writer.writerow({field: row.get(field) for field in fields})


def load_rows(path: Path) -> dict[tuple, dict]:
    if not path.exists(): return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("model_version") != MODEL_VERSION or data.get("process_assumption") != PROCESS_ASSUMPTION: return {}
        return {(r["scale"], int(r["seed"]), float(r["capacity_multiplier"])): r for r in data.get("rows", [])}
    except (OSError, ValueError, KeyError, TypeError): return {}


def main() -> None:
    args = args_parser(); mults = tuple(args.capacity_multipliers)
    if any(mult <= 0 for mult in mults): raise ValueError("capacity multipliers must be positive")
    result = ROOT / "result"; result.mkdir(exist_ok=True)
    tmp = result / f"_{args.output_prefix}_tmp"; tmp.mkdir(exist_ok=True)
    do_pricing = args.part in ("both", "pricing"); do_sensitivity = args.part in ("both", "sensitivity")
    pricing_json = result / f"{args.output_prefix}_pricing.json"; sensitivity_json = result / f"{args.output_prefix}_bpc.json"
    pricing_rows = load_rows(pricing_json) if do_pricing else {}
    sensitivity_rows = load_rows(sensitivity_json) if do_sensitivity else {}
    total = sum(len(SCALES[s].seeds) for s in args.scales) * len(mults); index = 0
    for scale in args.scales:
        for seed in SCALES[scale].seeds:
            for mult in mults:
                index += 1; key = (scale, seed, float(mult))
                if do_pricing and (key not in pricing_rows or args.rerun_incomplete and not pricing_rows[key].get("root_lb_matches")):
                    print(f"[4A {index}/{total}] {scale} seed={seed} lambda={mult}: EBRP vs Gurobi root CG", flush=True)
                    pricing_rows[key] = exact_root_row(scale, seed, mult, run_worker("root_ebrp", scale, seed, mult, tmp), run_worker("root_gurobi", scale, seed, mult, tmp))
                if do_sensitivity and (key not in sensitivity_rows or args.rerun_incomplete and sensitivity_rows[key].get("bpc_status") in ("error", "wall_time_limit")):
                    print(f"[4B {index}/{total}] {scale} seed={seed} lambda={mult}: complete BPC", flush=True)
                    sensitivity_rows[key] = bpc_row(scale, seed, mult, run_worker("bpc", scale, seed, mult, tmp))
                if do_pricing:
                    rows = list(pricing_rows.values()); summary = root_summary(rows)
                    payload = {"experiment":"E4A: EBRP versus Gurobi exact pricing","process_assumption":PROCESS_ASSUMPTION,"model_version":MODEL_VERSION,"capacity_multipliers":mults,"root_lb_tolerance":TOL,"rows":rows,"summary":summary}
                    pricing_json.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
                    write_csv(result/f"{args.output_prefix}_pricing.csv",rows,["scale","seed","T","I","capacity_multiplier","capacity_feasible","ebrp_status","ebrp_root_lb","ebrp_cg_time_sec","ebrp_pricing_time_sec","ebrp_iterations","ebrp_columns","gurobi_status","gurobi_root_lb","gurobi_cg_time_sec","gurobi_pricing_time_sec","gurobi_iterations","gurobi_columns","both_root_cg_converged","lb_difference_ebrp_minus_gurobi","root_lb_matches","cg_speedup_gurobi_over_ebrp","pricing_speedup_gurobi_over_ebrp","ebrp_error","gurobi_error"])
                    write_csv(result/f"{args.output_prefix}_pricing_summary.csv",summary,list(summary[0]) if summary else ["scale"])
                if do_sensitivity:
                    rows = list(sensitivity_rows.values()); summary = sensitivity_summary(rows)
                    payload = {"experiment":"E4B: autoclave-length sensitivity of complete BPC","process_assumption":PROCESS_ASSUMPTION,"model_version":MODEL_VERSION,"capacity_multipliers":mults,"rows":rows,"summary":summary}
                    sensitivity_json.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
                    write_csv(result/f"{args.output_prefix}_bpc.csv",rows,["scale","seed","T","I","capacity_multiplier","capacity_feasible","oversized_items","bpc_status","bpc_ub","bpc_lb","bpc_gap_pct","bpc_runtime_sec","bpc_nodes","bpc_nodes_pruned","bpc_nodes_cg_failed","bpc_root_lb","bpc_root_converged","bpc_error"])
                    write_csv(result/f"{args.output_prefix}_bpc_summary.csv",summary,list(summary[0]) if summary else ["scale"])
    print("Finished. E4 raw and summary tables are in result/.",flush=True)


if __name__ == "__main__": main()
