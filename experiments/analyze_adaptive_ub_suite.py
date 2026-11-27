"""Audit and tabulate every adaptive-UB benchmark case, including failures."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
from dataclasses import fields
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def target_reached(row):
    return (row.get("status") in ("optimal", "gap_limit")
            and row.get("gap_pct") is not None and row["gap_pct"] <= 1.000001
            and row.get("nodes_unresolved", 0) == 0)


def analyze(folder, audit=False):
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    rows, table, audits = [], [], []
    for case in manifest["cases"]:
        path = folder / f"{case['scale']}_{case['seed']}.json"
        if not path.exists():
            continue
        row = json.loads(path.read_text(encoding="utf-8"))
        if row["status"] == "running":
            continue
        rows.append(row)
        if row.get("source_sha256") != manifest["source_sha256"]:
            raise AssertionError(f"Source version mismatch: {path}")
        history = row.get("primal_history", [])
        for previous, current in zip(history, history[1:]):
            assert current["objective"] < previous["objective"] - 1e-5
            assert current["elapsed_sec"] >= previous["elapsed_sec"]
        if history:
            assert abs(history[-1]["objective"] - row["ub"]) < 1e-4
        stats, profiling = row.get("primal_stats", {}), row.get("profiling", {})
        root_irmp = next((x["objective"] for x in history if x["source"] == "irmp" and x["node_id"] == 0), None)
        root_cgfo = next((x["objective"] for x in history if x["source"] == "cgfo_root"), None)
        record = {k: row.get(k) for k in (
            "scale", "seed", "status", "ub", "lb", "gap_pct", "runtime_sec", "cpu_runtime_sec",
            "nodes", "nodes_created", "nodes_cg_failed", "nodes_unresolved", "primal_validated")}
        record.update(root_irmp_ub=root_irmp, root_cgfo_ub=root_cgfo,
                      irmp_calls=stats.get("irmp_calls"), cgfo_calls=stats.get("cgfo_calls"),
                      repair_calls=stats.get("cgfo_repair_calls"),
                      cgfo_improvements=stats.get("cgfo_improvements"),
                      irmp_seconds=profiling.get("irmp"), cgfo_seconds=profiling.get("cgfo"))
        for historical in row.get("historical", []):
            prefix = "corrected_baseline" if "bp_fixed_medium_large" in historical["path"] else "original_baseline"
            for name in ("status", "ub", "lb", "gap_pct", "runtime_sec", "nodes"):
                record[prefix + "_" + name] = historical["result"].get(name)
        table.append(record)
        if audit and row.get("solution_path"):
            import configuration as cfg
            from common.data_models import InstanceData, SolutionResult
            from experiments.paper_experiment_config import make_set_config, scale_spec
            from instance_generator.generator import generate_instance
            from preprocessor.preprocess import preprocess
            from model_and_algo.primal_heuristics import validate_primal
            cfg.apply_bench_generation()
            if row.get("input_path"):
                input_path = ROOT / row["input_path"]
                assert hashlib.sha256(input_path.read_bytes()).hexdigest() == row["input_sha256"]
                raw = InstanceData.from_dict(json.loads(input_path.read_text(encoding="utf-8")))
            else:
                raw = generate_instance(make_set_config(scale_spec(row["scale"])), row["seed"] % 1000, row["seed"])
            data = preprocess(raw)
            payload = json.loads((ROOT / row["solution_path"]).read_text(encoding="utf-8"))
            allowed = {f.name for f in fields(SolutionResult)}
            saved = SolutionResult(**{k: v for k, v in payload.items() if k in allowed})
            validate_primal(data, saved)
            assert abs(saved.objective - row["ub"]) < 1e-4
            audits.append({"scale": row["scale"], "seed": row["seed"], "passed": True})
    table.sort(key=lambda r: (r["scale"], r["seed"]))
    columns = list(dict.fromkeys(key for row in table for key in row))
    if table:
        with (folder / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=columns)
            writer.writeheader()
            writer.writerows(table)
    groups = {}
    for scale in ("medium", "large"):
        group = [r for r in rows if r["scale"] == scale]
        group_table = [r for r in table if r["scale"] == scale]
        paired_roots = [r for r in group_table if r["root_irmp_ub"] is not None and r["root_cgfo_ub"] is not None]
        root_reductions = [100 * (r["root_irmp_ub"] - r["root_cgfo_ub"]) / abs(r["root_irmp_ub"]) for r in paired_roots]
        groups[scale] = {
            "expected": sum(c["scale"] == scale for c in manifest["cases"]),
            "completed": len(group), "target_reached": sum(target_reached(r) for r in group),
            "status_counts": dict(Counter(r["status"] for r in group)),
            "optimal": sum(r["status"] == "optimal" for r in group),
            "time_limit": sum(r["status"] == "time_limit" for r in group),
            "errors": sum(r["status"] in ("error", "watchdog_timeout") for r in group),
            "unresolved_cases": sum(r.get("nodes_unresolved", 0) > 0 for r in group),
            "saved_primal_validated": sum(r.get("primal_validated", False) for r in group),
            "mean_runtime_sec": (statistics.mean(r["runtime_sec"] for r in group if r.get("runtime_sec") is not None)
                                 if any(r.get("runtime_sec") is not None for r in group) else None),
            "total_nodes": sum(r.get("nodes", 0) for r in group),
            "root_cgfo_improvements": sum(r["root_cgfo_ub"] is not None for r in group_table),
            "paired_root_irmp_cgfo": len(paired_roots),
            "median_root_ub_reduction_pct": statistics.median(root_reductions) if root_reductions else None,
            "repair_calls": sum(r.get("primal_stats", {}).get("cgfo_repair_calls", 0) for r in group),
            "repair_improvements": sum(sum(x["source"] == "cgfo_repair" for x in r.get("primal_history", [])) for r in group),
        }
        for prefix in ("original_baseline", "corrected_baseline"):
            matched = [r for r in group_table if r.get(prefix + "_ub") is not None and r.get("ub") is not None]
            groups[scale][prefix + "_ub_comparison"] = {
                "better": sum(r["ub"] < r[prefix + "_ub"] - 1e-4 for r in matched),
                "equal": sum(abs(r["ub"] - r[prefix + "_ub"]) <= 1e-4 for r in matched),
                "worse": sum(r["ub"] > r[prefix + "_ub"] + 1e-4 for r in matched),
                "note": "Historical controls differ; this is descriptive, not an isolated UB-effect estimate.",
            }
    summary = {"expected": len(manifest["cases"]), "completed": len(rows),
               "all_complete": len(rows) == len(manifest["cases"]), "groups": groups,
               "audit_count": len(audits), "audits": audits,
               "gap_definition": "100*(UB-LB)/abs(UB), with certified B&P frontier LB",
               "timing_note": (f"{manifest['workers']} case process(es), {manifest['gurobi_threads_per_case']} solver threads per case; historical controls differ.")}
    (folder / "analysis.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "result/diagnostics/ub_adaptive_serial_v1")
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    analyze(args.output.resolve(), args.audit)
