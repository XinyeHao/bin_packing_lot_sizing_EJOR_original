"""Rebuild all benchmark root column pools; record IRMP's own MIP gap.

This is a fresh root-only diagnostic, not a reconstruction of old tree calls.
Cases are sequential and use all 16 host Gurobi threads.
"""
import csv
import gc
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ["GRB_CS_QUIET"] = "1"

import gurobipy as gp
import configuration as cfg
from experiments.paper_experiment_config import make_set_config, scale_spec
from experiments.run_adaptive_ub_suite import source_hashes, write_json
from instance_generator.generator import generate_instance
from model_and_algo.branch_and_price import BranchAndPriceSolver, BranchNode
from preprocessor.preprocess import preprocess


def main():
    folder = ROOT / "result/diagnostics/irmp_root_gap_probe_v3"
    folder.mkdir(parents=True, exist_ok=True)
    gp.setParam("OutputFlag", 0)
    gp.setParam("Threads", os.cpu_count() or 1)
    gp.setParam("Seed", 0)
    cfg.apply_bench_generation()
    write_json(folder / "settings.json", {
        "scope": "Fresh root IRMP only; historical non-root MIP gaps unavailable",
        "workers": 1, "threads": os.cpu_count() or 1,
        "irmp_time_limit_sec": 5.0, "target_mip_gap": 0.01,
        "source_sha256": source_hashes(),
    })
    rows = []
    for scale in ("medium", "large"):
        spec = scale_spec(scale)
        for seed in spec.seeds:
            data = preprocess(generate_instance(make_set_config(spec), seed % 1000, seed))
            solver = BranchAndPriceSolver(
                data, branch_strategy="ryan_foster", strict_cg_certification=True,
                cg_kwargs=dict(pricing_mode="ebrp", ebrp_force=True, use_gurobi_init=True,
                               pricing_inprocess=True, skip_cross_check=True, lrmp_cuts="none",
                               max_iterations=120, ebrp_time_limit=20.0, verbose=False),
            )
            solver._start_pricing_pool()
            started = time.perf_counter()
            cg = solver._run_node_cg(BranchNode(node_id=0), time_limit=600.0)
            cg_seconds = time.perf_counter() - started
            cg.solve_rmp_ip(time_limit=5.0, mip_gap=0.01, threads=os.cpu_count() or 1)
            row = dict(cg.last_irmp_diagnostics)
            row.update(scale=scale, seed=seed, root_cg_seconds=cg_seconds,
                       root_cg_converged=cg.converged, root_lp=cg.lrmp_obj)
            rows.append(row)
            write_json(folder / "results.json", rows)
            print(json.dumps({k: row[k] for k in ("scale", "seed", "status", "mip_gap_pct", "mip_runtime_sec")}), flush=True)
            cg._model.dispose()
            solver._stop_pricing_pool()
            del cg, solver
            gc.collect()
    with (folder / "results.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("PROBE_COMPLETE", len(rows), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    main()
