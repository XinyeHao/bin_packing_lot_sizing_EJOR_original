"""Run all 15 medium + 15 large cases with adaptive primal heuristics.

Cases run sequentially by default, using all host solver threads, and keep a log, complete primal
solution, bounds, heuristic history and source/config hashes. Historical
results are read for context only, never overwritten or treated as an isolated
upper-bound ablation. --resume requires the same sources and protocol.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from experiments.paper_experiment_config import (
    MODEL_VERSION, PROCESS_ASSUMPTION, generation_metadata, make_set_config, scale_spec,
)


def source_hashes():
    paths = [*(p for folder in ("model_and_algo", "common", "preprocessor", "instance_generator", "checkers")
               for p in (ROOT / folder).glob("*.py")), ROOT / "configuration.py",
             ROOT / "experiments/paper_experiment_config.py",
             Path(__file__).resolve()]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths)}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def historical(scale, seed):
    paths = [ROOT / f"result/rf_{scale}_{seed}.json",
             ROOT / f"research_updates/bp_fixed_medium_large/result/rf_{scale}_{seed}.json"]
    return [{"path": str(p.relative_to(ROOT)), "result": json.loads(p.read_text(encoding="utf-8"))}
            for p in paths if p.exists()]


def run_case(scale, seed, folder, strategy, override_limit=None, threads=None):
    import gurobipy as gp
    import configuration as cfg
    from instance_generator.generator import generate_instance
    from preprocessor.preprocess import preprocess
    from model_and_algo.branch_and_price import BranchAndPriceSolver
    from model_and_algo.primal_heuristics import PrimalHeuristicConfig, validate_primal
    from model_and_algo.pricing_ebrp import EBRP_TIME

    threads = threads if threads is not None else (os.cpu_count() or 1)
    gp.setParam("OutputFlag", 0)
    gp.setParam("Threads", threads)
    gp.setParam("Seed", 0)
    spec = scale_spec(scale)
    limit = spec.time_limit_sec if override_limit is None else override_limit
    cfg.apply_bench_generation()
    raw = generate_instance(make_set_config(spec), seed % 1000, seed)
    data = preprocess(raw)
    options = dict(pricing_mode="ebrp", ebrp_force=True, use_gurobi_init=True,
                   pricing_inprocess=True, skip_cross_check=True, lrmp_cuts="none",
                   max_iterations=120, ebrp_time_limit=20.0, verbose=False)
    config = PrimalHeuristicConfig(threads=threads)
    input_path = folder / "inputs" / f"{scale}_{seed}.json"
    write_json(input_path, raw.to_dict())
    path = folder / f"{scale}_{seed}.json"
    row = {
        "experiment": "adaptive_ub_v1", "scale": scale, "seed": seed,
        "T": spec.periods, "I": spec.items, "machines": len(data.machines),
        "status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_version": MODEL_VERSION, "process_assumption": PROCESS_ASSUMPTION,
        "generation": generation_metadata(), "source_sha256": source_hashes(),
        "input_path": str(input_path.relative_to(ROOT)),
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "upper_bound_strategy": strategy, "primal_config": asdict(config),
        "time_limit_sec": limit, "target_mip_gap": 0.01, "max_nodes": 8000,
        "strict_cg_certification": True, "branch_strategy": "ryan_foster", "cuts": "none",
        "cg_kwargs": options,
        "environment": {"pid": os.getpid(), "python": platform.python_version(),
                        "gurobi": gp.gurobi.version(), "gurobi_threads": threads, "solver_seed": 0,
                        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "numba", "gurobipy")}},
        "historical": historical(scale, seed),
        "comparison_scope": "Historical timings have different pricing/safeguard/thread settings; descriptive, not causal UB ablation.",
    }
    write_json(path, row)
    solver = BranchAndPriceSolver(
        data, max_nodes=8000, time_limit=limit, mip_gap=0.01,
        branch_strategy="ryan_foster", use_node_im=False, root_im=False,
        final_im=False, strict_cg_certification=True, cg_kwargs=options,
        upper_bound_strategy=strategy, primal_config=config,
    )
    started, cpu_started = time.perf_counter(), time.process_time()
    try:
        result = solver.solve()
        solution_path = None
        if result.objective is not None:
            if result.best_solution is None:
                raise RuntimeError("Upper bound has no saved feasible schedule")
            validate_primal(data, result.best_solution)
            if abs(result.best_solution.objective - result.objective) > 1e-4:
                raise RuntimeError("Saved schedule objective differs from global UB")
            solution_path = folder / "solutions" / f"{scale}_{seed}.json"
            write_json(solution_path, result.best_solution.to_dict())
        if result.lower_bound is not None and result.objective is not None:
            if result.lower_bound > result.objective + 1e-4:
                raise RuntimeError("Reported lower bound exceeds feasible UB")
        row.update(
            status=result.status, ub=result.objective, lb=result.lower_bound,
            gap_pct=(100.0 * max(0.0, result.objective - result.lower_bound) / max(abs(result.objective), 1e-9)
                     if result.objective is not None and result.lower_bound is not None else None),
            nodes=result.nodes_explored, nodes_created=result.nodes_created,
            nodes_pruned=result.nodes_pruned, nodes_infeasible=result.nodes_infeasible,
            nodes_cg_failed=result.nodes_cg_failed, nodes_unresolved=result.nodes_unresolved,
            root_lb=result.root_valid_lb, root_lrmp=result.root_lrmp,
            root_converged=result.root_cg_converged, root_stop=result.root_cg_stop,
            runtime_sec=result.runtime_sec, audit_runtime_sec=result.audit_runtime_sec,
            wall_runtime_sec=time.perf_counter() - started,
            cpu_runtime_sec=time.process_time() - cpu_started,
            primal_history=result.primal_history, primal_stats=result.primal_stats,
            profiling={k: float(getattr(solver, "_time_" + k, 0.0))
                       for k in ("cg", "master", "pricing", "irmp", "cgfo")},
            ebrp_accumulated=dict(EBRP_TIME),
            primal_validated=result.best_solution is not None,
            solution_path=str(solution_path.relative_to(ROOT)) if solution_path else None,
        )
    except BaseException as exc:
        row.update(status="error", error=f"{type(exc).__name__}: {exc}",
                   wall_runtime_sec=time.perf_counter() - started)
        traceback.print_exc()
        raise
    finally:
        row["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(path, row)
        print("CASE_RESULT " + json.dumps({k: row.get(k) for k in (
            "scale", "seed", "status", "ub", "lb", "gap_pct", "runtime_sec", "nodes")}), flush=True)
        gp.disposeDefaultEnv()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scales", nargs="+", choices=("medium", "large"), default=["medium", "large"])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--output", type=Path, default=Path("result/diagnostics/ub_adaptive_serial_v1"))
    parser.add_argument("--strategy", choices=("adaptive", "cgfo"), default="adaptive")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--case", nargs=2, metavar=("SCALE", "SEED"))
    parser.add_argument("--time-limit", type=float)
    args = parser.parse_args()
    folder = args.output.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    if args.case:
        scale, seed_text = args.case
        seed = int(seed_text)
        if seed not in scale_spec(scale).seeds:
            parser.error("Seed does not belong to the benchmark scale")
        run_case(scale, seed, folder, args.strategy, args.time_limit, args.threads)
        return
    if args.time_limit is not None:
        parser.error("Full-suite budgets come from paper_experiment_config; --time-limit is for smoke cases only")
    if args.workers < 1 or args.threads < 1:
        parser.error("--workers and --threads must be positive")
    hashes = source_hashes()
    manifest_path = folder / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not args.resume:
            parser.error("Output already has a manifest; use --resume or choose another directory")
        if (previous["source_sha256"] != hashes or previous["strategy"] != args.strategy
                or previous["workers"] != args.workers or previous["gurobi_threads_per_case"] != args.threads):
            parser.error("Sources/strategy/thread protocol changed; use a new output directory")
    else:
        write_json(manifest_path, {
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "strategy": args.strategy, "workers": args.workers, "gurobi_threads_per_case": args.threads,
            "case_order": "medium then large, benchmark seed order" if args.workers == 1 else "longest historical case first",
            "source_sha256": hashes,
            "cases": [{"scale": s, "seed": seed, "time_limit_sec": scale_spec(s).time_limit_sec}
                      for s in args.scales for seed in scale_spec(s).seeds],
        })
        # Freeze reviewable source copies alongside the results.
        for name in hashes:
            dest = folder / "source" / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes((ROOT / name).read_bytes())

    def execute(scale, seed):
        path = folder / f"{scale}_{seed}.json"
        if path.exists():
            row = json.loads(path.read_text(encoding="utf-8"))
            if row.get("status") not in ("running", "error", "watchdog_timeout"):
                if row.get("source_sha256") != hashes:
                    raise RuntimeError(f"Cannot resume {scale}/{seed}: source mismatch")
                return row
            archive = folder / "attempts" / f"{scale}_{seed}_{time.time_ns()}.json"
            write_json(archive, row)
        if source_hashes() != hashes:
            raise RuntimeError("Sources changed during the suite")
        log_path = folder / f"{scale}_{seed}.log"
        command = [sys.executable, "-u", "-B", str(Path(__file__).resolve()),
                   "--case", scale, str(seed), "--output", str(folder), "--strategy", args.strategy,
                   "--threads", str(args.threads)]
        env = {**os.environ, "GRB_CS_QUIET": "1", "PYTHONIOENCODING": "utf-8",
               "OMP_NUM_THREADS": str(args.threads), "OPENBLAS_NUM_THREADS": str(args.threads),
               "MKL_NUM_THREADS": str(args.threads)}
        print(f"START {scale} {seed}", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
            try:
                code = process.wait(timeout=scale_spec(scale).time_limit_sec + 300)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                row = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"scale": scale, "seed": seed}
                row.update(status="watchdog_timeout", finished_at_utc=datetime.now(timezone.utc).isoformat())
                write_json(path, row)
                return row
        row = json.loads(path.read_text(encoding="utf-8"))
        if code != 0 and row.get("status") != "error":
            row.update(status="error", error=f"Subprocess exit {code}")
            write_json(path, row)
        return row

    cases = [(s, seed) for s in args.scales for seed in scale_spec(s).seeds]
    # Start historically long cases first to avoid leaving a single hard case last.
    def effort(case):
        history = historical(*case)
        return history[-1]["result"].get("runtime_sec", 0) if history else 0
    if args.workers > 1:
        cases.sort(key=effort, reverse=True)
    rows = []
    if args.workers == 1:
        for scale, seed in cases:
            row = execute(scale, seed)
            rows.append(row)
            write_json(folder / "summary.json", rows)
            print(f"DONE {len(rows)}/{len(cases)} " + json.dumps({k: row.get(k) for k in (
                "scale", "seed", "status", "ub", "lb", "gap_pct", "runtime_sec", "nodes")}), flush=True)
        print(f"SUITE_COMPLETE {len(rows)} cases; {folder}", flush=True)
        return
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        jobs = {executor.submit(execute, *case): case for case in cases}
        for future in as_completed(jobs):
            row = future.result()
            rows.append(row)
            rows.sort(key=lambda r: (r["scale"], r["seed"]))
            write_json(folder / "summary.json", rows)
            print(f"DONE {len(rows)}/{len(cases)} " + json.dumps({k: row.get(k) for k in (
                "scale", "seed", "status", "ub", "lb", "gap_pct", "runtime_sec", "nodes")}), flush=True)
    print(f"SUITE_COMPLETE {len(rows)} cases; {folder}", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    main()
