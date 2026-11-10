"""Small-scale lower-bound comparison: compact IM-LP versus pure root DW-CG.

Both relaxations use the common-unload process assumption: all items loaded in
one autoclave batch enter inventory together when the longest curing operation
in that batch finishes.  The CG side intentionally disables all LRMP cuts, so
the comparison isolates the strength of the Dantzig--Wolfe reformulation.

The script writes results after every seed and safely resumes completed rows.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import multiprocessing as mp
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
from configuration import SetConfig
from experiments.paper_experiment_config import (
    MODEL_VERSION,
    PROCESS_ASSUMPTION,
    generation_metadata,
    make_set_config,
    scale_spec,
)
from instance_generator.generator import generate_instance
from model_and_algo.column_generation import ColumnGenerationSolver
from model_and_algo.im_model import solve_lim
from preprocessor.preprocess import preprocess

def _cleanup(*objects: object) -> None:
    for obj in objects:
        try:
            if hasattr(obj, "dispose"):
                obj.dispose()
        except Exception:
            pass
    gc.collect()
    try:
        gp.disposeDefaultEnv()
    except Exception:
        pass
    gc.collect()


def _im_lp(data, limit: float) -> dict:
    t0 = time.perf_counter()
    result = solve_lim(data, time_limit=limit, verbose=False)
    return {
        "status": result.status,
        "lower_bound": result.lower_bound,
        "runtime_sec": round(time.perf_counter() - t0, 3),
    }


def _im_lp_worker(seed: int, T: int, I: int, limit: float, output: object) -> None:
    """Run IM-LP in a separate process so its full wall time is controllable."""
    raw = data = None
    try:
        cfg.apply_bench_generation()
        raw = generate_instance(SetConfig("small_cg", T, 5, 0, I, 5), seed % 1000, seed)
        data = preprocess(raw)
        output.put({"ok": True, **_im_lp(data, limit)})
    except Exception as exc:
        output.put({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    finally:
        _cleanup(data, raw)


def _im_lp_with_wall_limit(seed: int, T: int, I: int, limit: float) -> dict:
    """Cap generation, construction, and optimisation of IM-LP together."""
    ctx = mp.get_context("spawn")
    output = ctx.Queue(maxsize=1)
    worker = ctx.Process(target=_im_lp_worker, args=(seed, T, I, limit, output))
    start = time.perf_counter()
    worker.start()
    worker.join(timeout=limit)
    elapsed = round(time.perf_counter() - start, 3)
    if worker.is_alive():
        worker.terminate()
        worker.join(timeout=10)
        output.close()
        return {"status": "wall_time_limit", "lower_bound": None, "runtime_sec": elapsed}
    try:
        message = output.get(timeout=5)
    except Exception:
        message = {"ok": False, "error": "IM-LP worker ended without returning a result."}
    finally:
        output.close()
    if not message.get("ok"):
        return {"status": "worker_error", "lower_bound": None, "runtime_sec": elapsed, "note": message.get("error")}
    message.pop("ok", None)
    message["runtime_sec"] = elapsed
    return message


def _root_dw_cg(data, limit: float, max_iterations: int, pricing_limit: float) -> dict:
    """Solve only the root DW relaxation with exact Gurobi pricing and no cuts."""
    cg = ColumnGenerationSolver(
        data,
        max_iterations=max_iterations,
        time_limit=limit,
        pricing_mode="im_gurobi",
        use_gurobi_init=True,
        im_pricing_time_limit=pricing_limit,
        pricing_inprocess=True,
        skip_cross_check=True,
        lrmp_cuts="none",
        verbose=True,
        progress_interval_sec=30.0,
    )
    t0 = time.perf_counter()
    try:
        returned_value, rcs, columns = cg.run()
        # Only a converged CG run equals the pure DW LP optimum.  During an
        # interrupted run, last_valid_lb remains a valid Lagrangian bound but
        # is deliberately kept separate from the DW-LP comparison column.
        dw_lp = cg.lrmp_obj if cg.converged else None
        return {
            "status": "converged" if cg.converged else (cg.stop_reason or "not_converged"),
            "dw_lp_lower_bound": dw_lp,
            "last_valid_lower_bound": getattr(cg, "last_valid_lb", None),
            "runtime_sec": round(time.perf_counter() - t0, 3),
            "iterations": len(rcs),
            "columns": sum(len(v) for v in columns.values()),
            "last_min_rc": cg.last_min_rc,
        }
    finally:
        try:
            if cg._model is not None:
                cg._model.dispose()
        except Exception:
            pass


def _root_dw_worker(
    scale_name: str, seed: int, limit: float, max_iterations: int,
    pricing_limit: float, output: object,
) -> None:
    """Run root CG in isolation so preparation time counts toward its budget."""
    raw = data = None
    try:
        cfg.apply_bench_generation()
        spec = scale_spec(scale_name)
        raw = generate_instance(make_set_config(spec), seed % 1000, seed)
        data = preprocess(raw)
        output.put({"ok": True, **_root_dw_cg(data, limit, max_iterations, pricing_limit)})
    except Exception as exc:
        output.put({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    finally:
        _cleanup(data, raw)


def _root_dw_with_wall_limit(
    scale_name: str, seed: int, limit: float, max_iterations: int, pricing_limit: float,
) -> dict:
    """Cap generation, preprocessing, master construction, and CG together."""
    ctx = mp.get_context("spawn")
    output = ctx.Queue(maxsize=1)
    worker = ctx.Process(
        target=_root_dw_worker,
        args=(scale_name, seed, limit, max_iterations, pricing_limit, output),
    )
    start = time.perf_counter()
    worker.start()
    worker.join(timeout=limit)
    elapsed = round(time.perf_counter() - start, 3)
    if worker.is_alive():
        worker.terminate()
        worker.join(timeout=10)
        output.close()
        return {
            "status": "wall_time_limit", "dw_lp_lower_bound": None,
            "last_valid_lower_bound": None, "runtime_sec": elapsed,
            "iterations": None, "columns": None, "last_min_rc": None,
        }
    try:
        message = output.get(timeout=5)
    except Exception:
        message = {"ok": False, "error": "DW-CG worker ended without returning a result."}
    finally:
        output.close()
    if not message.get("ok"):
        return {
            "status": "worker_error", "dw_lp_lower_bound": None,
            "last_valid_lower_bound": None, "runtime_sec": elapsed,
            "iterations": None, "columns": None, "last_min_rc": None,
            "note": message.get("error"),
        }
    message.pop("ok", None)
    message["runtime_sec"] = elapsed
    return message


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare compact IM-LP and pure root DW-CG lower bounds under the common-unload assumption."
    )
    parser.add_argument("--scale", choices=("small", "medium", "large"), default="small")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Defaults to the 15 predeclared seeds of --scale.")
    parser.add_argument("--time-limit", type=float, default=None, help="Defaults to the predeclared time limit of --scale.")
    parser.add_argument("--max-iterations", type=int, default=500)
    parser.add_argument(
        "--pricing-time-limit", type=float, default=90.0,
        help="Maximum seconds for one Gurobi pricing MIP; an interrupted pricer never certifies DW convergence.",
    )
    parser.add_argument("--output-prefix", default=None)
    return parser.parse_args()


def _write(rows: list[dict], args: argparse.Namespace, json_path: Path, csv_path: Path) -> None:
    completed = [r for r in rows if not r.get("error")]
    comparable = [r for r in completed if r.get("dw_cg_converged") and r.get("im_lp_lb") is not None]
    improvements = [r["dw_over_im_lp_pct"] for r in comparable if r.get("dw_over_im_lp_pct") is not None]
    payload = {
        "experiment": "Lower-bound comparison: compact IM-LP versus pure root DW-CG",
        "process_assumption": PROCESS_ASSUMPTION,
        "model_version": MODEL_VERSION,
        "generation_settings": generation_metadata(),
        "scale": {"name": args.spec.name, "T": args.spec.periods, "I": args.spec.items, "U": 5, "M": 5},
        "seeds": args.seeds,
        "time_limit_sec_per_method": args.time_limit,
        "im_lp_limit_interpretation": "The wall-clock cap includes generation, preprocessing, model construction, and Gurobi optimisation.",
        "max_cg_iterations": args.max_iterations,
        "pricing_time_limit_sec": args.pricing_time_limit,
        "dw_settings": {
            "pricing": "im_gurobi (exact)",
            "root_only": True,
            "lrmp_cuts": "none",
            "interpretation": "A converged DW-CG value is the pure DW LP lower bound; interrupted runs report only last_valid_lower_bound.",
        },
        "summary": {
            "completed": len(completed),
            "dw_converged": len(comparable),
            "mean_dw_over_im_lp_pct": (sum(improvements) / len(improvements)) if improvements else None,
        },
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = [
        "seed", "T", "I", "im_lp_status", "im_lp_lb", "im_lp_time_sec",
        "dw_cg_status", "dw_cg_converged", "dw_cg_lp_lb", "dw_cg_last_valid_lb",
        "dw_cg_time_sec", "dw_cg_iterations", "dw_cg_columns", "dw_cg_last_min_rc",
        "dw_minus_im_lp", "dw_over_im_lp_pct", "error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = _parse_args()
    args.spec = scale_spec(args.scale)
    if args.seeds is None:
        args.seeds = list(args.spec.seeds)
    if args.time_limit is None:
        args.time_limit = args.spec.time_limit_sec
    if args.output_prefix is None:
        args.output_prefix = f"exp1_lb_{args.scale}_15"
    cfg.apply_bench_generation()
    result_dir = ROOT / "result"
    result_dir.mkdir(exist_ok=True)
    json_path = result_dir / f"{args.output_prefix}.json"
    csv_path = result_dir / f"{args.output_prefix}.csv"

    old_rows: dict[int, dict] = {}
    if json_path.exists():
        try:
            old = json.loads(json_path.read_text(encoding="utf-8"))
            if (
                old.get("process_assumption") == PROCESS_ASSUMPTION
                and old.get("model_version") == MODEL_VERSION
            ):
                old_scale = old.get("scale", {})
                compatible = (
                    old_scale.get("name") == args.spec.name
                    and old.get("time_limit_sec_per_method") == args.time_limit
                    and old.get("max_cg_iterations") == args.max_iterations
                    and old.get("pricing_time_limit_sec") == args.pricing_time_limit
                )
                if compatible:
                    old_rows = {
                        int(r["seed"]): r for r in old.get("rows", [])
                        if not r.get("error") and r.get("dw_cg_status") is not None
                    }
        except (OSError, ValueError, KeyError, TypeError):
            pass
    rows = [old_rows[s] for s in args.seeds if s in old_rows]

    T, I = args.spec.periods, args.spec.items
    print(f"DW lower-bound comparison: T={T}, I={I}, seeds={args.seeds}", flush=True)
    print(f"Per method limit={args.time_limit:.0f}s; pure DW root CG uses exact Gurobi pricing and no cuts.", flush=True)
    for index, seed in enumerate(args.seeds, start=1):
        if seed in old_rows:
            print(f"[{index}/{len(args.seeds)}] seed={seed}: reuse completed row", flush=True)
            continue
        row = {"seed": seed, "T": T, "I": I, "error": None}
        try:
            print(f"[{index}/{len(args.seeds)}] seed={seed}: IM-LP", flush=True)
            im = _im_lp_with_wall_limit(seed, T, I, args.time_limit)
            row.update({
                "im_lp_status": im["status"], "im_lp_lb": im["lower_bound"],
                "im_lp_time_sec": im["runtime_sec"],
            })
            print(f"  IM-LP={im['lower_bound']} status={im['status']}, time={im['runtime_sec']:.2f}s", flush=True)

            print(f"[{index}/{len(args.seeds)}] seed={seed}: root DW-CG", flush=True)
            dw = _root_dw_with_wall_limit(
                args.scale, seed, args.time_limit, args.max_iterations, args.pricing_time_limit,
            )
            row.update({
                "dw_cg_status": dw["status"], "dw_cg_converged": dw["dw_lp_lower_bound"] is not None,
                "dw_cg_lp_lb": dw["dw_lp_lower_bound"],
                "dw_cg_last_valid_lb": dw["last_valid_lower_bound"],
                "dw_cg_time_sec": dw["runtime_sec"], "dw_cg_iterations": dw["iterations"],
                "dw_cg_columns": dw["columns"], "dw_cg_last_min_rc": dw["last_min_rc"],
            })
            if row["dw_cg_converged"] and row["im_lp_lb"] is not None:
                delta = row["dw_cg_lp_lb"] - row["im_lp_lb"]
                row["dw_minus_im_lp"] = delta
                denom = max(abs(row["im_lp_lb"]), 1e-6)
                row["dw_over_im_lp_pct"] = 100.0 * delta / denom
            else:
                row["dw_minus_im_lp"] = None
                row["dw_over_im_lp_pct"] = None
            print(
                f"  DW-CG={row['dw_cg_lp_lb']} status={row['dw_cg_status']} "
                f"time={row['dw_cg_time_sec']:.2f}s improvement={row['dw_over_im_lp_pct']}",
                flush=True,
            )
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            print(f"  FAILED: {row['error']}", flush=True)
        rows.append(row)
        _write(rows, args, json_path, csv_path)

    print(f"Finished. JSON: {json_path}; CSV: {csv_path}", flush=True)


if __name__ == "__main__":
    main()
