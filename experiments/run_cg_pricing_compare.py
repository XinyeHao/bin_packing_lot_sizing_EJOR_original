"""CG subproblem pricing: Gurobi vs EBRP — compare across med_cg + small_cg instances."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_and_algo.column_generation import solve_column_generation
from preprocessor.preprocess import load_instance, preprocess

INSTANCE_NAMES = [
    "set_med_cg_01",
    "set_med_cg_02",
    "set_med_cg_03",
    "set_small_cg_01",
    "set_small_cg_02",
    "set_small_cg_03",
]


def run_one(inst_name: str, pricing_mode: str) -> dict:
    print(f"\n{'='*60}", flush=True)
    print(f"  {inst_name} | pricing={pricing_mode}", flush=True)
    print(f"{'='*60}", flush=True)
    data = preprocess(load_instance(inst_name))
    t0 = time.perf_counter()

    if pricing_mode == "im_gurobi":
        pricing_kwargs: dict = {"pricing_mode": "im_gurobi", "use_gurobi_init": False}
    else:
        pricing_kwargs = {"pricing_mode": "ebrp", "use_gurobi_init": False, "ebrp_force": True}

    try:
        result = solve_column_generation(
            data,
            max_iterations=5000,
            verbose=True,
            progress_interval_sec=0.0,
            time_limit=600.0,
            ebrp_time_limit=None,
            im_pricing_time_limit=30.0,
            pricing_workers=0,
            **pricing_kwargs,
        )
    except Exception as exc:
        print(f"  ERROR: {exc}", flush=True)
        return {
            "instance": inst_name,
            "pricing": pricing_mode,
            "error": str(exc),
        }
    elapsed = time.perf_counter() - t0
    ex = result.extra or {}
    return {
        "instance": inst_name,
        "pricing": pricing_mode,
        "T": len(data.periods),
        "I": len(data.cured_items),
        "M": len(data.machines),
        "max_Im": max(len(xs) for xs in data.items_by_machine),
        "LRMP": result.lower_bound,
        "iters": ex.get("num_iterations"),
        "converged": ex.get("converged"),
        "stop_reason": ex.get("stop_reason"),
        "last_min_rc": ex.get("last_min_rc"),
        "cols": sum((ex.get("num_columns") or {}).values()),
        "time_sec": round(elapsed, 1),
        "status": result.status,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="cg_pricing_compare")
    parser.add_argument("--instances", type=str, nargs="*")
    args = parser.parse_args()

    names = args.instances or INSTANCE_NAMES
    rows: list[dict] = []

    for name in names:
        for mode in ["im_gurobi", "ebrp"]:
            rows.append(run_one(name, mode))
            # Save incrementally
            out = ROOT / "result" / f"{args.name}.json"
            out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    # Summary
    print("\n" + "=" * 110)
    print(f"{'instance':<20} {'pricing':>10} {'LRMP':>10} {'iters':>5} {'cols':>5} {'conv':>5} {'time':>7}")
    print("-" * 80)
    for r in rows:
        if r.get("error"):
            print(f"{r['instance']:<20} {r['pricing']:>10}  ERROR: {r['error'][:50]}")
        else:
            print(
                f"{r['instance']:<20} {r['pricing']:>10} {r['LRMP']:10.2f} "
                f"{r['iters']:5d} {r['cols']:5d} {str(r['converged']):>5} "
                f"{r['time_sec']:6.1f}s"
            )

    print("\n=== Side-by-side ===")
    print(f"{'instance':<20} {'G_LRMP':>10} {'E_LRMP':>10} {'delta':>8} {'G_time':>7} {'E_time':>7} {'spd':>7}")
    print("-" * 80)
    seen: dict[str, dict] = {}
    for r in rows:
        if r.get("error"):
            continue
        seen.setdefault(r["instance"], {})[r["pricing"]] = r

    for name in names:
        g = seen.get(name, {}).get("im_gurobi")
        e = seen.get(name, {}).get("ebrp")
        if g and e:
            delta = e["LRMP"] - g["LRMP"]
            spd = f"{g['time_sec']/e['time_sec']:.1f}x" if e["time_sec"] > 0 else "N/A"
            mark = "  ***" if abs(delta) > 0.1 else ""
            print(
                f"{name:<20} {g['LRMP']:10.2f} {e['LRMP']:10.2f} "
                f"{delta:+8.4f} {g['time_sec']:6.1f}s {e['time_sec']:6.1f}s "
                f"{spd:>7}{mark}"
            )
        elif g:
            print(f"{name:<20} {g['LRMP']:10.2f} {'FAILED':>10}")
        elif e:
            print(f"{name:<20} {'FAILED':>10} {e['LRMP']:10.2f}")

    print(f"\nSaved: {ROOT / 'result' / f'{args.name}.json'}")


if __name__ == "__main__":
    main()
