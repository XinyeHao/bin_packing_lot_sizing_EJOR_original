"""Diagnostic: run_big5_02 slowness + run_big5_03 LRMP deviation fix test."""
import sys, time, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def run_one(name, pricing_mode):
    from preprocessor.preprocess import load_instance, preprocess
    from model_and_algo.column_generation import ColumnGenerationSolver

    data = preprocess(load_instance(name))
    t0 = time.perf_counter()

    if pricing_mode == "im_gurobi":
        kwargs = {
            "pricing_mode": "im_gurobi",
            "use_gurobi_init": False,
            "im_pricing_time_limit": 30.0,
        }
    else:
        kwargs = {
            "pricing_mode": "ebrp",
            "use_gurobi_init": False,
            "ebrp_force": True,
            "ebrp_time_limit": 30.0,
        }

    cg = ColumnGenerationSolver(
        data,
        max_iterations=5000,
        verbose=True,
        progress_interval_sec=5.0,
        time_limit=600.0,
        pricing_workers=0,
        pricing_inprocess=True,
        **kwargs,
    )
    lrmp, rcs, cols = cg.run()
    elapsed = time.perf_counter() - t0
    iterations = len(rcs) if rcs else 0
    return {
        "name": name,
        "mode": pricing_mode,
        "lrmp": round(lrmp, 2),
        "iterations": iterations,
        "elapsed_s": round(elapsed, 1),
    }

results = []
for name in ["run_big5_02", "run_big5_03"]:
    print(f"\n{'='*60}")
    print(f"=== {name} ===")
    print(f"{'='*60}")

    r_g = run_one(name, "im_gurobi")
    print(f"\n  >>> Gurobi: LRMP={r_g['lrmp']:.2f}  iter={r_g['iterations']}  {r_g['elapsed_s']:.1f}s")
    results.append(r_g)

    r_e = run_one(name, "ebrp")
    print(f"\n  >>> EBRP:   LRMP={r_e['lrmp']:.2f}  iter={r_e['iterations']}  {r_e['elapsed_s']:.1f}s")
    results.append(r_e)

    delta = r_g["lrmp"] - r_e["lrmp"]
    speedup = r_g["elapsed_s"] / r_e["elapsed_s"] if r_e["elapsed_s"] > 0 else 0
    print(f"\n  Delta={delta:+.2f}  speedup={speedup:.1f}x")

ROOT.joinpath("result").mkdir(parents=True, exist_ok=True)
ROOT.joinpath("result/big5_diag_verify.json").write_text(json.dumps(results, indent=2))
print(f"\nResults saved.")
