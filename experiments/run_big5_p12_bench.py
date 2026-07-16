"""Test all 10 big5 instances with EBRP after P1+P2 optimization."""
import sys, time, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "result" / "big5_ebrp_p12_bench.json"


def run_one(name):
    from preprocessor.preprocess import load_instance, preprocess
    from model_and_algo.column_generation import ColumnGenerationSolver

    data = preprocess(load_instance(name))
    t0 = time.perf_counter()

    cg = ColumnGenerationSolver(
        data,
        max_iterations=5000,
        verbose=False,
        progress_interval_sec=0.0,
        time_limit=600.0,
        pricing_mode="ebrp",
        use_gurobi_init=False,
        ebrp_force=True,
        ebrp_time_limit=30.0,
        pricing_workers=0,
        pricing_inprocess=True,
    )
    lrmp, rcs, cols = cg.run()
    elapsed = time.perf_counter() - t0
    iterations = len(rcs) if rcs else 0
    return {
        "name": name,
        "lrmp": round(lrmp, 2),
        "iterations": iterations,
        "elapsed_s": round(elapsed, 1),
        "ni": len(data.cured_items),
        "nt": len(data.periods),
        "nm": len(data.machines),
    }


def main():
    INSTANCES = [f"run_big5_{i:02d}" for i in range(1, 6)] + [
        f"run_big5v2_{i:02d}" for i in range(1, 6)
    ]
    results = []

    for name in INSTANCES:
        print(f"--- {name} ---", flush=True)
        r = run_one(name)
        print(f"  LRMP={r['lrmp']:.2f}  iter={r['iterations']}  {r['elapsed_s']:.1f}s", flush=True)
        results.append(r)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {OUT}")

    # Compare with previous Gurobi results
    prev_big5 = ROOT / "result" / "big5_cg_compare_v2.json"
    prev_big5v2 = ROOT / "result" / "big5v2_cg_compare.json"

    print(f"\n{'='*85}")
    print(f"{'Instance':<18} {'LRMP':>10} {'Iter':>5} {'EBRP_s':>8} {'Prev_EBRP_s':>12} {'Speedup':>8}")
    print(f"{'-'*85}")

    prev_results = {}
    for pf in [prev_big5, prev_big5v2]:
        if pf.exists():
            for pr in json.loads(pf.read_text()):
                if pr["mode"] == "ebrp":
                    prev_results[pr["name"]] = pr

    total_new = 0.0
    total_old = 0.0
    for r in results:
        name = r["name"]
        prev = prev_results.get(name)
        old_s = prev["elapsed_s"] if prev else 0
        speedup = old_s / r["elapsed_s"] if r["elapsed_s"] > 0 and old_s > 0 else 0
        total_new += r["elapsed_s"]
        total_old += old_s
        print(f"{name:<18} {r['lrmp']:>10.2f} {r['iterations']:>5} {r['elapsed_s']:>8.1f} {old_s:>12.1f} {speedup:>8.1f}x")

    if total_old > 0:
        print(f"{'-'*85}")
        print(f"{'TOTAL':<18} {'':>10} {'':>5} {total_new:>8.1f} {total_old:>12.1f} {total_old/total_new:>8.1f}x")


if __name__ == "__main__":
    main()
