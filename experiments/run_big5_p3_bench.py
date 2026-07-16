"""Benchmark all 10 big5 instances with EBRP after P3 (DP optimizations)."""
import sys, time, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "result" / "big5_ebrp_p3_bench.json"


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

    # Compare with previous results
    prev_files = {
        "original": ROOT / "result" / "big5_cg_compare_v2.json",
        "v2": ROOT / "result" / "big5v2_cg_compare.json",
        "p12": ROOT / "result" / "big5_ebrp_p12_bench.json",
    }
    prev = {}
    for label, pf in prev_files.items():
        if pf.exists():
            for pr in json.loads(pf.read_text()):
                key = (pr["name"], pr.get("mode", "ebrp"), label)
                prev[key] = pr

    # Gurobi baselines
    gb = {}
    for pf in [ROOT / "result" / "big5_cg_compare_v2.json", ROOT / "result" / "big5v2_cg_compare.json"]:
        if pf.exists():
            for pr in json.loads(pf.read_text()):
                if pr.get("mode") == "im_gurobi":
                    gb[pr["name"]] = pr

    print(f"\n{'='*95}")
    print(f"{'Instance':<18} {'LRMP':>10} {'Iter':>5} {'New_s':>7} {'Prev_s':>7} {'Speedup':>8} {'Ref_LRMP':>10} {'Diff':>8}")
    print(f"{'-'*95}")

    total_new = 0.0
    total_old = 0.0
    for r in results:
        name = r["name"]
        old = prev.get((name, "ebrp", "p12"))
        old_s = old["elapsed_s"] if old else 0
        speedup = old_s / r["elapsed_s"] if old_s > 0 else 0
        total_new += r["elapsed_s"]
        total_old += old_s
        ref = gb.get(name, {})
        ref_lrmp = ref.get("lrmp", 0)
        diff = r["lrmp"] - ref_lrmp
        print(f"{name:<18} {r['lrmp']:>10.2f} {r['iterations']:>5} {r['elapsed_s']:>7.1f} {old_s:>7.1f} {speedup:>8.1f}x {ref_lrmp:>10.2f} {diff:>+8.2f}")

    if total_old > 0:
        print(f"{'-'*95}")
        print(f"{'TOTAL':<18} {'':>10} {'':>5} {total_new:>7.1f} {total_old:>7.1f} {total_old/total_new:>8.1f}x")


if __name__ == "__main__":
    main()
