"""Benchmark: CG+Gurobi vs CG+EBRP on run_big5 instances (post TD-bounds fix)."""
import sys, math, time; from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from preprocessor.preprocess import load_instance, preprocess
from model_and_algo.column_generation import ColumnGenerationSolver

instances = [f"run_big5_{i:02d}" for i in range(1, 6)]
print(f"{'Instance':20s} {'G_LRMP':>10s} {'E_LRMP':>10s} {'D':>8s} {'G_s':>8s} {'E_s':>8s} {'Spd':>8s}")
print("-" * 78)

g_tot = 0.0; e_tot = 0.0
for name in instances:
    try:
        data = preprocess(load_instance(name))
    except Exception as exc:
        print(f"{name:20s} SKIP: {exc}")
        continue

    # CG+Gurobi
    t0 = time.perf_counter()
    cg_g = ColumnGenerationSolver(
        data, pricing_mode="im_gurobi", use_gurobi_init=False,
        im_pricing_time_limit=30.0, max_iterations=5000,
        verbose=False, progress_interval_sec=0.0, time_limit=600.0,
        pricing_workers=0, pricing_inprocess=True,
    )
    g_lrmp = cg_g.run()[0]
    g_t = time.perf_counter() - t0

    # CG+EBRP
    t0 = time.perf_counter()
    cg_e = ColumnGenerationSolver(
        data, pricing_mode="ebrp", use_gurobi_init=False,
        ebrp_force=True, ebrp_time_limit=30.0,
        max_iterations=5000, verbose=False,
        progress_interval_sec=0.0, time_limit=600.0,
        pricing_workers=0, pricing_inprocess=True,
    )
    e_lrmp = cg_e.run()[0]
    e_t = time.perf_counter() - t0

    d = g_lrmp - e_lrmp
    spd = g_t / e_t if e_t > 0 else 0
    g_tot += g_t; e_tot += e_t
    print(f"{name:20s} {g_lrmp:>10.2f} {e_lrmp:>10.2f} {d:>+8.2f} {g_t:>8.1f} {e_t:>8.1f} {spd:>7.1f}x")

print("-" * 78)
spd_tot = g_tot / e_tot if e_tot > 0 else 0
print(f"Total:   Gurobi={g_tot:.1f}s  EBRP={e_tot:.1f}s  Speedup={spd_tot:.1f}x")
