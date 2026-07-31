"""Test 5 instances (T=35/I=50/M=5/LT=(8,9,10)): CG+EBRP vs IM(900s)."""
import sys, time, gc, os, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ["GRB_CS_QUIET"] = "1"
sys.stderr = open(os.devnull, "w")

import configuration as cfg
cfg.apply_bench_generation()

from configuration import SetConfig
from instance_generator.generator import generate_instance
from preprocessor.preprocess import preprocess
from model_and_algo.column_generation import ColumnGenerationSolver
from model_and_algo.solver import solve

SEEDS = [101, 202, 303, 404, 505]
scfg = SetConfig("small_cg", 35, 5, 0, 50, 5)

results = []

for idx, seed in enumerate(SEEDS, 1):
    gc.collect(); time.sleep(2.0)
    print(f"\n{'='*60}")
    print(f"[{idx}/5] seed={seed}", flush=True)

    inst = generate_instance(scfg, 1, seed=seed)
    data = preprocess(inst)
    T = len(data.periods); I = len(data.cured_items); M = len(data.machines)
    total = sum(len(data.items_by_machine[m]) for m in range(M))
    print(f"  T={T} I={I} M={M} items/m={total/M:.1f}", flush=True)

    vols = [data.v_i[i] for i in range(I)]
    total_demand = sum(
        data.v_i[i] * statistics.mean([data.lt_it[i][t] for t in range(T) if data.lt_it[i][t] > 0])
        for i in range(I)
    )
    total_capacity = sum(data.q_m[m] for m in range(M)) * T
    util = total_demand / total_capacity * 100
    print(f"  v_i: [{min(vols):.0f},{max(vols):.0f}] util={util:.1f}%", flush=True)

    # --- CG+EBRP ---
    gc.collect(); time.sleep(1.0)
    print(f"  CG+EBRP running...", flush=True)
    t0 = time.perf_counter()
    cg = ColumnGenerationSolver(
        data, pricing_mode="ebrp", pricing_inprocess=True,
        use_gurobi_init=False, time_limit=600, skip_cross_check=True,
    )
    lrmp, _, _ = cg.run()
    t_ebrp = time.perf_counter() - t0
    conv = cg.converged; nit = len(cg.iteration_times)
    print(f"  CG+EBRP: LRMP={lrmp:.1f} iter={nit} conv={conv} time={t_ebrp:.1f}s", flush=True)
    del cg; gc.collect()

    # --- IM 900s ---
    gc.collect(); time.sleep(1.0)
    print(f"  IM 900s running...", flush=True)
    t0 = time.perf_counter()
    result = solve(data, method="im", time_limit=900, mip_gap=0.0, verbose=False)
    t_im = time.perf_counter() - t0
    gap = f"{result.mip_gap*100:.2f}%" if result.mip_gap else "N/A"
    print(f"  IM 900s: obj={result.objective} lb={result.lower_bound} gap={gap} status={result.status} time={t_im:.1f}s", flush=True)

    results.append({
        "seed": seed, "util": util,
        "lrmp": lrmp, "t_ebrp": t_ebrp, "nit": nit, "conv": conv,
        "im_obj": result.objective, "im_lb": result.lower_bound,
        "im_gap": gap, "im_status": result.status, "t_im": t_im,
    })

# === Summary ===
print(f"\n{'='*60}")
print(f"SUMMARY (T={T} I={I} M={M} LT=(8,9,10))")
print(f"{'='*60}")
print(f"{'seed':>5}  {'util%':>6}  {'EBRP_s':>8}  {'EBRP_iter':>9}  {'conv':>5}  {'LRMP':>10}  {'IM_obj':>10}  {'IM_gap':>8}  {'IM_s':>7}")
for r in results:
    print(f"{r['seed']:>5}  {r['util']:>6.1f}  {r['t_ebrp']:>8.1f}  {r['nit']:>9}  {str(r['conv']):>5}  {r['lrmp']:>10.1f}  {r['im_obj']:>10.1f}  {r['im_gap']:>8}  {r['t_im']:>7.1f}")

print(f"\nDone.", flush=True)
