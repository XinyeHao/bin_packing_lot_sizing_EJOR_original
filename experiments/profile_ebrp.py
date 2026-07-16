"""Benchmark: CG+EBRP vs CG+Gurobi on small/medium/large instances (3 each).
Memory-clean version. Uses "small_cg" set_type for correct generation params."""
import sys, time, gc, os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ["GRB_CS_QUIET"] = "1"
sys.stderr = open(os.devnull, "w")

from configuration import SetConfig
from instance_generator.generator import generate_instance
from preprocessor.preprocess import preprocess
from model_and_algo.column_generation import ColumnGenerationSolver


def clean_memory():
    time.sleep(1.0)
    gc.collect()
    gc.collect()
    time.sleep(0.5)


# All use small_cg set_type for correct window/generation params.
# items/m ≈ I * avg(2,4) / M ≈ I * 3 / M
SCALES = [
    # name      set_type     T   U end  I   M  → items/m
    ("small",  SetConfig("small_cg", 18, 4, 0, 24, 3)),  # → 24*3/3=24
    ("medium", SetConfig("small_cg", 24, 4, 0, 28, 4)),  # → 28*3/4=21
    ("large",  SetConfig("small_cg", 28, 5, 0, 30, 5)),  # → 30*3/5=18 (more machines, fewer items each)
]
INSTANCES_PER_SCALE = 3
CG_TIME_LIMIT = 600.0
BASE_SEED = 100

HDR = f"{'scale':>6} {'inst':>4} {'method':>9} {'iters':>6} {'lrmp_obj':>12} {'conv':>6} {'time':>8} {'cols':>5}"
print(HDR)
print("-" * 72)

for scale_name, scfg in SCALES:
    for inst_id in range(1, INSTANCES_PER_SCALE + 1):
        seed = BASE_SEED + inst_id
        clean_memory()

        t_gen0 = time.perf_counter()
        inst = generate_instance(scfg, inst_id, seed=seed)
        data = preprocess(inst)
        t_gen = time.perf_counter() - t_gen0
        T = len(data.periods); I = len(data.cured_items)
        M = len(data.machines); U = len(data.configs)
        total = sum(len(data.items_by_machine[m]) for m in range(M))
        print(f"  [info] {scale_name}#{inst_id}: T={T} I={I} M={M} U={U} items/m={total/M:.1f} gen={t_gen:.1f}s", flush=True)

        # ── EBRP ──
        clean_memory()
        cg_e = ColumnGenerationSolver(
            data, pricing_mode="ebrp", pricing_inprocess=True,
            time_limit=CG_TIME_LIMIT, skip_cross_check=True, progress_interval_sec=0,
        )
        t0 = time.perf_counter()
        lrmp_e, rcs_e, cols_e = cg_e.run()
        t_e = time.perf_counter() - t0
        it_e = len(cg_e.iteration_times)
        conv_e = "Y" if cg_e.converged else "N"
        ncol_e = sum(len(v) for v in cols_e.values())
        del cg_e, rcs_e, cols_e
        clean_memory()

        # ── Gurobi（子问题）──
        clean_memory()
        cg_g = ColumnGenerationSolver(
            data, pricing_mode="im_gurobi", pricing_inprocess=True,
            time_limit=CG_TIME_LIMIT, progress_interval_sec=0,
        )
        t0 = time.perf_counter()
        lrmp_g, rcs_g, cols_g = cg_g.run()
        t_g = time.perf_counter() - t0
        it_g = len(cg_g.iteration_times)
        conv_g = "Y" if cg_g.converged else "N"
        ncol_g = sum(len(v) for v in cols_g.values())
        del cg_g, rcs_g, cols_g
        clean_memory()

        del inst, data
        clean_memory()

        # ── Results ──
        print(
            f"{scale_name:>6} {inst_id:>4} {'EBRP':>9} {it_e:>6} {lrmp_e or 0:>12.1f} "
            f"{conv_e:>6} {t_e:>7.1f}s {ncol_e:>5}", flush=True)
        print(
            f"{scale_name:>6} {inst_id:>4} {'Gurobi':>9} {it_g:>6} {lrmp_g or 0:>12.1f} "
            f"{conv_g:>6} {t_g:>7.1f}s {ncol_g:>5}", flush=True)
        if lrmp_e and lrmp_g:
            gap = abs(lrmp_e - lrmp_g) / max(abs(lrmp_g), 0.01) * 100
            speedup = t_g / max(t_e, 0.001)
            print(f"         {'':>9} {'speedup':>6} {speedup:>12.1f}x  {'LRMP_gap':>8} {gap:>.2f}%", flush=True)
            if gap > 0.5:
                print(f"  ** WARNING: LRMP gap={gap:.2f}%", flush=True)
            if conv_e != conv_g:
                print(f"  ** WARNING: convergence mismatch", flush=True)

print("\nDone.", flush=True)
