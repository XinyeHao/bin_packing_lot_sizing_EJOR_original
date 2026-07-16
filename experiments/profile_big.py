"""Quick test: 120T 60I 5M — CG+EBRP vs CG+Gurobi (subproblem).
Memory-clean. Suppresses stderr noise."""
import sys, time, gc, os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ["GRB_CS_QUIET"] = "1"


from configuration import SetConfig
from instance_generator.generator import generate_instance
from preprocessor.preprocess import preprocess
from model_and_algo.column_generation import ColumnGenerationSolver


def clean():
    time.sleep(1.0)
    gc.collect()
    gc.collect()
    time.sleep(0.5)


scfg = SetConfig("small_cg", 120, 5, 0, 60, 5)  # T=120, U=5, I=60, M=5
CG_TIME_LIMIT = 1200.0  # 20 min per CG solve

clean()
inst = generate_instance(scfg, 1, seed=777)
data = preprocess(inst)
T = len(data.periods)
I = len(data.cured_items)
M = len(data.machines)
U = len(data.configs)
total = sum(len(data.items_by_machine[m]) for m in range(M))
print(f"Instance: T={T} I={I} M={M} U={U} items/m={total/M:.1f}", flush=True)

# ── EBRP ──
clean()
print("\n=== CG + EBRP ===", flush=True)
t0 = time.perf_counter()
cg_e = ColumnGenerationSolver(
    data, pricing_mode="ebrp", pricing_inprocess=True,
    time_limit=CG_TIME_LIMIT, skip_cross_check=True, progress_interval_sec=0,
)
lrmp_e, rcs_e, cols_e = cg_e.run()
t_e = time.perf_counter() - t0
it_e = len(cg_e.iteration_times)
conv_e = "Y" if cg_e.converged else "N"
ncol_e = sum(len(v) for v in cols_e.values())
print(f"EBRP: iter={it_e} lrmp={lrmp_e or 0:.1f} conv={conv_e} time={t_e:.1f}s cols={ncol_e}", flush=True)
del cg_e, rcs_e, cols_e
clean()

# ── Gurobi ──
clean()
print("\n=== CG + Gurobi ===", flush=True)
t0 = time.perf_counter()
cg_g = ColumnGenerationSolver(
    data, pricing_mode="im_gurobi", pricing_inprocess=True,
    time_limit=CG_TIME_LIMIT, progress_interval_sec=0,
)
lrmp_g, rcs_g, cols_g = cg_g.run()
t_g = time.perf_counter() - t0
it_g = len(cg_g.iteration_times)
conv_g = "Y" if cg_g.converged else "N"
ncol_g = sum(len(v) for v in cols_g.values())
print(f"Gurobi: iter={it_g} lrmp={lrmp_g or 0:.1f} conv={conv_g} time={t_g:.1f}s cols={ncol_g}", flush=True)
del cg_g, rcs_g, cols_g
clean()

# ── Summary ──
print(f"\n=== Summary ===", flush=True)
if lrmp_e and lrmp_g:
    gap = abs(lrmp_e - lrmp_g) / max(abs(lrmp_g), 0.01) * 100
    speedup = t_g / max(t_e, 0.001)
    print(f"EBRP:     {t_e:.1f}s  iter={it_e}  LRMP={lrmp_e:.1f}  cols={ncol_e}", flush=True)
    print(f"Gurobi:   {t_g:.1f}s  iter={it_g}  LRMP={lrmp_g:.1f}  cols={ncol_g}", flush=True)
    print(f"Speedup:  {speedup:.1f}x", flush=True)
    print(f"LRMP gap: {gap:.2f}%", flush=True)

print("\nDone.", flush=True)
