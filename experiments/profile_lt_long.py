"""LT=(6,7,8), T=60, I=42, M=5 — CG(EBRP), CG(Gurobi), IM (gap=0, 600s)."""
import sys, time, gc, os, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ["GRB_CS_QUIET"] = "1"
sys.stderr = open(os.devnull, "w")

import configuration as cfg
cfg.LEAD_TIME_CHOICES = (8, 9, 10)                     # ← 拉长 LT
cfg.WINDOW_GENERATION_BY_SET["small_cg"] = {            # ← 加宽窗口
    "length_frac": (0.45, 0.65), "rel_frac": (0.03, 0.25), "min_length": 6
}

from configuration import SetConfig
from instance_generator.generator import generate_instance
from preprocessor.preprocess import preprocess
from model_and_algo.column_generation import ColumnGenerationSolver
from model_and_algo.solver import solve
from model_and_algo.pricing_ebrp import _item_start_feasible


def clean():
    time.sleep(1.0); gc.collect(); gc.collect(); time.sleep(0.5)


def analyze(data):
    T = len(data.periods); I = len(data.cured_items); M = len(data.machines)
    vols = [data.v_i[i] for i in range(I)]
    lt_vals = []
    for i in range(I):
        for t in range(T):
            if data.lt_it[i][t] > 0:
                lt_vals.append(data.lt_it[i][t])
    earliest, latest, widths = [], [], []
    end_risk = 0
    for i in range(I):
        fea = [t for t in range(T) if _item_start_feasible(data, i, t, T)]
        if fea:
            earliest.append(fea[0]); latest.append(fea[-1])
            widths.append(fea[-1] - fea[0] + 1)
            if fea[-1] + data.lt_it[i][fea[-1]] > T - 3:
                end_risk += 1
    total_demand = sum(
        data.v_i[i] * statistics.mean([data.lt_it[i][t] for t in range(T) if data.lt_it[i][t] > 0])
        for i in range(I)
    )
    total_capacity = sum(data.q_m[m] for m in range(M)) * T
    util_theory = total_demand / total_capacity * 100

    print(f"  v_i: [{min(vols):.0f},{max(vols):.0f}] mean={statistics.mean(vols):.1f}", flush=True)
    print(f"  lt:  [{min(lt_vals)},{max(lt_vals)}] mean={statistics.mean(lt_vals):.1f} med={statistics.median(lt_vals):.0f}", flush=True)
    print(f"  window: E=[{min(earliest)},{max(earliest)}] D=[{min(latest)},{max(latest)}] width_mean={statistics.mean(widths):.0f}", flush=True)
    print(f"  end_risk: {end_risk}/{I}", flush=True)
    print(f"  time_util(theory): {util_theory:.1f}%", flush=True)
    return util_theory


# ═══════════════════════════════════════════════════════════
scfg = SetConfig("small_cg", 35, 5, 0, 50, 5)          # T=35  I=50  M=5  U=5
seed = 888

clean()
inst = generate_instance(scfg, 1, seed=seed)
data = preprocess(inst)
T = len(data.periods); I = len(data.cured_items); M = len(data.machines); U = len(data.configs)
total = sum(len(data.items_by_machine[m]) for m in range(M))
print(f"=== Instance: T={T} I={I} M={M} U={U} items/m={total/M:.1f} ===", flush=True)
analyze(data)

# ═══════════════════════════════════════════════════════════
# 1. CG + EBRP
# ═══════════════════════════════════════════════════════════
clean()
print(f"\n--- 1. CG+EBRP ---", flush=True)
t0 = time.perf_counter()
cg_e = ColumnGenerationSolver(
    data, pricing_mode="ebrp", pricing_inprocess=True,
    use_gurobi_init=False, time_limit=600, skip_cross_check=True,
)
lrmp_e, rcs_e, cols_e = cg_e.run()
t_e = time.perf_counter() - t0
it_e = len(cg_e.iteration_times)
conv_e = cg_e.converged

all_vol_utils, all_items, all_spans = [], [], []
for machine in data.machines:
    m_idx = int(machine[1:])
    for col in cols_e.get(machine, []):
        for (u, t), yv in col.y.items():
            if yv > 0.5:
                bv, bi, bs = 0, 0, 0
                for (item_id, t2), xv in col.x.items():
                    if xv > 0.5 and t2 == t:
                        i_idx = data.item_index.get(item_id)
                        if i_idx is None: continue
                        bv += data.v_i[i_idx]; bi += 1
                        bs = max(bs, data.lt_it[i_idx][t2])
                if bv > 0 and bi > 0:
                    all_vol_utils.append(bv / data.q_m[m_idx] * 100)
                    all_items.append(bi); all_spans.append(bs)

print(f"  EBRP: LRMP={lrmp_e:.1f}  iter={it_e}  conv={conv_e}  time={t_e:.1f}s", flush=True)
if all_vol_utils:
    print(f"  batches: {len(all_vol_utils)}, vol_util mean={statistics.mean(all_vol_utils):.1f}% med={statistics.median(all_vol_utils):.1f}%", flush=True)
    print(f"  items/batch mean={statistics.mean(all_items):.1f}, span mean={statistics.mean(all_spans):.1f}", flush=True)
del cg_e, rcs_e, cols_e
clean()

# ═══════════════════════════════════════════════════════════
# 2. CG + Gurobi (subproblem)
# ═══════════════════════════════════════════════════════════
clean()
print(f"\n--- 2. CG+Gurobi(subproblem) ---", flush=True)
lrmp_g, t_g, conv_g = None, 0, None
try:
    t0 = time.perf_counter()
    cg_g = ColumnGenerationSolver(
        data, pricing_mode="im_gurobi", pricing_inprocess=True,
        use_gurobi_init=False, time_limit=600,
    )
    lrmp_g, rcs_g, cols_g = cg_g.run()
    t_g = time.perf_counter() - t0
    conv_g = cg_g.converged
    print(f"  Gurobi: LRMP={lrmp_g:.1f}  iter={len(cg_g.iteration_times)}  conv={conv_g}  time={t_g:.1f}s", flush=True)
    del cg_g, rcs_g, cols_g
except Exception as ex:
    t_g = time.perf_counter() - t0
    print(f"  Gurobi CG FAILED ({t_g:.0f}s): {type(ex).__name__}: {ex}", flush=True)
clean()

# ═══════════════════════════════════════════════════════════
# 3. IM (direct, MIP gap=0, 600s hard limit)
# ═══════════════════════════════════════════════════════════
clean()
print(f"\n--- 3. IM (MIP gap=0, 600s hard limit) ---", flush=True)
t0 = time.perf_counter()
try:
    result = solve(data, method="im", time_limit=600, mip_gap=0.0, verbose=False)
    t_im = time.perf_counter() - t0
    gap_str = f"{result.mip_gap*100:.2f}%" if result.mip_gap else "N/A"
    print(f"  IM: obj={result.objective}  lb={result.lower_bound}  gap={gap_str}  status={result.status}  time={t_im:.1f}s", flush=True)
    im_obj = result.objective; im_lb = result.lower_bound; im_gap = result.mip_gap; im_status = result.status
except Exception as ex:
    t_im = time.perf_counter() - t0
    print(f"  IM FAILED ({t_im:.0f}s): {type(ex).__name__}: {ex}", flush=True)
    im_obj = im_lb = im_gap = im_status = None

# ═══════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════
print(f"\n=== Summary ===", flush=True)
print(f"  LT choices: (6,7,8)  T={T}  I={I}  M={M}  items/m={total/M:.1f}", flush=True)
print(f"  CG+EBRP:     {t_e:.1f}s  LRMP={lrmp_e:.1f}  iter={it_e}  conv={conv_e}", flush=True)
if lrmp_g is not None:
    print(f"  CG+Gurobi:   {t_g:.1f}s  LRMP={lrmp_g:.1f}  conv={conv_g}", flush=True)
    gap = abs(lrmp_e - lrmp_g) / max(abs(lrmp_g), 0.01) * 100
    print(f"  EBRP LRMP gap: {gap:.2f}%", flush=True)
else:
    print(f"  CG+Gurobi:   FAILED ({t_g:.0f}s)", flush=True)
if im_obj is not None:
    print(f"  IM:          {t_im:.1f}s  obj={im_obj}  lb={im_lb}  {im_status}", flush=True)
print(f"\nDone.", flush=True)
