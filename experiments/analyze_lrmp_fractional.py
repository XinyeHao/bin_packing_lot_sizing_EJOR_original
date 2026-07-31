"""Analyze fractional solution at last LRMP iteration for branch-and-price."""
import sys, time, gc, os, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ["GRB_CS_QUIET"] = "1"

import configuration as cfg
cfg.apply_bench_generation()

from configuration import SetConfig
from instance_generator.generator import generate_instance
from preprocessor.preprocess import preprocess
from model_and_algo.column_generation import ColumnGenerationSolver
from model_and_algo.branch_utils import (
    pick_setup_y0_branch, pick_wip_branch,
    get_rho_duals, FRAC_TOL,
)

SEED = 101
scfg = SetConfig("small_cg", 35, 5, 0, 50, 5)

gc.collect(); time.sleep(1.0)
print(f"Generating instance seed={SEED}...", flush=True)
inst = generate_instance(scfg, 1, seed=SEED)
data = preprocess(inst)
T = len(data.periods); I = len(data.cured_items); M = len(data.machines)
print(f"T={T} I={I} M={M}", flush=True)

# Suppress stderr only during CG (progress monitor heartbeat is noisy)
_stderr_backup = sys.stderr
sys.stderr = open(os.devnull, "w")
try:
    print("Running CG+EBRP (stderr suppressed)...", flush=True)
    t0 = time.perf_counter()
    cg = ColumnGenerationSolver(
        data, pricing_mode="ebrp", pricing_inprocess=True,
        use_gurobi_init=False, time_limit=600, skip_cross_check=True,
    )
    lrmp, _, columns = cg.run()
    t_cg = time.perf_counter() - t0
finally:
    sys.stderr.close()
    sys.stderr = _stderr_backup

print(f"CG done: LRMP={lrmp:.1f} iter={len(cg.iteration_times)} conv={cg.converged} time={t_cg:.1f}s", flush=True)

try:
    assert cg._model is not None and cg._artifacts is not None
    q_vars = cg._artifacts["Q"]

    print(f"\n{'='*70}")
    print(f"FRACTIONAL SOLUTION ANALYSIS: Last LRMP iteration")
    print(f"{'='*70}")

    # 1. Fractional column weights
    print(f"\n--- 1. Fractional Column Weights (q[m,c]) ---")
    frac_cols = []
    sel_cols = []
    for m_idx, machine in enumerate(data.machines):
        for c_idx, col in enumerate(columns[machine]):
            try:
                qv = q_vars[machine][c_idx].X
            except Exception:
                continue
            items_set = set()
            setups = []
            for (item_id, period), xv in col.x.items():
                if xv > 0.5:
                    i_idx = data.item_index.get(item_id)
                    items_set.add(i_idx)
            for (u_name, period), yv in col.y.items():
                if yv > 0.5:
                    setups.append((u_name, period))
            if FRAC_TOL < qv < 1.0 - FRAC_TOL:
                frac_cols.append({
                    "machine": machine, "m_idx": m_idx,
                    "col_idx": c_idx, "q": qv,
                    "items": sorted(items_set),
                    "setups": setups,
                })
            elif qv > 1.0 - FRAC_TOL:
                sel_cols.append({
                    "machine": machine, "m_idx": m_idx,
                    "col_idx": c_idx, "q": qv,
                    "items": sorted(items_set),
                    "setups": setups,
                })

    if frac_cols:
        print(f"  Fractional columns ({len(frac_cols)}):")
        for fc in frac_cols[:20]:
            its = str(fc["items"][:8])
            print(f"    m{fc['m_idx']} col#{fc['col_idx']}  q={fc['q']:.4f}  items={its}  setups={fc['setups']}", flush=True)
        if len(frac_cols) > 20:
            print(f"    ... and {len(frac_cols)-20} more", flush=True)
    else:
        print(f"  All columns with nonzero weight are integral.", flush=True)

    if sel_cols:
        print(f"\n  Selected (q=1) columns ({len(sel_cols)}):")
        for sc in sel_cols[:5]:
            print(f"    m{sc['m_idx']} col#{sc['col_idx']}  items={sc['items']}  setups={sc['setups']}", flush=True)
        if len(sel_cols) > 5:
            print(f"    ... and {len(sel_cols)-5} more", flush=True)

    # 2. Implied Y values
    print(f"\n--- 2. Implied Setup Assignments (y[u,m,t]) ---")
    implied_y = cg.get_implied_y()
    frac_y = []
    int_y1 = []
    for (u_idx, m_idx, t_idx), val in sorted(implied_y.items(), key=lambda kv: -abs(kv[1] - 0.5)):
        if FRAC_TOL < val < 1.0 - FRAC_TOL:
            frac_y.append((u_idx, m_idx, t_idx, val))
        elif val > 1.0 - FRAC_TOL:
            int_y1.append((u_idx, m_idx, t_idx, val))
    if frac_y:
        print(f"  Fractional Y ({len(frac_y)}):")
        for u, m, t, v in frac_y[:20]:
            cn = data.configs[u]
            print(f"    y({cn}, m{m}, t={data.periods[t]}) = {v:.4f}", flush=True)
        if len(frac_y) > 20:
            print(f"    ... and {len(frac_y)-20} more", flush=True)
    else:
        print(f"  All Y are integral. ({len(int_y1)} Y=1 assignments)", flush=True)

    # 3. Implied R values
    print(f"\n--- 3. Implied Scrap (R_i) ---")
    r_vals = cg.get_implied_r()
    frac_r = []
    for i_idx, rv in sorted(r_vals.items(), key=lambda kv: -abs(kv[1] - 0.5)):
        if FRAC_TOL < rv < 1.0 - FRAC_TOL:
            frac_r.append((i_idx, rv))
    if frac_r:
        print(f"  Fractional R ({len(frac_r)}):")
        for i_idx, rv in frac_r[:15]:
            print(f"    R_{i_idx}({data.cured_items[i_idx]}) = {rv:.4f}", flush=True)
        if len(frac_r) > 15:
            print(f"    ... and {len(frac_r)-15} more", flush=True)
    else:
        nonzero_r = [(i, v) for i, v in r_vals.items() if v > FRAC_TOL]
        print(f"  All R integrals. ({len(nonzero_r)} items with R>0)", flush=True)
        if nonzero_r:
            for i, v in nonzero_r[:5]:
                print(f"    R_{i}({data.cured_items[i]}) = {v:.1f}", flush=True)

    # 4. Implied WIP
    print(f"\n--- 4. Implied Window Processing (W_i) ---")
    wip_vals = cg.get_implied_wip()
    frac_wip = []
    for i_idx, wv in sorted(wip_vals.items(), key=lambda kv: -abs(kv[1] - round(kv[1]))):
        if wv > FRAC_TOL and abs(wv - round(wv)) > FRAC_TOL:
            frac_wip.append((i_idx, wv, data.w_i[i_idx]))
    if frac_wip:
        print(f"  Fractional WIP ({len(frac_wip)}):")
        for i_idx, wv, total in frac_wip[:20]:
            print(f"    W_{i_idx}({data.cured_items[i_idx]}) = {wv:.4f} / total={total:.1f}", flush=True)
        if len(frac_wip) > 20:
            print(f"    ... and {len(frac_wip)-20} more", flush=True)
    else:
        print(f"  All W_i are integral.", flush=True)

    # Branching recommendations
    print(f"\n{'='*70}")
    print(f"BRANCHING RECOMMENDATIONS")
    print(f"{'='*70}")

    rhos = get_rho_duals(cg)
    fixed_empty = {}

    y_branch = pick_setup_y0_branch(data, implied_y, fixed_empty, rhos)
    if y_branch:
        u_idx, m_idx, t_idx = y_branch.key
        print(f"\n[Setup Y=0 Branch] RECOMMENDED")
        print(f"  Variable: y(config={data.configs[u_idx]}, machine=m{m_idx}, t={data.periods[t_idx]})")
        print(f"  Implied LP value: {y_branch.value:.4f}")
        print(f"  Branch: LEFT child y=0  |  RIGHT child y=1")
        affected = data.items_by_config[u_idx]
        print(f"  Affected items ({len(affected)}): {[data.cured_items[i] for i in affected[:8]]}{'...' if len(affected)>8 else ''}")
    else:
        print(f"\n[Setup Y=0] No candidate found.")

    wip_branch = pick_wip_branch(data, wip_vals, {}, rhos)
    if wip_branch:
        i_idx = wip_branch.key
        sense, bound = wip_branch.wip_fix
        print(f"\n[WIP Branch]")
        print(f"  Variable: W_{i_idx}(item={data.cured_items[i_idx]})")
        print(f"  LP value: {wip_branch.value:.4f}, total demand: {data.w_i[i_idx]:.1f}")
        print(f"  Window: [{data.earliest_starts[i_idx]}, {data.deadlines[i_idx]}]")
        print(f"  Branch: LEFT W <= {bound}  |  RIGHT W >= {bound+1}")
    else:
        print(f"\n[WIP] No candidate found.")

    # Overall
    has_frac_y = any(FRAC_TOL < v < 1.0 - FRAC_TOL for v in implied_y.values())
    has_frac_wip = any(abs(w - round(w)) > FRAC_TOL for w in wip_vals.values() if w > FRAC_TOL)
    has_frac_r = any(FRAC_TOL < v < 1.0 - FRAC_TOL for v in r_vals.values())

    print(f"\n--- Overall ---")
    print(f"  Fractional Y: {'Yes' if has_frac_y else 'No'}  |  Fractional WIP: {'Yes' if has_frac_wip else 'No'}  |  Fractional R: {'Yes' if has_frac_r else 'No'}")

    if not has_frac_y and not has_frac_wip and not has_frac_r:
        print("  LP solution is completely integral! No branching needed (branch-and-price done).")
    else:
        if y_branch:
            print(f"  -> First branch: y({data.configs[y_branch.key[0]]}, m{y_branch.key[1]}, t={data.periods[y_branch.key[2]]}) = 0 vs 1")
        elif wip_branch:
            print(f"  -> First branch: W_{wip_branch.key} <= {wip_branch.wip_fix[1]} vs >= {wip_branch.wip_fix[1]+1}")

except Exception:
    traceback.print_exc()
    raise

print(f"\nDone.", flush=True)
