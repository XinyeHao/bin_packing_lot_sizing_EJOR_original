"""Test Branch-Price-Cut with BO (Bayesian Optimization) branching strategy.

Output terms:
  incumbent (upper bound): best integer feasible solution found
  root_lrmp (lower bound): root LP+cuts optimal value (valid LB if CG converged)
  cg_converged: whether root CG converged (True => root_lrmp is valid LB)
  gap: (incumbent - root_lrmp) / incumbent * 100%; optimal => gap=0%
"""
import sys, os, gc, time
ROOT = r'C:\Users\haoxi\PycharmProjects\bin_packing_and_scheduling'
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ['GRB_CS_QUIET'] = '1'

import configuration as cfg
cfg.apply_bench_generation()
from configuration import SetConfig
from instance_generator.generator import generate_instance
from preprocessor.preprocess import preprocess
from model_and_algo.branch_and_price import solve_branch_and_price

results = []
for seed in (101, 202, 303, 404, 505):
    gc.collect(); time.sleep(0.5)
    print(f"\n{'='*70}")
    print(f"BPC BO seed={seed}")
    print(f"{'='*70}", flush=True)
    scfg = SetConfig('small_cg', 35, 5, 0, 50, 5)
    data = generate_instance(scfg, instance_id=seed % 100, seed=seed)
    data = preprocess(data)
    print(f"T={len(data.periods)} I={len(data.cured_items)} M={len(data.machines)}", flush=True)

    cg_kwargs = {
        'pricing_mode': 'ebrp',
        'ebrp_force': True,
        'use_gurobi_init': False,
        'skip_cross_check': True,
        'pricing_inprocess': True,
        'verbose': False,
        'progress_interval_sec': 10.0,
        'time_limit': 600.0,
        'lrmp_cuts': 'standard',
    }
    t0 = time.perf_counter()
    r = solve_branch_and_price(
        data,
        max_nodes=5000,
        im_time_limit=90.0,
        time_limit=900.0,
        use_node_im=False,
        root_im=True,
        final_im=True,
        branch_strategy='bo',
        cg_kwargs=cg_kwargs,
    )
    elapsed = time.perf_counter() - t0
    nodes = r.extra.get('nodes_explored', '?') if r.extra else '?'
    max_depth = r.extra.get('max_depth', '?') if r.extra else '?'
    root_rmp = f"{r.extra.get('root_lrmp', '?'):.2f}" if r.extra and r.extra.get('root_lrmp') else "?"
    root_vlb = r.extra.get('root_valid_lb', None) if r.extra else None
    root_vlb_str = f"{root_vlb:.2f}" if root_vlb is not None else "?"
    cg_conv = r.extra.get('root_cg_converged', False) if r.extra else False
    gap_pct = f"{r.mip_gap:.2f}%" if r.mip_gap is not None else "N/A"
    status_cn = "optimal" if r.status == "optimal" else ("time_limit" if r.status == "time_limit" else r.status)
    print(f"\nBPC BO seed={seed} DONE: status={status_cn}  "
          f"incumbent={r.objective:.2f}  Lagrangian_LB={root_vlb_str}  root_RMP={root_rmp}  "
          f"gap={gap_pct}  nodes={nodes}  depth={max_depth}  "
          f"time={elapsed:.1f}s", flush=True)
    results.append((seed, status_cn, r.objective, root_vlb_str, root_rmp, cg_conv, gap_pct, nodes, max_depth, elapsed))
    del data, r
    gc.collect()

print(f"\n{'='*70}")
print(f"SUMMARY (BO branching)")
print(f"{'='*70}")
print(f"  incumbent=upper_bound / root_lrmp=lower_bound(LP+cuts at root)")
print(f"  gap = (incumbent - root_lrmp) / incumbent, optimal => gap=0%")
for (seed, status_cn, obj, root_vlb, root_rmp, cg_conv, gap, nodes, md, elapsed) in results:
    print(f"  seed={seed}: status={status_cn}  incumbent={obj:.2f}  "
          f"Lagrangian_LB={root_vlb}  root_RMP={root_rmp}  "
          f"cg_conv={cg_conv}  gap={gap}  nodes={nodes}  depth={md}  time={elapsed:.1f}s")
