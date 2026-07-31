"""Test Branch-Price-Cut with best_bound strategy."""
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
    print(f"BPC best_bound seed={seed}")
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
        im_time_limit=60.0,
        time_limit=900.0,
        use_node_im=False,
        root_im=True,
        final_im=True,
        branch_strategy='best_bound',
        cg_kwargs=cg_kwargs,
    )
    elapsed = time.perf_counter() - t0
    nodes = r.extra.get('nodes_explored', '?') if r.extra else '?'
    max_depth = r.extra.get('max_depth', '?') if r.extra else '?'
    best_node_lb = r.extra.get('best_node_lb', '?') if r.extra else '?'
    gap_pct = f"{r.mip_gap:.2f}%" if r.mip_gap else ""
    print(f"\nBPC best_bound seed={seed} DONE: status={r.status} obj={r.objective} "
          f"lb={r.lower_bound} best_node_lb={best_node_lb} "
          f"mip_gap={gap_pct} nodes={nodes} max_depth={max_depth} "
          f"runtime={elapsed:.1f}s", flush=True)
    results.append((seed, r.status, r.objective, r.lower_bound, best_node_lb, nodes, max_depth, elapsed))
    del data, r
    gc.collect()

print(f"\n{'='*70}")
print(f"SUMMARY (best_bound, Y0 budget=5, standard cuts)")
print(f"{'='*70}")
for (seed, status, obj, lb, bnl, nodes, md, elapsed) in results:
    print(f"  seed={seed}: status={status} obj={obj} root_lb={lb} best_node_lb={bnl} "
          f"nodes={nodes} max_depth={md} time={elapsed:.1f}s")
