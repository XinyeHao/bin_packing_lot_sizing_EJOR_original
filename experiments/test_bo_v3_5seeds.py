"""BO v3 improved benchmark: seeds 101, 202, 303, 404, 505. 1200s each."""
import sys, os, gc, time
ROOT = r'C:\Users\haoxi\PycharmProjects\bin_packing_and_scheduling'
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ['GRB_CS_QUIET'] = '1'

import warnings
warnings.filterwarnings('ignore')

import configuration as cfg
cfg.apply_bench_generation()
from configuration import SetConfig
from instance_generator.generator import generate_instance
from preprocessor.preprocess import preprocess
from model_and_algo.branch_and_price import solve_branch_and_price

SEEDS = [101, 202, 303, 404, 505]
TLIM = 1200

results = []
for seed in SEEDS:
    gc.collect(); time.sleep(1.0)
    print(f"\n{'='*60}")
    print(f"BO v3 improved - seed {seed}")
    print(f"{'='*60}", flush=True)

    scfg = SetConfig('small_cg', 35, 5, 0, 50, 5)
    data = generate_instance(scfg, instance_id=1, seed=seed)
    data = preprocess(data)
    print(f"T={len(data.periods)} I={len(data.cured_items)} M={len(data.machines)}", flush=True)

    cg_kwargs = {
        'pricing_mode': 'ebrp', 'ebrp_force': True,
        'use_gurobi_init': False, 'skip_cross_check': True,
        'pricing_inprocess': True, 'verbose': True,
        'progress_interval_sec': 30.0, 'time_limit': 600.0,
    }

    t0 = time.perf_counter()
    r = solve_branch_and_price(
        data, max_nodes=5000, im_time_limit=90.0, time_limit=TLIM,
        use_node_im=False, root_im=True, final_im=True,
        branch_strategy='bo', cg_kwargs=cg_kwargs,
    )
    elapsed = time.perf_counter() - t0

    nodes = r.extra.get('nodes_explored', '?') if r.extra else '?'
    root_rmp = r.extra.get('root_lrmp', None) if r.extra else None
    gap_pct = f"{r.mip_gap:.2f}%" if r.mip_gap is not None else "N/A"
    obj = f"{r.objective:.2f}" if r.objective else "None"

    print(f"\n--- seed={seed} done ---")
    print(f"  status={r.status}  incumbent={obj}  root_RMP={root_rmp:.2f}" if root_rmp else f"  status={r.status}  incumbent={obj}")
    print(f"  mip_gap={gap_pct}  nodes={nodes}  time={elapsed:.1f}s", flush=True)

    results.append({
        'seed': seed, 'status': r.status, 'incumbent': r.objective,
        'root_RMP': root_rmp, 'mip_gap': r.mip_gap, 'nodes': nodes,
        'time': elapsed,
    })
    del data, r; gc.collect()

print(f"\n{'='*60}")
print("FINAL SUMMARY")
print(f"{'='*60}")
print(f"{'Seed':>6}  {'Status':<12}  {'Incumbent':>12}  {'Root RMP':>12}  {'Gap':>8}  {'Nodes':>6}  {'Time':>8}")
for r in results:
    obj = f"{r['incumbent']:.2f}" if r['incumbent'] else "None"
    rrmp = f"{r['root_RMP']:.2f}" if r['root_RMP'] else "None"
    gap = f"{r['mip_gap']:.2f}%" if r['mip_gap'] is not None else "N/A"
    print(f"{r['seed']:>6}  {r['status']:<12}  {obj:>12}  {rrmp:>12}  {gap:>8}  {str(r['nodes']):>6}  {r['time']:>7.1f}s", flush=True)

# Average gap
feasible = [r for r in results if r['incumbent'] is not None and r['mip_gap'] is not None]
if feasible:
    avg_gap = sum(r['mip_gap'] for r in feasible) / len(feasible)
    print(f"\n  Average gap (feasible): {avg_gap:.2f}%")
    avg_time = sum(r['time'] for r in feasible) / len(feasible)
    print(f"  Average time: {avg_time:.1f}s")
print("Done.")
