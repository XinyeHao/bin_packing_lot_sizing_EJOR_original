"""Pure BP + improved BO, seed 101, 3600s."""
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

gc.collect(); time.sleep(1.0)
print("="*60)
print("Pure BP + improved BO, seed=101")
print("="*60, flush=True)

scfg = SetConfig('small_cg', 35, 5, 0, 50, 5)
data = generate_instance(scfg, instance_id=1, seed=101)
data = preprocess(data)
print(f"T={len(data.periods)} I={len(data.cured_items)} M={len(data.machines)}", flush=True)

cg_kwargs = {
    'pricing_mode': 'ebrp',
    'ebrp_force': True,
    'use_gurobi_init': False,
    'skip_cross_check': True,
    'pricing_inprocess': True,
    'verbose': True,
    'progress_interval_sec': 15.0,
    'time_limit': 600.0,
    # NO lrmp_cuts: pure BP
}

t0 = time.perf_counter()
r = solve_branch_and_price(
    data,
    max_nodes=5000,
    im_time_limit=90.0,
    time_limit=1800.0,
    use_node_im=False,
    root_im=True,
    final_im=True,
    branch_strategy='bo',
    cg_kwargs=cg_kwargs,
)
elapsed = time.perf_counter() - t0

nodes = r.extra.get('nodes_explored', '?') if r.extra else '?'
max_depth = r.extra.get('max_depth', '?') if r.extra else '?'
root_rmp = r.extra.get('root_lrmp', None) if r.extra else None
root_vlb = r.extra.get('root_valid_lb', None) if r.extra else None
cg_conv = r.extra.get('root_cg_converged', False) if r.extra else False
cg_stop = r.extra.get('root_cg_stop', '?') if r.extra else '?'
gap_pct = f"{r.mip_gap:.2f}%" if r.mip_gap is not None else "N/A"

print(f"\n=== seed=101 DONE ===")
print(f"  status={r.status}")
print(f"  incumbent={r.objective:.2f}" if r.objective else "  incumbent=None")
print(f"  root_RMP={root_rmp:.2f}" if root_rmp else "  root_RMP=None")
print(f"  valid_LB={root_vlb:.2f}" if root_vlb else "  valid_LB=None")
print(f"  cg_converged={cg_conv}  cg_stop={cg_stop}")
print(f"  mip_gap={gap_pct}")
print(f"  nodes={nodes}  max_depth={max_depth}")
print(f"  time={elapsed:.1f}s", flush=True)

del data, r; gc.collect()
