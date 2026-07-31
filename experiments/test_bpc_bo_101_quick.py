"""Quick comparison: BO with Matérn vs RBF kernel — seed 101 only."""
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

seed = 101
gc.collect(); time.sleep(0.5)
print(f"\n{'='*70}")
print(f"BPC BO seed={seed} (Matérn kernel, no column pruning)")
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
    branch_strategy='bo',
    cg_kwargs=cg_kwargs,
)
elapsed = time.perf_counter() - t0
nodes = r.extra.get('nodes_explored', '?') if r.extra else '?'
root_rmp = f"{r.lower_bound:.2f}" if r.lower_bound else "?"
gap_pct = f"{r.mip_gap:.2f}%" if r.mip_gap is not None else "N/A"
status_cn = "最优" if r.status == "optimal" else ("超时" if r.status == "time_limit" else r.status)
print(f"\nBPC BO Matérn seed={seed}: {status_cn}  可行解={r.objective:.2f}  下界={root_rmp}  间隙={gap_pct}  节点={nodes}  用时={elapsed:.1f}s", flush=True)
del data, r; gc.collect()
