"""Quick debug: check root CG converged flag in B&P."""
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
print(f"\nDEBUG BPC BO seed={seed}", flush=True)
scfg = SetConfig('small_cg', 35, 5, 0, 50, 5)
data = generate_instance(scfg, instance_id=seed % 100, seed=seed)
data = preprocess(data)

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
r = solve_branch_and_price(
    data, max_nodes=5000, im_time_limit=90.0, time_limit=900.0,
    use_node_im=False, root_im=True, final_im=True,
    branch_strategy='bo', cg_kwargs=cg_kwargs,
)
cg_conv = r.extra.get('root_cg_converged', False) if r.extra else False
print(f"\nFINAL: converged={cg_conv} status={r.status} obj={r.objective:.2f}", flush=True)
