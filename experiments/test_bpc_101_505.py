"""Test Branch-Price-Cut on seeds 101 and 505. Output per node."""
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

for seed in [101, 505]:
    gc.collect(); time.sleep(1)
    print(f"\n{'='*70}")
    print(f"BPC seed={seed}")
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
        branch_strategy='new_policy',
        cg_kwargs=cg_kwargs,
    )
    elapsed = time.perf_counter() - t0
    print(f"\nBPC seed={seed} DONE: status={r.status} obj={r.objective} lb={r.lower_bound} "
          f"gap={r.gap:.4%} nodes={r.nodes_explored} runtime={elapsed:.1f}s",
          flush=True)
    del data, r
    gc.collect()
