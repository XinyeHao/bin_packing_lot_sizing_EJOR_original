"""Pure BP + BO branching, NO root IM. 5 seeds, 900s each."""
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

seeds = [101, 202, 303, 404, 505]
results = []

for idx, seed in enumerate(seeds):
    gc.collect(); time.sleep(1.0)
    print(f"\n{'#'*60}")
    print(f"# [{idx+1}/5] Pure BP + BO (no root IM), seed={seed}")
    print(f"{'#'*60}", flush=True)

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
        'progress_interval_sec': 30.0,
        'time_limit': 600.0,
    }

    t0 = time.perf_counter()
    try:
        r = solve_branch_and_price(
            data,
            max_nodes=5000,
            im_time_limit=90.0,
            time_limit=900.0,
            use_node_im=False,
            root_im=False,
            final_im=False,
            branch_strategy='bo',
            cg_kwargs=cg_kwargs,
        )
    except Exception as e:
        print(f"  ERROR: {e}", flush=True)
        r = None
    elapsed = time.perf_counter() - t0

    if r is not None:
        nodes = r.extra.get('nodes_explored', '?') if r.extra else '?'
        max_depth = r.extra.get('max_depth', '?') if r.extra else '?'
        root_rmp = r.extra.get('root_lrmp', None) if r.extra else None
        root_vlb = r.extra.get('root_valid_lb', None) if r.extra else None
        cg_conv = r.extra.get('root_cg_converged', False) if r.extra else False
        gap_pct = f"{r.mip_gap:.2f}%" if r.mip_gap is not None else "N/A"
        status_cn = r.status
        obj = r.objective
    else:
        nodes, max_depth, root_rmp, root_vlb, cg_conv = '?', '?', None, None, False
        gap_pct, status_cn, obj = "ERROR", "error", None

    obj_str = f"{obj:.2f}" if obj is not None else "?"
    root_rmp_str = f"{root_rmp:.2f}" if root_rmp is not None else "?"
    root_vlb_str = f"{root_vlb:.2f}" if root_vlb is not None else "?"

    print(f"    DONE: status={status_cn}  incumbent={obj_str}  root_RMP={root_rmp_str}  "
          f"Lagr_LB={root_vlb_str}  cg_conv={cg_conv}  gap={gap_pct}  "
          f"nodes={nodes}  depth={max_depth}  time={elapsed:.1f}s", flush=True)

    results.append((seed, status_cn, obj_str, root_rmp_str, root_vlb_str, cg_conv,
                    gap_pct, nodes, max_depth, elapsed))

    del data
    if r is not None:
        del r
    gc.collect(); time.sleep(1.0)

print(f"\n{'#'*60}")
print(f"SUMMARY: Pure BP + BO (no root IM)")
print(f"{'#'*60}")
for row in results:
    print(f"  seed={row[0]:>4}  status={row[1]:<12}  incumbent={row[2]:>10}  "
          f"root_RMP={row[3]:>10}  Lagr_LB={row[4]:>10}  "
          f"cg_conv={row[5]}  gap={row[6]:>8}  nodes={str(row[7]):>4}  "
          f"depth={str(row[8]):>3}  time={row[9]:.1f}s")
