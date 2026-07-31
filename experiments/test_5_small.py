"""Test 5 smaller instances: CG+Gurobi vs CG+EBRP, 900s limit."""
import sys, os, gc, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(str(ROOT))
os.environ['GRB_CS_QUIET'] = '1'


def main():
    import configuration as cfg
    cfg.apply_bench_generation()

    from configuration import SetConfig
    from instance_generator.generator import generate_instance
    from preprocessor.preprocess import preprocess
    from model_and_algo.column_generation import solve_column_generation

    scfg = SetConfig('small_cg', 18, 3, 0, 24, 3)
    seeds = [51, 52, 53, 54, 55]
    TIME_LIMIT = 900.0

    results = []

    for seed in seeds:
        gc.collect()
        time.sleep(0.3)
        inst = generate_instance(scfg, 1, seed=seed)
        data = preprocess(inst)
        T = len(data.periods)
        I = len(data.cured_items)
        M = len(data.machines)
        print(f'\n{"="*60}')
        print(f'seed={seed}  T={T}  I={I}  M={M}')
        print(f'{"="*60}', flush=True)

        row = {'seed': seed, 'T': T, 'I': I, 'M': M}

        for mode_key, mode_opts in [
            ('Gurobi', {'pricing_mode': 'im_gurobi', 'use_gurobi_init': True,
                        'im_pricing_time_limit': 120.0}),
            ('EBRP',   {'pricing_mode': 'ebrp', 'use_gurobi_init': False,
                        'pricing_inprocess': True, 'skip_cross_check': True,
                        'ebrp_force': True}),
        ]:
            print(f'  --- {mode_key} ---', flush=True)
            t0 = time.perf_counter()
            try:
                r = solve_column_generation(
                    data,
                    max_iterations=5000,
                    verbose=False,
                    progress_interval_sec=0.0,
                    time_limit=TIME_LIMIT,
                    **mode_opts,
                )
            except Exception as e:
                print(f'  ERROR: {e}', flush=True)
                row[f'{mode_key}_lrmp'] = None
                row[f'{mode_key}_iters'] = None
                row[f'{mode_key}_time'] = None
                row[f'{mode_key}_status'] = str(e)[:80]
                continue

            elapsed = time.perf_counter() - t0
            lrmp = r.extra.get('lrmp_obj') if r.extra else r.lower_bound
            niter = r.extra.get('num_iterations') if r.extra else None
            converged = r.extra.get('converged') if r.extra else (r.status == 'completed')
            print(f'  LRMP={lrmp:.2f}  iters={niter}  converged={converged}  time={elapsed:.1f}s', flush=True)

            row[f'{mode_key}_lrmp'] = round(lrmp, 2) if lrmp is not None else None
            row[f'{mode_key}_iters'] = niter
            row[f'{mode_key}_time'] = round(elapsed, 1)
            row[f'{mode_key}_status'] = 'converged' if converged else r.status

        g_lrmp = row.get('Gurobi_lrmp')
        e_lrmp = row.get('EBRP_lrmp')
        if g_lrmp is not None and e_lrmp is not None:
            delta = abs(g_lrmp - e_lrmp)
            rel = delta / max(abs(g_lrmp), 1) * 100
            ok = delta <= 1.0 or rel <= 0.1
            row['match'] = 'YES' if ok else f'NO (d={delta:.2f})'
            print(f'  => LRMP delta={delta:.4f}  rel={rel:.4f}%  match={row["match"]}', flush=True)
        else:
            row['match'] = 'N/A'

        results.append(row)

    # Summary
    print(f'\n{"="*80}')
    print(f'{" SUMMARY ":^80}')
    print(f'{"="*80}')
    print(f'{"seed":>6} {"T":>4} {"I":>4} {"M":>4} {"Gurobi LRMP":>14} {"EBRP LRMP":>14} {"Delta":>10} {"G time":>9} {"E time":>9} {"Match":>6}')
    print(f'{"-"*80}')
    for r in results:
        g = r.get('Gurobi_lrmp')
        e = r.get('EBRP_lrmp')
        gt = r.get('Gurobi_time')
        et = r.get('EBRP_time')
        delta_str = f'{abs(g - e):.2f}' if (g is not None and e is not None) else 'N/A'
        g_str = f'{g:.2f}' if g is not None else 'N/A'
        e_str = f'{e:.2f}' if e is not None else 'N/A'
        gt_str = f'{gt:.1f}s' if gt is not None else 'N/A'
        et_str = f'{et:.1f}s' if et is not None else 'N/A'
        print(f'{r["seed"]:>6} {r["T"]:>4} {r["I"]:>4} {r["M"]:>4} '
              f'{g_str:>14} {e_str:>14} {delta_str:>10} '
              f'{gt_str:>9} {et_str:>9} {r.get("match","N/A"):>6}')

    matched = sum(1 for r in results if r.get('match') == 'YES')
    print(f'\n{matched}/{len(results)} matched.')


if __name__ == '__main__':
    main()
