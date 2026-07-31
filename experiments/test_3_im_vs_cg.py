"""Test 3 instances at T=35/I=50/M=5: IM (900s) vs CG+EBRP (900s)."""
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
    from model_and_algo.im_model import solve_im

    scfg = SetConfig('small_cg', 35, 5, 0, 50, 5)
    seeds = [601, 602, 603]
    TIME_LIMIT = 900.0

    results = []

    for seed in seeds:
        gc.collect()
        time.sleep(0.3)
        print(f'\n{"="*60}')
        print(f'Generating seed={seed} ...', flush=True)
        inst = generate_instance(scfg, 1, seed=seed)
        data = preprocess(inst)
        T = len(data.periods)
        I = len(data.cured_items)
        M = len(data.machines)
        print(f'seed={seed}  T={T}  I={I}  M={M}', flush=True)
        print(f'{"="*60}', flush=True)

        row = {'seed': seed, 'T': T, 'I': I, 'M': M}

        # --- IM ---
        print(f'  --- IM (Gurobi, {TIME_LIMIT}s) ---', flush=True)
        t0 = time.perf_counter()
        try:
            im_r = solve_im(data, time_limit=TIME_LIMIT, mip_gap=0.0, verbose=False)
        except Exception as e:
            print(f'  ERROR: {e}', flush=True)
            im_r = None
        elapsed = time.perf_counter() - t0

        if im_r:
            obj = im_r.objective
            lb = im_r.lower_bound
            gap = im_r.mip_gap
            status = im_r.status
            print(f'  IM: obj={obj:.2f}  LB={lb:.2f}  gap={gap:.4%}  status={status}  time={elapsed:.1f}s', flush=True)
            row['im_obj'] = round(obj, 2) if obj else None
            row['im_lb'] = round(lb, 2) if lb else None
            row['im_gap'] = round(gap, 4) if gap else None
            row['im_time'] = round(elapsed, 1)
            row['im_status'] = status
        else:
            print(f'  IM: FAILED  time={elapsed:.1f}s', flush=True)
            row['im_obj'] = None
            row['im_lb'] = None
            row['im_gap'] = None
            row['im_time'] = round(elapsed, 1)
            row['im_status'] = 'failed'

        # --- CG+EBRP ---
        print(f'  --- CG+EBRP ({TIME_LIMIT}s) ---', flush=True)
        t0 = time.perf_counter()
        try:
            cg_r = solve_column_generation(
                data,
                max_iterations=5000,
                verbose=False,
                progress_interval_sec=0.0,
                time_limit=TIME_LIMIT,
                pricing_mode='ebrp',
                use_gurobi_init=False,
                pricing_inprocess=True,
                skip_cross_check=True,
                ebrp_force=True,
            )
        except Exception as e:
            print(f'  ERROR: {e}', flush=True)
            cg_r = None
        elapsed = time.perf_counter() - t0

        if cg_r:
            lrmp = cg_r.extra.get('lrmp_obj') if cg_r.extra else cg_r.lower_bound
            niter = cg_r.extra.get('num_iterations') if cg_r.extra else None
            converged = cg_r.extra.get('converged') if cg_r.extra else (cg_r.status == 'completed')
            print(f'  CG: LRMP={lrmp:.2f}  iters={niter}  converged={converged}  time={elapsed:.1f}s', flush=True)
            row['cg_lrmp'] = round(lrmp, 2) if lrmp is not None else None
            row['cg_iters'] = niter
            row['cg_time'] = round(elapsed, 1)
            row['cg_status'] = 'converged' if converged else cg_r.status
        else:
            print(f'  CG: FAILED  time={elapsed:.1f}s', flush=True)
            row['cg_lrmp'] = None
            row['cg_iters'] = None
            row['cg_time'] = round(elapsed, 1)
            row['cg_status'] = 'failed'

        # Compare
        im_lb = row.get('im_lb')
        cg_lrmp = row.get('cg_lrmp')
        if im_lb is not None and cg_lrmp is not None:
            print(f'  => IM LB={im_lb:.2f}  CG LRMP={cg_lrmp:.2f}  gap={abs(im_lb - cg_lrmp)/max(abs(im_lb),1)*100:.2f}%', flush=True)

        results.append(row)

    # Summary
    print(f'\n{"="*100}')
    print(f'{" SUMMARY ":^100}')
    print(f'{"="*100}')
    header = (f'{"seed":>5} {"T":>3} {"I":>3} {"M":>3} '
              f'{"IM obj":>12} {"IM LB":>12} {"IM gap":>8} {"IM time":>9} '
              f'{"CG LRMP":>12} {"CG iters":>9} {"CG time":>9}')
    print(header)
    print(f'{"-"*100}')
    for r in results:
        print(f'{r["seed"]:>5} {r["T"]:>3} {r["I"]:>3} {r["M"]:>3} '
              f'{r.get("im_obj","N/A"):>12} {r.get("im_lb","N/A"):>12} '
              f'{r.get("im_gap","N/A"):>8} {r.get("im_time","N/A"):>9} '
              f'{r.get("cg_lrmp","N/A"):>12} {r.get("cg_iters","N/A"):>9} '
              f'{r.get("cg_time","N/A"):>9}')


if __name__ == '__main__':
    main()
