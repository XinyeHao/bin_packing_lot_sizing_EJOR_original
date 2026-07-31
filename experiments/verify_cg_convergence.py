"""Verify CG+EBRP convergence: after CG, check with Gurobi that no column has negative RC."""
import sys, os, gc, time

ROOT = r'C:\Users\haoxi\PycharmProjects\bin_packing_and_scheduling'
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ['GRB_CS_QUIET'] = '1'


def main():
    import configuration as cfg
    cfg.apply_bench_generation()
    from configuration import SetConfig
    from instance_generator.generator import generate_instance
    from preprocessor.preprocess import preprocess
    from model_and_algo.column_generation import ColumnGenerationSolver
    from model_and_algo.pricing_im_gurobi import solve_subproblem_im_gurobi

    seeds = [101, 202]
    scfg = SetConfig('small_cg', 35, 5, 0, 50, 5)

    for seed in seeds:
        gc.collect(); time.sleep(0.3)
        inst = generate_instance(scfg, 1, seed=seed)
        data = preprocess(inst)
        print(f'\n=== seed={seed} ===')

        cg = ColumnGenerationSolver(
            data, max_iterations=5000, verbose=False,
            progress_interval_sec=0.0, time_limit=600.0,
            pricing_mode='ebrp', use_gurobi_init=False,
            skip_cross_check=True, pricing_inprocess=True, ebrp_force=True,
        )
        t0 = time.perf_counter()
        lb, rcs, cols = cg.run()
        elapsed = time.perf_counter() - t0
        print(f'  CG+EBRP: LRMP={cg.lrmp_obj:.2f}  iters={len(rcs)}  time={elapsed:.1f}s')

        if cg._model is None or cg._artifacts is None:
            continue

        num_t = len(data.periods)
        num_i = len(data.cured_items)
        flow_constrs = cg._artifacts.get("flow_items")
        alpha = [[0.0] * num_t for _ in range(num_i)]
        for (i_idx, t_idx), c in flow_constrs.items():
            alpha[i_idx][t_idx] = c.Pi
        rho = [cg._artifacts["wip_balance"][i].Pi for i in range(num_i)]
        betas = [cg._artifacts["choose_one"][m].Pi for m in data.machines]

        any_bug = False
        for m_idx, machine in enumerate(data.machines):
            sp = solve_subproblem_im_gurobi(
                data, m_idx, alpha, betas[m_idx], rho,
                time_limit=60.0, progress_interval_sec=0.0,
            )
            if sp.reduced_cost < -1e-3:
                print(f'  BUG: {machine} Gurobi RC={sp.reduced_cost:.6f} < 0 (missed!)')
                any_bug = True
            else:
                print(f'  {machine}: Gurobi RC={sp.reduced_cost:.6f} OK')

        if not any_bug:
            print(f'  ALL OK: LRMP={cg.lrmp_obj:.2f} is optimal')


if __name__ == '__main__':
    main()
