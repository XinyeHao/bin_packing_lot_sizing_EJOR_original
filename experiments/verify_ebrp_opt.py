"""Verify EBRP correctness against Gurobi."""
import sys, os, gc, time, random

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
    from model_and_algo.pricing_ebrp import solve_subproblem_ebrp, EBRPContext, EBRPStats
    from model_and_algo.pricing_im_gurobi import solve_subproblem_im_gurobi

    scfg = SetConfig('small_cg', 35, 5, 0, 50, 5)
    inst = generate_instance(scfg, 1, seed=101)
    data = preprocess(inst)
    T = len(data.periods)
    I = len(data.cured_items)
    M = len(data.machines)
    print(f'T={T} I={I} M={M}')

    random.seed(42)
    alpha = [[random.uniform(-500, 500) for _ in range(T)] for _ in range(I)]
    rho = [random.uniform(0, 500) for _ in range(I)]
    beta = 500.0

    for m_idx in range(M):
        print(f'\n--- Machine {m_idx} ---')

        # EBRP
        st = EBRPStats()
        t0 = time.perf_counter()
        sp_ebrp = solve_subproblem_ebrp(
            data, m_idx, alpha, beta, rho=rho,
            time_limit=10.0, stats=st, progress_interval_sec=0.0,
            skip_cross_check=True,
        )
        t1 = time.perf_counter()
        print(f'  EBRP: RC={sp_ebrp.reduced_cost:.4f}  time={t1-t0:.3f}s  states={st.nodes_expanded}')

        # Gurobi
        t0 = time.perf_counter()
        sp_gb = solve_subproblem_im_gurobi(
            data, m_idx, alpha, beta, rho,
            time_limit=120.0, progress_interval_sec=0.0,
        )
        t1 = time.perf_counter()
        print(f'  Gurobi: RC={sp_gb.reduced_cost:.4f}  time={t1-t0:.3f}s')

        delta = sp_ebrp.reduced_cost - sp_gb.reduced_cost
        ok = abs(delta) < 1e-4
        print(f'  Delta={delta:.6f}  {"OK" if ok else "MISMATCH!"}')


if __name__ == '__main__':
    main()
