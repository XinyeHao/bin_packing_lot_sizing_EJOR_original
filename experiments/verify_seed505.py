"""Verify EBRP vs Gurobi on seed 505 (slowest speedup)."""
import sys, os, gc, time, random

ROOT = r'C:\Users\haoxi\PycharmProjects\bin_packing_and_scheduling'
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ['GRB_CS_QUIET'] = '1'


def main():
    import configuration as cfg
    cfg.LEAD_TIME_CHOICES = (8, 9, 10)
    cfg.WINDOW_GENERATION_BY_SET['small_cg'] = {
        'length_frac': (0.45, 0.65), 'rel_frac': (0.03, 0.25), 'min_length': 6
    }
    from configuration import SetConfig
    from instance_generator.generator import generate_instance
    from preprocessor.preprocess import preprocess
    from model_and_algo.pricing_ebrp import solve_subproblem_ebrp, EBRPStats
    from model_and_algo.pricing_im_gurobi import solve_subproblem_im_gurobi

    scfg = SetConfig('small_cg', 36, 5, 0, 50, 5)
    inst = generate_instance(scfg, 1, seed=505)
    data = preprocess(inst)
    T, I, M = len(data.periods), len(data.cured_items), len(data.machines)
    print(f'T={T} I={I} M={M}')

    random.seed(42)
    alpha = [[random.uniform(-500, 500) for _ in range(T)] for _ in range(I)]
    rho = [random.uniform(0, 500) for _ in range(I)]
    beta = 500.0

    all_ok = True
    for m_idx in range(M):
        st = EBRPStats()
        sp_ebrp = solve_subproblem_ebrp(
            data, m_idx, alpha, beta, rho=rho,
            time_limit=10.0, stats=st, progress_interval_sec=0.0,
            skip_cross_check=True,
        )
        sp_gb = solve_subproblem_im_gurobi(
            data, m_idx, alpha, beta, rho,
            time_limit=60.0, progress_interval_sec=0.0,
        )
        delta = abs(sp_ebrp.reduced_cost - sp_gb.reduced_cost)
        ok = delta < 1e-3
        status = "OK" if ok else f"MISMATCH ({delta:.6f})"
        if not ok:
            all_ok = False
        print(f'  M{m_idx}: EBRP={sp_ebrp.reduced_cost:.4f}  '
              f'Gurobi={sp_gb.reduced_cost:.4f}  states={st.nodes_expanded}  {status}')

    print(f'\nAll OK: {all_ok}')


if __name__ == '__main__':
    main()
