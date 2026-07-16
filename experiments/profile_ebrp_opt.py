"""Profile optimized EBRP stages."""
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
    from model_and_algo.pricing_ebrp import (EBRPContext, _index_batches_by_t,
        _solve_routing, EBRPStats)

    scfg = SetConfig('small_cg', 36, 5, 0, 50, 5)
    inst = generate_instance(scfg, 1, seed=101)
    data = preprocess(inst)
    T = len(data.periods)
    I = len(data.cured_items)
    M = len(data.machines)
    print(f'T={T} I={I} M={M}')

    ctx = EBRPContext.build(data, 0)
    total_subsets = sum(len(cache) for cache in ctx._subset_cache)
    print(f'Machine 0: pool_specs={len(ctx.pool_specs)} subsets={total_subsets} events={len(ctx.event_times)}')

    # Random duals
    random.seed(42)
    alpha = [[random.uniform(-500, 500) for _ in range(T)] for _ in range(I)]
    rho = [random.uniform(0, 500) for _ in range(I)]

    # Catalog
    t0 = time.perf_counter()
    catalog = ctx.build_catalog(data, alpha, rho)
    t1 = time.perf_counter()
    total_batches = sum(len(row) for u in catalog for row in u)
    print(f'Catalog: {t1-t0:.3f}s  batches={total_batches}')

    # IndexByT
    batches_by_t = _index_batches_by_t(catalog, ctx.num_u, T, set())
    total_bt = sum(len(bt) for bt in batches_by_t)
    print(f'IndexByT: total_by_t={total_bt}')

    # DP
    st = EBRPStats()
    t0 = time.perf_counter()
    batch_cost, memo = _solve_routing(
        batches_by_t, T,
        deadline=None,
        st=st,
        progress_interval_sec=0.0,
        machine='m0',
    )
    t1 = time.perf_counter()
    print(f'DP: {t1-t0:.3f}s  states={st.nodes_expanded}  batch_cost={batch_cost:.2f}  memo_size={len(memo)}')


if __name__ == '__main__':
    main()
