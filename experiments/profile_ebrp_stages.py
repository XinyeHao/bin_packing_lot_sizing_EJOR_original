"""Instrument EBRP to measure time per stage."""
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
        _solve_routing, _prune_batch_cell, EBRPStats, _item_mask_global)

    scfg = SetConfig('small_cg', 36, 5, 0, 50, 5)
    inst = generate_instance(scfg, 1, seed=101)
    data = preprocess(inst)
    T = len(data.periods)
    I = len(data.cured_items)
    M = len(data.machines)
    print(f'T={T} I={I} M={M}')

    # Build context once
    t0 = time.perf_counter()
    ctx = EBRPContext.build(data, 0)
    t1 = time.perf_counter()
    print(f'\n[Build] {t1-t0:.3f}s  pool_specs={len(ctx.pool_specs)}  events={len(ctx.event_times)}')
    total_subsets = sum(len(cache) for cache in ctx._subset_cache)
    print(f'  total_subsets={total_subsets}')
    # Show subset count per pool spec
    subset_counts = [len(cache) for cache in ctx._subset_cache]
    if subset_counts:
        print(f'  subsets per spec: min={min(subset_counts)} max={max(subset_counts)} avg={total_subsets/len(subset_counts):.0f}')

    # Generate random duals for testing
    random.seed(42)
    alpha = [[random.uniform(-500, 500) for _ in range(T)] for _ in range(I)]
    rho = [random.uniform(0, 500) for _ in range(I)]

    # Time catalog building
    t0 = time.perf_counter()
    catalog = ctx.build_catalog(data, alpha, rho)
    t1 = time.perf_counter()
    total_batches = sum(len(row) for u in catalog for row in u)
    print(f'\n[Catalog] {t1-t0:.3f}s  total_batches={total_batches}')

    # Time _index_batches_by_t
    from configuration import CHECKER_TOL
    t0 = time.perf_counter()
    batches_by_t = _index_batches_by_t(catalog, ctx.num_u, T, set())
    t1 = time.perf_counter()
    total_bt = sum(len(bt) for bt in batches_by_t)
    print(f'[IndexByT] {t1-t0:.3f}s  total_by_t={total_bt}')
    for t in range(T):
        if batches_by_t[t]:
            print(f'  t={t:2d}: {len(batches_by_t[t])} batches')

    # Time DP routing
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
    print(f'\n[DP] {t1-t0:.3f}s  nodes_expanded={st.nodes_expanded}  memo_hits={st.memo_hits}  '
          f'memo_size={len(memo)}  batch_cost={batch_cost:.2f}')
    print(f'  batch evals: {st.memo_hits + st.memo_misses + st.nodes_expanded} '
          f'(states expanded {st.nodes_expanded} + hits {st.memo_hits} + non-key loop iterations)')


if __name__ == '__main__':
    main()
