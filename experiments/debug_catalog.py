"""Isolate EBRP mismatch on machine 0."""
import sys, os, random

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
    from model_and_algo.pricing_ebrp import (EBRPContext, _index_batches_by_t,
        _solve_routing, _prune_batch_cell, EBRPStats, _item_mask_global)

    scfg = SetConfig('small_cg', 35, 5, 0, 50, 5)
    inst = generate_instance(scfg, 1, seed=101)
    data = preprocess(inst)
    T = len(data.periods)
    I = len(data.cured_items)
    print(f'T={T} I={I} M={len(data.machines)}')

    ctx = EBRPContext.build(data, 0)
    print(f'Machine 0: local_items={len(ctx.local_items)}')

    random.seed(42)
    alpha = [[random.uniform(-500, 500) for _ in range(T)] for _ in range(I)]
    rho = [random.uniform(0, 500) for _ in range(I)]

    # Compare old vs new build_catalog
    # New version (pre-computed duals, pre-computed masks)
    catalog_new = ctx.build_catalog(data, alpha, rho)

    # Old version (on-the-fly dual_sum and mask)
    from configuration import CHECKER_TOL
    catalog_old = [[[] for _ in range(T)] for _ in range(ctx.num_u)]

    def _dual_sum(item, start_t, comp_t):
        if comp_t < 0 or comp_t >= ctx.num_t:
            return rho[item]
        return rho[item] + alpha[item][comp_t]

    for spec_idx, spec in enumerate(ctx.pool_specs):
        u_idx, t_idx, span = spec.u_idx, spec.t_idx, spec.span
        # Note: _subset_cache stores (subset, mask) tuples now
        for subset, _mask in ctx._subset_cache[spec_idx]:
            comp_times = [t_idx + data.lt_it[i][t_idx] - 1 for i in subset]
            dual_sum_val = sum(_dual_sum(i, t_idx, ct) for i, ct in zip(subset, comp_times))
            delta = ctx.p_cum[u_idx] - dual_sum_val
            if delta >= 0 - CHECKER_TOL["pricing_dp"]:
                continue
            catalog_old[u_idx][t_idx].append(
                type('Batch', (), {
                    'u_idx': u_idx, 't_idx': t_idx, 'span': span,
                    'items': subset, 'delta': delta,
                    'item_mask': _item_mask_global(subset, ctx.global_to_local),
                })()
            )

    # Prune both
    for u in range(ctx.num_u):
        for t in range(T):
            catalog_old[u][t] = _prune_batch_cell(catalog_old[u][t])

    # Compare batch counts
    new_count = sum(len(row) for u in catalog_new for row in u)
    old_count = sum(len(row) for u in catalog_old for row in u)
    print(f'Catalog: new={new_count} old={old_count}')

    if new_count != old_count:
        print('Batch count differs! Finding mismatches...')
        for u in range(ctx.num_u):
            for t in range(T):
                if len(catalog_new[u][t]) != len(catalog_old[u][t]):
                    print(f'  u={u} t={t}: new={len(catalog_new[u][t])} old={len(catalog_old[u][t])}')
                    # Print first batch in each
                    for b in catalog_new[u][t][:3]:
                        print(f'    new: items={sorted(b.items)} delta={b.delta:.4f} mask={b.item_mask:b}')
                    for b in catalog_old[u][t][:3]:
                        print(f'    old: items={sorted(b.items)} delta={b.delta:.4f} mask={b.item_mask:b}')
    else:
        # Compare dp results
        batches_old = _index_batches_by_t(catalog_old, ctx.num_u, T, set())
        batches_new = _index_batches_by_t(catalog_new, ctx.num_u, T, set())

        st_old = EBRPStats()
        bc_old, memo_old = _solve_routing(batches_old, T, deadline=None, st=st_old,
                                           progress_interval_sec=0, machine='m0')

        st_new = EBRPStats()
        bc_new, memo_new = _solve_routing(batches_new, T, deadline=None, st=st_new,
                                           progress_interval_sec=0, machine='m0')

        print(f'DP: old_cost={bc_old:.4f} new_cost={bc_new:.4f}')
        if abs(bc_old - bc_new) > 1e-6:
            print('DP cost differs!')

        # Compare against Gurobi
        from model_and_algo.pricing_im_gurobi import solve_subproblem_im_gurobi
        beta = 500.0
        sp_gb = solve_subproblem_im_gurobi(data, 0, alpha, beta, rho, time_limit=60, progress_interval_sec=0)
        print(f'Gurobi: cost={sp_gb.reduced_cost + beta:.4f} (batch_cost)')


if __name__ == '__main__':
    main()
