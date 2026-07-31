"""Debug: compare old recursive DP vs new bottom-up DP on same data."""
import sys, os, gc, time, random

ROOT = r'C:\Users\haoxi\PycharmProjects\bin_packing_and_scheduling'
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ['GRB_CS_QUIET'] = '1'


def main():
    import configuration as cfg
    cfg.apply_bench_generation()
    from configuration import SetConfig, CHECKER_TOL
    from instance_generator.generator import generate_instance
    from preprocessor.preprocess import preprocess
    from model_and_algo.pricing_ebrp import (EBRPContext, _index_batches_by_t,
        Batch, EBRPStats, _MemoEntry)

    scfg = SetConfig('small_cg', 35, 5, 0, 50, 5)
    inst = generate_instance(scfg, 1, seed=101)
    data = preprocess(inst)
    T = len(data.periods)

    ctx = EBRPContext.build(data, 0)

    random.seed(42)
    I = len(data.cured_items)
    alpha = [[random.uniform(-500, 500) for _ in range(T)] for _ in range(I)]
    rho = [random.uniform(0, 500) for _ in range(I)]
    catalog = ctx.build_catalog(data, alpha, rho)
    batches_by_t = _index_batches_by_t(catalog, ctx.num_u, T, set())

    tol = CHECKER_TOL["pricing_dp"]

    # === Old recursive DP (known correct) ===
    next_event = [T] * (T + 1)
    nxt = T
    for t in range(T - 1, -1, -1):
        next_event[t] = nxt
        if batches_by_t[t]:
            nxt = t
    next_event[T] = T

    memo_old: dict[tuple[int, int], _MemoEntry] = {}

    def solve_old(t_idx, used_mask):
        if t_idx >= T:
            return 0.0
        key = (t_idx, used_mask)
        if key in memo_old:
            return memo_old[key].cost

        nxt = next_event[t_idx]
        idle_best = solve_old(nxt, used_mask)
        best = idle_best
        entry = _MemoEntry(best, "idle", next_t=nxt)

        for b in batches_by_t[t_idx]:
            if b.item_mask & used_mask:
                continue
            et = t_idx + b.span
            if et > T:
                continue
            cand = b.delta + solve_old(et, used_mask | b.item_mask)
            if cand < best - tol:
                best = cand
                entry = _MemoEntry(best, "batch", next_t=et, batch=b)

        memo_old[key] = entry
        return best

    t0 = time.perf_counter()
    bc_old = solve_old(0, 0)
    t1 = time.perf_counter()
    print(f'Old recursive DP: cost={bc_old:.4f}  memo_size={len(memo_old)}  time={t1-t0:.3f}s')

    # === New bottom-up DP ===
    dp: list[list[tuple[int, float, int, int, object]]] = [[] for _ in range(T + 1)]
    dp[0].append((0, 0.0, -1, -1, None))

    def prune_frontier(entries, new_mask, new_cost, new_pt, new_pm, new_action):
        for mask, cost, _, _, _ in entries:
            if (mask & new_mask) == mask and cost <= new_cost + tol:
                return
        entries[:] = [(m, c, pt, pm, act) for (m, c, pt, pm, act) in entries
                      if not ((new_mask & m) == new_mask and new_cost <= c + tol)]
        entries.append((new_mask, new_cost, new_pt, new_pm, new_action))

    t0 = time.perf_counter()
    for t in range(T):
        if not dp[t]:
            continue
        nxt = next_event[t]
        for mask, cost, pt, pm, action in dp[t]:
            prune_frontier(dp[nxt], mask, cost, t, mask, 'idle')
            for b in batches_by_t[t]:
                if b.item_mask & mask:
                    continue
                et = t + b.span
                if et > T:
                    continue
                new_mask = mask | b.item_mask
                new_cost = cost + b.delta
                prune_frontier(dp[et], new_mask, new_cost, t, mask, b)
    t1 = time.perf_counter()

    best_cost = float('inf')
    for mask, cost, pt, pm, action in dp[T]:
        if cost < best_cost - tol:
            best_cost = cost

    n_states = sum(len(e) for e in dp)
    print(f'New bottom-up DP: cost={best_cost:.4f}  states={n_states}  time={t1-t0:.3f}s')
    print(f'Delta: {best_cost - bc_old:.4f} (should be ~0)')

    # === Now try WITHOUT dominance pruning ===
    dp2: list[dict[int, tuple[float, int, int, object]]] = [{} for _ in range(T + 1)]
    dp2[0][0] = (0.0, -1, -1, None)

    t0 = time.perf_counter()
    for t in range(T):
        if not dp2[t]:
            continue
        nxt = next_event[t]
        for mask, (cost, pt, pm, action) in list(dp2[t].items()):
            # Idle
            old = dp2[nxt].get(mask)
            if old is None or cost < old[0]:
                dp2[nxt][mask] = (cost, t, mask, 'idle')
            # Batches
            for b in batches_by_t[t]:
                if b.item_mask & mask:
                    continue
                et = t + b.span
                if et > T:
                    continue
                new_mask = mask | b.item_mask
                new_cost = cost + b.delta
                old = dp2[et].get(new_mask)
                if old is None or new_cost < old[0]:
                    dp2[et][new_mask] = (new_cost, t, mask, b)
    t1 = time.perf_counter()

    best_cost2 = float('inf')
    for mask, (cost, pt, pm, action) in dp2[T].items():
        if cost < best_cost2 - tol:
            best_cost2 = cost

    n_states2 = sum(len(e) for e in dp2)
    print(f'New B-U DP (no prune): cost={best_cost2:.4f}  states={n_states2}  time={t1-t0:.3f}s')
    print(f'Delta vs old: {best_cost2 - bc_old:.4f} (should be ~0)')


if __name__ == '__main__':
    main()
