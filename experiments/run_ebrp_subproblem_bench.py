"""EBRP 子问题求解效率压测（set_c_01_new）。"""
from __future__ import annotations

import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configuration import EBRP_DEFAULT_TIME_LIMIT
from model_and_algo.pricing_ebrp import (
    EBRPStats,
    build_batch_catalog,
    build_ebrp_contexts,
    solve_subproblem_ebrp,
)
from preprocessor.preprocess import load_instance, preprocess


def _run_zero_dual(data, ctxs) -> None:
    num_i, num_t = len(data.cured_items), len(data.periods)
    zero_alpha = [[0.0] * num_t for _ in range(num_i)]
    zero_rho = [0.0] * num_i
    ts: list[float] = []
    for m, ctx in enumerate(ctxs):
        st = EBRPStats()
        t0 = time.perf_counter()
        solve_subproblem_ebrp(
            data, m, zero_alpha, 0.0, rho=zero_rho, stats=st, ctx=ctx, progress_interval_sec=0.0
        )
        ts.append(time.perf_counter() - t0)
    print(
        f"零对偶 exact: per_machine={[round(t, 3) for t in ts]} "
        f"max={max(ts):.3f}s mean={statistics.mean(ts):.3f}s nodes~{st.nodes_expanded}"
    )


def _run_real_dual(data, ctxs, alpha, rho, beta: float, limit: float | None, label: str) -> None:
    ts: list[float] = []
    nodes: list[int] = []
    exact: list[bool] = []
    tout: list[bool] = []
    for m, ctx in enumerate(ctxs):
        st = EBRPStats()
        t0 = time.perf_counter()
        solve_subproblem_ebrp(
            data,
            m,
            alpha,
            beta,
            rho=rho,
            stats=st,
            ctx=ctx,
            time_limit=limit,
            progress_interval_sec=0.0,
        )
        ts.append(time.perf_counter() - t0)
        nodes.append(st.nodes_expanded)
        exact.append(st.exact)
        tout.append(st.timed_out)
    lim = "exact" if limit is None else f"{limit:.0f}s"
    print(
        f"真实对偶 {label} ({lim}): per_machine={[round(t, 2) for t in ts]} "
        f"max={max(ts):.2f}s mean={statistics.mean(ts):.2f}s"
    )
    print(f"  nodes={nodes} exact={exact} timed_out={tout}")


def main() -> None:
    instance = "set_c_01_new"
    data = preprocess(load_instance(instance))
    num_i, num_t = len(data.cured_items), len(data.periods)
    print(f"=== EBRP 子问题效率 |I|={num_i} |M|={len(data.machines)} |T|={num_t} ===")

    t0 = time.perf_counter()
    ctxs = build_ebrp_contexts(data)
    print(f"build_ebrp_contexts: {time.perf_counter() - t0:.1f}s  pool_specs={len(ctxs[0].pool_specs)}")

    rng = random.Random(42)
    alpha = [[rng.uniform(0, 50) for _ in range(num_t)] for _ in range(num_i)]
    rho = [rng.uniform(0, 20) for _ in range(num_i)]
    beta = 5.0

    print("--- 零对偶 ---")
    _run_zero_dual(data, ctxs)

    print("--- 真实对偶 catalog 构建 (m0) ---")
    t0 = time.perf_counter()
    cat = build_batch_catalog(data, 0, alpha, rho, ctx=ctxs[0])
    nb = sum(len(cell) for u in cat for cell in u)
    print(f"catalog m0: {time.perf_counter() - t0:.1f}s  batches={nb}")

    print("--- 真实对偶 全机 ---")
    _run_real_dual(data, ctxs, alpha, rho, beta, 10.0, "限时")
    _run_real_dual(data, ctxs, alpha, rho, beta, EBRP_DEFAULT_TIME_LIMIT, "默认")
    _run_real_dual(data, ctxs, alpha, rho, beta, None, "精确")


if __name__ == "__main__":
    main()
