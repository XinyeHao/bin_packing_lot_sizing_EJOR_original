"""EBRP 并行定价墙钟压测（精确；墙钟≈max_m T_m）。"""
from __future__ import annotations

import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_and_algo.pricing_ebrp import EBRPStats, build_ebrp_contexts, solve_subproblem_ebrp
from preprocessor.preprocess import load_instance, preprocess


def bench(instance: str, label: str, alpha, rho, betas: list[float]) -> None:
    data = preprocess(load_instance(instance))
    ctxs = build_ebrp_contexts(data)
    sizes = [len(xs) for xs in data.items_by_machine]
    print(f"\n=== {label} | {instance} |I_m|={sizes} max={max(sizes)} ===")
    times: list[float] = []
    for m, ctx in enumerate(ctxs):
        st = EBRPStats()
        t0 = time.perf_counter()
        solve_subproblem_ebrp(
            data, m, alpha, betas[m], rho=rho, stats=st, ctx=ctx, time_limit=None, progress_interval_sec=0.0
        )
        sec = time.perf_counter() - t0
        times.append(sec)
        print(f"  m{m}: {sec:.3f}s exact={st.exact} nodes={st.nodes_expanded}")
    print(f"  parallel_wall≈max_m={max(times):.3f}s")


def main() -> None:
    instance = "set_c_01_elig"
    data = preprocess(load_instance(instance))
    num_i, num_t = len(data.cured_items), len(data.periods)
    rng = random.Random(42)
    alpha = [[rng.uniform(0, 50) for _ in range(num_t)] for _ in range(num_i)]
    rho = [rng.uniform(0, 20) for _ in range(num_i)]
    betas = [5.0] * len(data.machines)
    bench(instance, "random_dual", alpha, rho, betas)

    zero = [[0.0] * num_t for _ in range(num_i)]
    zrho = [0.0] * num_i
    bench(instance, "zero_dual", zero, zrho, [0.0] * len(data.machines))


if __name__ == "__main__":
    main()
