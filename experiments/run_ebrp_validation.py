"""EBRP 小算例验证 + Set-C CG 对比实验。"""



from __future__ import annotations



import argparse

import sys

import time

from pathlib import Path



ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:

    sys.path.insert(0, str(ROOT))



from configuration import SET_C_CG, SET_C_CG_EBRP_TIMED, SET_C_CG_IM

from model_and_algo.pricing_ebrp import EBRPStats, build_ebrp_contexts, solve_subproblem_ebrp

from model_and_algo.pricing_im_gurobi import solve_subproblem_im_gurobi

from model_and_algo.column_generation import solve_column_generation

from preprocessor.preprocess import load_instance, preprocess





def _zero_duals(data):

    num_i = len(data.cured_items)

    num_t = len(data.periods)

    alpha = [[0.0] * num_t for _ in range(num_i)]

    rho = [0.0] * num_i

    return alpha, rho





def benchmark_ebrp(instance_name: str, repeats: int = 5) -> None:

    """精确 EBRP 性能基准（毫秒级目标：小算例）。"""

    import statistics



    data = preprocess(load_instance(instance_name))

    contexts = build_ebrp_contexts(data)

    alpha, rho = _zero_duals(data)

    beta = 0.0

    num_i = len(data.cured_items)

    print(f"\n=== EBRP benchmark: {instance_name} |I|={num_i} |M|={len(data.machines)} ===")



    for m_idx, machine in enumerate(data.machines):

        times: list[float] = []

        nodes = 0

        for _ in range(repeats):

            stats = EBRPStats()

            t0 = time.perf_counter()

            sp = solve_subproblem_ebrp(

                data,

                m_idx,

                alpha,

                beta,

                rho=rho,

                stats=stats,

                ctx=contexts[m_idx],

                progress_interval_sec=0.0,

            )

            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            times.append(elapsed_ms)

            nodes = stats.nodes_expanded

        med = statistics.median(times)

        print(

            f"  {machine}: RC={sp.reduced_cost:.4f}  median={med:.2f}ms  "

            f"nodes={nodes} pareto_prunes={stats.pareto_prunes} exact={stats.exact}"

        )





def compare_pricing(instance_name: str, machine_idx: int = 0) -> None:

    data = preprocess(load_instance(instance_name))

    alpha, rho = _zero_duals(data)

    beta = 0.0

    num_i = len(data.cured_items)



    print(f"\n=== pricing compare: {instance_name} |I|={num_i} ===")



    t0 = time.perf_counter()

    sp_g = solve_subproblem_im_gurobi(data, machine_idx, alpha, beta, rho=rho, time_limit=120.0)

    t_g = time.perf_counter() - t0

    print(f"Gurobi §3.3  RC={sp_g.reduced_cost:.6f}  time={t_g:.3f}s")



    stats = EBRPStats()

    t0 = time.perf_counter()

    sp_e = solve_subproblem_ebrp(

        data, machine_idx, alpha, beta, rho=rho, stats=stats,

        time_limit=120.0 if num_i > 40 else None,

    )

    t_e = time.perf_counter() - t0

    print(

        f"EBRP       RC={sp_e.reduced_cost:.6f}  time={t_e:.3f}s  "

        f"exact={stats.exact} nodes={stats.nodes_expanded}"

    )

    diff = abs(sp_g.reduced_cost - sp_e.reduced_cost)

    print(f"RC match: {diff <= 1e-4}  (diff={diff:.2e})")





def smoke_ebrp_setc(instance_name: str = "set_c_01", time_limit: float = 60.0) -> None:

    """每台机器限时 EBRP 定价（零对偶），观察节点数与是否超时。"""

    data = preprocess(load_instance(instance_name))

    alpha, rho = _zero_duals(data)

    beta = 0.0

    num_i = len(data.cured_items)

    print(f"\n=== EBRP smoke (timed): {instance_name} |I|={num_i} limit={time_limit}s ===")

    for m_idx, machine in enumerate(data.machines):

        stats = EBRPStats()

        t0 = time.perf_counter()

        sp = solve_subproblem_ebrp(

            data, m_idx, alpha, beta, rho=rho, stats=stats, time_limit=time_limit,

        )

        elapsed = time.perf_counter() - t0

        print(

            f"  {machine}: RC={sp.reduced_cost:.4f}  time={elapsed:.1f}s  "

            f"exact={stats.exact} timed_out={stats.timed_out} nodes={stats.nodes_expanded} memo={stats.memo_misses}"

        )





def run_cg(

    instance_name: str,

    pricing_mode: str,

    time_limit: float,

    max_iters: int = 50,

    *,

    ebrp_time_limit: float | None = None,

) -> dict:

    data = preprocess(load_instance(instance_name))

    print(f"\n=== CG ({pricing_mode}): {instance_name} ===")

    t0 = time.perf_counter()

    kwargs = {

        "max_iterations": max_iters,

        "pricing_mode": pricing_mode,

        "time_limit": time_limit,

        "verbose": True,

        "use_gurobi_init": False,

        "pricing_inprocess": True,

        "pricing_workers": 0,

        "im_pricing_time_limit": 90.0,

        "ebrp_time_limit": 120.0,

    }

    if instance_name.startswith("set_c_") and pricing_mode == "ebrp":

        kwargs.update(SET_C_CG)

        kwargs["pricing_mode"] = "ebrp"

    if instance_name.startswith("set_c_") and pricing_mode == "im_gurobi":

        kwargs.update(SET_C_CG_IM)

    if instance_name.startswith("set_c_") and pricing_mode in ("ebrp_force", "ebrp_timed"):

        kwargs.update(SET_C_CG_EBRP_TIMED)

        kwargs["pricing_mode"] = "ebrp_force"

        kwargs["ebrp_force"] = True

        kwargs["max_iterations"] = min(max_iters, kwargs.get("max_iterations", max_iters))

    if ebrp_time_limit is not None:

        kwargs["ebrp_time_limit"] = ebrp_time_limit

    result = solve_column_generation(data, **kwargs)

    elapsed = time.perf_counter() - t0

    summary = {

        "pricing_mode": pricing_mode,

        "lb": result.lower_bound,

        "status": result.status,

        "iters": len(result.extra.get("iteration_rcs", [])),

        "time": elapsed,

    }

    print(f"DONE: {summary}")

    return summary





def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument("--instance", default="demo_small")

    parser.add_argument("--compare-pricing", action="store_true")

    parser.add_argument("--cg", action="store_true")

    parser.add_argument("--compare-setc", action="store_true")

    parser.add_argument("--benchmark", action="store_true", help="EBRP 精确求解性能基准")

    parser.add_argument("--setc-ebrp-smoke", action="store_true", help="每台机器限时 EBRP 定价 smoke test")

    parser.add_argument("--setc-ebrp-timed", action="store_true", help="set_c_01 CG with timed EBRP (ebrp_force)")

    parser.add_argument("--ebrp-time-limit", type=float, default=60.0)

    parser.add_argument("--pricing-mode", default="ebrp", choices=["ebrp", "ebrp_force", "im_gurobi"])

    parser.add_argument("--time-limit", type=float, default=900.0)

    parser.add_argument("--max-iters", type=int, default=30)

    args = parser.parse_args()



    if args.benchmark:

        benchmark_ebrp(args.instance)

        return



    if args.setc_ebrp_smoke:

        smoke_ebrp_setc("set_c_01", time_limit=args.ebrp_time_limit)

        return



    if args.setc_ebrp_timed:

        run_cg(

            "set_c_01",

            "ebrp_force",

            args.time_limit,

            args.max_iters,

            ebrp_time_limit=args.ebrp_time_limit,

        )

        return



    if args.compare_setc:

        inst = "set_c_01"

        ebrp = run_cg(inst, "ebrp", args.time_limit, args.max_iters)

        im = run_cg(inst, "im_gurobi", args.time_limit, args.max_iters)

        print("\n=== Set-C CG summary ===")

        print(f"EBRP:      LB={ebrp['lb']:.2f} iters={ebrp['iters']} time={ebrp['time']:.1f}s")

        print(f"IM/Gurobi: LB={im['lb']:.2f} iters={im['iters']} time={im['time']:.1f}s")

        return



    if args.compare_pricing:

        compare_pricing(args.instance)

    if args.cg:

        run_cg(args.instance, args.pricing_mode, args.time_limit, args.max_iters)

    if not args.compare_pricing and not args.cg and not args.compare_setc:

        compare_pricing(args.instance)





if __name__ == "__main__":

    main()

