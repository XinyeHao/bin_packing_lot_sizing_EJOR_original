"""项目主入口。"""

from __future__ import annotations

import argparse

from common.paths import INSTANCES_DIR, RESULT_DIR
from preprocessor.instance_db import instance_db_path
from configuration import (
    DEFAULT_CG_MAX_ITERATIONS,
    DEFAULT_METHOD,
    DEFAULT_MIP_GAP,
    DEFAULT_TIME_LIMIT,
    SET_C_CG_TIME_LIMIT,
    SET_C_IM_TIME_LIMIT,
    SOLVER_METHODS,
)
from instance_generator.generator import (
    capacity_metrics,
    generate_demo_instance,
    generate_instance_set,
    generate_large_instance,
    generate_set_c_instance_set,
)
from model_and_algo.solver import solve
from postprocessor.postprocess import format_summary, save_result
from preprocessor.preprocess import load_instance, preprocess


def main() -> None:
    parser = argparse.ArgumentParser(description="集成装箱与批量计划求解器")
    parser.add_argument("--generate", choices=["demo", "large", "C", "A", "B", "all"], help="生成算例")
    parser.add_argument("--instance", type=str, help="算例名称（对应 instances/{name}.db）")
    parser.add_argument(
        "--method",
        choices=list(SOLVER_METHODS),
        default=DEFAULT_METHOD,
    )
    parser.add_argument("--time-limit", type=float, default=None, help="求解时限（秒）；Set-C 算例默认 1200")
    parser.add_argument("--mip-gap", type=float, default=DEFAULT_MIP_GAP)
    parser.add_argument("--cg-iterations", type=int, default=DEFAULT_CG_MAX_ITERATIONS)
    parser.add_argument("--verbose", action="store_true", help="打印列生成等求解过程")
    args = parser.parse_args()

    if args.generate:
        if args.generate == "demo":
            generate_demo_instance()
            print(f"已生成演示算例 → {instance_db_path('demo_small')}")
        elif args.generate == "large":
            generate_large_instance()
            print(f"已生成大算例 → {instance_db_path('large_01')}")
        elif args.generate == "C":
            from configuration import SET_C_INSTANCE_COUNT
            from collections import Counter

            instances = generate_set_c_instance_set(count=SET_C_INSTANCE_COUNT)
            for inst in instances:
                m = capacity_metrics(inst)
                per_u = Counter(inst.item_config.values())
                u_summary = ", ".join(f"{u}:{per_u[u]}" for u in sorted(per_u))
                print(f"已生成 Set-C → {instance_db_path(inst.name)}")
                print(
                    f"  T={inst.num_periods}, M={len(inst.machine_ids)}, U={len(inst.config_ids)}, "
                    f"|I|={int(m['num_items'])}, 托盘总负载={m['tray_load_util']:.1%}, "
                    f"件数密度={m['slot_density']:.1%}, 估计每批≈{m.get('est_avg_items_per_batch', 0):.1f}件, "
                    f"各配置件数=[{u_summary}]"
                )
        elif args.generate == "all":
            generate_instance_set("A")
            generate_instance_set("B")
            print(f"已生成 Set-A 与 Set-B 各 20 个算例 → {INSTANCES_DIR}/*.db")
        else:
            generate_instance_set(args.generate)
            print(f"已生成 Set-{args.generate} 算例 → {INSTANCES_DIR}/*.db")
        return

    if not args.instance:
        args.instance = "demo_small"

    if args.time_limit is not None:
        time_limit = args.time_limit
    elif args.instance.startswith("set_c_"):
        if args.method in ("cg", "column_generation"):
            time_limit = SET_C_CG_TIME_LIMIT
        else:
            time_limit = SET_C_IM_TIME_LIMIT
    else:
        time_limit = DEFAULT_TIME_LIMIT

    data = preprocess(load_instance(args.instance))
    result = solve(
        data,
        method=args.method,
        time_limit=time_limit,
        mip_gap=args.mip_gap,
        cg_max_iterations=args.cg_iterations,
        verbose=args.verbose or args.instance.startswith("set_c_"),
    )
    output_path = save_result(result, RESULT_DIR)
    print(format_summary(result))
    print(f"结果已保存: {output_path}")


if __name__ == "__main__":
    main()
