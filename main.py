"""项目主入口。"""

from __future__ import annotations

import argparse

from common.paths import INSTANCES_DIR, RESULT_DIR
from configuration import (
    DEFAULT_CG_MAX_ITERATIONS,
    DEFAULT_METHOD,
    DEFAULT_MIP_GAP,
    DEFAULT_TIME_LIMIT,
    SOLVER_METHODS,
)
from instance_generator.generator import generate_demo_instance, generate_instance_set
from model_and_algo.solver import solve
from postprocessor.postprocess import format_summary, save_result
from preprocessor.preprocess import load_instance, preprocess


def main() -> None:
    parser = argparse.ArgumentParser(description="集成装箱与批量计划求解器")
    parser.add_argument("--generate", choices=["demo", "A", "B", "all"], help="生成算例")
    parser.add_argument("--instance", type=str, help="算例 JSON 路径或文件名")
    parser.add_argument(
        "--method",
        choices=list(SOLVER_METHODS),
        default=DEFAULT_METHOD,
    )
    parser.add_argument("--time-limit", type=float, default=DEFAULT_TIME_LIMIT)
    parser.add_argument("--mip-gap", type=float, default=DEFAULT_MIP_GAP)
    parser.add_argument("--cg-iterations", type=int, default=DEFAULT_CG_MAX_ITERATIONS)
    args = parser.parse_args()

    if args.generate:
        if args.generate == "demo":
            generate_demo_instance()
            print("已生成演示算例 demo_small.json")
        elif args.generate == "all":
            generate_instance_set("A")
            generate_instance_set("B")
            print("已生成 Set-A 与 Set-B 各 20 个算例")
        else:
            generate_instance_set(args.generate)
            print(f"已生成 Set-{args.generate} 算例")
        return

    if not args.instance:
        args.instance = str(INSTANCES_DIR / "demo_small.json")

    instance_path = args.instance
    if not instance_path.endswith(".json"):
        instance_path = str(INSTANCES_DIR / f"{instance_path}.json")

    data = preprocess(load_instance(instance_path))
    result = solve(
        data,
        method=args.method,
        time_limit=args.time_limit,
        mip_gap=args.mip_gap,
        cg_max_iterations=args.cg_iterations,
    )
    output_path = save_result(result, RESULT_DIR)
    print(format_summary(result))
    print(f"结果已保存: {output_path}")


if __name__ == "__main__":
    main()
