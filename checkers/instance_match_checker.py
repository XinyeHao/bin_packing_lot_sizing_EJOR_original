"""结果与算例一致性检查。"""

from __future__ import annotations

from common.data_models import ProcessedInstance, SolutionResult
from checkers.report import CheckReport


def check_instance_match(data: ProcessedInstance, result: SolutionResult) -> CheckReport:
    report = CheckReport(name=f"instance_match:{result.instance_name}:{result.method}")

    if result.instance_name != data.raw.name:
        report.add_violation(
            "instance_match",
            "name_mismatch",
            f"结果算例名 {result.instance_name} != 当前算例 {data.raw.name}",
        )

    saved_seed = result.extra.get("instance_seed")
    if saved_seed is not None and data.raw.seed is not None and int(saved_seed) != int(data.raw.seed):
        report.add_violation(
            "instance_match",
            "seed_mismatch",
            f"结果 seed={saved_seed} 与当前算例 seed={data.raw.seed} 不一致，可能是旧结果",
        )
        report.metrics["hint"] = "请对当前算例重新求解后再比较"

    saved_set = result.extra.get("set_type")
    if saved_set is not None and saved_set != data.raw.set_type:
        report.add_violation(
            "instance_match",
            "set_type_mismatch",
            f"结果 set_type={saved_set} != 当前 {data.raw.set_type}",
        )

    return report
