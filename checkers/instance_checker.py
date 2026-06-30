"""算例数据一致性检查。"""

from __future__ import annotations

from common.data_models import InstanceData, ProcessedInstance
from checkers.report import CheckReport
from configuration import BACKORDER_COST_MULTIPLIER, CHECKER_TOL, SCRAP_COST_MULTIPLIER


def check_instance_raw(instance: InstanceData) -> CheckReport:
    report = CheckReport(name=f"instance:{instance.name}")

    for i in instance.cured_item_ids:
        u = instance.item_config[i]
        if u not in instance.config_ids:
            report.add_violation("instance", "item_config", f"物料 {i} 的配置 {u} 不存在")
        elif instance.item_lead_times[i] != instance.config_lead_times[u]:
            report.add_violation(
                "instance",
                "lead_time_mismatch",
                f"物料 {i} 提前期 {instance.item_lead_times[i]} 与配置 {u} 的 {instance.config_lead_times[u]} 不一致",
            )

    for j in instance.end_item_ids:
        if sum(instance.bom[j].values()) == 0:
            report.add_violation("instance", "empty_bom", f"成品 {j} 的 BOM 全为 0")
        if instance.holding_costs[j] <= 0:
            report.add_violation("instance", "holding_cost", f"成品 {j} 的 holding 成本非正")

    for j in instance.end_item_ids:
        expected = BACKORDER_COST_MULTIPLIER * instance.holding_costs[j]
        if abs(instance.backorder_costs[j] - expected) > CHECKER_TOL["instance_backorder"]:
            report.add_violation(
                "instance",
                "backorder_cost",
                f"成品 {j} backorder 成本 {instance.backorder_costs[j]} != 10 * hc = {expected}",
            )

    for i in instance.cured_item_ids:
        bc = instance.backorder_costs.get(i, 0.0)
        if i in instance.scrap_costs:
            sc = instance.scrap_costs[i]
        else:
            sc = SCRAP_COST_MULTIPLIER * bc
        if sc <= bc + CHECKER_TOL["instance_backorder"]:
            report.add_violation(
                "instance",
                "scrap_cost",
                f"物料 {i} 报废成本 {sc} 须大于 backorder 成本 {bc}",
            )

    return report


def check_processed(data: ProcessedInstance) -> CheckReport:
    report = CheckReport(name=f"processed:{data.raw.name}")
    report.merge(check_instance_raw(data.raw))

    num_i = len(data.cured_items)
    for i_idx, item in enumerate(data.cured_items):
        u_idx = data.config_of_item[i_idx]
        if data.b_iu[i_idx][u_idx] != 1:
            report.add_violation("processed", "b_iu", f"物料 {item} 的 b_iu 与配置映射不一致")
        if int(data.l_ti[i_idx]) != data.l_u[u_idx]:
            report.add_violation("processed", "l_ti", f"物料 {item} 的 l_ti 与配置提前期不一致")

    for i_idx in range(num_i):
        if i_idx not in [x for group in data.items_by_config for x in group]:
            report.add_violation("processed", "items_by_config", f"物料 {data.cured_items[i_idx]} 未出现在 items_by_config")

    return report
