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
                f"物料 {i} item_lead_times {instance.item_lead_times[i]} 与配置 {u} 基准 {instance.config_lead_times[u]} 不一致",
            )
        for step in instance.item_wait_steps.get(i, []):
            if len(step) != 2 or step[0] <= 0 or step[1] <= 0:
                report.add_violation(
                    "instance",
                    "wait_steps",
                    f"物料 {i} 的等待阶梯 {step} 无效",
                )
        rel = instance.earliest_starts.get(i, 1)
        end = instance.deadlines.get(i, instance.num_periods)
        if i in instance.item_machines:
            for m in instance.item_machines[i]:
                if m not in instance.machine_ids:
                    report.add_violation("instance", "item_machines", f"物料 {i} 的可加工机 {m} 不存在")
            if not instance.item_machines[i]:
                report.add_violation("instance", "item_machines", f"物料 {i} 的 M_i 为空")
        span = end - rel
        for step in instance.item_wait_steps.get(i, []):
            if step[0] > span:
                report.add_violation(
                    "instance",
                    "wait_steps",
                    f"物料 {i} 阶梯阈值 ω={step[0]} 超出时间窗跨度 {span}",
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
        e_i = instance.earliest_starts.get(i, 1)
        d_i = instance.deadlines.get(i, instance.num_periods)
        if e_i > d_i:
            report.add_violation(
                "instance",
                "wip_window",
                f"物料 {i} 时间窗无效: rel_i={e_i} > end_i={d_i}",
            )
        if e_i < 1 or d_i > instance.num_periods:
            report.add_violation(
                "instance",
                "wip_window",
                f"物料 {i} 时间窗越界: rel_i={e_i}, end_i={d_i}, T={instance.num_periods}",
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
        rel = data.earliest_starts[i_idx]
        end = data.deadlines[i_idx]
        base = data.l_u[u_idx]
        prev = data.lt_it[i_idx][rel - 1]
        if prev < base:
            report.add_violation(
                "processed",
                "lt_it",
                f"物料 {item} 在 rel_i={rel} 的 lt={prev} 小于基准 l_u={base}",
            )
        for t in range(rel, end + 1):
            val = data.lt_it[i_idx][t - 1]
            if t > rel and val < data.lt_it[i_idx][t - 2]:
                report.add_violation(
                    "processed",
                    "lt_it_monotone",
                    f"物料 {item} 在 t={t} 的 lt_it 非单调",
                )
        if int(data.l_ti[i_idx]) != data.lt_it[i_idx][rel - 1]:
            report.add_violation(
                "processed",
                "l_ti",
                f"物料 {item} 的 l_ti 与 rel_i 时刻 lt_it 不一致",
            )

    for i_idx in range(num_i):
        if sum(data.b_im[i_idx]) < 1:
            report.add_violation(
                "processed",
                "item_machines",
                f"物料 {data.cured_items[i_idx]} 无允许加工机台",
            )
        if data.allowed_machine_count[i_idx] != sum(data.b_im[i_idx]):
            report.add_violation(
                "processed",
                "item_machines",
                f"物料 {data.cured_items[i_idx]} |M_i| 与 b_im 不一致",
            )

    for i_idx in range(num_i):
        if i_idx not in [x for group in data.items_by_config for x in group]:
            report.add_violation("processed", "items_by_config", f"物料 {data.cured_items[i_idx]} 未出现在 items_by_config")

    for i_idx in range(num_i):
        if data.sc_i[i_idx] <= data.bc_j[i_idx] + CHECKER_TOL["instance_backorder"]:
            report.add_violation(
                "processed",
                "scrap_cost",
                f"物料 {data.cured_items[i_idx]} 报废成本 {data.sc_i[i_idx]} 须大于 backorder {data.bc_j[i_idx]}",
            )

    for i_idx in range(num_i):
        e_i = data.earliest_starts[i_idx]
        d_i = data.deadlines[i_idx]
        if e_i > d_i:
            report.add_violation(
                "processed",
                "wip_window",
                f"物料 {data.cured_items[i_idx]} 时间窗无效: rel_i={e_i} > end_i={d_i}",
            )

    return report
