"""算例加载与预处理。"""

from __future__ import annotations

from pathlib import Path

from common.data_models import InstanceData, ProcessedInstance
from configuration import SCRAP_COST_MULTIPLIER
from preprocessor.instance_db import instance_name_from_arg, load_instance_from_db
from preprocessor.lead_time import compute_lt_it


def load_instance(path_or_name: str | Path) -> InstanceData:
    """从 instances/instances.db 按名称加载算例（兼容传入 *.json 路径，取 stem 为名称）。"""
    name = instance_name_from_arg(path_or_name)
    return load_instance_from_db(name)


def preprocess(instance: InstanceData) -> ProcessedInstance:
    """将原始算例转换为索引化矩阵结构，并生成阶跃占罐表 lt_it。"""
    periods = list(range(1, instance.num_periods + 1))
    configs = list(instance.config_ids)
    machines = list(instance.machine_ids)
    cured_items = list(instance.cured_item_ids)
    end_items = list(getattr(instance, "end_item_ids", [])) or []
    all_items = cured_items + end_items

    config_index = {u: idx for idx, u in enumerate(configs)}
    machine_index = {m: idx for idx, m in enumerate(machines)}
    cured_index = {i: idx for idx, i in enumerate(cured_items)}
    end_index = {j: idx for idx, j in enumerate(end_items)} if end_items else {}
    item_index = {item: idx for idx, item in enumerate(all_items)}

    num_u = len(configs)
    num_i = len(cured_items)
    num_j = len(end_items)
    num_t = len(periods)

    l_u = [instance.config_lead_times[u] for u in configs]
    v_i = [instance.tray_lengths[i] for i in cured_items]
    q_m = [instance.machine_capacities[m] for m in machines]
    min_q = min(q_m)

    b_iu = [[0] * num_u for _ in range(num_i)]
    config_of_item = [0] * num_i
    items_by_config: list[list[int]] = [[] for _ in range(num_u)]
    for i_idx, item in enumerate(cured_items):
        u_name = instance.item_config[item]
        u_idx = config_index[u_name]
        b_iu[i_idx][u_idx] = 1
        config_of_item[i_idx] = u_idx
        items_by_config[u_idx].append(i_idx)

    num_m = len(machines)
    b_im = [[0] * num_m for _ in range(num_i)]
    items_by_machine: list[list[int]] = [[] for _ in range(num_m)]
    allowed_machine_count = [num_m] * num_i
    for i_idx, item in enumerate(cured_items):
        allowed_names = instance.item_machines.get(item)
        if not allowed_names:
            allowed_names = list(machines)
        else:
            allowed_names = list(dict.fromkeys(allowed_names))
            for m_name in allowed_names:
                if m_name not in machine_index:
                    raise ValueError(f"物料 {item} 的可加工机 {m_name} 不存在")
        allowed_machine_count[i_idx] = len(allowed_names)
        for m_name in allowed_names:
            m_idx = machine_index[m_name]
            b_im[i_idx][m_idx] = 1
            items_by_machine[m_idx].append(i_idx)

    p_cum = [
        [instance.production_costs[m][u] for u in configs]
        for m in machines
    ]

    h_c = [instance.holding_costs.get(item, 0.0) for item in cured_items]
    bc_j = [instance.backorder_costs.get(i, 0.0) for i in cured_items]

    r_ij = [[0] * max(1, num_j) for _ in range(num_i)]
    bom_parents: list[list[tuple[int, int]]] = [[] for _ in range(num_i)]

    d_it = [[0] * num_t for _ in range(num_i)]
    for i_idx, i in enumerate(cured_items):
        for t, qty in instance.demand.get(i, {}).items():
            d_it[i_idx][t - 1] = qty

    deadlines = [instance.deadlines.get(i, num_t) for i in cured_items]
    earliest_starts = [instance.earliest_starts.get(i, 1) for i in cured_items]
    for i_idx, item in enumerate(cured_items):
        if earliest_starts[i_idx] > deadlines[i_idx]:
            raise ValueError(
                f"物料 {item} 的时间窗无效: rel_i={earliest_starts[i_idx]} > end_i={deadlines[i_idx]}"
            )

    lt_it = compute_lt_it(
        cured_items=cured_items,
        item_config=instance.item_config,
        config_lead_times=instance.config_lead_times,
        config_wait_steps=instance.config_wait_steps,
        earliest_starts=instance.earliest_starts,
        deadlines=instance.deadlines,
        num_periods=instance.num_periods,
        period_hours=instance.period_hours,
        item_wait_steps=instance.item_wait_steps,
    )
    # 兼容仍使用固定 l_ti 的求解器：取 rel_i 时刻的占罐长度
    l_ti = [
        float(lt_it[i_idx][earliest_starts[i_idx] - 1])
        for i_idx in range(num_i)
    ]

    w_i: list[int] = []
    sc_i: list[float] = []
    for i_idx, item in enumerate(cured_items):
        w_i.append(int(instance.wip_quantities.get(item, 1)))
        if item in instance.scrap_costs:
            sc_i.append(float(instance.scrap_costs[item]))
        else:
            sc_i.append(SCRAP_COST_MULTIPLIER * bc_j[i_idx])

    d_jt = [[0] * num_t for _ in range(max(1, num_j))]
    if end_items:
        for j_idx, j in enumerate(end_items):
            for t, qty in instance.demand.get(j, {}).items():
                d_jt[j_idx][t - 1] = qty

    return ProcessedInstance(
        raw=instance,
        periods=periods,
        configs=configs,
        machines=machines,
        cured_items=cured_items,
        end_items=end_items,
        all_items=all_items,
        config_index=config_index,
        machine_index=machine_index,
        cured_index=cured_index,
        end_index=end_index,
        item_index=item_index,
        l_u=l_u,
        l_ti=l_ti,
        b_iu=b_iu,
        v_i=v_i,
        q_m=q_m,
        p_cum=p_cum,
        h_c=h_c,
        bc_j=bc_j,
        r_ij=r_ij,
        d_jt=d_jt,
        d_it=d_it,
        bom_parents=bom_parents,
        items_by_config=items_by_config,
        config_of_item=config_of_item,
        b_im=b_im,
        items_by_machine=items_by_machine,
        allowed_machine_count=allowed_machine_count,
        min_q=min_q,
        deadlines=deadlines,
        earliest_starts=earliest_starts,
        w_i=w_i,
        sc_i=sc_i,
        lt_it=lt_it,
    )


def max_tray_count(q: float, v: float) -> int:
    return int(q // v)
