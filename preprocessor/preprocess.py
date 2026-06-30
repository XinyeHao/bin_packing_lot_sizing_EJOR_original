"""算例加载与预处理。"""

from __future__ import annotations

import json
from pathlib import Path

from common.data_models import InstanceData, ProcessedInstance
from configuration import SCRAP_COST_MULTIPLIER


def load_instance(path: str | Path) -> InstanceData:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return InstanceData.from_dict(data)


def preprocess(instance: InstanceData) -> ProcessedInstance:
    """将原始算例转换为索引化矩阵结构。"""
    periods = list(range(1, instance.num_periods + 1))
    configs = list(instance.config_ids)
    machines = list(instance.machine_ids)
    cured_items = list(instance.cured_item_ids)
    end_items = list(getattr(instance, "end_item_ids", [])) or []
    all_items = cured_items + end_items   # end_items 为空时即为 cured_items

    config_index = {u: idx for idx, u in enumerate(configs)}
    machine_index = {m: idx for idx, m in enumerate(machines)}
    cured_index = {i: idx for idx, i in enumerate(cured_items)}
    end_index = {j: idx for idx, j in enumerate(end_items)} if end_items else {}
    item_index = {item: idx for idx, item in enumerate(all_items)}

    num_u = len(configs)
    num_i = len(cured_items)
    num_j = len(end_items)
    num_m = len(machines)
    num_t = len(periods)

    l_u = [instance.config_lead_times[u] for u in configs]
    l_ti = [float(instance.item_lead_times[i]) for i in cured_items]
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

    p_cum = [
        [instance.production_costs[m][u] for u in configs]
        for m in machines
    ]

    # holding / backorder 只针对 cured_items（现为最终产品）
    h_c = [instance.holding_costs.get(item, 0.0) for item in cured_items]
    bc_j = [instance.backorder_costs.get(i, 0.0) for i in cured_items]   # 复用字段名，实际是 items 的 backorder cost

    # 不再需要 r_ij / bom_parents（无装配）
    r_ij = [[0] * max(1, num_j) for _ in range(num_i)]
    bom_parents: list[list[tuple[int, int]]] = [[] for _ in range(num_i)]

    # 需求现在直接在 cured items 上
    d_it = [[0] * num_t for _ in range(num_i)]
    for i_idx, i in enumerate(cured_items):
        for t, qty in instance.demand.get(i, {}).items():
            d_it[i_idx][t - 1] = qty

    # deadlines（1-based）
    deadlines = [instance.deadlines.get(i, num_t) for i in cured_items]

    w_i: list[int] = []
    sc_i: list[float] = []
    for i_idx, item in enumerate(cured_items):
        if item in instance.wip_quantities:
            w_i.append(int(instance.wip_quantities[item]))
        else:
            w_i.append(int(sum(d_it[i_idx])))
        if item in instance.scrap_costs:
            sc_i.append(float(instance.scrap_costs[item]))
        else:
            sc_i.append(SCRAP_COST_MULTIPLIER * bc_j[i_idx])

    # 兼容旧字段 d_jt（如果有 end 则保留，否则置空）
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
        min_q=min_q,
        deadlines=deadlines,
        w_i=w_i,
        sc_i=sc_i,
    )


def max_tray_count(q: float, v: float) -> int:
    return int(q // v)
