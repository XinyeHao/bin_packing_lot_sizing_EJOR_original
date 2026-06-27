"""算例加载与预处理。"""

from __future__ import annotations

import json
from pathlib import Path

from common.data_models import InstanceData, ProcessedInstance


def load_instance(path: str | Path) -> InstanceData:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return InstanceData.from_dict(data)


def preprocess(instance: InstanceData) -> ProcessedInstance:
    """将原始算例转换为索引化矩阵结构。"""
    periods = list(range(1, instance.num_periods + 1))
    configs = list(instance.config_ids)
    machines = list(instance.machine_ids)
    cured_items = list(instance.cured_item_ids)
    end_items = list(instance.end_item_ids)
    all_items = cured_items + end_items

    config_index = {u: idx for idx, u in enumerate(configs)}
    machine_index = {m: idx for idx, m in enumerate(machines)}
    cured_index = {i: idx for idx, i in enumerate(cured_items)}
    end_index = {j: idx for idx, j in enumerate(end_items)}
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

    h_c = [instance.holding_costs[item] for item in all_items]
    bc_j = [instance.backorder_costs[j] for j in end_items]

    r_ij = [[0] * num_j for _ in range(num_i)]
    bom_parents: list[list[tuple[int, int]]] = [[] for _ in range(num_i)]
    for j_idx, j in enumerate(end_items):
        for i_idx, i in enumerate(cured_items):
            qty = instance.bom[j].get(i, 0)
            r_ij[i_idx][j_idx] = qty
            if qty > 0:
                bom_parents[i_idx].append((j_idx, qty))

    d_jt = [[0] * num_t for _ in range(num_j)]
    for j_idx, j in enumerate(end_items):
        for t, qty in instance.demand[j].items():
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
        bom_parents=bom_parents,
        items_by_config=items_by_config,
        config_of_item=config_of_item,
        min_q=min_q,
    )


def max_tray_count(q: float, v: float) -> int:
    return int(q // v)
