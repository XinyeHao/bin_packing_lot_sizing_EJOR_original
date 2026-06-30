"""按论文 Section 5.1 规则生成算例。"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

from common.data_models import InstanceData
from common.paths import INSTANCES_DIR
from configuration import (
    BACKORDER_COST_MULTIPLIER,
    SCRAP_COST_MULTIPLIER,
    C_STAR_RANGE,
    DEADLINE_MIN_FRACTION,
    DEMAND_CHOICES,
    DEMAND_PERIODS_PER_ITEM,
    DEMO_CONFIG,
    DEMO_INSTANCE_NAME,
    DEMO_INSTANCE_SEED,
    HOLDING_COST_DIVISOR,
    INSTANCE_BASE_SEED,
    INSTANCE_COUNT_DEFAULT,
    LEAD_TIME_CHOICES,
    MACHINE_CAPACITY_RANGE,
    SET_CONFIGS,
    TRAY_LENGTH_LONG_PROB,
    TRAY_LENGTH_LONG_RANGE,
    TRAY_LENGTH_SHORT_RANGE,
    SetConfig,
)


def _generate_demands(item_ids: list[str], num_periods: int, rng: random.Random) -> dict[str, dict[int, int]]:
    """为每个物料随机抽取若干需求时段。"""
    demand: dict[str, dict[int, int]] = {i: {} for i in item_ids}
    candidate_periods = list(range(1, num_periods + 1))
    for i in item_ids:
        periods = rng.sample(candidate_periods, k=min(DEMAND_PERIODS_PER_ITEM, num_periods))
        for t in periods:
            demand[i][t] = rng.choice(DEMAND_CHOICES)
    return demand


def _generate_deadlines(item_ids: list[str], num_periods: int, rng: random.Random) -> dict[str, int]:
    """最晚进罐 D_i：在计划期后 30% 至 T 之间均匀随机（D_i 过小则不允许）。"""
    min_dl = max(1, math.ceil(DEADLINE_MIN_FRACTION * num_periods))
    return {i: rng.randint(min_dl, num_periods) for i in item_ids}


def generate_instance(
    set_config: SetConfig,
    instance_id: int,
    seed: int | None = None,
) -> InstanceData:
    """生成单个算例。"""
    rng = random.Random(seed)
    num_periods = set_config.num_periods

    config_ids = [f"u{k}" for k in range(set_config.num_configs)]
    machine_ids = [f"m{k}" for k in range(set_config.num_machines)]
    # 新问题：只有“cured items”，它们即为最终产品
    item_ids = [f"i{k}" for k in range(set_config.num_cured_items)]   # 仍用 cured_item_ids 字段名以减少代码改动

    config_lead_times = {u: rng.choice(LEAD_TIME_CHOICES) for u in config_ids}
    item_config: dict[str, str] = {}
    item_lead_times: dict[str, int] = {}
    for i in item_ids:
        u = rng.choice(config_ids)
        item_config[i] = u
        item_lead_times[i] = config_lead_times[u]

    tray_lengths: dict[str, float] = {}
    for i in item_ids:
        if rng.random() < TRAY_LENGTH_LONG_PROB:
            tray_lengths[i] = rng.uniform(*TRAY_LENGTH_LONG_RANGE)
        else:
            tray_lengths[i] = rng.uniform(*TRAY_LENGTH_SHORT_RANGE)

    machine_capacities = {m: rng.uniform(*MACHINE_CAPACITY_RANGE) for m in machine_ids}
    min_q = min(machine_capacities.values())
    c_star = rng.uniform(*C_STAR_RANGE)

    production_costs: dict[str, dict[str, float]] = {}
    for m in machine_ids:
        production_costs[m] = {}
        for u in config_ids:
            l_u = config_lead_times[u]
            production_costs[m][u] = c_star * l_u * machine_capacities[m] / min_q

    holding_costs: dict[str, float] = {}
    for i in item_ids:
        holding_costs[i] = c_star * item_lead_times[i] / (HOLDING_COST_DIVISOR * num_periods)

    backorder_costs = {i: BACKORDER_COST_MULTIPLIER * holding_costs[i] for i in item_ids}
    scrap_costs = {i: SCRAP_COST_MULTIPLIER * backorder_costs[i] for i in item_ids}

    demand = _generate_demands(item_ids, num_periods, rng)
    deadlines = _generate_deadlines(item_ids, num_periods, rng)
    wip_quantities = {i: int(sum(demand[i].values())) for i in item_ids}

    return InstanceData(
        name=f"set_{set_config.set_type.lower()}_{instance_id:02d}",
        set_type=set_config.set_type,
        num_periods=num_periods,
        config_ids=config_ids,
        machine_ids=machine_ids,
        cured_item_ids=item_ids,          # 这些 item 现在就是最终产品
        end_item_ids=[],                  # 无装配
        config_lead_times=config_lead_times,
        item_config=item_config,
        item_lead_times=item_lead_times,
        tray_lengths=tray_lengths,
        machine_capacities=machine_capacities,
        production_costs=production_costs,
        holding_costs=holding_costs,
        backorder_costs=backorder_costs,
        bom={},
        demand=demand,
        deadlines=deadlines,
        wip_quantities=wip_quantities,
        scrap_costs=scrap_costs,
        seed=seed,
    )


def generate_instance_set(
    set_type: str,
    count: int = INSTANCE_COUNT_DEFAULT,
    base_seed: int = INSTANCE_BASE_SEED,
    output_dir: Path | None = None,
) -> list[InstanceData]:
    """生成一组算例并写入 instances 目录。"""
    if set_type not in SET_CONFIGS:
        raise ValueError(f"Unknown set type: {set_type}")
    config = SET_CONFIGS[set_type]
    out = output_dir or INSTANCES_DIR
    out.mkdir(parents=True, exist_ok=True)

    instances: list[InstanceData] = []
    for idx in range(1, count + 1):
        instance = generate_instance(config, idx, seed=base_seed + idx)
        instances.append(instance)
        path = out / f"{instance.name}.json"
        path.write_text(json.dumps(instance.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return instances


def generate_demo_instance(output_dir: Path | None = None) -> InstanceData:
    """生成小规模演示算例，便于快速调试。"""
    instance = generate_instance(DEMO_CONFIG, 1, seed=DEMO_INSTANCE_SEED)
    instance.name = DEMO_INSTANCE_NAME
    out = output_dir or INSTANCES_DIR
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{instance.name}.json"
    path.write_text(json.dumps(instance.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return instance
