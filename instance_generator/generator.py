"""按论文 Section 5.1 规则生成算例，写入 instances/{name}.db。"""

from __future__ import annotations

import math
import random
from pathlib import Path

from common.data_models import InstanceData
from configuration import (
    BACKORDER_COST_MULTIPLIER,
    C_STAR_RANGE,
    DEFAULT_PERIOD_HOURS,
    DEMO_CONFIG,
    DEMO_INSTANCE_NAME,
    DEMO_INSTANCE_SEED,
    DEMAND_CHOICES,
    DEMAND_PERIODS_PER_ITEM,
    HOLDING_COST_DIVISOR,
    INSTANCE_BASE_SEED,
    INSTANCE_COUNT_DEFAULT,
    ITEM_STEP_COUNT_RANGE,
    ITEM_STEP_DELTA_RANGE,
    LARGE_CONFIG,
    LARGE_INSTANCE_NAME,
    LARGE_INSTANCE_SEED,
    LEAD_TIME_CHOICES,
    MACHINE_CAPACITY_RANGE,
    SET_C,
    SET_C_INSTANCE_BASE_SEED,
    SET_C_INSTANCE_COUNT,
    SET_C_ITEM_MACHINES_RANGE,
    SET_C_BATCH_ITEMS_RANGE,
    SET_C_CONFIG_ANCHOR_STAGGER,
    SET_C_CONFIG_WINDOW_JITTER,
    SET_C_DEMAND_START_FRACTION,
    SET_C_LEAD_TIME_CHOICES,
    SET_C_MACHINE_CAPACITY_RANGE,
    SET_C_MACHINE_TIME_TARGET,
    SET_C_MAX_ITEMS,
    SET_C_MIN_ITEMS,
    SET_C_SCRAP_SETUP_FACTOR,
    SET_C_SLOT_DENSITY_RANGE,
    SET_C_STEP_DELTA_RANGE,
    SET_C_TARGET_ITEMS_PER_BATCH,
    SET_C_TRAY_LENGTH_LONG_RANGE,
    SET_C_TRAY_LENGTH_SHORT_RANGE,
    SET_C_TRAY_LONG_PROB,
    SET_CONFIGS,
    MED_CG_GENERATION,
    MED_CG_ITEM_MACHINES_RANGE,
    MED_CG_SCRAP_SETUP_FACTOR,
    SMALL_CG_GENERATION,
    SMALL_CG_ITEM_MACHINES_RANGE,
    SMALL_CG_SCRAP_SETUP_FACTOR,
    SCRAP_COST_MULTIPLIER,
    SET_B_GENERATION,
    TRAY_LENGTH_LONG_PROB,
    TRAY_LENGTH_LONG_RANGE,
    TRAY_LENGTH_SHORT_RANGE,
    WINDOW_GENERATION_BY_SET,
    SetConfig,
)
from preprocessor.instance_db import init_db, instance_db_path, save_instance
from preprocessor.lead_time import compute_lt_it

_CG_BENCH_TYPES = frozenset({"small_cg", "med_cg"})


def _cg_item_machines_range(set_type: str) -> tuple[int, int]:
    if set_type == "med_cg":
        return MED_CG_ITEM_MACHINES_RANGE
    return SMALL_CG_ITEM_MACHINES_RANGE


def _cg_scrap_setup_factor(set_type: str) -> float:
    if set_type == "med_cg":
        return MED_CG_SCRAP_SETUP_FACTOR
    return SMALL_CG_SCRAP_SETUP_FACTOR


def _window_params(set_config: SetConfig) -> dict:
    key = set_config.set_type.lower()
    if key == "demo":
        return WINDOW_GENERATION_BY_SET["demo"]
    if set_config.set_type == "B":
        base = dict(WINDOW_GENERATION_BY_SET["B"])
        return base
    if set_config.set_type.lower() == "large":
        return WINDOW_GENERATION_BY_SET["large"]
    if set_config.set_type == "C":
        return WINDOW_GENERATION_BY_SET["C"]
    if set_config.set_type == "small_cg":
        return WINDOW_GENERATION_BY_SET["small_cg"]
    if set_config.set_type == "med_cg":
        return WINDOW_GENERATION_BY_SET["med_cg"]
    return WINDOW_GENERATION_BY_SET.get(set_config.set_type, WINDOW_GENERATION_BY_SET["A"])


def _generation_profile(set_config: SetConfig) -> dict:
    if set_config.set_type == "B":
        return {
            "machine_capacity_range": SET_B_GENERATION["machine_capacity_range"],
            "num_cured_items": SET_B_GENERATION.get("num_cured_items"),
            "demand_periods_per_item": SET_B_GENERATION["demand_periods_per_item"],
        }
    if set_config.set_type == "C":
        return {
            "machine_capacity_range": SET_C_MACHINE_CAPACITY_RANGE,
            "num_cured_items": None,
            "demand_periods_per_item": DEMAND_PERIODS_PER_ITEM,
        }
    if set_config.set_type == "small_cg":
        return {
            "machine_capacity_range": SMALL_CG_GENERATION["machine_capacity_range"],
            "num_cured_items": None,
            "demand_periods_per_item": SMALL_CG_GENERATION["demand_periods_per_item"],
        }
    if set_config.set_type == "med_cg":
        return {
            "machine_capacity_range": MED_CG_GENERATION["machine_capacity_range"],
            "num_cured_items": None,
            "demand_periods_per_item": MED_CG_GENERATION["demand_periods_per_item"],
        }
    return {
        "machine_capacity_range": MACHINE_CAPACITY_RANGE,
        "num_cured_items": None,
        "demand_periods_per_item": DEMAND_PERIODS_PER_ITEM,
    }


def _assign_item_machines(
    item_ids: list[str],
    machine_ids: list[str],
    rng: random.Random,
    *,
    machines_per_item_range: tuple[int, int],
) -> dict[str, list[str]]:
    """每件物料可在 1~|M| 台热压罐加工（随机子集）。"""
    lo, hi = machines_per_item_range
    lo = max(1, lo)
    hi = min(len(machine_ids), hi)
    item_machines: dict[str, list[str]] = {}
    for item in item_ids:
        k = rng.randint(lo, hi)
        item_machines[item] = rng.sample(machine_ids, k)
    return item_machines


def _assign_items_to_configs(
    item_ids: list[str],
    config_ids: list[str],
    rng: random.Random,
) -> dict[str, str]:
    """每种配置分配相近件数（非均匀随机），便于同配置合批。"""
    n = len(item_ids)
    num_u = len(config_ids)
    base, rem = divmod(n, num_u)
    counts = [base + (1 if i < rem else 0) for i in range(num_u)]
    pool = list(item_ids)
    rng.shuffle(pool)
    item_config: dict[str, str] = {}
    idx = 0
    for u, cnt in zip(config_ids, counts):
        for _ in range(cnt):
            item_config[pool[idx]] = u
            idx += 1
    return item_config


def _generate_set_c_wait_steps(
    rel: int,
    end: int,
    rng: random.Random,
) -> list[list[int]]:
    """Set-C：周计划窗内 1~2 档阶跃（小时粒度），单批占罐约 4~6h。"""
    span = end - rel
    if span < 8:
        return []
    if span < 24:
        return [[rng.randint(2, max(2, span // 3)), rng.randint(*SET_C_STEP_DELTA_RANGE)]]

    k1 = rng.randint(4, max(4, span // 4))
    k2 = rng.randint(min(span, k1 + 8), max(k1 + 8, span - 4))
    omegas = sorted({k1, k2})

    steps: list[list[int]] = []
    prev_delta = 0
    for omega in omegas:
        delta = rng.randint(*SET_C_STEP_DELTA_RANGE)
        if delta <= prev_delta:
            delta = prev_delta + 1
        prev_delta = delta
        steps.append([omega, delta])
    return steps


def _generate_set_c_tray_lengths(
    item_ids: list[str],
    rng: random.Random,
    q_ref: float,
) -> dict[str, float]:
    """托盘长度校准：满罐约 TARGET_ITEMS_PER_BATCH 件，单批典型 2~4 件。"""
    target_avg = q_ref / SET_C_TARGET_ITEMS_PER_BATCH
    lo = max(SET_C_TRAY_LENGTH_SHORT_RANGE[0], target_avg * 0.72)
    hi = min(SET_C_TRAY_LENGTH_LONG_RANGE[1], target_avg * 1.28)
    max_single = q_ref * (1.0 / SET_C_BATCH_ITEMS_RANGE[0])
    trays: dict[str, float] = {}
    for i in item_ids:
        if rng.random() < SET_C_TRAY_LONG_PROB:
            trays[i] = round(rng.uniform(max(lo, SET_C_TRAY_LENGTH_LONG_RANGE[0]), hi), 1)
        else:
            trays[i] = round(rng.uniform(
                SET_C_TRAY_LENGTH_SHORT_RANGE[0],
                min(hi, SET_C_TRAY_LENGTH_SHORT_RANGE[1]),
            ), 1)
        trays[i] = min(trays[i], max_single)
    return trays


def _generate_set_c_config_anchors(
    config_ids: list[str],
    num_periods: int,
    rng: random.Random,
    window_params: dict,
) -> dict[str, tuple[int, int]]:
    """每种配置一个锚点加工窗；可选沿周计划错峰，避免全周高度重叠。"""
    lo_f, hi_f = window_params["length_frac"]
    min_length = window_params["min_length"]
    w_lo = max(min_length, math.ceil(lo_f * num_periods))
    w_hi = max(w_lo, math.ceil(hi_f * num_periods))
    w_hi = min(w_hi, num_periods)

    num_u = len(config_ids)
    anchors: dict[str, tuple[int, int]] = {}
    for k, u in enumerate(config_ids):
        window_len = rng.randint(w_lo, w_hi)
        max_rel = max(1, num_periods - window_len + 1)
        if SET_C_CONFIG_ANCHOR_STAGGER and num_u > 1:
            seg_lo = 1 + k * (max_rel - 1) // num_u
            seg_hi = 1 + (k + 1) * (max_rel - 1) // num_u
            seg_hi = min(seg_hi, max_rel)
            seg_lo = max(1, min(seg_lo, max_rel))
            seg_hi = max(seg_lo, min(seg_hi, max_rel))
            rel = rng.randint(seg_lo, seg_hi) if seg_hi >= seg_lo else 1
        else:
            rel_frac = window_params["rel_frac"]
            rel_lo = max(1, math.ceil(rel_frac[0] * num_periods))
            rel_hi = max(rel_lo, math.ceil(rel_frac[1] * num_periods))
            rel_hi = min(rel_hi, max_rel)
            rel = rng.randint(rel_lo, rel_hi) if rel_hi >= rel_lo else 1
        end = min(num_periods, rel + window_len - 1)
        anchors[u] = (rel, end)
    return anchors


def _jitter_set_c_window(
    rel: int,
    end: int,
    num_periods: int,
    rng: random.Random,
    *,
    min_length: int,
    jitter: int,
) -> tuple[int, int]:
    if jitter <= 0:
        return rel, end
    rel_j = max(1, rel + rng.randint(-jitter, jitter))
    end_j = min(num_periods, end + rng.randint(-jitter, jitter))
    if end_j - rel_j + 1 < min_length:
        end_j = min(num_periods, rel_j + min_length - 1)
    if end_j < rel_j:
        return rel, end
    return rel_j, end_j


def _generate_set_c_windows_and_steps(
    item_ids: list[str],
    item_config: dict[str, str],
    config_ids: list[str],
    num_periods: int,
    rng: random.Random,
    window_params: dict,
) -> tuple[dict[str, int], dict[str, int], dict[str, list[list[int]]]]:
    anchors = _generate_set_c_config_anchors(config_ids, num_periods, rng, window_params)
    min_length = window_params["min_length"]
    earliest: dict[str, int] = {}
    deadlines: dict[str, int] = {}
    item_wait_steps: dict[str, list[list[int]]] = {}
    for i in item_ids:
        rel, end = anchors[item_config[i]]
        rel, end = _jitter_set_c_window(
            rel,
            end,
            num_periods,
            rng,
            min_length=min_length,
            jitter=SET_C_CONFIG_WINDOW_JITTER,
        )
        earliest[i] = rel
        deadlines[i] = end
        item_wait_steps[i] = _generate_set_c_wait_steps(rel, end, rng)
    return earliest, deadlines, item_wait_steps


def _generate_set_c_demands(
    item_ids: list[str],
    num_periods: int,
    rng: random.Random,
) -> dict[str, dict[int, int]]:
    demand: dict[str, dict[int, int]] = {i: {} for i in item_ids}
    t_lo = max(1, math.ceil(SET_C_DEMAND_START_FRACTION * num_periods))
    for i in item_ids:
        demand[i][rng.randint(t_lo, num_periods)] = 1
    return demand


def capacity_metrics(instance: InstanceData) -> dict[str, float]:
    """算例产能标量：托盘总负载率、件数密度与 Set-C 占罐压力估计。"""
    q = next(iter(instance.machine_capacities.values()))
    m = len(instance.machine_ids)
    t = instance.num_periods
    n = len(instance.cured_item_ids)
    tray_sum = sum(instance.tray_lengths.values())
    total_cap = m * t * q
    metrics = {
        "tray_load_util": tray_sum / total_cap if total_cap else 0.0,
        "slot_density": n / (m * t) if m * t else 0.0,
        "tray_sum": tray_sum,
        "total_capacity": total_cap,
        "num_items": float(n),
        "avg_tray": tray_sum / n if n else 0.0,
        "machine_capacity": q,
    }
    if instance.set_type == "C":
        q_min = min(instance.machine_capacities.values())
        est_batches = 0
        for u in instance.config_ids:
            items_u = [i for i in instance.cured_item_ids if instance.item_config[i] == u]
            if items_u:
                vol = sum(instance.tray_lengths[i] for i in items_u)
                est_batches += max(1, math.ceil(vol / q_min))
        metrics["est_batches"] = float(est_batches)
        metrics["est_avg_items_per_batch"] = n / est_batches if est_batches else 0.0
        metrics["est_machine_time_lb"] = _estimate_set_c_machine_time_lb(instance)
    return metrics


def _estimate_set_c_machine_time_lb(instance: InstanceData) -> float:
    """
    启发式占罐压力：各配置按容量下限估计最少批次数 × 窗内 max(lt)，再除以 M·T。
    高于 SET_C_MACHINE_TIME_TARGET 时，IM 解的工作率更可能达标。
    """
    lt_it = compute_lt_it(
        instance.cured_item_ids,
        instance.item_config,
        instance.config_lead_times,
        instance.config_wait_steps,
        instance.earliest_starts,
        instance.deadlines,
        instance.num_periods,
        instance.period_hours,
        instance.item_wait_steps,
    )
    q_min = min(instance.machine_capacities.values())
    batch_periods = 0.0
    for u in instance.config_ids:
        items_u = [i for i in instance.cured_item_ids if instance.item_config[i] == u]
        if not items_u:
            continue
        vol = sum(instance.tray_lengths[i] for i in items_u)
        n_batches = max(1, math.ceil(vol / q_min))
        max_lt = 0
        for i in items_u:
            i_idx = instance.cured_item_ids.index(i)
            rel = instance.earliest_starts[i]
            end = instance.deadlines[i]
            max_lt = max(max_lt, max(lt_it[i_idx][tp - 1] for tp in range(rel, end + 1)))
        batch_periods += n_batches * max_lt
    denom = len(instance.machine_ids) * instance.num_periods
    return batch_periods / denom if denom else 0.0


def _calibrate_tray_lengths(
    item_ids: list[str],
    rng: random.Random,
    *,
    target_sum: float,
    max_tray: float,
) -> dict[str, float]:
    """按目标总托盘长度分配 v_i（Set-C 已改用论文随机规则，本函数保留供其他用途）。"""
    if not item_ids:
        return {}
    weights = [rng.uniform(0.92, 1.08) for _ in item_ids]
    w_sum = sum(weights)
    tray_lengths = {i: round(min(max_tray, target_sum * w / w_sum), 1) for i, w in zip(item_ids, weights)}
    actual = sum(tray_lengths.values())
    if actual > 0 and abs(actual - target_sum) > 1e-4:
        scale = target_sum / actual
        tray_lengths = {i: round(min(max_tray, v * scale), 1) for i, v in tray_lengths.items()}
    return tray_lengths


def _generate_item_window(
    num_periods: int,
    rng: random.Random,
    *,
    length_frac: tuple[float, float],
    rel_frac: tuple[float, float],
    min_length: int,
) -> tuple[int, int]:
    """生成单件进罐时间窗 rel_i, end_i；窗长适中且可容纳阶跃。"""
    lo_f, hi_f = length_frac
    w_lo = max(min_length, math.ceil(lo_f * num_periods))
    w_hi = max(w_lo, math.ceil(hi_f * num_periods))
    w_hi = min(w_hi, num_periods)
    window_len = rng.randint(w_lo, w_hi)

    rel_lo = max(1, math.ceil(rel_frac[0] * num_periods))
    rel_hi = max(rel_lo, math.ceil(rel_frac[1] * num_periods))
    max_rel = max(1, num_periods - window_len + 1)
    rel_hi = min(rel_hi, max_rel)
    rel = rng.randint(rel_lo, rel_hi) if rel_hi >= rel_lo else 1
    end = min(num_periods, rel + window_len - 1)
    return rel, end


def _generate_item_wait_steps(
    rel: int,
    end: int,
    rng: random.Random,
) -> list[list[int]]:
    """
  为单件生成等待阶梯 [[ω, Δ], ...]。
  ω 为等待小时（h=1 时即 t-rel），须落在 (0, end-rel] 内，使窗内出现 1~2 次 lt 阶跃。
    """
    span = end - rel
    if span < 2:
        return []

    n_steps = rng.randint(*ITEM_STEP_COUNT_RANGE)
    if span < 4:
        n_steps = 1

  # 在窗内选取阶跃点（相对 rel 的等待时段数）
    if n_steps == 1:
        omegas = [rng.randint(1, max(1, span - 1))]
    else:
        k1 = rng.randint(1, max(1, span // 2))
        k2 = rng.randint(min(span, k1 + 1), span)
        omegas = sorted({k1, k2})

    steps: list[list[int]] = []
    prev_delta = 0
    for omega in omegas:
        delta = rng.randint(*ITEM_STEP_DELTA_RANGE)
        if delta <= prev_delta:
            delta = prev_delta + 1
        prev_delta = delta
        steps.append([omega, delta])
    return steps


def _generate_windows_and_steps(
    item_ids: list[str],
    num_periods: int,
    rng: random.Random,
    window_params: dict,
) -> tuple[dict[str, int], dict[str, int], dict[str, list[list[int]]]]:
    earliest: dict[str, int] = {}
    deadlines: dict[str, int] = {}
    item_wait_steps: dict[str, list[list[int]]] = {}
    for i in item_ids:
        rel, end = _generate_item_window(
            num_periods,
            rng,
            length_frac=window_params["length_frac"],
            rel_frac=window_params["rel_frac"],
            min_length=window_params["min_length"],
        )
        earliest[i] = rel
        deadlines[i] = end
        item_wait_steps[i] = _generate_item_wait_steps(rel, end, rng)
    return earliest, deadlines, item_wait_steps


def _generate_demands(
    item_ids: list[str],
    num_periods: int,
    rng: random.Random,
    demand_periods_per_item: int,
) -> dict[str, dict[int, int]]:
    """单件模型：每件恰 1 单位需求，落在 1 个随机时段。"""
    demand: dict[str, dict[int, int]] = {i: {} for i in item_ids}
    candidate_periods = list(range(1, num_periods + 1))
    for i in item_ids:
        t = rng.choice(candidate_periods)
        demand[i][t] = 1
    return demand


def generate_instance(
    set_config: SetConfig,
    instance_id: int,
    seed: int | None = None,
) -> InstanceData:
    rng = random.Random(seed)
    num_periods = set_config.num_periods
    profile = _generation_profile(set_config)
    window_params = _window_params(set_config)
    cap_range = profile["machine_capacity_range"]

    config_ids = [f"u{k}" for k in range(set_config.num_configs)]
    machine_ids = [f"m{k}" for k in range(set_config.num_machines)]

    slot_density: float | None = None
    if set_config.set_type == "C":
        d_lo, d_hi = SET_C_SLOT_DENSITY_RANGE
        slot_density = round(rng.uniform(d_lo, d_hi), 1)
        num_items = max(1, round(slot_density * set_config.num_machines * set_config.num_periods))
        num_items = max(SET_C_MIN_ITEMS, min(SET_C_MAX_ITEMS, num_items))
    else:
        num_items = profile.get("num_cured_items") or set_config.num_cured_items
    item_ids = [f"i{k}" for k in range(num_items)]

    if set_config.set_type == "C":
        config_lead_times = {u: rng.choice(SET_C_LEAD_TIME_CHOICES) for u in config_ids}
    else:
        config_lead_times = {u: rng.choice(LEAD_TIME_CHOICES) for u in config_ids}
    if set_config.set_type == "C":
        item_config = _assign_items_to_configs(item_ids, config_ids, rng)
        item_machines = _assign_item_machines(
            item_ids, machine_ids, rng, machines_per_item_range=SET_C_ITEM_MACHINES_RANGE
        )
    elif set_config.set_type in _CG_BENCH_TYPES:
        item_config = _assign_items_to_configs(item_ids, config_ids, rng)
        item_machines = _assign_item_machines(
            item_ids,
            machine_ids,
            rng,
            machines_per_item_range=_cg_item_machines_range(set_config.set_type),
        )
    else:
        item_config = {}
        for i in item_ids:
            item_config[i] = rng.choice(config_ids)
        item_machines = _assign_item_machines(
            item_ids, machine_ids, rng, machines_per_item_range=(1, len(machine_ids))
        )
    item_lead_times: dict[str, int] = {}
    for i in item_ids:
        item_lead_times[i] = config_lead_times[item_config[i]]

    if set_config.set_type == "C":
        earliest_starts, deadlines, item_wait_steps = _generate_set_c_windows_and_steps(
            item_ids, item_config, config_ids, num_periods, rng, window_params
        )
    else:
        earliest_starts, deadlines, item_wait_steps = _generate_windows_and_steps(
            item_ids, num_periods, rng, window_params
        )

    machine_capacities = {m: round(rng.uniform(*cap_range), 1) for m in machine_ids}
    min_q = min(machine_capacities.values())

    if set_config.set_type == "C":
        tray_lengths = _generate_set_c_tray_lengths(item_ids, rng, min_q)
    else:
        tray_lengths = {}
        for i in item_ids:
            if rng.random() < TRAY_LENGTH_LONG_PROB:
                tray_lengths[i] = round(rng.uniform(*TRAY_LENGTH_LONG_RANGE), 1)
            else:
                tray_lengths[i] = round(rng.uniform(*TRAY_LENGTH_SHORT_RANGE), 1)

    c_star = round(rng.uniform(*C_STAR_RANGE), 1)

    production_costs: dict[str, dict[str, float]] = {}
    for m in machine_ids:
        production_costs[m] = {}
        for u in config_ids:
            l_u = config_lead_times[u]
            production_costs[m][u] = round(c_star * l_u * machine_capacities[m] / min_q, 1)

    holding_costs: dict[str, float] = {}
    for i in item_ids:
        holding_costs[i] = round(c_star * item_lead_times[i] / (HOLDING_COST_DIVISOR * num_periods), 1)

    backorder_costs = {i: round(BACKORDER_COST_MULTIPLIER * holding_costs[i], 1) for i in item_ids}
    if set_config.set_type == "C":
        scrap_costs = {}
        for i in item_ids:
            u = item_config[i]
            min_pc = min(production_costs[m][u] for m in machine_ids)
            scrap_costs[i] = round(max(
                SCRAP_COST_MULTIPLIER * backorder_costs[i],
                SET_C_SCRAP_SETUP_FACTOR * min_pc,
            ), 1)
    elif set_config.set_type in _CG_BENCH_TYPES:
        scrap_factor = _cg_scrap_setup_factor(set_config.set_type)
        scrap_costs = {}
        for i in item_ids:
            u = item_config[i]
            elig = item_machines[i]
            min_pc = min(production_costs[m][u] for m in elig)
            scrap_costs[i] = round(max(
                SCRAP_COST_MULTIPLIER * backorder_costs[i],
                scrap_factor * min_pc,
            ), 1)
    else:
        scrap_costs = {
            i: round(SCRAP_COST_MULTIPLIER * backorder_costs[i], 1)
            for i in item_ids
        }

    if set_config.set_type == "C":
        demand = _generate_set_c_demands(item_ids, num_periods, rng)
    else:
        demand = _generate_demands(
            item_ids,
            num_periods,
            rng,
            profile["demand_periods_per_item"],
        )
    wip_quantities = {i: 1 for i in item_ids}

    return InstanceData(
        name=f"set_{set_config.set_type.lower()}_{instance_id:02d}",
        set_type=set_config.set_type,
        num_periods=num_periods,
        config_ids=config_ids,
        machine_ids=machine_ids,
        cured_item_ids=item_ids,
        end_item_ids=[],
        config_lead_times=config_lead_times,
        item_config=item_config,
        item_machines=item_machines,
        item_lead_times=item_lead_times,
        tray_lengths=tray_lengths,
        machine_capacities=machine_capacities,
        production_costs=production_costs,
        holding_costs=holding_costs,
        backorder_costs=backorder_costs,
        bom={},
        demand=demand,
        deadlines=deadlines,
        earliest_starts=earliest_starts,
        wip_quantities=wip_quantities,
        scrap_costs=scrap_costs,
        config_wait_steps={},
        item_wait_steps=item_wait_steps,
        period_hours=DEFAULT_PERIOD_HOURS,
        seed=seed,
    )


def persist_instance(instance: InstanceData, db_path: Path | None = None) -> Path:
    """将算例写入 instances/{name}.db。"""
    return save_instance(instance, db_path=db_path or instance_db_path(instance.name))


def generate_instance_set(
    set_type: str,
    count: int = INSTANCE_COUNT_DEFAULT,
    base_seed: int = INSTANCE_BASE_SEED,
    db_path: Path | None = None,
) -> list[InstanceData]:
    if set_type not in SET_CONFIGS:
        raise ValueError(f"Unknown set type: {set_type}")
    config = SET_CONFIGS[set_type]
    init_db()

    instances: list[InstanceData] = []
    for idx in range(1, count + 1):
        instance = generate_instance(config, idx, seed=base_seed + idx)
        instances.append(instance)
        if db_path is not None:
            persist_instance(instance, db_path=db_path)
        else:
            persist_instance(instance)
    return instances


def generate_demo_instance(db_path: Path | None = None) -> InstanceData:
    instance = generate_instance(DEMO_CONFIG, 1, seed=DEMO_INSTANCE_SEED)
    instance.name = DEMO_INSTANCE_NAME
    if db_path is not None:
        persist_instance(instance, db_path=db_path)
    else:
        persist_instance(instance)
    return instance


def generate_large_instance(db_path: Path | None = None) -> InstanceData:
    """生成较大测试算例 large_01（T=20, 28 items, 6 configs, 4 machines）。"""
    instance = generate_instance(LARGE_CONFIG, 1, seed=LARGE_INSTANCE_SEED)
    instance.name = LARGE_INSTANCE_NAME
    if db_path is not None:
        persist_instance(instance, db_path=db_path)
    else:
        persist_instance(instance)
    return instance


def generate_set_c_instance_set(
    count: int = SET_C_INSTANCE_COUNT,
    base_seed: int = SET_C_INSTANCE_BASE_SEED,
) -> list[InstanceData]:
    """生成 set_c_01 … set_c_{count:02d}（Set-C：T=120, M=5, U=5, |I|≈225）。"""
    instances: list[InstanceData] = []
    for idx in range(1, count + 1):
        instance = generate_instance(SET_C, idx, seed=base_seed + idx - 1)
        instance.name = f"set_c_{idx:02d}"
        persist_instance(instance)
        m = capacity_metrics(instance)
        print(
            f"{instance.name}: |I|={int(m['num_items'])} T={instance.num_periods} "
            f"M={len(instance.machine_ids)} 托盘负载={m['tray_load_util']:.1%} "
            f"估计批次数={int(m.get('est_batches', 0))} "
            f"估计每批件数={m.get('est_avg_items_per_batch', 0):.1f} "
            f"占罐压力={m.get('est_machine_time_lb', 0):.1%}"
        )
        instances.append(instance)
    return instances


def generate_small_cg_instances(
    count: int | None = None,
    base_seed: int | None = None,
) -> list[InstanceData]:
    """生成小规模多轮 CG 测试算例 set_small_cg_01 …。"""
    from configuration import SMALL_CG_BASE_SEED, SMALL_CG_INSTANCE_COUNT, SET_SMALL_CG

    n = count or SMALL_CG_INSTANCE_COUNT
    seed0 = base_seed if base_seed is not None else SMALL_CG_BASE_SEED
    init_db()
    instances: list[InstanceData] = []
    for idx in range(1, n + 1):
        inst = generate_instance(SET_SMALL_CG, idx, seed=seed0 + idx)
        persist_instance(inst)
        instances.append(inst)
    return instances


def generate_med_cg_instances(
    count: int | None = None,
    base_seed: int | None = None,
) -> list[InstanceData]:
    """生成中规模 CG 测试算例 set_med_cg_01 …。"""
    from configuration import MED_CG_BASE_SEED, MED_CG_INSTANCE_COUNT, SET_MED_CG

    n = count or MED_CG_INSTANCE_COUNT
    seed0 = base_seed if base_seed is not None else MED_CG_BASE_SEED
    init_db()
    instances: list[InstanceData] = []
    for idx in range(1, n + 1):
        inst = generate_instance(SET_MED_CG, idx, seed=seed0 + idx)
        persist_instance(inst)
        instances.append(inst)
    return instances
