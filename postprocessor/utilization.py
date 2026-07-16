"""从 IM 解统计热压罐工作率与罐内空间利用率。"""

from __future__ import annotations

from typing import Any

from common.data_models import InstanceData, SolutionResult


def analyze_im_utilization(
    instance: InstanceData,
    result: SolutionResult,
) -> dict[str, Any]:
    """
    指标说明（1-based 时段）：
    - 热压罐工作率：各机台在规划期内「有批次占罐」的时段占比（跨期占罐由 span 展开）。
    - 罐内空间利用率：所有开工时刻 sum(v_i)/q_m 的加权平均（按开工批次数）；
      以及占罐时段内按「开工时装箱率」折算的平均空间利用。
    """
    if not result.setup or not result.extra.get("span"):
        return {"error": "解中缺少 setup/span"}

    periods = list(range(1, instance.num_periods + 1))
    machines = list(instance.machine_ids)
    span = result.extra["span"]
    packing = result.packing or {}
    q_map = instance.machine_capacities
    v_map = instance.tray_lengths

    per_machine: dict[str, dict[str, float]] = {}
    all_busy_periods = 0
    all_total_periods = len(machines) * len(periods)
    start_fills: list[float] = []
    batch_item_counts: list[int] = []
    busy_weighted_fill = 0.0
    busy_span_periods = 0

    for m in machines:
        q_m = q_map[m]
        busy_set: set[int] = set()
        for u, y_by_t in result.setup.get(m, {}).items():
            l_by_t = span.get(m, {}).get(u, {})
            for t, y_val in y_by_t.items():
                t_int = int(t)
                if int(round(y_val)) <= 0:
                    continue
                l_val = int(round(l_by_t.get(t, l_by_t.get(str(t_int), 0))))
                if l_val <= 0:
                    continue
                for dt in range(l_val):
                    tp = t_int + dt
                    if tp in periods:
                        busy_set.add(tp)
                load = 0.0
                item_cnt = 0
                for i, x_by_t in packing.get(m, {}).items():
                    qty = float(x_by_t.get(t, x_by_t.get(str(t_int), 0)))
                    load += v_map.get(i, 0.0) * qty
                    if qty > 0.5:
                        item_cnt += 1
                fill = load / q_m if q_m > 0 else 0.0
                if load > 1e-6:
                    start_fills.append(fill)
                    batch_item_counts.append(item_cnt)
                    busy_weighted_fill += fill * l_val
                    busy_span_periods += l_val

        busy_count = len(busy_set)
        all_busy_periods += busy_count
        per_machine[m] = {
            "busy_periods": busy_count,
            "total_periods": len(periods),
            "time_util": busy_count / len(periods) if periods else 0.0,
        }

    scrap = result.extra.get("scrap") or {}
    scrap_qty = sum(float(v) for v in scrap.values() if v > 0.5)
    n_items = len(instance.cured_item_ids)

    return {
        "machine_time_util": all_busy_periods / all_total_periods if all_total_periods else 0.0,
        "per_machine_time_util": per_machine,
        "avg_start_space_util": sum(start_fills) / len(start_fills) if start_fills else 0.0,
        "span_weighted_space_util": busy_weighted_fill / busy_span_periods if busy_span_periods else 0.0,
        "num_batches": len(start_fills),
        "avg_items_per_batch": (
            sum(batch_item_counts) / len(batch_item_counts) if batch_item_counts else 0.0
        ),
        "scrap_qty": scrap_qty,
        "scheduled_qty": n_items - scrap_qty,
        "meets_time_80pct": (all_busy_periods / all_total_periods >= 0.60) if all_total_periods else False,
        "meets_space_50pct": (
            (sum(start_fills) / len(start_fills) >= 0.50) if start_fills else False
        ),
    }
