"""算例、预处理数据与求解结果的数据结构。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class InstanceData:
    """原始算例（JSON 序列化格式）。"""

    name: str
    set_type: str
    num_periods: int
    config_ids: list[str]
    machine_ids: list[str]
    cured_item_ids: list[str]
    config_lead_times: dict[str, int]
    item_config: dict[str, str]
    item_lead_times: dict[str, int]
    tray_lengths: dict[str, float]
    machine_capacities: dict[str, float]
    production_costs: dict[str, dict[str, float]]
    holding_costs: dict[str, float]
    backorder_costs: dict[str, float]
    demand: dict[str, dict[int, int]]
    # 以下为可选/新字段（有默认值，必须放在最后）
    end_item_ids: list[str] = field(default_factory=list)   # 已废弃（无装配）
    bom: dict[str, dict[str, int]] = field(default_factory=dict)  # 已废弃
    deadlines: dict[str, int] = field(default_factory=dict)   # 新：最晚加工时段（1-based）
    seed: int | None = None

    def to_dict(self) -> dict[str, Any]:
        d = {
            "name": self.name,
            "set_type": self.set_type,
            "num_periods": self.num_periods,
            "config_ids": self.config_ids,
            "machine_ids": self.machine_ids,
            "cured_item_ids": self.cured_item_ids,
            "end_item_ids": self.end_item_ids,
            "config_lead_times": self.config_lead_times,
            "item_config": self.item_config,
            "item_lead_times": self.item_lead_times,
            "tray_lengths": self.tray_lengths,
            "machine_capacities": self.machine_capacities,
            "production_costs": self.production_costs,
            "holding_costs": self.holding_costs,
            "backorder_costs": self.backorder_costs,
            "bom": self.bom,
            "demand": {k: {str(t): v for t, v in vals.items()} for k, vals in self.demand.items()},
            "deadlines": self.deadlines,
            "seed": self.seed,
        }
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InstanceData:
        demand = {
            item: {int(t): qty for t, qty in periods.items()}
            for item, periods in data["demand"].items()
        }
        return cls(
            name=data["name"],
            set_type=data["set_type"],
            num_periods=data["num_periods"],
            config_ids=list(data["config_ids"]),
            machine_ids=list(data["machine_ids"]),
            cured_item_ids=list(data["cured_item_ids"]),
            end_item_ids=list(data.get("end_item_ids", [])),
            config_lead_times={k: int(v) for k, v in data["config_lead_times"].items()},
            item_config=dict(data["item_config"]),
            item_lead_times={k: int(v) for k, v in data["item_lead_times"].items()},
            tray_lengths={k: float(v) for k, v in data["tray_lengths"].items()},
            machine_capacities={k: float(v) for k, v in data["machine_capacities"].items()},
            production_costs={
                m: {u: float(v) for u, v in costs.items()}
                for m, costs in data["production_costs"].items()
            },
            holding_costs={k: float(v) for k, v in data["holding_costs"].items()},
            backorder_costs={k: float(v) for k, v in data["backorder_costs"].items()},
            bom={
                j: {i: int(v) for i, v in items.items()}
                for j, items in data.get("bom", {}).items()
            },
            demand=demand,
            deadlines={k: int(v) for k, v in data.get("deadlines", {}).items()},
            seed=data.get("seed"),
        )


@dataclass
class ProcessedInstance:
    """预处理后的算法友好型数据结构。"""

    raw: InstanceData
    periods: list[int]
    configs: list[str]
    machines: list[str]
    cured_items: list[str]
    end_items: list[str]
    all_items: list[str]
    config_index: dict[str, int]
    machine_index: dict[str, int]
    cured_index: dict[str, int]
    end_index: dict[str, int]
    item_index: dict[str, int]
    l_u: list[int]
    l_ti: list[float]
    b_iu: list[list[int]]
    v_i: list[float]
    q_m: list[float]
    p_cum: list[list[float]]
    h_c: list[float]
    bc_j: list[float]          # 现用于 cured_items（final items）
    r_ij: list[list[int]]     # 废弃（无装配）
    d_jt: list[list[int]]     # 废弃，改用 d_it
    d_it: list[list[int]]     # 新：cured items（现为最终产品）的需求
    bom_parents: list[list[tuple[int, int]]]
    items_by_config: list[list[int]]
    config_of_item: list[int]
    min_q: float
    deadlines: list[int]      # 新：每个 cured item 的最晚加工时段（1-based）


@dataclass
class ScheduleColumn:
    """单台热压罐的可行调度列（Dantzig-Wolfe 子问题解）。"""

    machine: str
    column_id: int
    x: dict[tuple[str, int], int]
    y: dict[tuple[str, int], int]
    production_cost: float
    reduced_cost: float = 0.0


@dataclass
class SolutionResult:
    """求解结果。"""

    method: str
    instance_name: str
    status: str
    objective: float | None
    lower_bound: float | None
    mip_gap: float | None
    runtime_sec: float
    holding_cost: float | None = None
    backorder_cost: float | None = None
    production_cost: float | None = None
    inventory: dict[str, dict[int, float]] = field(default_factory=dict)
    backorder: dict[str, dict[int, float]] = field(default_factory=dict)
    assembly: dict[str, dict[int, float]] = field(default_factory=dict)
    packing: dict[str, dict[str, dict[int, float]]] = field(default_factory=dict)
    setup: dict[str, dict[str, dict[int, int]]] = field(default_factory=dict)
    columns_selected: list[int] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "instance_name": self.instance_name,
            "status": self.status,
            "objective": self.objective,
            "lower_bound": self.lower_bound,
            "mip_gap": self.mip_gap,
            "runtime_sec": self.runtime_sec,
            "holding_cost": self.holding_cost,
            "backorder_cost": self.backorder_cost,
            "production_cost": self.production_cost,
            "inventory": self.inventory,
            "backorder": self.backorder,
            "assembly": self.assembly,
            "packing": self.packing,
            "setup": self.setup,
            "columns_selected": self.columns_selected,
            "extra": self.extra,
        }
