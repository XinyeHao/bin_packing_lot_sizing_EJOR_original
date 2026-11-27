"""Primal-only helpers; these restrictions never enter the B&P bound model."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from common.data_models import ProcessedInstance, SolutionResult

if TYPE_CHECKING:
    from model_and_algo.column_generation import ColumnGenerationSolver


@dataclass(frozen=True)
class PrimalHeuristicConfig:
    threads: int = 0  # Gurobi automatic: all available host threads.
    root_irmp_seconds: float = 5.0
    root_cgfo_seconds: float = 30.0
    repair_seconds: float = 10.0
    repair_interval: int = 120
    stall_nodes: int = 120
    max_backoff: int = 8
    repair_machines: int = 2
    node_time_fraction: float = 0.15


def validate_primal(data: ProcessedInstance, result: SolutionResult) -> None:
    """Reject a numerical objective without a matching original-feasible schedule."""
    # checkers.__init__ also imports the CLI solver; defer until initialization
    # of the model_and_algo package is complete.
    from checkers.im_solution_checker import check_im_feasibility
    from checkers.objective_checker import check_objective
    if result.objective is None or not math.isfinite(result.objective):
        raise ValueError("Primal solution has no finite objective")
    for report in (check_im_feasibility(data, result), check_objective(data, result)):
        if not report.passed:
            details = "; ".join(v.message for v in report.violations[:5])
            raise ValueError(f"Invalid primal solution ({report.name}): {details}")


def snapshot_integer_master(cg: ColumnGenerationSolver, method: str = "irmp") -> SolutionResult:
    """Copy Q/X/Y/R *before* restoring the LP; reconstruct unit-item stock flows.

    The snapshot owns ordinary Python dictionaries, not Gurobi variables or a
    mutable CG object. Inventory is independently reconstructed using each
    batch's common unloading date, then checked against the master objective.
    """
    from checkers.objective_checker import recompute_costs
    data = cg.data
    if cg._model is None or cg._artifacts is None or cg._model.SolCount == 0:
        raise ValueError("No master solution to snapshot")
    packing = {m: {} for m in data.machines}
    setup = {m: {} for m in data.machines}
    selected = []
    assignments: dict[str, tuple[str, int]] = {}
    for machine in data.machines:
        count = 0
        for col, var in zip(cg.columns[machine], cg._artifacts["Q"][machine], strict=True):
            value = float(var.X)
            if abs(value - round(value)) > 1e-5 or value < -1e-5 or value > 1.0 + 1e-5:
                raise ValueError("Cannot snapshot a fractional master")
            if value < 0.5:
                continue
            count += 1
            selected.append(col.column_id)
            for (item, period), quantity in col.x.items():
                if quantity <= 0:
                    continue
                if quantity != 1 or item in assignments:
                    raise ValueError("Selected columns process a unit item more than once")
                assignments[item] = (machine, period)
                packing[machine].setdefault(item, {})[period] = 1.0
            for (config, period), value_y in col.y.items():
                if value_y:
                    setup[machine].setdefault(config, {})[period] = 1
        if count > 1:
            raise ValueError("More than one schedule selected for a machine")

    spans = {m: {} for m in data.machines}
    for machine, config_map in setup.items():
        for config, starts in config_map.items():
            indices = data.items_by_config[data.config_index[config]]
            for period in starts:
                ell = max((data.lt_it[i][period - 1] for i in indices
                           if packing[machine].get(data.cured_items[i], {}).get(period, 0) > 0.5),
                          default=0)
                if ell == 0:
                    raise ValueError("Selected schedule contains an empty batch")
                spans[machine].setdefault(config, {})[period] = ell

    inventory, backorder, scrap = {}, {}, {}
    for i, item in enumerate(data.cured_items):
        r_value = float(cg._artifacts["R"][i].X)
        expected_r = 0 if item in assignments else 1
        if abs(r_value - expected_r) > 1e-5:
            raise ValueError("Master scrap quantity disagrees with the selected schedules")
        scrap[item] = float(expected_r)
        arrival = None
        if item in assignments:
            machine, start = assignments[item]
            config = data.configs[data.config_of_item[i]]
            arrival = start + spans[machine][config][start] - 1
        net = 0.0
        inventory[item], backorder[item] = {}, {}
        for t_idx, period in enumerate(data.periods):
            net += float(arrival == period) - data.d_it[i][t_idx] * (1 - expected_r)
            inventory[item][period] = max(net, 0.0)
            backorder[item][period] = max(-net, 0.0)

    result = SolutionResult(
        method=method, instance_name=data.raw.name, status="feasible",
        objective=float(cg._model.ObjVal), lower_bound=None, mip_gap=None,
        runtime_sec=0.0, packing=packing, setup=setup,
        inventory=inventory, backorder=backorder, columns_selected=selected,
        extra={"scrap": scrap, "span": spans, "instance_seed": data.raw.seed},
    )
    h, b, p, s, _ = recompute_costs(data, result)
    result.holding_cost, result.backorder_cost, result.production_cost = h, b, p
    result.extra.update(scrap_cost=s, total_scrap_qty=sum(scrap.values()))
    validate_primal(data, result)
    return result


def lp_setup_support(cg: ColumnGenerationSolver) -> set[tuple[int, int, int]]:
    """Read the solved LP without another optimize() or any model mutation."""
    support = set()
    if cg._artifacts is None:
        return support
    for m_idx, machine in enumerate(cg.data.machines):
        for col, q in zip(cg.columns[machine], cg._artifacts["Q"][machine], strict=True):
            if q.X <= 1e-6:
                continue
            support.update((cg.data.config_index[u], m_idx, t - 1)
                           for (u, t), y in col.y.items() if y)
    return support


def solution_setup_support(data: ProcessedInstance, solution: SolutionResult | None) -> set[tuple[int, int, int]]:
    if solution is None:
        return set()
    return {(u_idx, m_idx, t_idx)
            for m_idx, m in enumerate(data.machines)
            for u_idx, u in enumerate(data.configs)
            for t_idx, t in enumerate(data.periods)
            if solution.setup.get(m, {}).get(u, {}).get(t, 0) > 0.5}


def cgfo_restrictions(
    data: ProcessedInstance,
    support: set[tuple[int, int, int]],
    incumbent: SolutionResult | None,
    release_count: int | None = None,
    rotation: int = 0,
) -> tuple[dict, dict, list[str]]:
    """LP-supported setups + incumbent; freeze X and Y on other machines.

    No branch fixes are inherited: this is a global original-problem primal
    heuristic. Including all incumbent setups keeps its MIP start feasible.
    """
    incumbent_support = solution_setup_support(data, incumbent)
    allowed = set(support) | incumbent_support
    machines = list(range(len(data.machines)))
    if incumbent is not None and release_count is not None:
        # Prefer machines whose LP-supported setup pattern differs from the
        # incumbent; rotate ties to explore different pairs deterministically.
        m_count = len(machines)
        machines.sort(key=lambda m: (
            -sum(1 for key in support ^ incumbent_support if key[1] == m),
            (m - rotation) % m_count,
        ))
        machines = machines[:max(1, release_count)]
    released = set(machines)
    fixed_y, fixed_x = {}, {}
    for m_idx, machine in enumerate(data.machines):
        for u_idx, u in enumerate(data.configs):
            for t_idx, period in enumerate(data.periods):
                key = (u_idx, m_idx, t_idx)
                if m_idx not in released and incumbent is not None:
                    fixed_y[key] = int(incumbent.setup.get(machine, {}).get(u, {}).get(period, 0) > 0.5)
                elif key not in allowed:
                    fixed_y[key] = 0
        if m_idx not in released and incumbent is not None:
            for i_idx, item in enumerate(data.cured_items):
                for t_idx, period in enumerate(data.periods):
                    fixed_x[i_idx, m_idx, t_idx] = int(
                        incumbent.packing.get(machine, {}).get(item, {}).get(period, 0) > 0.5)
    return fixed_y, fixed_x, [data.machines[m] for m in sorted(released)]
