"""Dantzig-Wolfe 列生成。"""

from __future__ import annotations

import time

import gurobipy as gp
from gurobipy import GRB

from common.data_models import ProcessedInstance, ScheduleColumn, SolutionResult
from configuration import (
    DEFAULT_CG_IMPROVE_TOLERANCE,
    DEFAULT_CG_INIT_TIME_LIMIT,
    DEFAULT_CG_MAX_ITERATIONS,
    DEFAULT_CG_RC_TOLERANCE,
    CHECKER_TOL,
)
from model_and_algo.bldp import solve_subproblem_bldp, subproblem_to_column
from model_and_algo.subproblem_gurobi import solve_subproblem_gurobi

IMPROVE_TOLERANCE = DEFAULT_CG_IMPROVE_TOLERANCE


class ColumnGenerationSolver:
    """集成模型列生成求解器。"""

    def __init__(
        self,
        data: ProcessedInstance,
        max_iterations: int = DEFAULT_CG_MAX_ITERATIONS,
        rc_tolerance: float = DEFAULT_CG_RC_TOLERANCE,
        init_time_limit: float = DEFAULT_CG_INIT_TIME_LIMIT,
        use_gurobi_init: bool = True,
    ) -> None:
        self.data = data
        self.max_iterations = max_iterations
        self.rc_tolerance = rc_tolerance
        self.init_time_limit = init_time_limit
        self.use_gurobi_init = use_gurobi_init
        self.columns: dict[str, list[ScheduleColumn]] = {m: [] for m in data.machines}
        self._next_col_id = 0
        self._model: gp.Model | None = None
        self._artifacts: dict | None = None

    def _reset_master(self) -> None:
        self._model = None
        self._artifacts = None

    def _add_column(self, machine: str, column: ScheduleColumn) -> None:
        self.columns[machine].append(column)

    def _create_empty_column(self, machine: str) -> ScheduleColumn:
        col = ScheduleColumn(
            machine=machine,
            column_id=self._next_col_id,
            x={},
            y={},
            production_cost=0.0,
            reduced_cost=0.0,
        )
        self._next_col_id += 1
        return col

    def initialize_columns(self) -> None:
        alpha = [[0.0] * len(self.data.periods) for _ in self.data.cured_items]
        for m_idx, machine in enumerate(self.data.machines):
            self._add_column(machine, self._create_empty_column(machine))
            if self.use_gurobi_init:
                sp = solve_subproblem_gurobi(
                    self.data,
                    m_idx,
                    alpha,
                    beta=0.0,
                    time_limit=self.init_time_limit,
                )
            else:
                sp = solve_subproblem_bldp(self.data, m_idx, alpha, beta=0.0)
            col = subproblem_to_column(self.data, m_idx, self._next_col_id, sp)
            self._next_col_id += 1
            self._add_column(machine, col)

    def _init_master(self, relaxed: bool = True) -> None:
        """首次构建 LRMP；后续通过 _add_column_to_master 增量加列。"""
        if self._model is not None:
            return

        data = self.data
        model = gp.Model("LRMP" if relaxed else "RMP")
        model.Params.OutputFlag = 0

        num_t = len(data.periods)
        num_j = len(data.end_items)
        num_all = len(data.all_items)
        num_i = len(data.cured_items)
        end_offset = num_i
        q_vtype = GRB.CONTINUOUS if relaxed else GRB.BINARY

        s_plus = model.addVars(num_all, num_t, vtype=GRB.CONTINUOUS, lb=0, name="S_plus")
        s_minus = model.addVars(num_j, num_t, vtype=GRB.CONTINUOUS, lb=0, name="S_minus")
        a_vars = model.addVars(num_j, num_t, vtype=GRB.CONTINUOUS, lb=0, name="A")

        q_vars: dict[str, list[gp.Var]] = {machine: [] for machine in data.machines}
        obj = gp.LinExpr()
        obj += gp.quicksum(data.h_c[i] * s_plus[i, t] for i in range(num_all) for t in range(num_t))
        obj += gp.quicksum(data.bc_j[j] * s_minus[j, t] for j in range(num_j) for t in range(num_t))

        for machine in data.machines:
            for col in self.columns[machine]:
                q_var = model.addVar(vtype=q_vtype, lb=0, ub=1, name=f"Q_{machine}_{len(q_vars[machine])}")
                q_vars[machine].append(q_var)
                obj += col.production_cost * q_var

        model.setObjective(obj, GRB.MINIMIZE)

        flow_end: dict[tuple[int, int], gp.Constr] = {}
        for j_idx in range(num_j):
            item_idx = end_offset + j_idx
            for t_idx in range(num_t):
                prev_inv = 0 if t_idx == 0 else s_plus[item_idx, t_idx - 1]
                prev_bo = 0 if t_idx == 0 else s_minus[j_idx, t_idx - 1]
                flow_end[(j_idx, t_idx)] = model.addConstr(
                    prev_inv - prev_bo + a_vars[j_idx, t_idx]
                    == data.d_jt[j_idx][t_idx] + s_plus[item_idx, t_idx] - s_minus[j_idx, t_idx]
                )

        flow_cured: dict[tuple[int, int], gp.Constr] = {}
        for i_idx in range(num_i):
            lead = int(data.l_ti[i_idx])
            item_name = data.cured_items[i_idx]
            for t_idx in range(num_t):
                prev_inv = 0 if t_idx == 0 else s_plus[i_idx, t_idx - 1]
                prod_in = gp.LinExpr()
                prod_t = t_idx - lead
                if prod_t >= 0:
                    period = data.periods[prod_t]
                    for machine in data.machines:
                        for col_idx, col in enumerate(self.columns[machine]):
                            qty = col.x.get((item_name, period), 0)
                            if qty:
                                prod_in += qty * q_vars[machine][col_idx]
                demand = gp.quicksum(data.r_ij[i_idx][j_idx] * a_vars[j_idx, t_idx] for j_idx in range(num_j))
                flow_cured[(i_idx, t_idx)] = model.addConstr(prev_inv + prod_in == demand + s_plus[i_idx, t_idx])

        choose_one: dict[str, gp.Constr] = {}
        for machine in data.machines:
            choose_one[machine] = model.addConstr(gp.quicksum(q_vars[machine]) <= 1)

        self._model = model
        self._artifacts = {
            "S_plus": s_plus,
            "S_minus": s_minus,
            "A": a_vars,
            "Q": q_vars,
            "flow_end": flow_end,
            "flow_cured": flow_cured,
            "choose_one": choose_one,
        }

    def _add_column_to_master(self, machine: str, col: ScheduleColumn) -> None:
        """向已有 LRMP 增量添加一列。"""
        if self._model is None or self._artifacts is None:
            raise RuntimeError("Master problem not initialized")

        data = self.data
        model = self._model
        q_vars = self._artifacts["Q"]
        flow_cured = self._artifacts["flow_cured"]
        choose_one = self._artifacts["choose_one"]

        q_var = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name=f"Q_{machine}_{len(q_vars[machine])}")
        q_vars[machine].append(q_var)

        model.setObjective(model.getObjective() + col.production_cost * q_var, GRB.MINIMIZE)

        for i_idx in range(len(data.cured_items)):
            lead = int(data.l_ti[i_idx])
            item_name = data.cured_items[i_idx]
            for t_idx in range(len(data.periods)):
                prod_t = t_idx - lead
                if prod_t < 0:
                    continue
                qty = col.x.get((item_name, data.periods[prod_t]), 0)
                if qty:
                    model.chgCoeff(flow_cured[(i_idx, t_idx)], q_var, float(qty))

        model.chgCoeff(choose_one[machine], q_var, 1.0)
        model.update()

    def run(self) -> tuple[float, list[float], dict[str, list[ScheduleColumn]]]:
        self.columns = {m: [] for m in self.data.machines}
        self._next_col_id = 0
        self._reset_master()

        self.initialize_columns()
        self._init_master(relaxed=True)

        iteration_rcs: list[float] = []
        best_lb = float("-inf")
        model = self._model
        assert model is not None and self._artifacts is not None

        for _ in range(self.max_iterations):
            model.optimize()
            if model.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
                break

            lrmp_obj = model.ObjVal
            alpha = [[0.0] * len(self.data.periods) for _ in self.data.cured_items]
            for (i_idx, t_idx), constr in self._artifacts["flow_cured"].items():
                alpha[i_idx][t_idx] = constr.Pi

            sum_rc = 0.0
            lb_correction = 0.0
            new_columns: list[tuple[str, ScheduleColumn]] = []
            for m_idx, machine in enumerate(self.data.machines):
                beta = self._artifacts["choose_one"][machine].Pi
                sp = solve_subproblem_bldp(self.data, m_idx, alpha, beta)
                sum_rc += sp.reduced_cost
                lb_correction += min(sp.reduced_cost, 0.0)
                if sp.reduced_cost < -IMPROVE_TOLERANCE:
                    col = subproblem_to_column(self.data, m_idx, self._next_col_id, sp)
                    self._next_col_id += 1
                    new_columns.append((machine, col))

            for machine, col in new_columns:
                self._add_column(machine, col)
                self._add_column_to_master(machine, col)

            iteration_rcs.append(sum_rc)
            valid_lb = lrmp_obj + lb_correction
            best_lb = max(best_lb, valid_lb)

            # 论文终止条件：所有子问题 reduced cost 之和 > -0.1
            if sum_rc > self.rc_tolerance:
                return valid_lb, iteration_rcs, self.columns

        return best_lb, iteration_rcs, self.columns

    def solve_restricted_master(self, relaxed: bool = False) -> tuple[gp.Model, dict]:
        if relaxed and self._model is not None and self._artifacts is not None:
            self._model.optimize()
            return self._model, self._artifacts
        self._reset_master()
        self._init_master(relaxed=relaxed)
        assert self._model is not None and self._artifacts is not None
        self._model.optimize()
        return self._model, self._artifacts

    def get_selected_columns(self, threshold: float = 0.0) -> dict[str, list[ScheduleColumn]]:
        if self._model is None or self._artifacts is None:
            self._init_master(relaxed=True)
        assert self._model is not None and self._artifacts is not None
        self._model.optimize()
        selected: dict[str, list[ScheduleColumn]] = {m: [] for m in self.data.machines}
        for machine in self.data.machines:
            for col_idx, col in enumerate(self.columns[machine]):
                if self._artifacts["Q"][machine][col_idx].X > threshold + CHECKER_TOL["cg_column_select"]:
                    selected[machine].append(col)
        return selected


def solve_column_generation(
    data: ProcessedInstance,
    max_iterations: int = DEFAULT_CG_MAX_ITERATIONS,
    init_time_limit: float = DEFAULT_CG_INIT_TIME_LIMIT,
) -> SolutionResult:
    start = time.perf_counter()
    cg = ColumnGenerationSolver(data, max_iterations=max_iterations, init_time_limit=init_time_limit)
    lb, rcs, columns = cg.run()
    runtime = time.perf_counter() - start
    return SolutionResult(
        method="column_generation",
        instance_name=data.raw.name,
        status="completed",
        objective=None,
        lower_bound=lb,
        mip_gap=None,
        runtime_sec=runtime,
        extra={
            "iteration_rcs": rcs,
            "num_columns": {m: len(cols) for m, cols in columns.items()},
            "num_iterations": len(rcs),
            "instance_seed": data.raw.seed,
            "set_type": data.raw.set_type,
            "num_periods": data.raw.num_periods,
        },
    )
