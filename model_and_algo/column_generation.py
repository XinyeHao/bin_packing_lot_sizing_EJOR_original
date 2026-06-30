"""Dantzig-Wolfe 列生成。"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor

import gurobipy as gp
from gurobipy import GRB

from common.data_models import ProcessedInstance, ScheduleColumn, SolutionResult
from configuration import (
    DEFAULT_CG_IMPROVE_TOLERANCE,
    DEFAULT_CG_INIT_TIME_LIMIT,
    DEFAULT_CG_MAX_ITERATIONS,
    DEFAULT_CG_PRICING_TIME_LIMIT,
    DEFAULT_CG_PRICING_WORKERS,
    DEFAULT_CG_RC_TOLERANCE,
    CHECKER_TOL,
)
from model_and_algo.branch_utils import SetupFix, column_compatible, filter_columns, machine_fixed_y
from model_and_algo.bldp import SubproblemResult, solve_subproblem_bldp, subproblem_to_column
from model_and_algo.subproblem_gurobi import solve_subproblem_gurobi

IMPROVE_TOLERANCE = DEFAULT_CG_IMPROVE_TOLERANCE


def _execute_pricing_job(
    job: tuple[
        ProcessedInstance,
        int,
        list[list[float]],
        list[float],
        float,
        bool,
        float,
        float,
        dict[SetupFix, int],
    ],
) -> tuple[int, SubproblemResult]:
    """单台机器定价（供进程池调用，须为模块级函数）。"""
    (
        data,
        m_idx,
        alpha,
        rho,
        beta,
        use_gurobi,
        init_time_limit,
        pricing_time_limit,
        fixed_y,
    ) = job
    machine_fixes = machine_fixed_y(fixed_y, m_idx)
    if use_gurobi or machine_fixes:
        sp = solve_subproblem_gurobi(
            data,
            m_idx,
            alpha,
            beta,
            rho=rho,
            time_limit=init_time_limit if not machine_fixes else pricing_time_limit,
            fixed_y=machine_fixes or None,
        )
    else:
        sp = solve_subproblem_bldp(data, m_idx, alpha, beta, rho=rho)
    return m_idx, sp


class ColumnGenerationSolver:
    """集成模型列生成求解器。"""

    def __init__(
        self,
        data: ProcessedInstance,
        max_iterations: int = DEFAULT_CG_MAX_ITERATIONS,
        rc_tolerance: float = DEFAULT_CG_RC_TOLERANCE,
        init_time_limit: float = DEFAULT_CG_INIT_TIME_LIMIT,
        use_gurobi_init: bool = True,
        stabilization_theta: float = 0.0,
        pricing_workers: int = DEFAULT_CG_PRICING_WORKERS,
        pricing_time_limit: float = DEFAULT_CG_PRICING_TIME_LIMIT,
        fixed_y: dict[SetupFix, int] | None = None,
        inherited_columns: dict[str, list[ScheduleColumn]] | None = None,
    ) -> None:
        self.data = data
        self.max_iterations = max_iterations
        self.rc_tolerance = rc_tolerance
        self.init_time_limit = init_time_limit
        self.use_gurobi_init = use_gurobi_init
        self.stabilization_theta = max(0.0, min(0.9, stabilization_theta))
        self.pricing_workers = pricing_workers
        self.pricing_time_limit = pricing_time_limit
        self.fixed_y: dict[SetupFix, int] = dict(fixed_y or {})
        self.inherited_columns = inherited_columns
        self.columns: dict[str, list[ScheduleColumn]] = {m: [] for m in data.machines}
        self._next_col_id = 0
        self._model: gp.Model | None = None
        self._artifacts: dict | None = None
        self.deadline_cuts: dict[int, dict] = {}
        self._prev_alpha = None
        self._prev_gamma = None
        self.converged = False
        self.lrmp_obj: float | None = None
        self.last_min_rc: float | None = None
        self.iteration_times: list[float] = []
        self.iteration_time_detail: list[dict[str, float]] = []

    def _reset_master(self) -> None:
        self._model = None
        self._artifacts = None

    def _add_column(self, machine: str, column: ScheduleColumn) -> None:
        self.columns[machine].append(column)

    def _get_column_contrib_to_tau(self, col: ScheduleColumn, tau: int, I: list[int]) -> float:
        """Return total qty of items in I produced by this column with arrival <= tau."""
        if not I:
            return 0.0
        contrib = 0.0
        for (name, prod_period), qty in col.x.items():
            if qty <= 0:
                continue
            try:
                ii = self.data.cured_items.index(name)
            except ValueError:
                continue
            if ii not in I:
                continue
            lead = int(self.data.l_ti[ii])
            arrival = prod_period + lead
            if arrival <= tau:
                contrib += qty
        return contrib

    def _get_column_wip_input(self, col: ScheduleColumn, i_idx: int) -> float:
        """列在 deadline 前对物料 i 的进罐总量。"""
        item_name = self.data.cured_items[i_idx]
        dl = self.data.deadlines[i_idx]
        total = 0.0
        for (name, period), qty in col.x.items():
            if name == item_name and period <= dl and qty > 0:
                total += qty
        return total

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

    def _pricing_pool_size(self) -> int:
        num_m = len(self.data.machines)
        if self.pricing_workers > 0:
            return min(self.pricing_workers, num_m)
        return num_m

    def _build_pricing_jobs(
        self,
        alpha: list[list[float]],
        rho: list[float],
        betas: list[float],
        *,
        use_gurobi_when_no_fixes: bool,
        init_time_limit: float | None = None,
    ) -> list[tuple]:
        init_tl = self.init_time_limit if init_time_limit is None else init_time_limit
        jobs = []
        for m_idx, _machine in enumerate(self.data.machines):
            jobs.append(
                (
                    self.data,
                    m_idx,
                    alpha,
                    rho,
                    betas[m_idx],
                    use_gurobi_when_no_fixes,
                    init_tl,
                    self.pricing_time_limit,
                    self.fixed_y,
                )
            )
        return jobs

    def _price_machines_parallel(
        self,
        alpha: list[list[float]],
        rho: list[float],
        betas: list[float],
        *,
        use_gurobi_when_no_fixes: bool,
        init_time_limit: float | None = None,
        executor: ProcessPoolExecutor,
    ) -> list[tuple[int, SubproblemResult]]:
        jobs = self._build_pricing_jobs(
            alpha,
            rho,
            betas,
            use_gurobi_when_no_fixes=use_gurobi_when_no_fixes,
            init_time_limit=init_time_limit,
        )
        return list(executor.map(_execute_pricing_job, jobs))

    def _setup_branch_expr(
        self,
        q_vars: dict[str, list[gp.Var]],
        machine: str,
        u_idx: int,
        t_idx: int,
    ) -> gp.LinExpr:
        data = self.data
        u_name = data.configs[u_idx]
        period = data.periods[t_idx]
        expr = gp.LinExpr()
        for col_idx, col in enumerate(self.columns[machine]):
            if col.y.get((u_name, period), 0) > 0.5:
                expr += q_vars[machine][col_idx]
        return expr

    def _add_branch_constraints(
        self,
        model: gp.Model,
        q_vars: dict[str, list[gp.Var]],
    ) -> list[tuple[gp.Constr, int, int, int, int]]:
        """对 fixed_y 添加 implied Y 分支约束，返回 (constr, u, m, t, val)。"""
        branch_rows: list[tuple[gp.Constr, int, int, int, int]] = []
        data = self.data
        for (u_idx, m_idx, t_idx), val in self.fixed_y.items():
            machine = data.machines[m_idx]
            expr = self._setup_branch_expr(q_vars, machine, u_idx, t_idx)
            if val == 0:
                constr = model.addConstr(expr == 0, name=f"branch_y0_{u_idx}_{m_idx}_{t_idx}")
            else:
                constr = model.addConstr(expr == 1, name=f"branch_y1_{u_idx}_{m_idx}_{t_idx}")
            branch_rows.append((constr, u_idx, m_idx, t_idx, val))
        return branch_rows

    def initialize_columns(self, executor: ProcessPoolExecutor) -> None:
        data = self.data
        if self.inherited_columns:
            self.columns = filter_columns(self.inherited_columns, data, self.fixed_y)
            max_id = max(
                (col.column_id for cols in self.columns.values() for col in cols),
                default=-1,
            )
            self._next_col_id = max_id + 1
        else:
            self.columns = {m: [] for m in data.machines}
            self._next_col_id = 0

        for m_idx, machine in enumerate(data.machines):
            has_empty = any(not col.y and not col.x for col in self.columns.get(machine, []))
            if not has_empty:
                self._add_column(machine, self._create_empty_column(machine))

        if self.inherited_columns:
            return

        alpha = [[0.0] * len(data.periods) for _ in data.cured_items]
        rho = [0.0] * len(data.cured_items)
        betas = [0.0] * len(data.machines)
        for machine in data.machines:
            if not any(not col.y and not col.x for col in self.columns[machine]):
                self._add_column(machine, self._create_empty_column(machine))

        priced = self._price_machines_parallel(
            alpha,
            rho,
            betas,
            use_gurobi_when_no_fixes=self.use_gurobi_init or bool(self.fixed_y),
            executor=executor,
        )
        for m_idx, sp in priced:
            machine = data.machines[m_idx]
            col = subproblem_to_column(data, m_idx, self._next_col_id, sp)
            self._next_col_id += 1
            if column_compatible(col, m_idx, data, self.fixed_y):
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

        # 简化：仅 cured_items（最终产品）
        s_plus = model.addVars(num_i, num_t, vtype=GRB.CONTINUOUS, lb=0, name="S_plus")
        s_minus = model.addVars(num_i, num_t, vtype=GRB.CONTINUOUS, lb=0, name="S_minus")
        r_vars = model.addVars(num_i, vtype=GRB.CONTINUOUS, lb=0, name="R")

        q_vars: dict[str, list[gp.Var]] = {machine: [] for machine in data.machines}
        obj = gp.LinExpr()
        obj += gp.quicksum(data.h_c[i] * s_plus[i, t] for i in range(num_i) for t in range(num_t))
        obj += gp.quicksum(data.bc_j[i] * s_minus[i, t] for i in range(num_i) for t in range(num_t))
        obj += gp.quicksum(data.sc_i[i] * r_vars[i] for i in range(num_i))

        for machine in data.machines:
            for col in self.columns[machine]:
                q_var = model.addVar(vtype=q_vtype, lb=0, ub=1, name=f"Q_{machine}_{len(q_vars[machine])}")
                q_vars[machine].append(q_var)
                obj += col.production_cost * q_var

        model.setObjective(obj, GRB.MINIMIZE)

        # 直接库存平衡
        flow_items: dict[tuple[int, int], gp.Constr] = {}
        for i_idx in range(num_i):
            lead = int(data.l_ti[i_idx])
            item_name = data.cured_items[i_idx]
            for t_idx in range(num_t):
                prev_inv = 0 if t_idx == 0 else s_plus[i_idx, t_idx - 1]
                prev_bo = 0 if t_idx == 0 else s_minus[i_idx, t_idx - 1]
                prod_in = gp.LinExpr()
                prod_t = t_idx - lead
                if prod_t >= 0:
                    period = data.periods[prod_t]
                    for machine in data.machines:
                        for col_idx, col in enumerate(self.columns[machine]):
                            qty = col.x.get((item_name, period), 0)
                            if qty:
                                prod_in += qty * q_vars[machine][col_idx]
                dem = data.d_it[i_idx][t_idx] if hasattr(data, "d_it") and data.d_it else 0
                flow_items[(i_idx, t_idx)] = model.addConstr(
                    prev_inv - prev_bo + prod_in == dem + s_plus[i_idx, t_idx] - s_minus[i_idx, t_idx]
                )

        choose_one: dict[str, gp.Constr] = {}
        for machine in data.machines:
            choose_one[machine] = model.addConstr(gp.quicksum(q_vars[machine]) <= 1)

        wip_balance: dict[int, gp.Constr] = {}
        for i_idx in range(num_i):
            wip_expr = gp.LinExpr()
            wip_expr += r_vars[i_idx]
            for machine in data.machines:
                for col_idx, col in enumerate(self.columns[machine]):
                    coeff = self._get_column_wip_input(col, i_idx)
                    if coeff > 0:
                        wip_expr += coeff * q_vars[machine][col_idx]
            wip_balance[i_idx] = model.addConstr(wip_expr == data.w_i[i_idx], name=f"wip_balance_{i_idx}")

        distinct_taus = sorted(set(data.deadlines)) if hasattr(data, "deadlines") and data.deadlines else []
        for tau in distinct_taus:
            I = [ii for ii in range(num_i) if data.deadlines[ii] <= tau]
            if not I:
                continue
            # Better RHS: number of effective setups * max trays per setup
            rhs = 0.0
            min_l = min(data.l_u) if data.l_u else 1
            for m_idx in range(len(data.machines)):
                max_trays_per_setup = int(data.q_m[m_idx] / max(min(data.v_i), 0.1)) if data.v_i else 5
                num_possible_setups = max(1, int(tau / max(min_l, 1)))
                rhs += num_possible_setups * max_trays_per_setup
            cut = model.addConstr(gp.LinExpr() <= rhs, name=f"deadline_cover_tau{tau}")
            self.deadline_cuts[tau] = {"constr": cut, "rhs": rhs, "I": I}

        for m_idx, machine in enumerate(data.machines):
            for cidx, col in enumerate(self.columns[machine]):
                qv = q_vars[machine][cidx]
                for tau, cut_info in self.deadline_cuts.items():
                    contrib = self._get_column_contrib_to_tau(col, tau, cut_info.get("I", []))
                    if contrib > 0:
                        model.chgCoeff(cut_info["constr"], qv, float(contrib))

        branch_rows = self._add_branch_constraints(model, q_vars)

        self._model = model
        self._artifacts = {
            "S_plus": s_plus,
            "S_minus": s_minus,
            "R": r_vars,
            "Q": q_vars,
            "flow_items": flow_items,
            "choose_one": choose_one,
            "wip_balance": wip_balance,
            "deadline_cuts": self.deadline_cuts,
            "branch_rows": branch_rows,
        }

    def _add_column_to_master(self, machine: str, col: ScheduleColumn) -> None:
        """向已有 LRMP 增量添加一列。"""
        if self._model is None or self._artifacts is None:
            raise RuntimeError("Master problem not initialized")

        data = self.data
        model = self._model
        q_vars = self._artifacts["Q"]
        flow_items = self._artifacts["flow_items"]
        choose_one = self._artifacts["choose_one"]
        wip_balance = self._artifacts["wip_balance"]

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
                    model.chgCoeff(flow_items[(i_idx, t_idx)], q_var, float(qty))

        model.chgCoeff(choose_one[machine], q_var, 1.0)

        for i_idx in range(len(data.cured_items)):
            coeff = self._get_column_wip_input(col, i_idx)
            if coeff > 0:
                model.chgCoeff(wip_balance[i_idx], q_var, float(coeff))

        for tau, cut_info in self.deadline_cuts.items():
            contrib = self._get_column_contrib_to_tau(col, tau, cut_info.get("I", []))
            if contrib > 0:
                model.chgCoeff(cut_info["constr"], q_var, float(contrib))

        u_name_lookup = data.configs
        for constr, u_idx, m_idx, t_idx, _val in self._artifacts.get("branch_rows", []):
            if data.machines[m_idx] != machine:
                continue
            u_name = u_name_lookup[u_idx]
            period = data.periods[t_idx]
            if col.y.get((u_name, period), 0) > 0.5:
                model.chgCoeff(constr, q_var, 1.0)

        model.update()

    def run(self) -> tuple[float, list[float], dict[str, list[ScheduleColumn]]]:
        """列生成直至 LRMP 最优（所有机器 RC >= -rc_tolerance，无法再改进列）。"""
        if not self.inherited_columns:
            self.columns = {m: [] for m in self.data.machines}
            self._next_col_id = 0
        self._reset_master()
        self.converged = False
        self.lrmp_obj = None
        self.last_min_rc = None
        self.iteration_times = []
        self.iteration_time_detail = []

        pool_size = self._pricing_pool_size()
        with ProcessPoolExecutor(max_workers=pool_size) as pricing_pool:
            self.initialize_columns(pricing_pool)
            self._init_master(relaxed=True)

            iteration_rcs: list[float] = []
            last_valid_lb = float("-inf")
            model = self._model
            assert model is not None and self._artifacts is not None

            for _ in range(self.max_iterations):
                iter_start = time.perf_counter()

                t_master = time.perf_counter()
                model.optimize()
                master_sec = time.perf_counter() - t_master
                if model.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
                    break

                lrmp_obj = model.ObjVal
                alpha = [[0.0] * len(self.data.periods) for _ in self.data.cured_items]
                flow_constrs = self._artifacts.get("flow_items") or self._artifacts.get("flow_cured", {})
                for (i_idx, t_idx), constr in flow_constrs.items():
                    alpha[i_idx][t_idx] = constr.Pi

                if self.stabilization_theta > 0 and self._prev_alpha is not None:
                    theta = self.stabilization_theta
                    for i_idx in range(len(self.data.cured_items)):
                        for t_idx in range(len(self.data.periods)):
                            alpha[i_idx][t_idx] = (
                                theta * self._prev_alpha[i_idx][t_idx]
                                + (1.0 - theta) * alpha[i_idx][t_idx]
                            )
                self._prev_alpha = [row[:] for row in alpha]

                betas = [self._artifacts["choose_one"][machine].Pi for machine in self.data.machines]
                rhos = [self._artifacts["wip_balance"][i_idx].Pi for i_idx in range(len(self.data.cured_items))]

                sum_rc = 0.0
                lb_correction = 0.0
                min_rc = float("inf")
                new_columns: list[tuple[str, ScheduleColumn]] = []
                t_pricing = time.perf_counter()
                priced = self._price_machines_parallel(
                    alpha,
                    rhos,
                    betas,
                    use_gurobi_when_no_fixes=bool(self.fixed_y),
                    executor=pricing_pool,
                )
                for m_idx, sp in priced:
                    machine = self.data.machines[m_idx]
                    sum_rc += sp.reduced_cost
                    min_rc = min(min_rc, sp.reduced_cost)
                    lb_correction += min(sp.reduced_cost, 0.0)
                    if sp.reduced_cost < -self.rc_tolerance:
                        col = subproblem_to_column(self.data, m_idx, self._next_col_id, sp)
                        self._next_col_id += 1
                        if column_compatible(col, m_idx, self.data, self.fixed_y):
                            new_columns.append((machine, col))
                pricing_sec = time.perf_counter() - t_pricing

                t_add = time.perf_counter()
                for machine, col in new_columns:
                    self._add_column(machine, col)
                    self._add_column_to_master(machine, col)
                add_col_sec = time.perf_counter() - t_add

                iter_sec = time.perf_counter() - iter_start
                self.iteration_times.append(iter_sec)
                self.iteration_time_detail.append(
                    {
                        "total": iter_sec,
                        "master": master_sec,
                        "pricing": pricing_sec,
                        "add_columns": add_col_sec,
                        "num_new_columns": float(len(new_columns)),
                        "pricing_workers": float(pool_size),
                    }
                )

                iteration_rcs.append(sum_rc)
                self.last_min_rc = min_rc if min_rc != float("inf") else None
                last_valid_lb = max(last_valid_lb, lrmp_obj + lb_correction)
                self.lrmp_obj = lrmp_obj

                if not new_columns:
                    self.converged = True
                    return lrmp_obj, iteration_rcs, self.columns

        self.converged = False
        return last_valid_lb, iteration_rcs, self.columns

    def solve_restricted_master(self) -> tuple[gp.Model, dict]:
        if self._model is None or self._artifacts is None:
            self._init_master(relaxed=True)
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

    def get_implied_y(self) -> dict[tuple[int, int, int], float]:
        """从当前主问题 LP 解重构 implied Y 值 {(m_idx, u_idx, t_idx): val}。"""
        if self._model is None or self._artifacts is None:
            return {}
        data = self.data
        implied: dict[tuple[int, int, int], float] = {}
        for m_idx, machine in enumerate(data.machines):
            qvars = self._artifacts["Q"][machine]
            for col_idx, col in enumerate(self.columns[machine]):
                try:
                    qv = qvars[col_idx].X
                except Exception:
                    qv = 0.0
                if qv < 1e-9:
                    continue
                for (u_name, period), yb in col.y.items():
                    if yb > 0.5:
                        u_idx = data.config_index.get(u_name, -1)
                        if u_idx < 0:
                            continue
                        t_idx = period - 1
                        key = (m_idx, u_idx, t_idx)
                        implied[key] = implied.get(key, 0.0) + qv
        return implied


def solve_column_generation(
    data: ProcessedInstance,
    max_iterations: int = DEFAULT_CG_MAX_ITERATIONS,
    init_time_limit: float = DEFAULT_CG_INIT_TIME_LIMIT,
) -> SolutionResult:
    start = time.perf_counter()
    cg = ColumnGenerationSolver(data, max_iterations=max_iterations, init_time_limit=init_time_limit)
    lb, rcs, columns = cg.run()
    runtime = time.perf_counter() - start

    scrap_qty: dict[str, float] = {}
    scrap_cost = None
    if cg.converged and cg._artifacts is not None:
        r_vars = cg._artifacts.get("R")
        if r_vars is not None:
            scrap_qty = {
                data.cured_items[i]: r_vars[i].X for i in range(len(data.cured_items)) if r_vars[i].X > 1e-9
            }
            scrap_cost = sum(data.sc_i[i] * r_vars[i].X for i in range(len(data.cured_items)))

    return SolutionResult(
        method="column_generation",
        instance_name=data.raw.name,
        status="completed" if cg.converged else "max_iterations",
        objective=None,
        lower_bound=lb,
        mip_gap=None,
        runtime_sec=runtime,
        extra={
            "iteration_rcs": rcs,
            "num_columns": {m: len(cols) for m, cols in columns.items()},
            "num_iterations": len(rcs),
            "converged": cg.converged,
            "lrmp_obj": cg.lrmp_obj,
            "last_min_rc": cg.last_min_rc,
            "iteration_times_sec": [round(t, 4) for t in cg.iteration_times],
            "iteration_time_detail": [
                {k: round(v, 4) if k != "num_new_columns" else int(v) for k, v in d.items()}
                for d in cg.iteration_time_detail
            ],
            "instance_seed": data.raw.seed,
            "set_type": data.raw.set_type,
            "num_periods": data.raw.num_periods,
            "scrap": scrap_qty,
            "scrap_cost": scrap_cost,
            "total_scrap_qty": sum(scrap_qty.values()) if scrap_qty else 0.0,
        },
    )
