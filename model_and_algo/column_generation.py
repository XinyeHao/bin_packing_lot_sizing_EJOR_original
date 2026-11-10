"""Dantzig-Wolfe 列生成。"""

from __future__ import annotations

import math
import multiprocessing as mp
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
    DEFAULT_PROGRESS_INTERVAL_SEC,
    EBRP_DEFAULT_TIME_LIMIT,
    EBRP_EXACT_MAX_ITEMS,
    EBRP_HEURISTIC_TIME_LIMIT,
    IM_PRICING_DEFAULT_TIME_LIMIT,
    SET_B_CG_FAST,
    SET_C_CG,
    CHECKER_TOL,
)
from model_and_algo.branch_utils import (
    SetupFix,
    WipFix,
    column_compatible,
    filter_columns,
    machine_fixed_y,
    machine_needs_setup_seed,
)
from model_and_algo.subproblem_common import SubproblemResult, column_fingerprint, subproblem_to_column
from model_and_algo.pricing_ebrp import build_ebrp_contexts, solve_subproblem_ebrp
from model_and_algo.pricing_ebrp_heuristic import solve_subproblem_ebrp_heuristic
from model_and_algo.pricing_im_gurobi import solve_subproblem_im_gurobi
from model_and_algo.progress_monitor import ProgressMonitor, heartbeat_while
from model_and_algo.lrmp_cuts import (
    DEFAULT_LRMP_CUTS,
    FRAC_TOL as CUT_FRAC_TOL,
    add_pair_cut_coeffs,
    add_setup_expr,
    build_static_setup_pair_cuts,
    column_setup_indicator,
    deadline_cover_rhs,
    find_violated_setup_pairs,
    max_process_for_item,
    merge_cut_options,
    cuts_enabled,
)
from model_and_algo.cut_SRI import SRICut, coefficient as sri_coefficient, separate_root_triples

IMPROVE_TOLERANCE = DEFAULT_CG_IMPROVE_TOLERANCE

_PRICING_WORKER_STATE: dict = {}


def _max_items_on_any_machine(data: ProcessedInstance) -> int:
    if getattr(data, "items_by_machine", None):
        return max((len(xs) for xs in data.items_by_machine), default=len(data.cured_items))
    return len(data.cured_items)


def _resolve_pricing_mode(data: ProcessedInstance, pricing_mode: str, *, ebrp_force: bool = False) -> str:
    mode = pricing_mode.lower()
    if mode in ("ebrp_heuristic", "ebrp_h"):
        return "ebrp_heuristic"
    if mode == "ebrp_force" or ebrp_force:
        return "ebrp"
    if mode == "ebrp":
        if _max_items_on_any_machine(data) > EBRP_EXACT_MAX_ITEMS:
            return "ebrp_heuristic"
        return "ebrp"
    return "im_gurobi"


def _solve_pricing_subproblem(
    data: ProcessedInstance,
    machine_idx: int,
    alpha: list[list[float]] | None,
    beta: float,
    rho: list[float],
    *,
    pricing_mode: str,
    im_pricing_time_limit: float,
    ebrp_time_limit: float | None,
    ebrp_force: bool = False,
    alpha_flat=None,
    num_t_for_flat: int | None = None,
    fixed_y: dict[tuple[int, int], int] | None = None,
    ebrp_contexts=None,
    ebrp_progress_sec: float = 0.0,
    skip_cross_check: bool = False,
    setup_cost_delta: list[list[float]] | None = None,
    sri_terms: tuple[tuple[int, float], ...] = (),
) -> SubproblemResult:
    """统一定价入口：ebrp | ebrp_heuristic | im_gurobi（规范 §3.3）。"""
    mode = _resolve_pricing_mode(data, pricing_mode, ebrp_force=ebrp_force)
    if sri_terms and mode != "ebrp":
        raise RuntimeError("SRI cuts currently require exact EBRP pricing (pricing_mode='ebrp_force')")
    if mode == "im_gurobi":
        return solve_subproblem_im_gurobi(
            data,
            machine_idx,
            alpha if alpha is not None else [[0.0] * len(data.periods) for _ in data.cured_items],
            beta,
            rho=rho,
            time_limit=im_pricing_time_limit,
            fixed_y=fixed_y,
            setup_cost_delta=setup_cost_delta,
        )
    ctx = ebrp_contexts[machine_idx] if ebrp_contexts is not None else None
    if mode == "ebrp_heuristic":
        limit = ebrp_time_limit if ebrp_time_limit is not None else EBRP_HEURISTIC_TIME_LIMIT
        return solve_subproblem_ebrp_heuristic(
            data,
            machine_idx,
            alpha if alpha is not None else [[0.0] * len(data.periods) for _ in data.cured_items],
            beta,
            rho=rho,
            fixed_y=fixed_y,
            time_limit=limit,
            ctx=ctx,
            alpha_flat=alpha_flat,
            num_t_for_flat=num_t_for_flat,
            setup_cost_delta=setup_cost_delta,
        )
    return solve_subproblem_ebrp(
        data,
        machine_idx,
        alpha if alpha is not None else [[]],
        beta,
        rho=rho,
        fixed_y=fixed_y,
        time_limit=ebrp_time_limit,
        ctx=ctx,
        alpha_flat=alpha_flat,
        num_t_for_flat=num_t_for_flat,
        progress_interval_sec=ebrp_progress_sec,
        skip_cross_check=skip_cross_check,
        setup_cost_delta=setup_cost_delta,
        sri_terms=sri_terms,
    )


def _init_pricing_worker(
    data: ProcessedInstance,
    fixed_y: dict[SetupFix, int],
    pricing_time_limit: float,
    alpha_flat: mp.Array | None,
    rho_arr: mp.Array | None,
    beta_arr: mp.Array | None,
    rc_arr: mp.Array | None,
    num_i: int,
    num_t: int,
    pricing_mode: str,
    ebrp_force: bool,
    ebrp_time_limit: float | None,
    im_pricing_time_limit: float,
    ebrp_contexts,
    skip_cross_check: bool = False,
) -> None:
    """进程池 initializer：每个 worker 只加载一次算例，避免每轮定价重复 pickle。"""
    global _PRICING_WORKER_STATE
    _PRICING_WORKER_STATE = {
        "data": data,
        "fixed_y": fixed_y,
        "pricing_time_limit": pricing_time_limit,
        "pricing_mode": pricing_mode,
        "ebrp_force": ebrp_force,
        "ebrp_time_limit": ebrp_time_limit,
        "im_pricing_time_limit": im_pricing_time_limit,
        "ebrp_contexts": ebrp_contexts,
        "alpha_flat": alpha_flat,
        "rho_arr": rho_arr,
        "beta_arr": beta_arr,
        "rc_arr": rc_arr,
        "num_i": num_i,
        "num_t": num_t,
        "skip_cross_check": skip_cross_check,
    }


def _nested_alpha_from_shm() -> tuple[list[list[float]], list[float]]:
    state = _PRICING_WORKER_STATE
    num_i = state["num_i"]
    num_t = state["num_t"]
    alpha_flat = state["alpha_flat"]
    rho_arr = state["rho_arr"]
    assert alpha_flat is not None and rho_arr is not None
    alpha = [[0.0] * num_t for _ in range(num_i)]
    for i_idx in range(num_i):
        base = i_idx * num_t
        row = alpha[i_idx]
        for t_idx in range(num_t):
            row[t_idx] = alpha_flat[base + t_idx]
    rho = [rho_arr[i_idx] for i_idx in range(num_i)]
    return alpha, rho


def _price_machine_in_worker(
    m_idx: int,
    use_gurobi_when_no_fixes: bool,
    init_time_limit: float,
    rc_tolerance: float,
) -> tuple[int, SubproblemResult | None]:
    """在 worker 内对单台机器定价；仅当 RC 显著为负时返回完整列（减小 pickle）。"""
    state = _PRICING_WORKER_STATE
    data = state["data"]
    fixed_y = state["fixed_y"]
    num_i = state["num_i"]
    num_t = state["num_t"]
    alpha_flat = state["alpha_flat"]
    rho_arr = state["rho_arr"]
    beta_arr = state["beta_arr"]
    rc_arr = state["rc_arr"]
    assert alpha_flat is not None and rho_arr is not None and beta_arr is not None
    beta = beta_arr[m_idx]
    machine_fixes = machine_fixed_y(fixed_y, m_idx)
    pricing_mode = state.get("pricing_mode", "ebrp")
    ebrp_force = state.get("ebrp_force", False)
    resolved = _resolve_pricing_mode(data, pricing_mode, ebrp_force=ebrp_force)
    if resolved in ("ebrp", "ebrp_heuristic", "im_gurobi") and alpha_flat is not None:
        if resolved in ("ebrp", "ebrp_heuristic"):
            alpha = None
            rho = [rho_arr[i_idx] for i_idx in range(num_i)]
        else:
            alpha, rho = _nested_alpha_from_shm()
    elif resolved in ("ebrp", "ebrp_heuristic", "im_gurobi"):
        alpha, rho = _nested_alpha_from_shm()
    else:
        alpha, rho = None, list(rho_arr)
    setup_deltas = state.get("setup_cost_deltas") or {}
    sp = _solve_pricing_subproblem(
        data,
        m_idx,
        alpha,
        beta,
        rho,
        pricing_mode=pricing_mode,
        im_pricing_time_limit=state.get("im_pricing_time_limit", IM_PRICING_DEFAULT_TIME_LIMIT),
        ebrp_time_limit=state.get("ebrp_time_limit"),
        ebrp_force=ebrp_force,
        alpha_flat=alpha_flat,
        num_t_for_flat=num_t,
        fixed_y=machine_fixes or None,
        ebrp_contexts=state.get("ebrp_contexts"),
        skip_cross_check=state.get("skip_cross_check", False),
        setup_cost_delta=setup_deltas.get(m_idx),
        sri_terms=(state.get("sri_terms_by_machine") or {}).get(m_idx, ()),
    )
    if rc_arr is not None:
        rc_arr[m_idx] = sp.reduced_cost
    if not sp.is_optimal or sp.reduced_cost < -rc_tolerance:
        return m_idx, sp
    return m_idx, None


def _execute_pricing_round(job: tuple[bool, float, float]) -> list[tuple[int, SubproblemResult | None]]:
    """一轮定价：单次 map 内顺序处理全部机器（机器数少时通常不如并行 map）。"""
    use_gurobi_when_no_fixes, init_time_limit, rc_tolerance = job
    num_m = len(_PRICING_WORKER_STATE["data"].machines)
    return [
        _price_machine_in_worker(m_idx, use_gurobi_when_no_fixes, init_time_limit, rc_tolerance)
        for m_idx in range(num_m)
    ]


def _execute_pricing_job(
    job: tuple[int, bool, float, float]
    | tuple[int, list[list[float]], list[float], float, bool, float],
) -> tuple[int, SubproblemResult | None]:
    """单台机器定价（供进程池调用；算例由 initializer 注入）。"""
    state = _PRICING_WORKER_STATE
    data = state["data"]
    fixed_y = state["fixed_y"]
    pricing_time_limit = state["pricing_time_limit"]
    num_i = state["num_i"]
    num_t = state["num_t"]

    if len(job) == 4:
        m_idx, use_gurobi, init_time_limit, rc_tolerance = job
        return _price_machine_in_worker(m_idx, use_gurobi, init_time_limit, rc_tolerance)

    m_idx, alpha, rho, beta, use_gurobi, init_time_limit = job
    machine_fixes = machine_fixed_y(fixed_y, m_idx)
    pricing_mode = state.get("pricing_mode", "ebrp")
    ebrp_force = state.get("ebrp_force", False)
    sp = _solve_pricing_subproblem(
        data,
        m_idx,
        alpha,
        beta,
        rho,
        pricing_mode=pricing_mode,
        im_pricing_time_limit=state.get("im_pricing_time_limit", IM_PRICING_DEFAULT_TIME_LIMIT),
        ebrp_time_limit=state.get("ebrp_time_limit"),
        ebrp_force=ebrp_force,
        fixed_y=machine_fixes or None,
        ebrp_contexts=state.get("ebrp_contexts"),
        skip_cross_check=state.get("skip_cross_check", False),
        setup_cost_delta=(state.get("setup_cost_deltas") or {}).get(m_idx),
        sri_terms=(state.get("sri_terms_by_machine") or {}).get(m_idx, ()),
    )
    rc_arr = state.get("rc_arr")
    if rc_arr is not None:
        rc_arr[m_idx] = sp.reduced_cost
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
        pricing_inprocess: bool = False,
        pricing_batch: bool = True,
        time_limit: float | None = None,
        fixed_y: dict[SetupFix, int] | None = None,
        fixed_wip: dict[int, WipFix] | None = None,
        fixed_scrap: dict[int, int] | None = None,
        inherited_columns: dict[str, list[ScheduleColumn]] | None = None,
        inherited_sri_cuts: tuple[tuple[tuple[int, ...], int, int, str], ...] = (),
        lrmp_cuts: dict | str | None = None,
        verbose: bool = False,
        pricing_mode: str = "ebrp",
        initial_pricing_mode: str | None = None,
        ebrp_time_limit: float | None = EBRP_DEFAULT_TIME_LIMIT,
        im_pricing_time_limit: float = IM_PRICING_DEFAULT_TIME_LIMIT,
        progress_interval_sec: float | None = DEFAULT_PROGRESS_INTERVAL_SEC,
        ebrp_force: bool = False,
        max_columns_per_iter: int | None = None,
        skip_cross_check: bool = False,
        sum_rc_stop_threshold: float | None = None,
    ) -> None:
        self.data = data
        self.max_iterations = max_iterations
        self.rc_tolerance = rc_tolerance
        self.init_time_limit = init_time_limit
        self.use_gurobi_init = use_gurobi_init
        self.max_columns_per_iter = max_columns_per_iter
        self.pricing_mode = pricing_mode.lower()
        self.initial_pricing_mode = (
            initial_pricing_mode.lower() if initial_pricing_mode else None
        )
        self.ebrp_time_limit = ebrp_time_limit
        self.im_pricing_time_limit = im_pricing_time_limit
        self.ebrp_force = ebrp_force or pricing_mode.lower() == "ebrp_force"
        self.stabilization_theta = max(0.0, min(0.9, stabilization_theta))
        self.pricing_workers = pricing_workers
        self.pricing_time_limit = pricing_time_limit
        self.pricing_inprocess = pricing_inprocess
        self.pricing_batch = pricing_batch
        self.time_limit = time_limit
        self.fixed_y: dict[SetupFix, int] = dict(fixed_y or {})
        self.fixed_wip: dict[int, WipFix] = dict(fixed_wip or {})
        self.fixed_scrap: dict[int, int] = dict(fixed_scrap or {})
        self.inherited_columns = inherited_columns
        # A Gurobi constraint belongs to one master model only.  Branch nodes
        # therefore pass the mathematical SRI specification, which is rebuilt
        # against every child's fresh master below.
        self._inherited_sri_specs = tuple(inherited_sri_cuts)
        self.lrmp_cut_opts = merge_cut_options(lrmp_cuts)
        self.verbose = verbose
        self._progress = ProgressMonitor(
            f"CG:{data.raw.name}",
            interval_sec=progress_interval_sec or DEFAULT_PROGRESS_INTERVAL_SEC,
            enabled=verbose,
        )
        self._pricing_iter = 0
        resolved = _resolve_pricing_mode(data, self.pricing_mode, ebrp_force=self.ebrp_force)
        self._ebrp_contexts = (
            build_ebrp_contexts(data)
            if resolved in ("ebrp", "ebrp_heuristic") else None
        )
        self._ebrp_progress_sec = (
            progress_interval_sec or DEFAULT_PROGRESS_INTERVAL_SEC
            if verbose
            else 0.0
        )
        self.skip_cross_check = skip_cross_check
        # Optional paper-style approximate stop: terminate when the sum of
        # pricing reduced costs exceeds a threshold (e.g. -0.1).  This is not
        # a proof that every reduced cost is non-negative.
        self.sum_rc_stop_threshold = sum_rc_stop_threshold
        self._flow_pi_slots: list[tuple[int, gp.Constr]] = []
        self._item_leads = [int(v) for v in data.l_ti]
        self._alpha_flat: mp.Array | None = None
        self._rho_arr: mp.Array | None = None
        self._beta_arr: mp.Array | None = None
        self._rc_arr: mp.Array | None = None
        self.columns: dict[str, list[ScheduleColumn]] = {m: [] for m in data.machines}
        self._column_sigs: dict[str, set[tuple]] = {m: set() for m in data.machines}
        self._next_col_id = 0
        self._model: gp.Model | None = None
        self._artifacts: dict | None = None
        self.deadline_cuts: dict[int, dict] = {}
        # Root-only subset-row cuts.  They are optional and disabled in the
        # default experiment configuration.
        self.sri_cuts: list[SRICut] = self._clone_inherited_sri_cuts()
        self.same_mt_cuts: list[dict] = []
        self.wip_proc_lb_rows: list[dict] = []
        self.setup_pair_cuts: list[dict] = []
        self._prev_alpha = None
        self._prev_gamma = None
        self.converged = False
        self.lrmp_obj: float | None = None
        self.last_min_rc: float | None = None
        self.last_pool_min_rc: float | None = None
        self.last_improvable_pool_rc: float | None = None
        self.last_effective_pricing_rc: float | None = None
        self.stop_reason: str | None = None
        self.iteration_times: list[float] = []
        self.iteration_time_detail: list[dict[str, float]] = []
        self.incumbent: float | None = None  # for early termination in B&P nodes
        self._setup_cost_deltas: dict[int, list[list[float]]] = {}

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message, flush=True)

    def _total_columns(self) -> int:
        return sum(len(cols) for cols in self.columns.values())

    def _pricing_kwargs(self) -> dict:
        return {
            "pricing_mode": self.pricing_mode,
            "init_time_limit": self.init_time_limit,
            "im_pricing_time_limit": self.im_pricing_time_limit,
            "ebrp_time_limit": self.ebrp_time_limit,
        }

    def _dispatch_pricing(
        self,
        m_idx: int,
        alpha: list[list[float]] | None,
        rho: list[float],
        beta: float,
        *,
        machine_fixes: dict[tuple[int, int], int] | None,
        alpha_flat=None,
        num_t_for_flat: int | None = None,
    ) -> SubproblemResult:
        return _solve_pricing_subproblem(
            self.data,
            m_idx,
            alpha,
            beta,
            rho,
            pricing_mode=self.pricing_mode,
            im_pricing_time_limit=self.im_pricing_time_limit,
            ebrp_time_limit=self.ebrp_time_limit,
            ebrp_force=self.ebrp_force,
            alpha_flat=alpha_flat,
            num_t_for_flat=num_t_for_flat,
            fixed_y=machine_fixes,
            ebrp_contexts=self._ebrp_contexts,
            ebrp_progress_sec=self._ebrp_progress_sec,
            skip_cross_check=self.skip_cross_check,
            setup_cost_delta=self._setup_cost_deltas.get(m_idx),
            sri_terms=self._sri_pricing_terms(m_idx),
        )

    def _sri_pricing_terms(self, m_idx: int) -> tuple[tuple[int, float], ...]:
        """Return machine-local ``(item_mask, master_dual)`` SRI pairs.

        ``SRICut.item_mask`` is expressed in the global ``cured_items``
        index.  EBRP's routing state, however, uses the local index of the
        items feasible on the current machine.  Passing a global mask into
        EBRP silently applies an SRI dual to unrelated local items (or drops
        it altogether), so translate every mask before pricing.
        """
        terms: list[tuple[int, float]] = []
        if self._ebrp_contexts is not None:
            local_items = self._ebrp_contexts[m_idx].local_items
        else:
            local_items = self.data.items_by_machine[m_idx]
        for cut in self.sri_cuts:
            if cut.constr is None:
                continue
            try:
                pi = float(cut.constr.Pi)
            except (AttributeError, gp.GurobiError):
                continue
            if abs(pi) <= 1e-12:
                continue
            local_mask = 0
            for local_idx, global_idx in enumerate(local_items):
                if cut.item_mask & (1 << global_idx):
                    local_mask |= 1 << local_idx
            if local_mask:
                terms.append((local_mask, pi))
        return tuple(terms)

    def _separate_sri_cuts(self) -> int:
        """Add violated triple SRIs at the root master only.

        Child-node CG solvers do not inherit dynamically separated master rows,
        so this conservative first implementation confines separation to the
        root.  New root columns still receive every stored SRI coefficient.
        """
        if not self.lrmp_cut_opts.get("sri", False):
            return 0
        if self.fixed_y or self.fixed_wip or self.fixed_scrap:
            return 0
        if self._model is None or self._artifacts is None:
            return 0
        remaining = int(self.lrmp_cut_opts.get("sri_max_cuts", 3)) - len(self.sri_cuts)
        if remaining <= 0:
            return 0
        q_vars = self._artifacts["Q"]
        q_values = {
            machine: [float(qv.X) for qv in q_vars[machine]]
            for machine in self.data.machines
        }
        candidates = separate_root_triples(
            self.data,
            self.columns,
            q_values,
            existing_masks={cut.item_mask for cut in self.sri_cuts},
            max_cuts=remaining,
            violation_tol=float(self.lrmp_cut_opts.get("sri_violation_tol", 1e-6)),
        )
        for cut in candidates:
            expr = gp.LinExpr()
            for machine in self.data.machines:
                for idx, col in enumerate(self.columns[machine]):
                    coeff = sri_coefficient(self.data, col, cut)
                    if coeff:
                        expr += coeff * q_vars[machine][idx]
            cut.constr = self._model.addConstr(expr <= cut.rhs, name=cut.name)
            self.sri_cuts.append(cut)
        if candidates:
            self._artifacts["sri_cuts"] = self.sri_cuts
            self._log(f"[SRI] added {len(candidates)} violated root triple cut(s)")
        return len(candidates)

    def _fold_deadline_duals_into_alpha(self, alpha: list[list[float]]) -> list[list[float]]:
        """把 deadline cover 割对偶 π_τ 折入 α：RC -= π·A  ⟺  α'[i][t] = α[i][t] + π_τ（arrival≤τ）。"""
        if not self.deadline_cuts:
            return alpha
        adj = [row[:] for row in alpha]
        periods = self.data.periods
        for tau, info in self.deadline_cuts.items():
            try:
                pi = info["constr"].Pi
            except Exception:
                continue
            if abs(pi) < 1e-12:
                continue
            for i_idx in info.get("I") or []:
                row = adj[i_idx]
                for t_idx, period in enumerate(periods):
                    if period <= tau:
                        row[t_idx] += pi
        return adj

    def _fold_wip_branch_duals_into_rho(self, rho: list[float]) -> list[float]:
        """WIP 分支 Σ q·wip_input ≷ bound 的对偶折入 ρ：RC -= π·x_i ⟺ ρ'[i] = ρ[i] + π。"""
        rows = (self._artifacts or {}).get("wip_branch_rows") or []
        if not rows:
            return rho
        adj = list(rho)
        for constr, i_idx, _sense, _bound in rows:
            try:
                pi = constr.Pi
            except Exception:
                continue
            if abs(pi) < 1e-12:
                continue
            adj[i_idx] += pi
        return adj

    def _refresh_setup_cut_cost_deltas(self) -> None:
        """setup_pair / same_mt / Y-branch 对偶 → setup 额外成本 σ[u][t] = -Σ π。"""
        self._setup_cost_deltas = {}
        num_u = len(self.data.configs)
        num_t = len(self.data.periods)
        num_m = len(self.data.machines)
        sigmas: list[list[list[float]] | None] = [None] * num_m

        def _sigma(m_idx: int) -> list[list[float]]:
            if sigmas[m_idx] is None:
                sigmas[m_idx] = [[0.0] * num_t for _ in range(num_u)]
            return sigmas[m_idx]

        for info in self.setup_pair_cuts:
            try:
                pi = info["constr"].Pi
            except Exception:
                continue
            if abs(pi) < 1e-12:
                continue
            m_idx = info["m_idx"]
            u1, t1 = info["a"]
            u2, t2 = info["b"]
            s = _sigma(m_idx)
            s[u1][t1] -= pi
            s[u2][t2] -= pi

        for info in self.same_mt_cuts:
            try:
                pi = info["constr"].Pi
            except Exception:
                continue
            if abs(pi) < 1e-12:
                continue
            m_idx = info["m_idx"]
            t_idx = info["t_idx"]
            s = _sigma(m_idx)
            for u_idx in range(num_u):
                s[u_idx][t_idx] -= pi

        for constr, u_idx, m_idx, t_idx, _val in (self._artifacts or {}).get("branch_rows") or []:
            try:
                pi = constr.Pi
            except Exception:
                continue
            if abs(pi) < 1e-12:
                continue
            _sigma(m_idx)[u_idx][t_idx] -= pi

        self._setup_cost_deltas = {m: s for m, s in enumerate(sigmas) if s is not None}

    def _pricing_label(self, use_gurobi: bool = False) -> str:
        mode = _resolve_pricing_mode(self.data, self.pricing_mode, ebrp_force=self.ebrp_force)
        if mode == "ebrp_heuristic":
            return "EBRP-H"
        if mode == "ebrp":
            return "EBRP"
        return "Gurobi(IM)"

    def _reset_master(self) -> None:
        self._model = None
        self._artifacts = None
        self.same_mt_cuts = []
        self.wip_proc_lb_rows = []
        self.setup_pair_cuts = []
        self.deadline_cuts = {}
        self.sri_cuts = self._clone_inherited_sri_cuts()

    def _clone_inherited_sri_cuts(self) -> list[SRICut]:
        """Instantiate node-local SRI rows from inherited math-only specs."""
        return [
            SRICut(items=items, item_mask=item_mask, rhs=rhs, name=name)
            for items, item_mask, rhs, name in self._inherited_sri_specs
        ]

    def _rebuild_column_sigs(self) -> None:
        self._column_sigs = {m: set() for m in self.data.machines}
        for machine, cols in self.columns.items():
            for col in cols:
                self._column_sigs[machine].add(column_fingerprint(col))

    def _add_column(self, machine: str, column: ScheduleColumn) -> None:
        fp = column_fingerprint(column)
        if fp in self._column_sigs[machine]:
            raise RuntimeError(
                f"duplicate column add for machine={machine}: fingerprint already in pool "
                f"(pool_size={len(self.columns.get(machine, []))})"
            )
        self.columns[machine].append(column)
        self._column_sigs[machine].add(fp)

    def _column_seen(self, machine: str, column: ScheduleColumn) -> bool:
        return column_fingerprint(column) in self._column_sigs[machine]

    def _uses_im_flow(self) -> bool:
        return _resolve_pricing_mode(self.data, self.pricing_mode, ebrp_force=self.ebrp_force) in (
            "ebrp",
            "ebrp_heuristic",
            "im_gurobi",
        )

    def _batch_span_at_start(self, col: ScheduleColumn, start_period: int) -> int:
        """列内在 start_period 开工批次的占罐跨期 L。"""
        t_idx = start_period - 1
        span = 0
        for (name, period), qty in col.x.items():
            if qty <= 0 or period != start_period:
                continue
            i_idx = self.data.cured_index[name]
            span = max(span, self.data.lt_it[i_idx][t_idx])
        return span

    def _column_arrival_t_idx(self, col: ScheduleColumn, item_name: str, start_period: int) -> int | None:
        """批次共同出罐的完工时刻：使用该批次的最大固化跨期。"""
        t_idx = start_period - 1
        span = self._batch_span_at_start(col, start_period)
        arr = t_idx + span - 1
        if 0 <= arr < len(self.data.periods):
            return arr
        return None

    def _column_prod_coeffs(self, col: ScheduleColumn) -> list[tuple[int, int, float]]:
        """返回列对完工流平衡的 (i_idx, arrival_t_idx, qty)。"""
        coeffs: list[tuple[int, int, float]] = []
        for (item_name, start_period), qty in col.x.items():
            if qty <= 0:
                continue
            i_idx = self.data.cured_index[item_name]
            if self._uses_im_flow():
                arr_t = self._column_arrival_t_idx(col, item_name, start_period)
            else:
                prod_t = start_period - 1
                arr_t = prod_t + self._item_leads[i_idx]
                if not (0 <= arr_t < len(self.data.periods)):
                    arr_t = None
            if arr_t is not None:
                coeffs.append((i_idx, arr_t, float(qty)))
        return coeffs

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
            if self._uses_im_flow():
                arr_t = self._column_arrival_t_idx(col, name, prod_period)
                arrival = self.data.periods[arr_t] if arr_t is not None else prod_period + 9999
            else:
                lead = int(self.data.l_ti[ii])
                arrival = prod_period + lead
            if arrival <= tau:
                contrib += qty
        return contrib

    def _get_column_wip_input(self, col: ScheduleColumn, i_idx: int) -> float:
        """列在时间窗 [E_i, D_i] 内对物料 i 的开始加工总量。"""
        item_name = self.data.cured_items[i_idx]
        e_i = self.data.earliest_starts[i_idx]
        d_i = self.data.deadlines[i_idx]
        total = 0.0
        for (name, period), qty in col.x.items():
            if name == item_name and e_i <= period <= d_i and qty > 0:
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
        use_shared_duals: bool = False,
    ) -> list[tuple]:
        init_tl = self.init_time_limit if init_time_limit is None else init_time_limit
        jobs = []
        for m_idx, _machine in enumerate(self.data.machines):
            machine_fixes = machine_fixed_y(self.fixed_y, m_idx)
            use_gurobi = use_gurobi_when_no_fixes or bool(machine_fixes)
            if use_shared_duals and not use_gurobi:
                jobs.append((m_idx, use_gurobi_when_no_fixes, init_tl, self.rc_tolerance))
            else:
                jobs.append(
                    (
                        m_idx,
                        alpha,
                        rho,
                        betas[m_idx],
                        use_gurobi_when_no_fixes,
                        init_tl,
                    )
                )
        return jobs

    def _write_pricing_duals(
        self,
        alpha: list[list[float]],
        rho: list[float],
        betas: list[float],
    ) -> None:
        if self._alpha_flat is None or self._rho_arr is None or self._beta_arr is None:
            return
        num_i = len(self.data.cured_items)
        num_t = len(self.data.periods)
        alpha_flat = self._alpha_flat
        for i_idx in range(num_i):
            base = i_idx * num_t
            row = alpha[i_idx]
            for t_idx in range(num_t):
                alpha_flat[base + t_idx] = row[t_idx]
        for i_idx in range(num_i):
            self._rho_arr[i_idx] = rho[i_idx]
        for m_idx in range(len(self.data.machines)):
            self._beta_arr[m_idx] = betas[m_idx]

    def _price_one_machine(
        self,
        m_idx: int,
        alpha: list[list[float]],
        rho: list[float],
        beta: float,
        *,
        use_gurobi_when_no_fixes: bool,
        init_time_limit: float,
    ) -> tuple[int, SubproblemResult]:
        machine_fixes = machine_fixed_y(self.fixed_y, m_idx)
        sp = self._dispatch_pricing(
            m_idx,
            alpha,
            rho,
            beta,
            machine_fixes=machine_fixes or None,
        )
        return m_idx, sp

    def _bind_pricing_worker_state(self) -> None:
        """主进程定价时同步 worker 全局状态（与进程池 initializer 一致）。"""
        global _PRICING_WORKER_STATE
        _PRICING_WORKER_STATE = {
            "data": self.data,
            "fixed_y": self.fixed_y,
            "pricing_time_limit": self.pricing_time_limit,
            "pricing_mode": self.pricing_mode,
            "ebrp_force": self.ebrp_force,
            "ebrp_time_limit": self.ebrp_time_limit,
            "im_pricing_time_limit": self.im_pricing_time_limit,
            "ebrp_contexts": self._ebrp_contexts,
            "alpha_flat": self._alpha_flat,
            "rho_arr": self._rho_arr,
            "beta_arr": self._beta_arr,
            "rc_arr": self._rc_arr,
            "num_i": len(self.data.cured_items),
            "num_t": len(self.data.periods),
            "skip_cross_check": self.skip_cross_check,
            "setup_cost_deltas": self._setup_cost_deltas,
            # Shared-memory pricing bypasses ``_dispatch_pricing``.  Carry
            # the machine-local SRI dual terms explicitly so that this path
            # uses the same reduced-cost expression as direct pricing.
            "sri_terms_by_machine": {
                m_idx: self._sri_pricing_terms(m_idx)
                for m_idx in range(len(self.data.machines))
            },
        }

    def _price_machines_inprocess(
        self,
        alpha: list[list[float]],
        rho: list[float],
        betas: list[float],
        *,
        use_gurobi_when_no_fixes: bool,
        init_time_limit: float | None = None,
        use_shared_duals: bool = False,
        duals_preloaded: bool = False,
    ) -> list[tuple[int, SubproblemResult]]:
        init_tl = self.init_time_limit if init_time_limit is None else init_time_limit

        def _pricing_status() -> str:
            rc_parts = []
            if self._rc_arr is not None:
                for m_idx, machine in enumerate(self.data.machines):
                    rc_parts.append(f"{machine}={self._rc_arr[m_idx]:.2f}")
            rc_txt = " ".join(rc_parts) if rc_parts else "pricing..."
            return (
                f"iter={self._pricing_iter}  pricing={self._pricing_label(False)}  "
                f"cols={self._total_columns()}  {rc_txt}"
            )

        with heartbeat_while(
            f"CG:{self.data.raw.name}",
            _pricing_status,
            interval_sec=self._progress.interval_sec,
            enabled=self._progress.enabled,
        ):
            if use_shared_duals and self._alpha_flat is not None:
                if not duals_preloaded:
                    self._write_pricing_duals(alpha, rho, betas)
                self._bind_pricing_worker_state()
                return _execute_pricing_round((use_gurobi_when_no_fixes, init_tl, self.rc_tolerance))
            return [
                self._price_one_machine(
                    m_idx,
                    alpha,
                    rho,
                    betas[m_idx],
                    use_gurobi_when_no_fixes=use_gurobi_when_no_fixes,
                    init_time_limit=init_tl,
                )
                for m_idx in range(len(self.data.machines))
            ]

    def _price_machines(
        self,
        alpha: list[list[float]],
        rho: list[float],
        betas: list[float],
        *,
        use_gurobi_when_no_fixes: bool,
        init_time_limit: float | None = None,
        executor: ProcessPoolExecutor | None = None,
        use_shared_duals: bool = False,
        duals_preloaded: bool = False,
    ) -> list[tuple[int, SubproblemResult]]:
        # SRI duals are path-dependent.  Keep this first implementation
        # in-process so worker pricing can never omit the SRI contribution.
        if self.sri_cuts or self.pricing_inprocess or executor is None:
            return self._price_machines_inprocess(
                alpha,
                rho,
                betas,
                use_gurobi_when_no_fixes=use_gurobi_when_no_fixes,
                init_time_limit=init_time_limit,
                use_shared_duals=use_shared_duals,
                duals_preloaded=duals_preloaded,
            )
        return self._price_machines_parallel(
            alpha,
            rho,
            betas,
            use_gurobi_when_no_fixes=use_gurobi_when_no_fixes,
            init_time_limit=init_time_limit,
            executor=executor,
            use_shared_duals=use_shared_duals,
            duals_preloaded=duals_preloaded,
        )

    def _price_machines_parallel(
        self,
        alpha: list[list[float]],
        rho: list[float],
        betas: list[float],
        *,
        use_gurobi_when_no_fixes: bool,
        init_time_limit: float | None = None,
        executor: ProcessPoolExecutor,
        use_shared_duals: bool = False,
        duals_preloaded: bool = False,
    ) -> list[tuple[int, SubproblemResult]]:
        if use_shared_duals and not duals_preloaded:
            self._write_pricing_duals(alpha, rho, betas)
        init_tl = self.init_time_limit if init_time_limit is None else init_time_limit
        if use_shared_duals and self.pricing_batch:
            return list(
                executor.map(
                    _execute_pricing_round,
                    [(use_gurobi_when_no_fixes, init_tl, self.rc_tolerance)],
                )
            )[0]
        jobs = self._build_pricing_jobs(
            alpha,
            rho,
            betas,
            use_gurobi_when_no_fixes=use_gurobi_when_no_fixes,
            init_time_limit=init_time_limit,
            use_shared_duals=use_shared_duals,
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

    def _wip_branch_expr(
        self,
        q_vars: dict[str, list[gp.Var]],
        i_idx: int,
    ) -> gp.LinExpr:
        """窗内加工量 W_i = sum_{m,c} q * column_wip_input(c,i)。"""
        data = self.data
        expr = gp.LinExpr()
        for machine in data.machines:
            for col_idx, col in enumerate(self.columns[machine]):
                coeff = self._get_column_wip_input(col, i_idx)
                if coeff > 0:
                    expr += float(coeff) * q_vars[machine][col_idx]
        return expr

    def _add_branch_constraints(
        self,
        model: gp.Model,
        q_vars: dict[str, list[gp.Var]],
        r_vars: gp.Var | dict,
    ) -> tuple[
        list[tuple[gp.Constr, int, int, int, int]],
        list[tuple[gp.Constr, int, str, int]],
        list[tuple[gp.Constr, int, int]],
    ]:
        """对 fixed_y / fixed_wip / fixed_scrap 添加分支约束。"""
        branch_rows: list[tuple[gp.Constr, int, int, int, int]] = []
        wip_rows: list[tuple[gp.Constr, int, str, int]] = []
        scrap_rows: list[tuple[gp.Constr, int, int]] = []
        data = self.data
        for (u_idx, m_idx, t_idx), val in self.fixed_y.items():
            machine = data.machines[m_idx]
            expr = self._setup_branch_expr(q_vars, machine, u_idx, t_idx)
            if val == 0:
                constr = model.addConstr(expr == 0, name=f"branch_y0_{u_idx}_{m_idx}_{t_idx}")
            else:
                constr = model.addConstr(expr == 1, name=f"branch_y1_{u_idx}_{m_idx}_{t_idx}")
            branch_rows.append((constr, u_idx, m_idx, t_idx, val))
        for i_idx, (sense, bound) in self.fixed_wip.items():
            expr = self._wip_branch_expr(q_vars, i_idx)
            if sense == "le":
                constr = model.addConstr(expr <= bound, name=f"branch_wip_le_{i_idx}_{bound}")
            else:
                constr = model.addConstr(expr >= bound, name=f"branch_wip_ge_{i_idx}_{bound}")
            wip_rows.append((constr, i_idx, sense, bound))
        for i_idx, val in self.fixed_scrap.items():
            if val > 0:
                constr = model.addConstr(r_vars[i_idx] >= val, name=f"branch_scrap_{i_idx}")
            else:
                # R=0 侧：必须强制加工全部 WIP（R + W = w_i ⇒ W = w_i）
                constr = model.addConstr(r_vars[i_idx] <= 0, name=f"branch_scrap0_{i_idx}")
            scrap_rows.append((constr, i_idx, val))
        return branch_rows, wip_rows, scrap_rows

    def _covering_items(self) -> list[int]:
        """需要列池覆盖加工量的物料：R=0（W=w_i）或 W≥bound。"""
        items: list[int] = []
        for i_idx, val in self.fixed_scrap.items():
            if val == 0:
                items.append(i_idx)
        for i_idx, (sense, bound) in self.fixed_wip.items():
            if sense == "ge" and bound > 0:
                items.append(i_idx)
        return sorted(set(items))

    def _needs_feasibility_seed(self) -> bool:
        if self._covering_items():
            return True
        if any(v == 1 for v in self.fixed_y.values()):
            return True
        return bool(self.fixed_wip)

    def initialize_columns(self, executor: ProcessPoolExecutor | None = None) -> None:
        data = self.data
        if self.inherited_columns:
            self.columns = filter_columns(self.inherited_columns, data, self.fixed_y)
            self._rebuild_column_sigs()
            max_id = max(
                (col.column_id for cols in self.columns.values() for col in cols),
                default=-1,
            )
            self._next_col_id = max_id + 1
        else:
            self.columns = {m: [] for m in data.machines}
            self._rebuild_column_sigs()
            self._next_col_id = 0

        for m_idx, machine in enumerate(data.machines):
            has_empty = any(not col.y and not col.x for col in self.columns.get(machine, []))
            if not has_empty:
                self._add_column(machine, self._create_empty_column(machine))

        if self.inherited_columns:
            self._seed_forced_setup_columns()
            if self._needs_feasibility_seed():
                self._seed_covering_columns(executor)
                self._seed_branch_columns(executor)
            return

        alpha = [[0.0] * len(data.periods) for _ in data.cured_items]
        rho = [0.0] * len(data.cured_items)
        betas = [0.0] * len(data.machines)
        for machine in data.machines:
            if not any(not col.y and not col.x for col in self.columns[machine]):
                self._add_column(machine, self._create_empty_column(machine))

        if self._alpha_flat is not None:
            self._write_pricing_duals(alpha, rho, betas)

        use_gurobi_init = self.use_gurobi_init and not self.fixed_y
        self._log(
            f"初始列定价: {self._pricing_label(use_gurobi_init)} "
            f"(|M|={len(data.machines)}, |I|={len(data.cured_items)}, |T|={len(data.periods)})"
        )
        if use_gurobi_init:
            original_mode = self.pricing_mode
            original_force = self.ebrp_force
            if self.initial_pricing_mode is not None:
                self.pricing_mode = self.initial_pricing_mode
                self.ebrp_force = False
            try:
                priced = self._price_machines(
                    alpha,
                    rho,
                    betas,
                    use_gurobi_when_no_fixes=True,
                    executor=executor,
                    use_shared_duals=self._alpha_flat is not None,
                )
            finally:
                self.pricing_mode = original_mode
                self.ebrp_force = original_force
            for priced_idx, (m_idx, sp) in enumerate(priced):
                if sp is None:
                    continue
                machine = data.machines[m_idx]
                col = subproblem_to_column(data, m_idx, self._next_col_id, sp)
                if not column_compatible(col, m_idx, data, self.fixed_y):
                    continue
                if self._column_seen(machine, col):
                    continue  # init seed may rediscover; CG loop forbids duplicates
                self._next_col_id += 1
                self._add_column(machine, col)
        else:
            self._log("初始列: 仅空列（典型 CG 多轮起点）")
        self._log(f"初始列完成: 共 {self._total_columns()} 列")

    def _try_add_priced_column(self, m_idx: int, sp: SubproblemResult | None) -> bool:
        if sp is None or not math.isfinite(getattr(sp, "reduced_cost", float("inf"))):
            return False
        data = self.data
        machine = data.machines[m_idx]
        col = subproblem_to_column(data, m_idx, self._next_col_id, sp)
        if not column_compatible(col, m_idx, data, self.fixed_y):
            return False
        if self._column_seen(machine, col):
            return False
        self._next_col_id += 1
        self._add_column(machine, col)
        return True

    def _seed_forced_setup_columns(self) -> None:
        """Y=1 分支后若列池缺少 forced setup 列，用规范定价生成可行列以启动 CG。"""
        data = self.data
        num_t = len(data.periods)
        num_i = len(data.cured_items)
        covering = self._covering_items()
        # 大 ρ 鼓励在 forced setup 上同时覆盖 R=0 / W≥ 物料
        rho = [0.0] * num_i
        for i_idx in covering:
            rho[i_idx] = 1.0e6
        alpha = [[0.0] * num_t for _ in range(num_i)]
        for m_idx, machine in enumerate(data.machines):
            cols = self.columns.get(machine, [])
            fixes = machine_needs_setup_seed(data, m_idx, cols, self.fixed_y)
            if fixes is None:
                continue
            sp = self._dispatch_pricing(
                m_idx,
                alpha,
                rho,
                0.0,
                machine_fixes=fixes,
            )
            if not sp.y or not math.isfinite(sp.reduced_cost):
                continue
            self._try_add_priced_column(m_idx, sp)

    def _seed_covering_columns(self, executor: ProcessPoolExecutor | None = None) -> None:
        """对 R=0 / W≥ 物料抬高 ρ 定价补种，先让 LRMP 可行再进入正式 CG。"""
        items = self._covering_items()
        if not items:
            return
        data = self.data
        num_i = len(data.cured_items)
        num_t = len(data.periods)
        seed_rho = 1.0e6

        rho_passes: list[list[float]] = []
        for i_idx in items:
            rho = [0.0] * num_i
            rho[i_idx] = seed_rho
            rho_passes.append(rho)
        if len(items) > 1:
            rho = [0.0] * num_i
            for i_idx in items:
                rho[i_idx] = seed_rho
            rho_passes.append(rho)

        added = 0
        for rho in rho_passes:
            alpha = [[0.0] * num_t for _ in range(num_i)]
            betas = [0.0] * len(data.machines)
            priced = self._price_machines(
                alpha,
                rho,
                betas,
                use_gurobi_when_no_fixes=True,
                executor=executor,
                use_shared_duals=False,
            )
            for priced_idx, (m_idx, sp) in enumerate(priced):
                if self._try_add_priced_column(m_idx, sp):
                    added += 1
        if added:
            self._log(
                f"补种覆盖列: +{added} 列 (forced items={items}, "
                f"pool={self._total_columns()})"
            )

    def _seed_branch_columns(self, executor: ProcessPoolExecutor | None = None) -> None:
        """scrap/Y/WIP 新分支后，用零对偶（或当前对偶）再补一轮列。"""
        data = self.data
        alpha = [[0.0] * len(data.periods) for _ in data.cured_items]
        rho = [0.0] * len(data.cured_items)
        betas = [0.0] * len(data.machines)
        if self._alpha_flat is not None:
            self._write_pricing_duals(alpha, rho, betas)
        priced = self._price_machines(
            alpha,
            rho,
            betas,
            use_gurobi_when_no_fixes=True,
            executor=executor,
            use_shared_duals=False,
        )
        for m_idx, sp in priced:
            self._try_add_priced_column(m_idx, sp)

    def _init_master(self, relaxed: bool = True) -> None:
        """首次构建 LRMP；后续通过 _add_column_to_master 增量加列。"""
        if self._model is not None:
            return

        data = self.data
        model = gp.Model("LRMP" if relaxed else "RMP")
        model.Params.OutputFlag = 0
        model.Params.Method = 1
        model.Params.LPWarmStart = 1

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
                q_var = model.addVar(vtype=q_vtype, lb=0, name=f"Q_{machine}_{len(q_vars[machine])}")
                # 注意：不要设 ub=1。已有 Σ_j q_j ≤ 1，再加逐列 ub=1 会引入上界对偶，
                # 使最优 LRMP 上出现 q=1 且 RC<0，定价误以为可再改进并触发“重复列”假阳性。
                if not relaxed:
                    q_var.UB = 1.0
                q_vars[machine].append(q_var)
                obj += col.production_cost * q_var

        model.setObjective(obj, GRB.MINIMIZE)

        # 直接库存平衡
        flow_items: dict[tuple[int, int], gp.Constr] = {}
        for i_idx in range(num_i):
            item_name = data.cured_items[i_idx]
            for t_idx in range(num_t):
                prev_inv = 0 if t_idx == 0 else s_plus[i_idx, t_idx - 1]
                prev_bo = 0 if t_idx == 0 else s_minus[i_idx, t_idx - 1]
                prod_in = gp.LinExpr()
                if self._uses_im_flow():
                    for machine in data.machines:
                        for col_idx, col in enumerate(self.columns[machine]):
                            for ci, arr_t, qty in self._column_prod_coeffs(col):
                                if ci == i_idx and arr_t == t_idx:
                                    prod_in += qty * q_vars[machine][col_idx]
                else:
                    lead = int(data.l_ti[i_idx])
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
                    prev_inv - prev_bo + prod_in
                    == dem - dem * r_vars[i_idx] + s_plus[i_idx, t_idx] - s_minus[i_idx, t_idx]
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

        if self.lrmp_cut_opts.get("deadline_cover", True):
            distinct_taus = sorted(set(data.deadlines)) if hasattr(data, "deadlines") and data.deadlines else []
            for tau in distinct_taus:
                # Paper (37)--(38): E_tau contains precisely the items whose
                # latest feasible start e_i is no later than tau.  The same
                # set is stored in ``deadline_cuts`` and is subsequently used
                # for column coefficients and the pricing-dual fold.
                I = [ii for ii in range(num_i) if data.deadlines[ii] <= tau]
                if not I:
                    continue
                if self.lrmp_cut_opts.get("tight_deadline", False):
                    rhs = deadline_cover_rhs(data, tau, I)
                else:
                    rhs = 0.0
                    min_l = min(data.l_u) if data.l_u else 1
                    for m_idx in range(len(data.machines)):
                        max_trays_per_setup = int(data.q_m[m_idx] / max(min(data.v_i), 0.1)) if data.v_i else 5
                        num_possible_setups = max(1, int(tau / max(min_l, 1)))
                        rhs += num_possible_setups * max_trays_per_setup
                cut = model.addConstr(gp.LinExpr() <= rhs, name=f"deadline_cover_tau{tau}")
                self.deadline_cuts[tau] = {"constr": cut, "rhs": rhs, "I": I}

        # Root-separated SRIs are inherited by branch nodes as mathematical
        # specifications.  Rebuild their node-local Gurobi rows before any
        # newly priced column is appended to this master.
        for sri_cut in self.sri_cuts:
            expr = gp.LinExpr()
            for machine in data.machines:
                for col_idx, col in enumerate(self.columns[machine]):
                    coeff = sri_coefficient(data, col, sri_cut)
                    if coeff:
                        expr += coeff * q_vars[machine][col_idx]
            sri_cut.constr = model.addConstr(expr <= sri_cut.rhs, name=sri_cut.name)

        self._add_lrmp_cuts(model, q_vars, r_vars)

        for m_idx, machine in enumerate(data.machines):
            for cidx, col in enumerate(self.columns[machine]):
                qv = q_vars[machine][cidx]
                for tau, cut_info in self.deadline_cuts.items():
                    contrib = self._get_column_contrib_to_tau(col, tau, cut_info.get("I", []))
                    if contrib > 0:
                        model.chgCoeff(cut_info["constr"], qv, float(contrib))

        branch_rows, wip_rows, scrap_rows = self._add_branch_constraints(model, q_vars, r_vars)

        self._flow_pi_slots = [
            (i_idx * num_t + t_idx, constr) for (i_idx, t_idx), constr in flow_items.items()
        ]

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
            "sri_cuts": self.sri_cuts,
            "same_mt_cuts": self.same_mt_cuts,
            "wip_proc_lb_rows": self.wip_proc_lb_rows,
            "setup_pair_cuts": self.setup_pair_cuts,
            "branch_rows": branch_rows,
            "wip_branch_rows": wip_rows,
            "scrap_rows": scrap_rows,
        }

    def _add_lrmp_cuts(self, model: gp.Model, q_vars: dict[str, list], r_vars) -> None:
        if not cuts_enabled(self.lrmp_cut_opts):
            return
        data = self.data
        opts = self.lrmp_cut_opts
        num_t = len(data.periods)
        num_u = len(data.configs)

        if opts.get("same_mt_setup", False):
            for m_idx, machine in enumerate(data.machines):
                for t_idx in range(num_t):
                    expr = gp.LinExpr()
                    for col_idx, col in enumerate(self.columns[machine]):
                        for u_idx in range(num_u):
                            coeff = column_setup_indicator(col, data, u_idx, t_idx)
                            if coeff > 0:
                                expr.addTerms(coeff, q_vars[machine][col_idx])
                    constr = model.addConstr(expr <= 1.0, name=f"same_mt_m{m_idx}_t{t_idx}")
                    self.same_mt_cuts.append(
                        {"constr": constr, "m_idx": m_idx, "t_idx": t_idx}
                    )

        if opts.get("wip_process_lb", False):
            for i_idx in range(len(data.cured_items)):
                max_proc = max_process_for_item(data, i_idx)
                lb = max(0.0, data.w_i[i_idx] - max_proc)
                if lb <= CUT_FRAC_TOL:
                    continue
                constr = model.addConstr(r_vars[i_idx] >= lb, name=f"wip_proc_lb_{i_idx}")
                self.wip_proc_lb_rows.append({"constr": constr, "i_idx": i_idx, "lb": lb})

        if opts.get("setup_pair_static", False):
            for cut_def in build_static_setup_pair_cuts(data):
                m_idx = cut_def["m_idx"]
                machine = data.machines[m_idx]
                expr = gp.LinExpr()
                for col_idx, col in enumerate(self.columns[machine]):
                    add_pair_cut_coeffs(expr, col, data, cut_def, q_vars[machine][col_idx])
                if expr.size() == 0:
                    continue
                constr = model.addConstr(expr <= 1.0, name=cut_def["name"])
                cut_def = dict(cut_def)
                cut_def["constr"] = constr
                self.setup_pair_cuts.append(cut_def)

        for m_idx, machine in enumerate(data.machines):
            for cidx, col in enumerate(self.columns[machine]):
                qv = q_vars[machine][cidx]
                for cut_info in self.same_mt_cuts:
                    if cut_info["m_idx"] != m_idx:
                        continue
                    t_idx = cut_info["t_idx"]
                    for u_idx in range(num_u):
                        coeff = column_setup_indicator(col, data, u_idx, t_idx)
                        if coeff > 0:
                            model.chgCoeff(cut_info["constr"], qv, float(coeff))
                for cut_info in self.setup_pair_cuts:
                    if cut_info["m_idx"] != m_idx:
                        continue
                    u1, t1 = cut_info["a"]
                    u2, t2 = cut_info["b"]
                    pair_coeff = column_setup_indicator(col, data, u1, t1) + column_setup_indicator(
                        col, data, u2, t2
                    )
                    if pair_coeff > 0:
                        model.chgCoeff(cut_info["constr"], qv, float(pair_coeff))

    def _separate_setup_pair_cuts(self) -> int:
        # Setup Pair 割已停用；仅保留 Deadline Cover
        if not self.lrmp_cut_opts.get("setup_pair_separate", False):
            return 0
        if self._model is None or self._artifacts is None:
            return 0
        existing = {c["name"] for c in self.setup_pair_cuts}
        max_cuts = int(self.lrmp_cut_opts.get("setup_pair_sep_max", 40))
        viol_tol = float(self.lrmp_cut_opts.get("setup_pair_viol_tol", 1e-4))
        new_defs = find_violated_setup_pairs(
            self.data,
            self._artifacts["Q"],
            self.columns,
            existing,
            viol_tol=viol_tol,
            max_cuts=max_cuts,
        )
        if not new_defs:
            return 0
        model = self._model
        for cut_def in new_defs:
            m_idx = cut_def["m_idx"]
            machine = self.data.machines[m_idx]
            expr = gp.LinExpr()
            for col_idx, col in enumerate(self.columns[machine]):
                add_pair_cut_coeffs(expr, col, self.data, cut_def, self._artifacts["Q"][machine][col_idx])
            constr = model.addConstr(expr <= 1.0, name=cut_def["name"])
            cut_def = dict(cut_def)
            cut_def["constr"] = constr
            self.setup_pair_cuts.append(cut_def)
            self._artifacts["setup_pair_cuts"] = self.setup_pair_cuts
        model.update()
        return len(new_defs)

    def _extract_duals_to_shm(self) -> None:
        """从 Gurobi 约束 Pi 直接写入共享内存，避免构造嵌套 alpha 列表。"""
        if self._alpha_flat is None or self._rho_arr is None or self._beta_arr is None:
            return
        assert self._artifacts is not None
        alpha_flat = self._alpha_flat
        for flat_idx, constr in self._flow_pi_slots:
            alpha_flat[flat_idx] = constr.Pi
        for i_idx, constr in self._artifacts["wip_balance"].items():
            self._rho_arr[i_idx] = constr.Pi
        for m_idx, machine in enumerate(self.data.machines):
            self._beta_arr[m_idx] = self._artifacts["choose_one"][machine].Pi

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

        col_expr = gp.Column()
        num_t = len(data.periods)
        for i_idx, arr_t, qty in self._column_prod_coeffs(col):
            if 0 <= arr_t < num_t:
                col_expr.addTerms(qty, flow_items[(i_idx, arr_t)])

        col_expr.addTerms(1.0, choose_one[machine])

        for i_idx in range(len(data.cured_items)):
            coeff = self._get_column_wip_input(col, i_idx)
            if coeff > 0:
                col_expr.addTerms(float(coeff), wip_balance[i_idx])

        for tau, cut_info in self.deadline_cuts.items():
            contrib = self._get_column_contrib_to_tau(col, tau, cut_info.get("I", []))
            if contrib > 0:
                col_expr.addTerms(float(contrib), cut_info["constr"])

        for cut in self.sri_cuts:
            if cut.constr is None:
                continue
            coeff = sri_coefficient(data, col, cut)
            if coeff:
                col_expr.addTerms(float(coeff), cut.constr)

        m_idx = data.machines.index(machine)
        num_u = len(data.configs)
        for cut_info in self._artifacts.get("same_mt_cuts", []):
            if cut_info["m_idx"] != m_idx:
                continue
            t_idx = cut_info["t_idx"]
            for u_idx in range(num_u):
                coeff = column_setup_indicator(col, data, u_idx, t_idx)
                if coeff > 0:
                    col_expr.addTerms(float(coeff), cut_info["constr"])
        for cut_info in self._artifacts.get("setup_pair_cuts", []):
            if cut_info["m_idx"] != m_idx:
                continue
            u1, t1 = cut_info["a"]
            u2, t2 = cut_info["b"]
            pair_coeff = column_setup_indicator(col, data, u1, t1) + column_setup_indicator(
                col, data, u2, t2
            )
            if pair_coeff > 0:
                col_expr.addTerms(float(pair_coeff), cut_info["constr"])

        u_name_lookup = data.configs
        for constr, u_idx, m_idx, t_idx, _val in self._artifacts.get("branch_rows", []):
            if data.machines[m_idx] != machine:
                continue
            u_name = u_name_lookup[u_idx]
            period = data.periods[t_idx]
            if col.y.get((u_name, period), 0) > 0.5:
                col_expr.addTerms(1.0, constr)

        for constr, i_idx, _sense, _bound in self._artifacts.get("wip_branch_rows", []):
            coeff = self._get_column_wip_input(col, i_idx)
            if coeff > 0:
                col_expr.addTerms(float(coeff), constr)

        q_var = model.addVar(
            vtype=GRB.CONTINUOUS,
            lb=0,
            obj=col.production_cost,
            name=f"Q_{machine}_{len(q_vars[machine])}",
            column=col_expr,
        )
        # 同 _init_master：连续松弛不设 ub=1（Σq≤1 已足够）
        q_vars[machine].append(q_var)
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
        self.stop_reason = None
        self._feasibility_seed_retried = False
        self.iteration_times = []
        self.iteration_time_detail = []

        pool_size = 0 if self.pricing_inprocess else self._pricing_pool_size()
        pool_workers = 1 if (self.pricing_batch and not self.pricing_inprocess) else pool_size
        run_start = time.perf_counter()
        num_i = len(self.data.cured_items)
        num_t = len(self.data.periods)
        num_m = len(self.data.machines)
        self._alpha_flat = mp.Array("d", num_i * num_t)
        self._rho_arr = mp.Array("d", num_i)
        self._beta_arr = mp.Array("d", num_m)
        self._rc_arr = mp.Array("d", num_m)
        use_shared_duals = True

        init_pricing = self._pricing_label(self.use_gurobi_init and not self.fixed_y)
        loop_pricing = self._pricing_label(False)
        self._log(
            f"列生成启动: {self.data.raw.name}  |I|={num_i} |M|={num_m} |T|={num_t} "
            f"init={init_pricing}  iter={loop_pricing}"
        )

        if self.pricing_inprocess:
            self._bind_pricing_worker_state()
            self.initialize_columns(None)
            self._init_master(relaxed=True)
            iteration_rcs, last_valid_lb = self._cg_main_loop(None, pool_size, run_start, use_shared_duals)
        else:
            with ProcessPoolExecutor(
                max_workers=pool_workers,
                initializer=_init_pricing_worker,
                initargs=(
                    self.data,
                    self.fixed_y,
                    self.pricing_time_limit,
                    self._alpha_flat,
                    self._rho_arr,
                    self._beta_arr,
                    self._rc_arr,
                    len(self.data.cured_items),
                    len(self.data.periods),
                    self.pricing_mode,
                    self.ebrp_force,
                    self.ebrp_time_limit,
                    self.im_pricing_time_limit,
                    self._ebrp_contexts,
                    self.skip_cross_check,
                ),
            ) as pricing_pool:
                self.initialize_columns(pricing_pool)
                self._init_master(relaxed=True)
                iteration_rcs, last_valid_lb = self._cg_main_loop(
                    pricing_pool, pool_size, run_start, use_shared_duals
                )

        self.last_valid_lb = last_valid_lb
        if iteration_rcs and self.converged:
            return self.lrmp_obj if self.lrmp_obj is not None else last_valid_lb, iteration_rcs, self.columns
        self.converged = False
        return self.lrmp_obj if self.lrmp_obj is not None else last_valid_lb, iteration_rcs, self.columns

    def ensure_master_lp(self) -> bool:
        """Ensure RMP is a solved continuous LP with readable primal/duals.

        Returns True if ObjVal/Pi are safe to read.
        """
        if self._model is None:
            return False
        model = self._model
        try:
            # RMP must stay continuous; MIP leftovers break .Pi access.
            model.Params.Method = -1
            model.optimize()
            if model.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL) and model.SolCount > 0:
                try:
                    self.lrmp_obj = float(model.ObjVal)
                except Exception:
                    pass
                return True
        except Exception:
            return False
        return False

    def reinforce_with_cuts(self, *, max_repricing_iters: int = 10) -> float:
        """Gomory R / gap-cuts 已停用；仅保留 Deadline Cover。"""
        return self.lrmp_obj or 0.0

    def _reinforce_with_cuts_legacy(self, *, max_repricing_iters: int = 10) -> float:
        """BPC 切平面强化：对 fractional R_i 加 Gomory 取整切后重新定价（已停用，保留备查）。

        返回加切并重新定价后的 LRMP 目标值。
        若加切后 RMP 不可行/无 dual，回滚切平面并恢复 LP，避免后续分支读 .Pi 崩溃。
        """
        if not self.converged or self._model is None or self._artifacts is None:
            return self.lrmp_obj or 0.0

        model = self._model
        _artifacts = self._artifacts
        r_vars = _artifacts["R"]
        num_i = len(self.data.cured_items)
        num_t = len(self.data.periods)
        num_m = len(self.data.machines)
        prev_lrmp = self.lrmp_obj or 0.0

        if not self.ensure_master_lp():
            return prev_lrmp

        # 1) 添加 Gomory R-cuts（记下约束以便失败回滚）
        added_constrs: list[gp.Constr] = []
        model.update()
        for i_idx in range(num_i):
            try:
                r_val = float(r_vars[i_idx].X)
            except Exception:
                continue
            frac = r_val - int(r_val)
            if frac < 1e-4:
                continue
            ceil_val = int(r_val) + 1
            c = model.addConstr(r_vars[i_idx] >= ceil_val, name=f"gomory_r_{i_idx}")
            added_constrs.append(c)

        if not added_constrs:
            return prev_lrmp

        print(
            f"  [BPC] added {len(added_constrs)} Gomory R-cuts, re-optimizing + repricing...",
            flush=True,
        )

        def _rollback() -> float:
            for c in added_constrs:
                try:
                    model.remove(c)
                except Exception:
                    pass
            model.update()
            self.ensure_master_lp()
            return self.lrmp_obj if self.lrmp_obj is not None else prev_lrmp

        # 2) 重新求解 RMP
        model.optimize()
        if model.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL) or model.SolCount <= 0:
            self.stop_reason = "cut_made_infeasible"
            print("    [BPC] cuts made RMP infeasible/unstable — rollback", flush=True)
            return _rollback()

        try:
            self.lrmp_obj = float(model.ObjVal)
        except Exception:
            print("    [BPC] cuts left RMP without ObjVal — rollback", flush=True)
            return _rollback()

        # 3) 提取新 duals 并重新定价（让列池适配增强后的 RMP）
        use_shared_duals = self._alpha_flat is not None
        try:
            for _ in range(max_repricing_iters):
                if not self.ensure_master_lp():
                    raise RuntimeError("RMP lost LP duals during cut reinforce")
                if use_shared_duals:
                    self._extract_duals_to_shm()
                    alpha = [
                        [self._alpha_flat[i * num_t + t] for t in range(num_t)]
                        for i in range(num_i)
                    ]
                    rhos = [self._rho_arr[i] for i in range(num_i)]
                    betas = [self._beta_arr[m] for m in range(num_m)]
                else:
                    flow_items = _artifacts.get("flow_items") or {}
                    alpha = [[0.0] * num_t for _ in range(num_i)]
                    for (i_idx, t_idx), constr in flow_items.items():
                        try:
                            alpha[i_idx][t_idx] = float(constr.Pi)
                        except Exception:
                            alpha[i_idx][t_idx] = 0.0
                    rhos = []
                    for idx in range(num_i):
                        try:
                            rhos.append(float(_artifacts["wip_balance"][idx].Pi))
                        except Exception:
                            rhos.append(0.0)
                    betas = []
                    for m in self.data.machines:
                        try:
                            betas.append(float(_artifacts["choose_one"][m].Pi))
                        except Exception:
                            betas.append(0.0)

                new_cols = self._price_machines(
                    alpha, rhos, betas,
                    use_gurobi_when_no_fixes=False, executor=None,
                    use_shared_duals=use_shared_duals, duals_preloaded=use_shared_duals,
                )
                if not new_cols:
                    break
                best_rc = self.last_min_rc
                for m, col in new_cols:
                    self._add_column_to_master(m, col)
                    self.iteration_times.append(time.perf_counter())
                model.optimize()
                if model.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL) or model.SolCount <= 0:
                    raise RuntimeError("RMP unstable after cut reprice")
                self.lrmp_obj = float(model.ObjVal)
                if best_rc is not None and best_rc >= -1e-6:
                    break
        except Exception as exc:
            print(f"    [BPC] cut reinforce failed ({exc}) — rollback", flush=True)
            return _rollback()

        if not self.ensure_master_lp():
            print("    [BPC] no duals after cuts — rollback", flush=True)
            return _rollback()
        return self.lrmp_obj if self.lrmp_obj is not None else prev_lrmp

    def _q_for_column(self, machine: str, col: ScheduleColumn) -> gp.Var | None:
        """列池中与 col 同指纹的 q 变量。"""
        if self._artifacts is None:
            return None
        q_vars = self._artifacts.get("Q", {}).get(machine)
        if not q_vars:
            return None
        fp = column_fingerprint(col)
        for qi, pool_col in enumerate(self.columns.get(machine, [])):
            if column_fingerprint(pool_col) == fp:
                return q_vars[qi]
        return None

    def _min_pool_column_rc(self) -> float:
        """列池约简成本（原始）：所有 q 的 Gurobi RC 最小值。"""
        if self._model is None or self._artifacts is None:
            return 0.0
        q_vars = self._artifacts.get("Q")
        if not q_vars:
            return 0.0
        min_rc = float("inf")
        for machine in self.data.machines:
            for qv in q_vars.get(machine, []):
                try:
                    min_rc = min(min_rc, float(qv.RC))
                except (AttributeError, gp.GurobiError):
                    continue
        return min_rc if math.isfinite(min_rc) else 0.0

    def _min_improvable_pool_rc(self) -> float:
        """可改进列池约简成本：仅 q≈0 且 RC<0（尚可增大 q）；q=1 贴界负 RC 视为饱和。"""
        if self._model is None or self._artifacts is None:
            return 0.0
        q_vars = self._artifacts.get("Q")
        if not q_vars:
            return 0.0
        eps = max(self.rc_tolerance, CHECKER_TOL["cg_column_select"])
        min_rc = float("inf")
        for machine in self.data.machines:
            for qv in q_vars.get(machine, []):
                try:
                    qx = float(qv.X)
                    rc = float(qv.RC)
                except (AttributeError, gp.GurobiError):
                    continue
                if qx <= eps and rc < -self.rc_tolerance:
                    min_rc = min(min_rc, rc)
        return min_rc if math.isfinite(min_rc) else 0.0

    def _effective_pricing_min_rc(
        self,
        priced: list[tuple[int, SubproblemResult | None]],
    ) -> float:
        """可改进定价约简成本：忽略与 fixed_y 不兼容的定价列（不应计入收敛判据）。"""
        min_rc = float("inf")
        for m_idx, sp in priced:
            rc = sp.reduced_cost if sp is not None else self._rc_arr[m_idx]
            if rc >= -self.rc_tolerance:
                continue
            if sp is None:
                min_rc = min(min_rc, rc)
                continue
            col = subproblem_to_column(self.data, m_idx, -1, sp)
            if not column_compatible(col, m_idx, self.data, self.fixed_y):
                continue
            min_rc = min(min_rc, rc)
        return min_rc if math.isfinite(min_rc) else 0.0

    def _cg_main_loop(
        self,
        pricing_pool: ProcessPoolExecutor | None,
        pool_size: int,
        run_start: float,
        use_shared_duals: bool,
    ) -> tuple[list[float], float]:
        iteration_rcs: list[float] = []
        last_valid_lb = float("-inf")
        model = self._model
        assert model is not None and self._artifacts is not None

        for _ in range(self.max_iterations):
            if self.time_limit is not None and time.perf_counter() - run_start >= self.time_limit:
                self.stop_reason = "time_limit"
                break

            self._pricing_iter = len(iteration_rcs) + 1
            self._progress.report(
                f"iter={self._pricing_iter}  phase=start  "
                f"cols={self._total_columns()}  LRMP={self.lrmp_obj if self.lrmp_obj is not None else '—'}"
            )

            iter_start = time.perf_counter()

            t_master = time.perf_counter()
            with self._progress.phase("master", iter=self._pricing_iter):
                model.optimize()
            master_sec = time.perf_counter() - t_master
            if model.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
                if (
                    not iteration_rcs
                    and self._needs_feasibility_seed()
                    and not getattr(self, "_feasibility_seed_retried", False)
                ):
                    self._feasibility_seed_retried = True
                    self._reset_master()
                    self._seed_forced_setup_columns()
                    self._seed_covering_columns(pricing_pool)
                    self._seed_branch_columns(pricing_pool)
                    self._init_master(relaxed=True)
                    model = self._model
                    assert model is not None
                    continue
                self.stop_reason = "infeasible_master"
                break

            lrmp_obj = model.ObjVal
            if self._separate_sri_cuts() > 0:
                model.optimize()
                if model.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
                    self.stop_reason = "infeasible_master"
                    break
                lrmp_obj = model.ObjVal
            if self._separate_setup_pair_cuts() > 0:
                model.optimize()
                if model.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
                    self.stop_reason = "infeasible_master"
                    break
                lrmp_obj = model.ObjVal
            if use_shared_duals and self._alpha_flat is not None:
                self._extract_duals_to_shm()
                num_i = len(self.data.cured_items)
                num_t = len(self.data.periods)
                alpha = [
                    [self._alpha_flat[i_idx * num_t + t_idx] for t_idx in range(num_t)]
                    for i_idx in range(num_i)
                ]
                rhos = [self._rho_arr[i_idx] for i_idx in range(num_i)]
                betas = [self._beta_arr[m_idx] for m_idx in range(len(self.data.machines))]
            else:
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

                rhos = [self._artifacts["wip_balance"][i_idx].Pi for i_idx in range(len(self.data.cured_items))]
                betas = [self._artifacts["choose_one"][machine].Pi for machine in self.data.machines]

            # LRMP 割 / 分支对偶必须进入定价，否则会把池中已有列误判为负 RC
            alpha = self._fold_deadline_duals_into_alpha(alpha)
            rhos = self._fold_wip_branch_duals_into_rho(rhos)
            self._refresh_setup_cut_cost_deltas()
            duals_ready = False
            if use_shared_duals and self._alpha_flat is not None:
                self._write_pricing_duals(alpha, rhos, betas)
                duals_ready = True

            sum_rc = 0.0
            lb_correction = 0.0
            min_rc = float("inf")
            new_columns: list[tuple[str, ScheduleColumn]] = []
            t_pricing = time.perf_counter()
            # A CG time limit is global.  Divide the remaining budget among
            # machines for this pricing round so one MIP cannot overrun the
            # whole experiment by several multiples of its stated limit.
            saved_im_pricing_limit = self.im_pricing_time_limit
            if self.pricing_mode.lower() == "im_gurobi" and self.time_limit is not None:
                remaining = max(0.0, self.time_limit - (time.perf_counter() - run_start))
                per_machine = max(1.0, 0.9 * remaining / max(1, len(self.data.machines)))
                self.im_pricing_time_limit = min(saved_im_pricing_limit, per_machine)
            try:
                priced = self._price_machines(
                    alpha,
                    rhos,
                    betas,
                    use_gurobi_when_no_fixes=False,
                    executor=pricing_pool,
                    use_shared_duals=use_shared_duals,
                    duals_preloaded=duals_ready,
                )
            finally:
                self.im_pricing_time_limit = saved_im_pricing_limit

            # An interrupted pricing MIP cannot prove that no negative column
            # exists.  Stop safely; do not report the RMP value as a converged
            # DW bound.
            incomplete = [m_idx for m_idx, sp in priced if sp is not None and not sp.is_optimal]
            if incomplete:
                self.stop_reason = "pricing_time_limit"
                self._log(
                    f"CG stops without certification: incomplete pricing on machines {incomplete}"
                )
                return iteration_rcs, last_valid_lb

            for priced_idx, (m_idx, sp) in enumerate(priced):
                machine = self.data.machines[m_idx]
                rc = sp.reduced_cost if sp is not None else self._rc_arr[m_idx]
                # A column already selected at q=1 cannot be increased because
                # of Σq≤1.  Its raw negative RC is a bound effect, not a new
                # improving direction; do not mistake it for a pricing error.
                if rc < -self.rc_tolerance and sp is not None:
                    candidate = subproblem_to_column(self.data, m_idx, -1, sp)
                    if self._column_seen(machine, candidate):
                        qv = self._q_for_column(machine, candidate)
                        try:
                            saturated = qv is not None and float(qv.X) >= 1.0 - self.rc_tolerance
                        except (AttributeError, gp.GurobiError):
                            saturated = False
                        if saturated:
                            rc = 0.0
                            # Keep downstream convergence tests consistent
                            # with the effective (non-improvable) direction.
                            sp = SubproblemResult(0.0, sp.x, sp.y, sp.production_cost)
                            priced[priced_idx] = (m_idx, sp)
                sum_rc += rc
                min_rc = min(min_rc, rc)
                lb_correction += min(rc, 0.0)
                if rc < -self.rc_tolerance:
                    if sp is None:
                        raise RuntimeError(f"missing pricing column for machine {machine} with rc={rc}")
                    col = subproblem_to_column(self.data, m_idx, self._next_col_id, sp)
                    if column_compatible(col, m_idx, self.data, self.fixed_y):
                        if self._column_seen(machine, col):
                            # The pricing oracle has returned a structural
                            # duplicate.  Do not blindly ignore it: a truly
                            # negative duplicate would mean that the oracle
                            # and the master use inconsistent coefficients.
                            # Instead, ask the existing master variable for
                            # its reduced cost.  This distinguishes harmless
                            # numerical rediscovery from a modelling defect.
                            qv = self._q_for_column(machine, col)
                            try:
                                master_rc = float(qv.RC) if qv is not None else float("-inf")
                            except (AttributeError, gp.GurobiError):
                                master_rc = float("-inf")
                            if (
                                master_rc >= -self.rc_tolerance
                                and abs(master_rc - rc) <= 10.0 * self.rc_tolerance
                            ):
                                # Same column is non-improving in the actual
                                # RMP and differs only at numerical tolerance.
                                sp = SubproblemResult(
                                    0.0, sp.x, sp.y, sp.production_cost,
                                    is_optimal=sp.is_optimal,
                                )
                                priced[priced_idx] = (m_idx, sp)
                            else:
                                # Never claim convergence when master and
                                # pricing disagree materially.  In particular,
                                # this catches a missing cut-dual term.
                                self.converged = False
                                self.stop_reason = "duplicate_column_inconsistency"
                                self._log(
                                    f"CG停止：重复列与主问题约简成本不一致 "
                                    f"machine={machine} pricing_rc={rc:.6f} "
                                    f"master_rc={master_rc:.6f} "
                                    f"iter={len(iteration_rcs) + 1}"
                                )
                                return iteration_rcs, last_valid_lb
                            continue
                        self._next_col_id += 1
                        new_columns.append((machine, col))
            pricing_sec = time.perf_counter() - t_pricing

            if self.max_columns_per_iter == 1 and new_columns:
                machine, col = min(new_columns, key=lambda mc: mc[1].reduced_cost)
                new_columns = [(machine, col)]

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

            iter_no = len(iteration_rcs) + 1
            self._log(
                f"[CG {iter_no:4d}] LRMP={lrmp_obj:12.2f}  min_RC={min_rc:11.4f}  "
                f"new_cols={len(new_columns):3d}  total_cols={self._total_columns():4d}  "
                f"master={master_sec:6.2f}s  pricing={pricing_sec:6.2f}s  "
                f"({self._pricing_label(False)})"
            )

            iteration_rcs.append(sum_rc)
            self.last_min_rc = min_rc if min_rc != float("inf") else None
            last_valid_lb = max(last_valid_lb, lrmp_obj + lb_correction)
            self.lrmp_obj = lrmp_obj

            if (
                self.sum_rc_stop_threshold is not None
                and sum_rc > self.sum_rc_stop_threshold
            ):
                self.converged = False
                self.stop_reason = "sum_reduced_cost_threshold"
                self._log(
                    f"CG近似停止：sum_RC={sum_rc:.4f} > "
                    f"{self.sum_rc_stop_threshold:.4f}; iter={iter_no}"
                )
                return iteration_rcs, last_valid_lb

            # Lagrangian lower bound pruning: lrmp_obj + Σ min(RC_i, 0) ≤ MP_obj ≤ IP_obj
            # The per-machine Lagrangian correction is safe for the base
            # master, but the first SRI implementation changes reduced costs
            # through a path-dependent rank-1 term.  Do not use this partial
            # correction to prune an SRI node before full CG convergence.
            # A diagnostic on seed 10102 found a reference-compatible node
            # incorrectly eliminated by this shortcut.  Once CG converges,
            # the full master objective remains the bound used by B&P.
            if (
                not self.sri_cuts
                and self.incumbent is not None
                and last_valid_lb >= self.incumbent - 1e-5
            ):
                self.stop_reason = "pruned_by_incumbent"
                self.converged = True
                self._log(
                    f"CG 提前终止(Lagrangian下界): LB={last_valid_lb:.2f} >= incumbent={self.incumbent:.2f}  "
                    f"iter={iter_no}  cols={self._total_columns()}"
                )
                return iteration_rcs, last_valid_lb

            if not new_columns:
                pool_min_rc = self._min_pool_column_rc()
                improvable_pool_rc = self._min_improvable_pool_rc()
                effective_pricing_rc = self._effective_pricing_min_rc(priced)
                self.last_pool_min_rc = pool_min_rc
                self.last_improvable_pool_rc = improvable_pool_rc
                self.last_effective_pricing_rc = effective_pricing_rc
                pricing_ok = effective_pricing_rc >= -self.rc_tolerance
                pool_ok = improvable_pool_rc >= -self.rc_tolerance
                if pricing_ok and pool_ok:
                    # After a fully priced master has converged, its current
                    # objective is itself an exact DW lower bound.  This is
                    # especially important after root SRI separation: the
                    # Lagrangian correction computed earlier in the iteration
                    # can still refer to pre-cut reduced costs and is then
                    # unnecessarily weaker than ``lrmp_obj``.
                    last_valid_lb = max(last_valid_lb, lrmp_obj)
                    self.converged = True
                    self.last_min_rc = effective_pricing_rc
                    self._log(
                        f"列生成收敛: LRMP={lrmp_obj:.2f}  迭代={iter_no}  "
                        f"列数={self._total_columns()}  耗时={time.perf_counter() - run_start:.1f}s"
                    )
                    return iteration_rcs, last_valid_lb
                if pool_ok:
                    self.converged = True
                    self.last_min_rc = effective_pricing_rc
                    self.stop_reason = "pool_optimal_saturated"
                    self._log(
                        f"列生成收敛(列饱和): LRMP={lrmp_obj:.2f}  迭代={iter_no}  "
                        f"定价RC={effective_pricing_rc:.4f}  列池RC={improvable_pool_rc:.4f}  "
                        f"列数={self._total_columns()}"
                    )
                    return iteration_rcs, last_valid_lb
                self.converged = False
                self.last_min_rc = min(effective_pricing_rc, improvable_pool_rc)
                self.stop_reason = "pricing_stall"
                self._log(
                    f"列生成停滞: LRMP={lrmp_obj:.2f}  定价RC={effective_pricing_rc:.4f}  "
                    f"列池RC={improvable_pool_rc:.4f}  (原始池RC={pool_min_rc:.4f})  无新列可加入"
                )
                return iteration_rcs, last_valid_lb

        if self.stop_reason == "time_limit":
            lrmp_show = f"{self.lrmp_obj:.2f}" if self.lrmp_obj is not None else "—"
            self._log(
                f"列生成时限停止: LRMP={lrmp_show}  迭代={len(iteration_rcs)}  min_RC={self.last_min_rc}"
            )
        elif self.stop_reason:
            self._log(f"列生成停止: {self.stop_reason}")

        return iteration_rcs, last_valid_lb

    def solve_restricted_master(self) -> tuple[gp.Model, dict]:
        if self._model is None or self._artifacts is None:
            self._init_master(relaxed=True)
        assert self._model is not None and self._artifacts is not None
        self._model.optimize()
        return self._model, self._artifacts

    def solve_rmp_ip(
        self,
        time_limit: float = 30.0,
        mip_gap: float = 0.01,
    ) -> float | None:
        """Solve current restricted master with binary Q → valid UB from column pool.

        Temporarily flips Q to binary, solves the MIP, then restores continuous Q
        and re-solves the LP so subsequent dual/implied queries stay valid.
        """
        if self._model is None or self._artifacts is None:
            return None
        model = self._model
        q_vars = self._artifacts["Q"]
        old_tl = model.Params.TimeLimit
        old_gap = model.Params.MIPGap
        old_method = model.Params.Method
        changed: list[tuple[gp.Var, str]] = []
        for qlist in q_vars.values():
            for q in qlist:
                changed.append((q, q.VType))
                q.VType = GRB.BINARY
        model.Params.TimeLimit = max(1.0, float(time_limit))
        model.Params.MIPGap = max(0.0, float(mip_gap))
        model.Params.Method = -1  # automatic for MIP
        model.Params.OutputFlag = 0
        obj: float | None = None
        try:
            model.optimize()
            if model.SolCount > 0 and model.ObjVal is not None:
                obj = float(model.ObjVal)
        finally:
            for q, vt in changed:
                q.VType = vt
            model.Params.TimeLimit = old_tl
            model.Params.MIPGap = old_gap
            model.Params.Method = old_method
            # Must restore a continuous LP with duals for branching features.
            self.ensure_master_lp()
        return obj

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
        """从当前主问题 LP 解重构 implied Y 值 {(u_idx, m_idx, t_idx): val}。"""
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
                        key = (u_idx, m_idx, t_idx)
                        implied[key] = implied.get(key, 0.0) + qv
        return implied

    def get_implied_r(self) -> dict[int, float]:
        """当前 LRMP 中各物料报废量 R_i。"""
        if self._model is None or self._artifacts is None:
            return {}
        r_vars = self._artifacts["R"]
        result = {}
        for i in range(len(self.data.cured_items)):
            try:
                result[i] = float(r_vars[i].X)
            except Exception:
                result[i] = 0.0
        return result

    def get_implied_wip(self) -> dict[int, float]:
        """当前 LRMP 窗内加工量 W_i（列贡献部分，不含 R）。"""
        if self._model is None or self._artifacts is None:
            return {}
        data = self.data
        wip: dict[int, float] = {}
        for i_idx in range(len(data.cured_items)):
            total = 0.0
            for machine in data.machines:
                qvars = self._artifacts["Q"][machine]
                for col_idx, col in enumerate(self.columns[machine]):
                    try:
                        qv = qvars[col_idx].X
                    except Exception:
                        qv = 0.0
                    if qv < 1e-9:
                        continue
                    total += qv * self._get_column_wip_input(col, i_idx)
            wip[i_idx] = total
        return wip


def solve_column_generation(
    data: ProcessedInstance,
    max_iterations: int = DEFAULT_CG_MAX_ITERATIONS,
    init_time_limit: float = DEFAULT_CG_INIT_TIME_LIMIT,
    fast: bool = False,
    set_c: bool = False,
    verbose: bool | None = None,
    **cg_kwargs: object,
) -> SolutionResult:
    start = time.perf_counter()
    if set_c:
        opts = {**SET_C_CG, **cg_kwargs}
        if verbose is not None:
            opts["verbose"] = verbose
        cg = ColumnGenerationSolver(data, init_time_limit=init_time_limit, **opts)
    elif fast:
        opts = {**SET_B_CG_FAST, **cg_kwargs}
        if verbose is not None:
            opts["verbose"] = verbose
        cg = ColumnGenerationSolver(data, init_time_limit=init_time_limit, **opts)
    else:
        cg = ColumnGenerationSolver(
            data,
            max_iterations=max_iterations,
            init_time_limit=init_time_limit,
            verbose=bool(verbose),
            **cg_kwargs,
        )
    lb, rcs, columns = cg.run()
    runtime = time.perf_counter() - start

    scrap_qty: dict[str, float] = {}
    scrap_cost = None
    if cg._artifacts is not None:
        r_vars = cg._artifacts.get("R")
        if r_vars is not None and cg._model is not None:
            try:
                scrap_qty = {
                    data.cured_items[i]: r_vars[i].X
                    for i in range(len(data.cured_items))
                    if r_vars[i].X > 1e-9
                }
                scrap_cost = sum(data.sc_i[i] * r_vars[i].X for i in range(len(data.cured_items)))
            except (AttributeError, gp.GurobiError):
                pass

    if cg.converged:
        status = "completed"
    elif cg.stop_reason == "time_limit":
        status = "time_limit"
    else:
        status = "max_iterations"

    if cg.verbose:
        print(
            f"CG 结束: status={status}  LRMP={lb:.2f}  iters={len(rcs)}  "
            f"cols={sum(len(c) for c in columns.values())}  time={runtime:.1f}s",
            flush=True,
        )

    return SolutionResult(
        method="column_generation",
        instance_name=data.raw.name,
        status=status,
        objective=None,
        lower_bound=lb,
        mip_gap=None,
        runtime_sec=runtime,
        extra={
            "iteration_rcs": rcs,
            "num_columns": {m: len(cols) for m, cols in columns.items()},
            "num_iterations": len(rcs),
            "converged": cg.converged,
            "stop_reason": cg.stop_reason,
            "lrmp_obj": cg.lrmp_obj,
            "last_valid_lb": getattr(cg, "last_valid_lb", None),
            "last_min_rc": cg.last_min_rc,
            "last_pool_min_rc": getattr(cg, "last_pool_min_rc", None),
            "last_improvable_pool_rc": getattr(cg, "last_improvable_pool_rc", None),
            "last_effective_pricing_rc": getattr(cg, "last_effective_pricing_rc", None),
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
