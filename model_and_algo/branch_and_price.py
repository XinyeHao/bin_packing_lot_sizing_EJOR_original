"""Branch-and-price：三层分支策略 + DFS/lazy 混合搜索。"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field

from common.data_models import ProcessedInstance, ScheduleColumn, SolutionResult
from configuration import (
    BP_STRATEGIES,
    DEFAULT_BP_FINAL_IM,
    DEFAULT_BP_LAZY_Y1_CG_CAP,
    DEFAULT_BP_LB_STALL_EPS,
    DEFAULT_BP_MAX_NODES,
    DEFAULT_BP_NODE_IM_TIME_LIMIT,
    DEFAULT_BP_ROOT_IM,
    DEFAULT_BP_STRATEGY,
    DEFAULT_BP_TIME_LIMIT,
    DEFAULT_BP_USE_NODE_IM,
    DEFAULT_BP_Y0_DEPTH_BUDGET,
    DEFAULT_CG_INIT_TIME_LIMIT,
    DEFAULT_CG_MAX_ITERATIONS,
    DEFAULT_MIP_GAP,
)
from model_and_algo.branch_utils import (
    Y1_DEFER_PENALTY,
    BranchDecision,
    BranchPolicy,
    SetupFix,
    WipFix,
    copy_column_pool,
    fixed_y_feasible,
    scrap_fix_feasible,
    wip_fix_feasible,
)
from model_and_algo.column_generation import ColumnGenerationSolver
from model_and_algo.im_model import solve_im

INT_TOL = 1e-5
FRAC_TOL = 1e-4


@dataclass
class BranchNode:
    node_id: int
    fixed_y: dict[SetupFix, int] = field(default_factory=dict)
    fixed_scrap: dict[int, int] = field(default_factory=dict)
    fixed_wip: dict[int, WipFix] = field(default_factory=dict)
    inherited_columns: dict[str, list[ScheduleColumn]] | None = None
    refresh_columns: bool = False
    is_lazy: bool = False
    lower_bound: float = float("-inf")
    lb_estimate: float = float("-inf")
    depth: int = 0
    branch_kind: str | None = None
    branch_on: SetupFix | int | None = None
    branch_value: int | None = None


def _is_int(v: float, tol: float = INT_TOL) -> bool:
    return abs(v - round(v)) <= tol


def _implied_y_integer(implied_y: dict[SetupFix, float]) -> bool:
    for val in implied_y.values():
        if val > FRAC_TOL and not _is_int(val):
            return False
    return True


def _q_integer(cg: ColumnGenerationSolver) -> bool:
    if cg._artifacts is None:
        return False
    for machine in cg.data.machines:
        for qv in cg._artifacts["Q"][machine]:
            if qv.X > FRAC_TOL and not _is_int(qv.X):
                return False
    return True


def _r_integer(cg: ColumnGenerationSolver) -> bool:
    if cg._artifacts is None:
        return False
    for rvar in cg._artifacts["R"]:
        if rvar.X > FRAC_TOL and not _is_int(rvar.X):
            return False
    return True


def _node_feasible(data: ProcessedInstance, node: BranchNode) -> bool:
    return (
        fixed_y_feasible(data, node.fixed_y)
        and scrap_fix_feasible(data, node.fixed_scrap)
        and wip_fix_feasible(data, node.fixed_wip)
    )


@dataclass
class BranchAndPriceResult:
    status: str
    objective: float | None
    lower_bound: float | None
    best_node_lb: float | None
    incumbent: float | None
    nodes_explored: int
    nodes_pruned: int
    nodes_infeasible: int
    nodes_cg_failed: int
    max_depth: int
    runtime_sec: float
    best_fixed_y: dict[SetupFix, int]
    root_lrmp: float | None
    cg_solver: ColumnGenerationSolver | None = None
    nodes_created: int = 0
    y1_lazy_created: int = 0
    lazy_popped: int = 0
    dfs_popped: int = 0


class BranchAndPriceSolver:
    """B&P：baseline Y=0 DFS / old_hybrid / new_policy 三层分支。"""

    def __init__(
        self,
        data: ProcessedInstance,
        max_nodes: int = DEFAULT_BP_MAX_NODES,
        cg_max_iterations: int = DEFAULT_CG_MAX_ITERATIONS,
        cg_init_time_limit: float = DEFAULT_CG_INIT_TIME_LIMIT,
        im_time_limit: float = DEFAULT_BP_NODE_IM_TIME_LIMIT,
        mip_gap: float = DEFAULT_MIP_GAP,
        time_limit: float | None = DEFAULT_BP_TIME_LIMIT,
        cg_kwargs: dict | None = None,
        use_node_im: bool = DEFAULT_BP_USE_NODE_IM,
        root_im: bool = DEFAULT_BP_ROOT_IM,
        final_im: bool = DEFAULT_BP_FINAL_IM,
        branch_strategy: str = DEFAULT_BP_STRATEGY,
        y0_depth_budget: int = DEFAULT_BP_Y0_DEPTH_BUDGET,
        lb_stall_eps: float = DEFAULT_BP_LB_STALL_EPS,
        lazy_y1_cg_cap: float = DEFAULT_BP_LAZY_Y1_CG_CAP,
    ) -> None:
        if branch_strategy not in BP_STRATEGIES:
            raise ValueError(f"branch_strategy must be one of {BP_STRATEGIES}")
        self.data = data
        self.max_nodes = max_nodes
        self.cg_max_iterations = cg_max_iterations
        self.cg_init_time_limit = cg_init_time_limit
        self.im_time_limit = im_time_limit
        self.mip_gap = mip_gap
        self.time_limit = time_limit
        self.cg_kwargs = dict(cg_kwargs or {})
        self.use_node_im = use_node_im
        self.root_im = root_im
        self.final_im = final_im
        self.branch_strategy = branch_strategy
        self.lazy_y1_cg_cap = lazy_y1_cg_cap
        self.policy = BranchPolicy(
            data,
            strategy=branch_strategy,
            y0_depth_budget=y0_depth_budget,
            lb_stall_eps=lb_stall_eps,
        )

    def _run_node_cg(
        self,
        node: BranchNode,
        time_limit: float | None = None,
    ) -> ColumnGenerationSolver:
        inherited = None if node.refresh_columns else node.inherited_columns
        use_gurobi_init = node.refresh_columns or not inherited
        if "use_gurobi_init" in self.cg_kwargs and not node.refresh_columns:
            use_gurobi_init = bool(self.cg_kwargs["use_gurobi_init"]) and not inherited
        kwargs = {k: v for k, v in self.cg_kwargs.items() if k != "use_gurobi_init"}
        if time_limit is not None:
            if node.is_lazy and any(v == 1 for v in node.fixed_y.values()):
                kwargs["time_limit"] = min(time_limit, self.lazy_y1_cg_cap)
            else:
                kwargs["time_limit"] = time_limit
        max_iters = int(kwargs.pop("max_iterations", self.cg_max_iterations))
        max_iters = max(max_iters, self.cg_max_iterations)
        if node.is_lazy and any(v == 1 for v in node.fixed_y.values()):
            max_iters = max(max_iters, 2500)
        elif any(v == 1 for v in node.fixed_y.values()) or node.fixed_scrap or node.fixed_wip:
            max_iters = max(max_iters, 2500)
        cg = ColumnGenerationSolver(
            self.data,
            max_iterations=max_iters,
            init_time_limit=self.cg_init_time_limit,
            fixed_y=node.fixed_y,
            fixed_scrap=node.fixed_scrap,
            fixed_wip=node.fixed_wip,
            inherited_columns=inherited,
            use_gurobi_init=use_gurobi_init,
            **kwargs,
        )
        cg.run()
        return cg

    def _try_incumbent(
        self,
        fixed_y: dict[SetupFix, int],
        incumbent: float | None,
        node_im_limit: float | None = None,
    ) -> tuple[float | None, dict[SetupFix, int]]:
        im_limit = self.im_time_limit if node_im_limit is None else node_im_limit
        im = solve_im(
            self.data,
            time_limit=im_limit,
            mip_gap=self.mip_gap,
            fixed_y=fixed_y or None,
            method="branch_and_price",
        )
        if im.objective is not None and im.status not in ("infeasible", "infeasible_or_unbounded"):
            if incumbent is None or im.objective < incumbent - INT_TOL:
                return im.objective, fixed_y
        return incumbent, fixed_y

    def _push_lazy(
        self,
        lazy_heap: list[tuple[float, int, BranchNode]],
        child: BranchNode,
        parent_lb: float,
    ) -> None:
        child.lb_estimate = parent_lb + Y1_DEFER_PENALTY
        heapq.heappush(lazy_heap, (child.lb_estimate, child.node_id, child))

    def _push_heap(
        self,
        open_heap: list[tuple[float, int, BranchNode]],
        child: BranchNode,
        parent_lb: float,
        defer: bool,
    ) -> None:
        child.lb_estimate = parent_lb + (Y1_DEFER_PENALTY if defer else 0.0)
        heapq.heappush(open_heap, (child.lb_estimate, child.node_id, child))

    def _decision_label(self, dec: BranchDecision) -> str:
        if dec.kind == "scrap":
            return f"scrap i={dec.key} R={dec.value:.4f}"
        if dec.kind == "wip":
            assert dec.wip_fix is not None
            sense, bound = dec.wip_fix
            return f"wip i={dec.key} W={dec.value:.4f} {sense}{bound}"
        if dec.kind in ("setup_y0", "setup"):
            return f"Y{dec.key}={dec.value:.4f}"
        return str(dec)

    def _create_children(
        self,
        node: BranchNode,
        dec: BranchDecision,
        inherited: dict[str, list[ScheduleColumn]],
        next_id: int,
        dfs_stack: list[BranchNode],
        lazy_heap: list[tuple[float, int, BranchNode]],
        open_heap: list[tuple[float, int, BranchNode]] | None,
        stats: dict[str, int],
    ) -> int:
        parent_lb = node.lower_bound

        if dec.kind in ("setup_y0", "setup"):
            branch_key = dec.key
            assert isinstance(branch_key, tuple)

            if self.branch_strategy == "old_hybrid":
                assert open_heap is not None
                for val in (1, 0):
                    child_fixed = dict(node.fixed_y)
                    child_fixed[branch_key] = val
                    if not fixed_y_feasible(self.data, child_fixed):
                        continue
                    child = BranchNode(
                        node_id=next_id,
                        fixed_y=child_fixed,
                        fixed_scrap=dict(node.fixed_scrap),
                        fixed_wip=dict(node.fixed_wip),
                        inherited_columns=inherited,
                        depth=node.depth + 1,
                        branch_kind="setup",
                        branch_on=branch_key,
                        branch_value=val,
                        is_lazy=(val == 1),
                    )
                    next_id += 1
                    self._push_heap(open_heap, child, parent_lb, defer=(val == 1))
                    if val == 1:
                        stats["y1_lazy_created"] += 1
                return next_id

            child_fixed = dict(node.fixed_y)
            child_fixed[branch_key] = 0
            if fixed_y_feasible(self.data, child_fixed):
                dfs_stack.append(
                    BranchNode(
                        node_id=next_id,
                        fixed_y=child_fixed,
                        fixed_scrap=dict(node.fixed_scrap),
                        fixed_wip=dict(node.fixed_wip),
                        inherited_columns=inherited,
                        depth=node.depth + 1,
                        branch_kind="setup_y0",
                        branch_on=branch_key,
                        branch_value=0,
                    )
                )
                next_id += 1

            child_fixed_y1 = dict(node.fixed_y)
            child_fixed_y1[branch_key] = 1
            if fixed_y_feasible(self.data, child_fixed_y1):
                child1 = BranchNode(
                    node_id=next_id,
                    fixed_y=child_fixed_y1,
                    fixed_scrap=dict(node.fixed_scrap),
                    fixed_wip=dict(node.fixed_wip),
                    inherited_columns=inherited,
                    depth=node.depth + 1,
                    branch_kind="setup_y1_lazy",
                    branch_on=branch_key,
                    branch_value=1,
                    is_lazy=True,
                )
                next_id += 1
                stats["y1_lazy_created"] += 1
                self._push_lazy(lazy_heap, child1, parent_lb)
            return next_id

        if dec.kind == "scrap":
            i_idx = int(dec.key)
            child_scrap = dict(node.fixed_scrap)
            child_scrap[i_idx] = 1
            if scrap_fix_feasible(self.data, child_scrap):
                child = BranchNode(
                    node_id=next_id,
                    fixed_y=dict(node.fixed_y),
                    fixed_scrap=child_scrap,
                    fixed_wip=dict(node.fixed_wip),
                    inherited_columns=inherited,
                    depth=node.depth + 1,
                    branch_kind="scrap",
                    branch_on=i_idx,
                    branch_value=1,
                )
                next_id += 1
                if self.branch_strategy == "old_hybrid" and open_heap is not None:
                    self._push_heap(open_heap, child, parent_lb, defer=False)
                else:
                    self._push_lazy(lazy_heap, child, parent_lb)

            child_scrap0 = dict(node.fixed_scrap)
            child_scrap0[i_idx] = 0
            if scrap_fix_feasible(self.data, child_scrap0):
                child0 = BranchNode(
                    node_id=next_id,
                    fixed_y=dict(node.fixed_y),
                    fixed_scrap=child_scrap0,
                    fixed_wip=dict(node.fixed_wip),
                    inherited_columns=None,
                    refresh_columns=True,
                    depth=node.depth + 1,
                    branch_kind="scrap_r0_lazy",
                    branch_on=i_idx,
                    branch_value=0,
                    is_lazy=True,
                )
                next_id += 1
                stats["y1_lazy_created"] += 1
                self._push_lazy(lazy_heap, child0, parent_lb)
            return next_id

        if dec.kind == "wip":
            i_idx = int(dec.key)
            assert dec.wip_fix is not None
            sense, bound = dec.wip_fix

            child_wip = dict(node.fixed_wip)
            child_wip[i_idx] = (sense, bound)
            if wip_fix_feasible(self.data, child_wip):
                dfs_stack.append(
                    BranchNode(
                        node_id=next_id,
                        fixed_y=dict(node.fixed_y),
                        fixed_scrap=dict(node.fixed_scrap),
                        fixed_wip=child_wip,
                        inherited_columns=inherited,
                        depth=node.depth + 1,
                        branch_kind="wip",
                        branch_on=i_idx,
                        branch_value=bound,
                    )
                )
                next_id += 1

            ceil_bound = bound + 1
            if ceil_bound <= self.data.w_i[i_idx]:
                child_wip_ge = dict(node.fixed_wip)
                child_wip_ge[i_idx] = ("ge", ceil_bound)
                if wip_fix_feasible(self.data, child_wip_ge):
                    child_ge = BranchNode(
                        node_id=next_id,
                        fixed_y=dict(node.fixed_y),
                        fixed_scrap=dict(node.fixed_scrap),
                        fixed_wip=child_wip_ge,
                        inherited_columns=inherited,
                        depth=node.depth + 1,
                        branch_kind="wip_ge_lazy",
                        branch_on=i_idx,
                        branch_value=ceil_bound,
                        is_lazy=True,
                    )
                    next_id += 1
                    stats["y1_lazy_created"] += 1
                    self._push_lazy(lazy_heap, child_ge, parent_lb)
            return next_id

        return next_id

    def _pop_node(
        self,
        dfs_stack: list[BranchNode],
        lazy_heap: list[tuple[float, int, BranchNode]],
        open_heap: list[tuple[float, int, BranchNode]] | None,
        stats: dict[str, int],
    ) -> BranchNode | None:
        if self.branch_strategy == "old_hybrid":
            assert open_heap is not None
            if not open_heap:
                return None
            _est, _nid, node = heapq.heappop(open_heap)
            return node

        if dfs_stack:
            stats["dfs_popped"] += 1
            return dfs_stack.pop()
        if lazy_heap:
            stats["lazy_popped"] += 1
            _est, _nid, node = heapq.heappop(lazy_heap)
            return node
        return None

    def _has_pending(
        self,
        dfs_stack: list[BranchNode],
        lazy_heap: list,
        open_heap: list | None,
    ) -> bool:
        if self.branch_strategy == "old_hybrid":
            return bool(open_heap)
        return bool(dfs_stack or lazy_heap)

    def solve(self) -> BranchAndPriceResult:
        start = time.perf_counter()
        deadline = start + self.time_limit if self.time_limit else None
        root = BranchNode(node_id=0, lb_estimate=float("-inf"))

        dfs_stack: list[BranchNode] = [root]
        lazy_heap: list[tuple[float, int, BranchNode]] = []
        open_heap: list[tuple[float, int, BranchNode]] | None = None
        if self.branch_strategy == "old_hybrid":
            open_heap = []
            heapq.heappush(open_heap, (root.lb_estimate, root.node_id, root))
            dfs_stack = []

        nodes_explored = 0
        nodes_pruned = 0
        nodes_infeasible = 0
        nodes_cg_failed = 0
        max_depth = 0
        incumbent: float | None = None
        best_fixed_y: dict[SetupFix, int] = {}
        root_lrmp: float | None = None
        best_node_lb: float | None = None
        best_cg: ColumnGenerationSolver | None = None
        next_id = 1
        timed_out = False
        proved_optimal = False
        stats = {"y1_lazy_created": 0, "lazy_popped": 0, "dfs_popped": 0}

        if self.root_im and deadline is not None:
            remaining = max(5.0, deadline - time.perf_counter())
            incumbent, best_fixed_y = self._try_incumbent({}, incumbent, min(self.im_time_limit, remaining * 0.35))
            if incumbent is not None:
                print(f"  BP root IM incumbent={incumbent:.2f}", flush=True)

        while (
            self._has_pending(dfs_stack, lazy_heap, open_heap)
            and nodes_explored < self.max_nodes
            and not proved_optimal
        ):
            if deadline is not None and time.perf_counter() >= deadline:
                timed_out = True
                break

            node = self._pop_node(dfs_stack, lazy_heap, open_heap, stats)
            if node is None:
                break

            nodes_explored += 1
            max_depth = max(max_depth, node.depth)

            if not _node_feasible(self.data, node):
                nodes_pruned += 1
                nodes_infeasible += 1
                continue

            node_cg_limit = None
            if deadline is not None:
                remaining = deadline - time.perf_counter()
                if remaining <= 1.0:
                    timed_out = True
                    break
                node_cg_limit = max(10.0, remaining * 0.75)

            node_t0 = time.perf_counter()
            cg = self._run_node_cg(node, time_limit=node_cg_limit)
            cg_sec = time.perf_counter() - node_t0

            if not cg.converged or cg.lrmp_obj is None:
                reason = cg.stop_reason or "not_converged"
                if reason == "infeasible_master" or (cg.lrmp_obj is None and not cg.iteration_times):
                    nodes_pruned += 1
                    nodes_infeasible += 1
                    print(
                        f"  BP [{nodes_explored}/{self.max_nodes}] node {node.node_id} "
                        f"kind={node.branch_kind} depth={node.depth} infeasible ({cg_sec:.1f}s)",
                        flush=True,
                    )
                else:
                    nodes_cg_failed += 1
                    print(
                        f"  BP [{nodes_explored}/{self.max_nodes}] node {node.node_id} "
                        f"kind={node.branch_kind} depth={node.depth} CG not converged ({cg_sec:.1f}s)",
                        flush=True,
                    )
                continue

            node.lower_bound = cg.lrmp_obj
            self.policy.note_lb(node.lower_bound)
            if best_node_lb is None or node.lower_bound > best_node_lb:
                best_node_lb = node.lower_bound
            if node.node_id == 0:
                root_lrmp = cg.lrmp_obj

            if incumbent is not None and node.lower_bound >= incumbent - INT_TOL:
                nodes_pruned += 1
                continue

            if self.use_node_im:
                node_im_limit = self.im_time_limit
                if deadline is not None:
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0:
                        timed_out = True
                        break
                    node_im_limit = min(
                        self.im_time_limit,
                        max(5.0, remaining * (0.45 if node.depth == 0 else 0.15)),
                    )
                incumbent, best_fixed_y = self._try_incumbent(node.fixed_y, incumbent, node_im_limit)

            if incumbent is not None and node.lower_bound >= incumbent - INT_TOL:
                nodes_pruned += 1
                continue

            implied = cg.get_implied_y()
            if _implied_y_integer(implied) and _q_integer(cg) and _r_integer(cg):
                best_cg = cg
                if incumbent is not None and abs(node.lower_bound - incumbent) <= INT_TOL:
                    proved_optimal = True
                continue

            dec = self.policy.pick(
                node.depth,
                cg,
                node.fixed_y,
                node.fixed_scrap,
                node.fixed_wip,
            )
            if dec is None:
                best_cg = cg
                continue

            inherited = copy_column_pool(cg.columns)
            print(
                f"  BP [{nodes_explored}/{self.max_nodes}] node {node.node_id} "
                f"depth={node.depth} LB={node.lower_bound:.2f} "
                f"branch {self._decision_label(dec)} strategy={self.branch_strategy} ({cg_sec:.1f}s)",
                flush=True,
            )

            next_id = self._create_children(
                node,
                dec,
                inherited,
                next_id,
                dfs_stack,
                lazy_heap,
                open_heap,
                stats,
            )

            if (
                incumbent is not None
                and best_node_lb is not None
                and best_node_lb >= incumbent - INT_TOL
                and not self._has_pending(dfs_stack, lazy_heap, open_heap)
            ):
                proved_optimal = True

        if self.final_im and deadline is not None:
            remaining = max(5.0, deadline - time.perf_counter())
            new_ub, new_fy = self._try_incumbent(best_fixed_y, incumbent, remaining)
            if new_ub is not None:
                incumbent, best_fixed_y = new_ub, new_fy

        runtime = time.perf_counter() - start
        pending = self._has_pending(dfs_stack, lazy_heap, open_heap)
        if proved_optimal or (
            incumbent is not None
            and best_node_lb is not None
            and incumbent <= best_node_lb + INT_TOL
            and not pending
        ):
            status = "optimal"
        elif timed_out:
            status = "time_limit"
        elif nodes_explored >= self.max_nodes:
            status = "max_nodes"
        else:
            status = "no_incumbent"

        return BranchAndPriceResult(
            status=status,
            objective=incumbent,
            lower_bound=root_lrmp,
            best_node_lb=best_node_lb,
            incumbent=incumbent,
            nodes_explored=nodes_explored,
            nodes_pruned=nodes_pruned,
            nodes_infeasible=nodes_infeasible,
            nodes_cg_failed=nodes_cg_failed,
            max_depth=max_depth,
            runtime_sec=runtime,
            best_fixed_y=best_fixed_y,
            root_lrmp=root_lrmp,
            cg_solver=best_cg,
            nodes_created=next_id,
            y1_lazy_created=stats["y1_lazy_created"],
            lazy_popped=stats["lazy_popped"],
            dfs_popped=stats["dfs_popped"],
        )


def solve_branch_and_price(
    data: ProcessedInstance,
    max_nodes: int = DEFAULT_BP_MAX_NODES,
    cg_max_iterations: int = DEFAULT_CG_MAX_ITERATIONS,
    im_time_limit: float = DEFAULT_BP_NODE_IM_TIME_LIMIT,
    mip_gap: float = DEFAULT_MIP_GAP,
    time_limit: float | None = DEFAULT_BP_TIME_LIMIT,
    cg_kwargs: dict | None = None,
    use_node_im: bool = DEFAULT_BP_USE_NODE_IM,
    root_im: bool = DEFAULT_BP_ROOT_IM,
    final_im: bool = DEFAULT_BP_FINAL_IM,
    branch_strategy: str = DEFAULT_BP_STRATEGY,
) -> SolutionResult:
    bp = BranchAndPriceSolver(
        data,
        max_nodes=max_nodes,
        cg_max_iterations=cg_max_iterations,
        im_time_limit=im_time_limit,
        mip_gap=mip_gap,
        time_limit=time_limit,
        cg_kwargs=cg_kwargs,
        use_node_im=use_node_im,
        root_im=root_im,
        final_im=final_im,
        branch_strategy=branch_strategy,
    )
    result = bp.solve()
    return SolutionResult(
        method="branch_and_price",
        instance_name=data.raw.name,
        status=result.status,
        objective=result.objective,
        lower_bound=result.lower_bound,
        mip_gap=(
            (result.objective - result.best_node_lb) / result.objective * 100
            if result.objective and result.best_node_lb
            else None
        ),
        runtime_sec=result.runtime_sec,
        extra={
            "nodes_explored": result.nodes_explored,
            "nodes_created": result.nodes_created,
            "nodes_pruned": result.nodes_pruned,
            "nodes_infeasible": result.nodes_infeasible,
            "nodes_cg_failed": result.nodes_cg_failed,
            "max_depth": result.max_depth,
            "root_lrmp": result.root_lrmp,
            "best_node_lb": result.best_node_lb,
            "best_fixed_y_count": len(result.best_fixed_y),
            "y1_lazy_created": result.y1_lazy_created,
            "lazy_popped": result.lazy_popped,
            "dfs_popped": result.dfs_popped,
            "branch_strategy": branch_strategy,
        },
    )
