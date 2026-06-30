"""Branch-and-price：对 implied setup Y 分支 + 节点列生成。"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from common.data_models import ProcessedInstance, ScheduleColumn, SolutionResult
from configuration import (
    DEFAULT_BP_MAX_NODES,
    DEFAULT_CG_MAX_ITERATIONS,
    DEFAULT_CG_INIT_TIME_LIMIT,
    DEFAULT_MIP_GAP,
    DEFAULT_BP_NODE_IM_TIME_LIMIT,
)
from model_and_algo.branch_utils import SetupFix, copy_column_pool
from model_and_algo.column_generation import ColumnGenerationSolver
from model_and_algo.im_model import solve_im

INT_TOL = 1e-5
FRAC_TOL = 1e-4


@dataclass
class BranchNode:
    """搜索树节点。"""

    node_id: int
    fixed_y: dict[SetupFix, int] = field(default_factory=dict)
    inherited_columns: dict[str, list[ScheduleColumn]] | None = None
    lower_bound: float = float("-inf")
    depth: int = 0
    branch_on: SetupFix | None = None
    branch_value: int | None = None


def _is_int(v: float, tol: float = INT_TOL) -> bool:
    return abs(v - round(v)) <= tol


def _frac_part(v: float) -> float:
    return abs(v - round(v))


def _pick_branch(implied_y: dict[SetupFix, float]) -> tuple[SetupFix, float] | None:
    """选 fractionality 最大的 implied Y。"""
    best: tuple[SetupFix, float] | None = None
    best_frac = FRAC_TOL
    for key, val in implied_y.items():
        if val <= FRAC_TOL:
            continue
        fp = _frac_part(val)
        if fp > best_frac:
            best_frac = fp
            best = (key, val)
    return best


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
            val = qv.X
            if val > FRAC_TOL and not _is_int(val):
                return False
    return True


@dataclass
class BranchAndPriceResult:
    status: str
    objective: float | None
    lower_bound: float | None
    incumbent: float | None
    nodes_explored: int
    nodes_pruned: int
    max_depth: int
    runtime_sec: float
    best_fixed_y: dict[SetupFix, int]
    root_lrmp: float | None
    cg_solver: ColumnGenerationSolver | None = None


class BranchAndPriceSolver:
    """对 implied Y 做 Ryan-Foster 风格分支的 B&P 求解器。"""

    def __init__(
        self,
        data: ProcessedInstance,
        max_nodes: int = DEFAULT_BP_MAX_NODES,
        cg_max_iterations: int = DEFAULT_CG_MAX_ITERATIONS,
        cg_init_time_limit: float = DEFAULT_CG_INIT_TIME_LIMIT,
        im_time_limit: float = DEFAULT_BP_NODE_IM_TIME_LIMIT,
        mip_gap: float = DEFAULT_MIP_GAP,
    ) -> None:
        self.data = data
        self.max_nodes = max_nodes
        self.cg_max_iterations = cg_max_iterations
        self.cg_init_time_limit = cg_init_time_limit
        self.im_time_limit = im_time_limit
        self.mip_gap = mip_gap

    def _run_node_cg(
        self,
        node: BranchNode,
    ) -> ColumnGenerationSolver:
        cg = ColumnGenerationSolver(
            self.data,
            max_iterations=self.cg_max_iterations,
            init_time_limit=self.cg_init_time_limit,
            fixed_y=node.fixed_y,
            inherited_columns=node.inherited_columns,
            use_gurobi_init=not node.inherited_columns,
        )
        cg.run()
        return cg

    def _try_incumbent(
        self,
        fixed_y: dict[SetupFix, int],
        incumbent: float | None,
    ) -> tuple[float | None, dict[SetupFix, int]]:
        im = solve_im(
            self.data,
            time_limit=self.im_time_limit,
            mip_gap=self.mip_gap,
            fixed_y=fixed_y or None,
            method="branch_and_price",
        )
        if im.status == "optimal" and im.objective is not None:
            if incumbent is None or im.objective < incumbent - INT_TOL:
                return im.objective, fixed_y
        return incumbent, fixed_y

    def solve(self) -> BranchAndPriceResult:
        start = time.perf_counter()
        root = BranchNode(node_id=0)
        stack: list[BranchNode] = [root]
        nodes_explored = 0
        nodes_pruned = 0
        max_depth = 0
        incumbent: float | None = None
        best_fixed_y: dict[SetupFix, int] = {}
        root_lrmp: float | None = None
        best_cg: ColumnGenerationSolver | None = None
        next_id = 1

        while stack and nodes_explored < self.max_nodes:
            node = stack.pop()
            nodes_explored += 1
            max_depth = max(max_depth, node.depth)

            cg = self._run_node_cg(node)
            if not cg.converged or cg.lrmp_obj is None:
                continue

            node.lower_bound = cg.lrmp_obj
            if node.node_id == 0:
                root_lrmp = cg.lrmp_obj

            if incumbent is not None and node.lower_bound >= incumbent - INT_TOL:
                nodes_pruned += 1
                continue

            incumbent, best_fixed_y = self._try_incumbent(node.fixed_y, incumbent)

            if incumbent is not None and node.lower_bound >= incumbent - INT_TOL:
                nodes_pruned += 1
                continue

            implied = cg.get_implied_y()
            if _implied_y_integer(implied) and _q_integer(cg):
                best_cg = cg
                continue

            pick = _pick_branch(implied)
            if pick is None:
                best_cg = cg
                continue

            branch_key, _ = pick
            inherited = copy_column_pool(cg.columns)

            for val in (0, 1):
                child_fixed = dict(node.fixed_y)
                child_fixed[branch_key] = val
                child = BranchNode(
                    node_id=next_id,
                    fixed_y=child_fixed,
                    inherited_columns=inherited,
                    depth=node.depth + 1,
                    branch_on=branch_key,
                    branch_value=val,
                )
                next_id += 1
                stack.append(child)

        runtime = time.perf_counter() - start
        status = "optimal" if incumbent is not None else "no_incumbent"
        if nodes_explored >= self.max_nodes:
            status = "max_nodes"

        return BranchAndPriceResult(
            status=status,
            objective=incumbent,
            lower_bound=root_lrmp,
            incumbent=incumbent,
            nodes_explored=nodes_explored,
            nodes_pruned=nodes_pruned,
            max_depth=max_depth,
            runtime_sec=runtime,
            best_fixed_y=best_fixed_y,
            root_lrmp=root_lrmp,
            cg_solver=best_cg,
        )


def solve_branch_and_price(
    data: ProcessedInstance,
    max_nodes: int = DEFAULT_BP_MAX_NODES,
    cg_max_iterations: int = DEFAULT_CG_MAX_ITERATIONS,
    im_time_limit: float = DEFAULT_BP_NODE_IM_TIME_LIMIT,
    mip_gap: float = DEFAULT_MIP_GAP,
) -> SolutionResult:
    """B&P 求解入口。"""
    bp = BranchAndPriceSolver(
        data,
        max_nodes=max_nodes,
        cg_max_iterations=cg_max_iterations,
        im_time_limit=im_time_limit,
        mip_gap=mip_gap,
    )
    result = bp.solve()
    return SolutionResult(
        method="branch_and_price",
        instance_name=data.raw.name,
        status=result.status,
        objective=result.objective,
        lower_bound=result.lower_bound,
        mip_gap=(
            (result.objective - result.lower_bound) / result.objective * 100
            if result.objective and result.lower_bound
            else None
        ),
        runtime_sec=result.runtime_sec,
        extra={
            "nodes_explored": result.nodes_explored,
            "nodes_pruned": result.nodes_pruned,
            "max_depth": result.max_depth,
            "root_lrmp": result.root_lrmp,
            "best_fixed_y_count": len(result.best_fixed_y),
        },
    )
