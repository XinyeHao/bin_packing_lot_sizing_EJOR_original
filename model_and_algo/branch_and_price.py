"""Branch-and-price：三层分支策略 + DFS/lazy 混合搜索。"""

from __future__ import annotations

import gc
import heapq
import math
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
    DEFAULT_BP_MIP_GAP,
    DEFAULT_MIP_GAP,
    ML_SB_CHILD_TLIM,
    ML_SB_MIN_REMAINING,
    ML_SB_REFINE_K,
    FRAC_SB_REFINE_K,
)
from model_and_algo.branch_utils import (
    Y1_DEFER_PENALTY,
    BranchDecision,
    BranchPolicy,
    SetupFix,
    WipFix,
    prune_column_pool,
    copy_column_pool,
    fixed_y_feasible,
    scrap_fix_feasible,
    wip_fix_feasible,
)
from configuration import CGFO_THRESHOLD_II
from model_and_algo.cgfo import _build_allowed_setup, _build_fixed_y
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
    parent_lb: float | None = None  # valid LB inherited from parent (dual-safe floor)
    sibling_group_id: int | None = None  # parent.node_id; used to co-locate sibling frontier
    depth: int = 0
    branch_kind: str | None = None
    branch_on: SetupFix | int | None = None
    branch_value: int | None = None
    cg_fail_retries: int = 0
    cg_budget_override: float | None = None


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
    try:
        for machine in cg.data.machines:
            for qv in cg._artifacts["Q"][machine]:
                if qv.X > FRAC_TOL and not _is_int(qv.X):
                    return False
    except Exception:
        return False
    return True


def _r_integer(cg: ColumnGenerationSolver) -> bool:
    if cg._artifacts is None:
        return False
    try:
        for rvar in cg._artifacts["R"].values():
            val = rvar.X
            if val > FRAC_TOL and not _is_int(val):
                return False
    except Exception:
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
    """根节点 RMP 目标值（在 q=ub=1 时可能偏高，不是有效下界）。"""
    root_valid_lb: float | None
    """根节点 Lagrangian 下界（有效下界 = LRMP + Σ min(RC_m, 0)，用于计算间隙）。"""
    root_cg_converged: bool
    """根节点 CG 是否收敛。"""
    root_cg_stop: str | None = None
    """根节点 CG 停止原因（classical / pool_optimal_saturated / time_limit 等）。"""
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
        mip_gap: float = DEFAULT_BP_MIP_GAP,
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
        # Kept for API compat; UB path ignores these (CGFO + integer LRMP only).
        self.use_node_im = use_node_im
        self.root_im = root_im
        self.final_im = final_im
        self.branch_strategy = branch_strategy
        self.lazy_y1_cg_cap = lazy_y1_cg_cap
        self.ml_sb_refine_k = ML_SB_REFINE_K
        self.ml_sb_child_tlim = ML_SB_CHILD_TLIM
        self.ml_sb_min_remaining = ML_SB_MIN_REMAINING
        if branch_strategy == "frac_sb":
            self.ml_sb_refine_k = FRAC_SB_REFINE_K
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
        incumbent: float | None = None,
    ) -> ColumnGenerationSolver:
        inherited = None if node.refresh_columns else node.inherited_columns
        use_gurobi_init = node.refresh_columns or not inherited
        if "use_gurobi_init" in self.cg_kwargs and not node.refresh_columns:
            use_gurobi_init = bool(self.cg_kwargs["use_gurobi_init"]) and not inherited
        kwargs = {k: v for k, v in self.cg_kwargs.items() if k != "use_gurobi_init"}
        if time_limit is not None:
            # Only cap deferred Y=1 setup sides — NOT scrap_r0 / wip_ge (those need full budget).
            kind = getattr(node, "branch_kind", "") or ""
            if (
                node.is_lazy
                and kind in ("setup_y1_lazy", "setup")
                and any(v == 1 for v in node.fixed_y.values())
            ):
                kwargs["time_limit"] = min(time_limit, self.lazy_y1_cg_cap)
            else:
                kwargs["time_limit"] = time_limit
        max_iters = int(kwargs.pop("max_iterations", self.cg_max_iterations))
        depth = node.depth
        hard_kind = getattr(node, "branch_kind", "") in (
            "scrap_r0_lazy",
            "wip_ge_lazy",
        )
        # Hard sides / CG retries: do not crush EBRP; need a real dual LB.
        if node.cg_budget_override is not None or hard_kind or node.cg_fail_retries > 0:
            max_iters = min(max_iters, self.cg_max_iterations)
            base_ebrp = float(kwargs.get("ebrp_time_limit") or 30.0)
            kwargs["ebrp_time_limit"] = min(base_ebrp, max(5.0, (time_limit or 30.0) * 0.35))
        elif depth >= 8:
            max_iters = min(max_iters, 15)
            kwargs["ebrp_time_limit"] = min(float(kwargs.get("ebrp_time_limit") or 30.0), 0.5)
        elif depth >= 4:
            max_iters = min(max_iters, 25)
            kwargs["ebrp_time_limit"] = min(float(kwargs.get("ebrp_time_limit") or 30.0), 0.8)
        elif depth >= 2:
            max_iters = min(max_iters, 40)
            kwargs["ebrp_time_limit"] = min(float(kwargs.get("ebrp_time_limit") or 30.0), 1.5)
        elif depth >= 1:
            max_iters = min(max_iters, 80)
            kwargs["ebrp_time_limit"] = min(float(kwargs.get("ebrp_time_limit") or 30.0), 3.0)
        else:
            max_iters = min(max_iters, self.cg_max_iterations)
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
        cg.incumbent = incumbent
        cg.run()
        return cg

    def _probe_child_lb(
        self,
        fixed_y: dict[SetupFix, int],
        fixed_scrap: dict[int, int],
        fixed_wip: dict[int, WipFix],
        inherited: dict[str, list[ScheduleColumn]],
        tlim: float,
        incumbent: float | None,
    ) -> float | None:
        """Short CG probe for strong-branch refine (no Gurobi init)."""
        if tlim < 1.0:
            return None
        kwargs = {k: v for k, v in self.cg_kwargs.items() if k != "use_gurobi_init"}
        kwargs["time_limit"] = tlim
        ebrp_cap = float(kwargs.get("ebrp_time_limit") or 30.0)
        kwargs["ebrp_time_limit"] = min(ebrp_cap, max(0.5, tlim * 0.4))
        max_iters = min(40, self.cg_max_iterations)
        try:
            child = ColumnGenerationSolver(
                self.data,
                max_iterations=max_iters,
                init_time_limit=self.cg_init_time_limit,
                fixed_y=fixed_y,
                fixed_scrap=fixed_scrap,
                fixed_wip=fixed_wip,
                inherited_columns=inherited,
                use_gurobi_init=False,
                **kwargs,
            )
            child.incumbent = incumbent
            child.run()
            return child.lrmp_obj
        except Exception:
            return None

    def _ml_sb_refine(
        self,
        node: BranchNode,
        cg: ColumnGenerationSolver,
        incumbent: float | None,
        deadline: float | None,
        fallback: BranchDecision | None,
    ) -> BranchDecision | None:
        """ML rank → Top-K short strong branch; pick max min(ΔL, ΔR)."""
        from model_and_algo.ml_branch_selector import (
            apply_decision_side,
            pick_ml_ranked,
            sb_y_min,
        )

        rem = (
            (deadline - time.perf_counter())
            if deadline is not None
            else self.ml_sb_min_remaining + 1.0
        )
        if rem < self.ml_sb_min_remaining:
            return fallback
        try:
            sel = self.policy._get_ml_selector()
            if not sel.ready:
                return fallback
            ranked = pick_ml_ranked(
                self.data, cg, node.fixed_y, node.fixed_scrap, node.fixed_wip, sel
            )
        except Exception as exc:
            print(f"    [ml-sb] rank failed ({exc}); keep ML pick", flush=True)
            return fallback
        return self._sb_refine_ranked(
            node, cg, incumbent, deadline, fallback, ranked, tag="ml-sb"
        )

    def _frac_sb_refine(
        self,
        node: BranchNode,
        cg: ColumnGenerationSolver,
        incumbent: float | None,
        deadline: float | None,
        fallback: BranchDecision | None,
    ) -> BranchDecision | None:
        """Frac Top-K → short strong branch; pick max min(ΔL, ΔR)."""
        from model_and_algo.ml_branch_selector import pick_frac_ranked

        rem = (
            (deadline - time.perf_counter())
            if deadline is not None
            else self.ml_sb_min_remaining + 1.0
        )
        if rem < self.ml_sb_min_remaining:
            return fallback
        try:
            ranked = pick_frac_ranked(
                self.data, cg, node.fixed_y, node.fixed_scrap, node.fixed_wip
            )
        except Exception as exc:
            print(f"    [frac-sb] rank failed ({exc}); keep frac pick", flush=True)
            return fallback
        return self._sb_refine_ranked(
            node, cg, incumbent, deadline, fallback, ranked, tag="frac-sb"
        )

    def _sb_refine_ranked(
        self,
        node: BranchNode,
        cg: ColumnGenerationSolver,
        incumbent: float | None,
        deadline: float | None,
        fallback: BranchDecision | None,
        ranked,
        tag: str = "sb",
    ) -> BranchDecision | None:
        from model_and_algo.ml_branch_selector import apply_decision_side, sb_y_min

        k = max(1, int(self.ml_sb_refine_k))
        top = ranked[:k]
        if len(top) <= 1:
            return top[0].decision if top else fallback

        parent_lb = float(node.lower_bound)
        inherited = copy_column_pool(cg.columns)
        best_dec: BranchDecision | None = None
        best_y = -1.0
        for cand in top:
            rem = (
                (deadline - time.perf_counter())
                if deadline is not None
                else self.ml_sb_child_tlim * 3
            )
            if rem < 2.5:
                break
            side_tlim = min(self.ml_sb_child_tlim, rem / 2.5)
            lbs: list[float | None] = []
            ok = True
            for side in ("L", "R"):
                rem = (
                    (deadline - time.perf_counter())
                    if deadline is not None
                    else side_tlim + 1.0
                )
                if rem < 1.5:
                    ok = False
                    break
                fy, fs, fw = apply_decision_side(
                    cand.decision,
                    node.fixed_y,
                    node.fixed_scrap,
                    node.fixed_wip,
                    side,
                )
                lbs.append(
                    self._probe_child_lb(
                        fy, fs, fw, inherited, min(side_tlim, rem - 0.5), incumbent
                    )
                )
            if not ok or len(lbs) < 2:
                break
            y = sb_y_min(parent_lb, lbs[0], lbs[1])
            if y > best_y:
                best_y = y
                best_dec = cand.decision
        if best_dec is None:
            return fallback
        if fallback is not None and best_dec != fallback:
            print(
                f"    [{tag}] refine k={len(top)} picked "
                f"{self._decision_label(best_dec)} y_min={best_y:.3f} "
                f"(coarse={self._decision_label(fallback)})",
                flush=True,
            )
        else:
            print(
                f"    [{tag}] refine k={len(top)} confirm "
                f"{self._decision_label(best_dec)} y_min={best_y:.3f}",
                flush=True,
            )
        return best_dec

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

    def _try_cgfo(
        self,
        cg: ColumnGenerationSolver,
        incumbent: float | None,
        budget: float,
        *,
        merge_fixed_y: dict[SetupFix, int] | None = None,
    ) -> tuple[float | None, dict[SetupFix, int] | None]:
        """CGFO：由分数列支撑构造允许 setup，其余 Y=0，再解受限 IM。"""
        if budget <= 1.0:
            return incumbent, None
        selected = cg.get_selected_columns(threshold=CGFO_THRESHOLD_II)
        allowed = _build_allowed_setup(self.data, selected)
        if not allowed:
            return incumbent, None
        cgfo_fy = _build_fixed_y(self.data, allowed)
        if merge_fixed_y:
            cgfo_fy = {**cgfo_fy, **merge_fixed_y}
        new_ub, new_fy = self._try_incumbent(cgfo_fy, incumbent, budget)
        if new_ub is not None and (incumbent is None or new_ub < incumbent - INT_TOL):
            return new_ub, new_fy
        return incumbent, None

    def _push_lazy(
        self,
        lazy_heap: list[tuple[float, int, BranchNode]],
        child: BranchNode,
        parent_lb: float,
    ) -> None:
        child.parent_lb = parent_lb
        child.lb_estimate = parent_lb + Y1_DEFER_PENALTY
        heapq.heappush(lazy_heap, (child.lb_estimate, child.node_id, child))

    def _push_heap(
        self,
        open_heap: list[tuple[float, int, BranchNode]],
        child: BranchNode,
        parent_lb: float,
        defer: bool,
    ) -> None:
        child.parent_lb = parent_lb
        child.lb_estimate = parent_lb + (Y1_DEFER_PENALTY if defer else 0.0)
        heapq.heappush(open_heap, (child.lb_estimate, child.node_id, child))

    @staticmethod
    def _heap_min_lb(heap: list[tuple[float, int, BranchNode]] | None) -> float | None:
        if not heap:
            return None
        return min(est for est, _nid, _n in heap)

    def _tree_dual_bound(
        self,
        open_heap: list[tuple[float, int, BranchNode]] | None,
        lazy_heap: list[tuple[float, int, BranchNode]],
        dfs_stack: list[BranchNode],
        current_lb: float | None = None,
    ) -> float | None:
        """全局对偶下界 = 开节点（含 lazy/DFS）LB 估计的最小值（+ 可选当前手中节点）。"""
        vals: list[float] = []
        if current_lb is not None and math.isfinite(current_lb):
            vals.append(current_lb)
        for h in (open_heap, lazy_heap):
            m = self._heap_min_lb(h)
            if m is not None and math.isfinite(m):
                vals.append(m)
        for n in dfs_stack:
            est = n.lb_estimate if n.lb_estimate > float("-inf") else n.lower_bound
            if math.isfinite(est):
                vals.append(est)
        return min(vals) if vals else None

    @staticmethod
    def _rel_gap(ub: float | None, lb: float | None) -> float | None:
        if ub is None or lb is None or not math.isfinite(ub) or not math.isfinite(lb):
            return None
        if ub <= 1e-9:
            return None
        return max(0.0, (ub - lb) / ub)

    def _gap_reached(
        self,
        incumbent: float | None,
        open_heap: list[tuple[float, int, BranchNode]] | None,
        lazy_heap: list[tuple[float, int, BranchNode]],
        dfs_stack: list[BranchNode],
        current_lb: float | None = None,
    ) -> tuple[bool, float | None, float | None]:
        """若 mip_gap>0 且 (UB−全局LB)/UB ≤ mip_gap，返回 (True, gap, tree_lb)。"""
        if self.mip_gap <= 0 or incumbent is None:
            return False, None, None
        tree_lb = self._tree_dual_bound(open_heap, lazy_heap, dfs_stack, current_lb)
        gap = self._rel_gap(incumbent, tree_lb)
        if gap is None:
            return False, None, tree_lb
        return gap <= self.mip_gap + 1e-12, gap, tree_lb

    def _note_gap_stop(
        self,
        incumbent: float | None,
        open_heap,
        lazy_heap,
        dfs_stack,
        nodes_explored: int,
        current_lb: float | None = None,
    ) -> bool:
        hit, rel_gap, tree_lb = self._gap_reached(
            incumbent, open_heap, lazy_heap, dfs_stack, current_lb
        )
        if not hit:
            return False
        print(
            f"  BP stop: gap={100.0 * rel_gap:.3f}% ≤ {100.0 * self.mip_gap:.1f}% "
            f"(UB={incumbent:.2f}, tree_LB={tree_lb:.2f}, nodes={nodes_explored})",
            flush=True,
        )
        return True

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
    ) -> tuple[int, int]:
        """Return (next_id, num_children_created)."""
        parent_lb = node.lower_bound
        n_created = 0

        def _tag(child: BranchNode, side: int) -> BranchNode:
            nonlocal n_created
            n_created += 1
            child.sibling_group_id = node.node_id
            return child

        if dec.kind in ("setup_y0", "setup"):
            branch_key = dec.key
            assert isinstance(branch_key, tuple)

            if self.branch_strategy in ("old_hybrid", "best_bound", "ml", "frac_sb"):
                assert open_heap is not None
                for val in (1, 0):
                    child_fixed = dict(node.fixed_y)
                    child_fixed[branch_key] = val
                    if not fixed_y_feasible(self.data, child_fixed):
                        continue
                    side = 0 if val == 0 else 1
                    child = _tag(BranchNode(
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
                    ), side)
                    next_id += 1
                    self._push_heap(open_heap, child, parent_lb, defer=(val == 1))
                    if val == 1:
                        stats["y1_lazy_created"] += 1
                return next_id, n_created

            child_fixed = dict(node.fixed_y)
            child_fixed[branch_key] = 0
            if fixed_y_feasible(self.data, child_fixed):
                dfs_stack.append(
                    _tag(BranchNode(
                        node_id=next_id,
                        fixed_y=child_fixed,
                        fixed_scrap=dict(node.fixed_scrap),
                        fixed_wip=dict(node.fixed_wip),
                        inherited_columns=inherited,
                        depth=node.depth + 1,
                        branch_kind="setup_y0",
                        branch_on=branch_key,
                        branch_value=0,
                    ), 0)
                )
                next_id += 1

            child_fixed_y1 = dict(node.fixed_y)
            child_fixed_y1[branch_key] = 1
            if fixed_y_feasible(self.data, child_fixed_y1):
                child1 = _tag(BranchNode(
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
                ), 1)
                next_id += 1
                stats["y1_lazy_created"] += 1
                self._push_lazy(lazy_heap, child1, parent_lb)
            return next_id, n_created

        if dec.kind == "scrap":
            i_idx = int(dec.key)
            child_scrap = dict(node.fixed_scrap)
            child_scrap[i_idx] = 1
            if scrap_fix_feasible(self.data, child_scrap):
                child = _tag(BranchNode(
                    node_id=next_id,
                    fixed_y=dict(node.fixed_y),
                    fixed_scrap=child_scrap,
                    fixed_wip=dict(node.fixed_wip),
                    inherited_columns=inherited,
                    depth=node.depth + 1,
                    branch_kind="scrap",
                    branch_on=i_idx,
                    branch_value=1,
                ), 0)
                next_id += 1
                if self.branch_strategy in ("old_hybrid", "best_bound", "ml", "frac_sb") and open_heap is not None:
                    self._push_heap(open_heap, child, parent_lb, defer=False)
                else:
                    self._push_lazy(lazy_heap, child, parent_lb)

            child_scrap0 = dict(node.fixed_scrap)
            child_scrap0[i_idx] = 0
            if scrap_fix_feasible(self.data, child_scrap0):
                # Inherit columns: R=0 is hard; wiping the pool caused CG fail / early fake stops.
                child0 = _tag(BranchNode(
                    node_id=next_id,
                    fixed_y=dict(node.fixed_y),
                    fixed_scrap=child_scrap0,
                    fixed_wip=dict(node.fixed_wip),
                    inherited_columns=inherited,
                    refresh_columns=False,
                    depth=node.depth + 1,
                    branch_kind="scrap_r0_lazy",
                    branch_on=i_idx,
                    branch_value=0,
                    is_lazy=True,
                ), 1)
                next_id += 1
                stats["y1_lazy_created"] += 1
                if self.branch_strategy in ("old_hybrid", "best_bound", "ml", "frac_sb") and open_heap is not None:
                    self._push_heap(open_heap, child0, parent_lb, defer=True)
                else:
                    self._push_lazy(lazy_heap, child0, parent_lb)
            return next_id, n_created

        if dec.kind == "wip":
            i_idx = int(dec.key)
            assert dec.wip_fix is not None
            sense, bound = dec.wip_fix

            child_wip = dict(node.fixed_wip)
            child_wip[i_idx] = (sense, bound)
            if wip_fix_feasible(self.data, child_wip):
                wip_child = _tag(BranchNode(
                    node_id=next_id,
                    fixed_y=dict(node.fixed_y),
                    fixed_scrap=dict(node.fixed_scrap),
                    fixed_wip=child_wip,
                    inherited_columns=inherited,
                    depth=node.depth + 1,
                    branch_kind="wip",
                    branch_on=i_idx,
                    branch_value=bound,
                ), 0)
                if self.branch_strategy in ("old_hybrid", "best_bound", "ml", "frac_sb") and open_heap is not None:
                    self._push_heap(open_heap, wip_child, parent_lb, defer=False)
                else:
                    dfs_stack.append(wip_child)
                next_id += 1

            ceil_bound = bound + 1
            if ceil_bound <= self.data.w_i[i_idx]:
                child_wip_ge = dict(node.fixed_wip)
                child_wip_ge[i_idx] = ("ge", ceil_bound)
                if wip_fix_feasible(self.data, child_wip_ge):
                    child_ge = _tag(BranchNode(
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
                    ), 1)
                    next_id += 1
                    stats["y1_lazy_created"] += 1
                    if self.branch_strategy in ("old_hybrid", "best_bound", "ml", "frac_sb") and open_heap is not None:
                        self._push_heap(open_heap, child_ge, parent_lb, defer=True)
                    else:
                        self._push_lazy(lazy_heap, child_ge, parent_lb)
            return next_id, n_created

        return next_id, n_created

    def _pop_node(
        self,
        dfs_stack: list[BranchNode],
        lazy_heap: list[tuple[float, int, BranchNode]],
        open_heap: list[tuple[float, int, BranchNode]] | None,
        stats: dict[str, int],
    ) -> BranchNode | None:
        if self.branch_strategy in ("old_hybrid", "best_bound", "ml", "frac_sb"):
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
        if self.branch_strategy in ("old_hybrid", "best_bound", "ml", "frac_sb"):
            return bool(open_heap)
        return bool(dfs_stack or lazy_heap)

    def solve(self) -> BranchAndPriceResult:
        start = time.perf_counter()
        deadline = start + self.time_limit if self.time_limit else None
        root = BranchNode(node_id=0, lb_estimate=float("-inf"))

        dfs_stack: list[BranchNode] = [root]
        lazy_heap: list[tuple[float, int, BranchNode]] = []
        open_heap: list[tuple[float, int, BranchNode]] | None = None
        if self.branch_strategy in ("old_hybrid", "best_bound", "ml", "frac_sb"):
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
        root_valid_lb: float | None = None
        root_cg_converged = False
        root_cg_stop: str | None = None
        best_node_lb: float | None = None
        best_cg: ColumnGenerationSolver | None = None
        next_id = 1
        timed_out = False
        proved_optimal = False
        gap_stopped = False
        stats = {"y1_lazy_created": 0, "lazy_popped": 0, "dfs_popped": 0}
        root_ub_done = False

        while (
            self._has_pending(dfs_stack, lazy_heap, open_heap)
            and nodes_explored < self.max_nodes
            and not proved_optimal
            and not gap_stopped
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
                if node.cg_budget_override is not None:
                    node_cg_limit = min(float(node.cg_budget_override), remaining * 0.9)
                elif node.depth == 0:
                    node_cg_limit = max(10.0, min(remaining * 0.75, 600.0))
                elif getattr(node, "branch_kind", "") == "scrap_r0_lazy":
                    # R=0 side needs real budget (was 5–12s → frequent CG fail / early stop)
                    node_cg_limit = max(25.0, min(remaining * 0.55, 90.0))
                elif getattr(node, "branch_kind", "") in ("wip_ge_lazy",):
                    node_cg_limit = max(15.0, min(remaining * 0.45, 40.0))
                elif node.depth >= 8:
                    node_cg_limit = max(4.0, min(remaining * 0.4, 8.0))
                elif node.depth >= 4:
                    node_cg_limit = max(5.0, min(remaining * 0.45, 12.0))
                elif node.depth >= 2:
                    node_cg_limit = max(6.0, min(remaining * 0.5, 20.0))
                else:
                    node_cg_limit = max(8.0, min(remaining * 0.6, 40.0))

            node_t0 = time.perf_counter()
            cg = self._run_node_cg(node, time_limit=node_cg_limit, incumbent=incumbent)
            cg_sec = time.perf_counter() - node_t0

            raw_lb = getattr(cg, "last_valid_lb", None)
            if raw_lb is None and cg.lrmp_obj is not None:
                raw_lb = cg.lrmp_obj

            if cg.stop_reason in ("infeasible_master", "pruned_by_incumbent") or (
                cg.lrmp_obj is None and not cg.iteration_times and raw_lb is None
            ):
                reason = cg.stop_reason or "infeasible"
                nodes_pruned += 1
                if reason == "pruned_by_incumbent":
                    print(
                        f"  BP [{nodes_explored}/{self.max_nodes}] node {node.node_id} "
                        f"kind={node.branch_kind} depth={node.depth} pruned "
                        f"(Lagrangian LB ≥ 当前最好可行解, {cg_sec:.1f}s)",
                        flush=True,
                    )
                else:
                    nodes_infeasible += 1
                    print(
                        f"  BP [{nodes_explored}/{self.max_nodes}] node {node.node_id} "
                        f"kind={node.branch_kind} depth={node.depth} infeasible ({cg_sec:.1f}s)",
                        flush=True,
                    )
                continue

            if raw_lb is None or not math.isfinite(raw_lb):
                nodes_cg_failed += 1
                rem = (
                    (deadline - time.perf_counter())
                    if deadline is not None
                    else 0.0
                )
                max_retries = 2
                # 有剩余时间则加大预算重入队（有上限，避免同一节点烧光时限）
                if (
                    rem > 8.0
                    and open_heap is not None
                    and node.cg_fail_retries < max_retries
                ):
                    node.cg_fail_retries += 1
                    used = float(cg_sec)
                    # 单调加预算，仅用剩余时间做硬顶（不要用 rem*0.55 把预算越做越小）
                    prev = float(
                        node.cg_budget_override
                        if node.cg_budget_override is not None
                        else (node_cg_limit or used or 20.0)
                    )
                    node.cg_budget_override = min(rem * 0.85, max(prev * 1.5, used * 1.5, 60.0))
                    node.refresh_columns = False
                    floor_lb = (
                        node.parent_lb
                        if node.parent_lb is not None
                        else (node.lb_estimate if math.isfinite(node.lb_estimate) else 0.0)
                    )
                    self._push_heap(open_heap, node, float(floor_lb), defer=False)
                    print(
                        f"  BP [{nodes_explored}/{self.max_nodes}] node {node.node_id} "
                        f"kind={node.branch_kind} depth={node.depth} CG failed "
                        f"(used={cg_sec:.1f}s, limit={node_cg_limit}); "
                        f"retry#{node.cg_fail_retries}/{max_retries} "
                        f"next_budget={node.cg_budget_override:.0f}s (remain={rem:.0f}s)",
                        flush=True,
                    )
                else:
                    why = (
                        f"max retries={max_retries}"
                        if node.cg_fail_retries >= max_retries
                        else "no time to retry"
                    )
                    print(
                        f"  BP [{nodes_explored}/{self.max_nodes}] node {node.node_id} "
                        f"kind={node.branch_kind} depth={node.depth} CG failed "
                        f"(used={cg_sec:.1f}s); abandon ({why})",
                        flush=True,
                    )
                continue

            # Parent LB is dual-safe for the child (child IP OPT ≥ parent IP OPT).
            # Floor incomplete CG duals so best-bound / tree_lb cannot collapse.
            floor = node.parent_lb
            if floor is None and node.lb_estimate > float("-inf"):
                floor = node.lb_estimate
            if floor is not None and math.isfinite(floor):
                node.lower_bound = max(float(raw_lb), float(floor))
            else:
                node.lower_bound = float(raw_lb)
            node.lb_estimate = node.lower_bound
            self.policy.note_lb(node.lower_bound)
            if best_node_lb is None or node.lower_bound > best_node_lb:
                best_node_lb = node.lower_bound
            if node.node_id == 0:
                root_lrmp = cg.lrmp_obj
                root_valid_lb = node.lower_bound
                root_cg_converged = getattr(cg, "converged", False)
                root_cg_stop = getattr(cg, "stop_reason", None)
            if not cg.converged:
                print(
                    f"  BP [{nodes_explored}/{self.max_nodes}] node {node.node_id} "
                    f"depth={node.depth} partial-CG LB={node.lower_bound:.2f} "
                    f"(raw={float(raw_lb):.2f}, {cg_sec:.1f}s)",
                    flush=True,
                )
                has_primal = False
                if cg._model is not None:
                    try:
                        if cg._model.SolCount <= 0:
                            cg._model.optimize()
                        has_primal = cg._model.SolCount > 0
                    except Exception:
                        has_primal = False
                if not has_primal:
                    nodes_cg_failed += 1
                    rem = (
                        (deadline - time.perf_counter())
                        if deadline is not None
                        else 0.0
                    )
                    max_retries = 2
                    if (
                        rem > 8.0
                        and open_heap is not None
                        and node.cg_fail_retries < max_retries
                    ):
                        node.cg_fail_retries += 1
                        prev = float(
                            node.cg_budget_override
                            if node.cg_budget_override is not None
                            else (node_cg_limit or 20.0)
                        )
                        node.cg_budget_override = min(
                            rem * 0.85, max(prev * 1.5, float(cg_sec) * 1.5, 60.0)
                        )
                        floor_lb = (
                            node.parent_lb
                            if node.parent_lb is not None
                            else node.lower_bound
                        )
                        self._push_heap(open_heap, node, float(floor_lb), defer=False)
                        print(
                            f"  BP [{nodes_explored}/{self.max_nodes}] node {node.node_id} "
                            f"partial-CG no primal; retry#{node.cg_fail_retries}/{max_retries} "
                            f"next_budget={node.cg_budget_override:.0f}s",
                            flush=True,
                        )
                    else:
                        print(
                            f"  BP [{nodes_explored}/{self.max_nodes}] node {node.node_id} "
                            f"partial-CG has no primal; abandon ({cg_sec:.1f}s)",
                            flush=True,
                        )
                    continue

            if incumbent is not None and node.lower_bound >= incumbent - INT_TOL:
                nodes_pruned += 1
                if self._note_gap_stop(
                    incumbent, open_heap, lazy_heap, dfs_stack, nodes_explored
                ):
                    gap_stopped = True
                continue

            # --- 上界：仅 CGFO + 整数 LRMP（见下方整数检测）---
            # 根节点：只做一次 CGFO
            if node.node_id == 0 and not root_ub_done and deadline is not None:
                root_ub_done = True
                remaining = deadline - time.perf_counter()
                if remaining > 12.0:
                    cgfo_budget = min(40.0, remaining * 0.12)
                    new_ub, new_fy = self._try_cgfo(cg, incumbent, cgfo_budget)
                    if new_fy is not None and new_ub is not None:
                        incumbent, best_fixed_y = new_ub, new_fy
                        print(
                            f"    [CGFO] incumbent={incumbent:.2f} (budget={cgfo_budget:.0f}s)",
                            flush=True,
                        )
                if incumbent is not None and node.lower_bound >= incumbent - INT_TOL:
                    nodes_pruned += 1
                    proved_optimal = True
                    print(
                        f"  BP root proved: LB={node.lower_bound:.2f} ≥ UB={incumbent:.2f}",
                        flush=True,
                    )
                    continue
                if (
                    getattr(cg, "converged", False)
                    and incumbent is not None
                    and cg.lrmp_obj is not None
                    and abs(incumbent - cg.lrmp_obj) <= INT_TOL
                ):
                    proved_optimal = True
                    print(
                        f"  BP root column-optimal: UB={incumbent:.2f} = LRMP "
                        f"(CG converged)",
                        flush=True,
                    )
                    continue
                if self._note_gap_stop(
                    incumbent, open_heap, lazy_heap, dfs_stack, nodes_explored,
                    current_lb=node.lower_bound,
                ):
                    gap_stopped = True
                    continue

            # 非根节点：有剩余时间则再试 CGFO
            if node.node_id != 0 and deadline is not None:
                remaining = deadline - time.perf_counter()
                if remaining > 20.0:
                    cgfo_budget = min(20.0, remaining * 0.08)
                    new_ub, new_fy = self._try_cgfo(
                        cg, incumbent, cgfo_budget, merge_fixed_y=node.fixed_y
                    )
                    if new_fy is not None and new_ub is not None:
                        incumbent, best_fixed_y = new_ub, new_fy
                        print(
                            f"    [CGFO node] incumbent={incumbent:.2f} "
                            f"(depth={node.depth}, budget={cgfo_budget:.0f}s)",
                            flush=True,
                        )

            if incumbent is not None and node.lower_bound >= incumbent - INT_TOL:
                nodes_pruned += 1
                if self._note_gap_stop(
                    incumbent, open_heap, lazy_heap, dfs_stack, nodes_explored
                ):
                    gap_stopped = True
                continue

            # 整数 LRMP：诱导 Y、Q、R 全整数 ⇒ 主问题目标即可行上界
            implied = cg.get_implied_y()
            if _implied_y_integer(implied) and _q_integer(cg) and _r_integer(cg):
                best_cg = cg
                int_cost = cg.lrmp_obj
                if int_cost is not None:
                    if incumbent is None or int_cost < incumbent - INT_TOL:
                        incumbent = int_cost
                        print(f"    [integer LRMP] incumbent={incumbent:.2f}", flush=True)
                    if abs(node.lower_bound - incumbent) <= INT_TOL:
                        proved_optimal = True
                    elif self._note_gap_stop(
                        incumbent, open_heap, lazy_heap, dfs_stack, nodes_explored
                    ):
                        gap_stopped = True
                continue

            dec = self.policy.pick(
                node.depth, cg, node.fixed_y, node.fixed_scrap, node.fixed_wip,
            )
            if self.branch_strategy == "ml" and dec is not None:
                dec = self._ml_sb_refine(node, cg, incumbent, deadline, dec)
            elif self.branch_strategy == "frac_sb" and dec is not None:
                dec = self._frac_sb_refine(node, cg, incumbent, deadline, dec)
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

            next_id, n_sides = self._create_children(
                node,
                dec,
                inherited,
                next_id,
                dfs_stack,
                lazy_heap,
                open_heap,
                stats,
            )

            # Memory management: cap heap sizes to prevent OOM
            if open_heap and len(open_heap) > 300:
                open_heap.sort(key=lambda x: x[0])  # best-bound first
                open_heap = open_heap[:300]
                heapq.heapify(open_heap)
            if lazy_heap and len(lazy_heap) > 300:
                lazy_heap.sort(key=lambda x: x[0])
                lazy_heap = lazy_heap[:300]
                heapq.heapify(lazy_heap)
            if nodes_explored % 20 == 0:
                gc.collect()

            if (
                incumbent is not None
                and not self._has_pending(dfs_stack, lazy_heap, open_heap)
                and nodes_cg_failed == 0
            ):
                proved_optimal = True
                continue

            # 全局下界 = 开节点 LB 最小；与当前最好可行解相对 gap ≤ mip_gap 则停
            if self._note_gap_stop(
                incumbent, open_heap, lazy_heap, dfs_stack, nodes_explored
            ):
                gap_stopped = True

        # 结束：保留搜索中得到的最好可行解上界（不再跑 final IM）
        runtime = time.perf_counter() - start
        pending = self._has_pending(dfs_stack, lazy_heap, open_heap)
        tree_lb = self._tree_dual_bound(open_heap, lazy_heap, dfs_stack)
        if tree_lb is None and root_valid_lb is not None:
            tree_lb = root_valid_lb
        # 开节点队列已空 + 全程无 CG 失败 ⇒ 可证优
        if proved_optimal or (
            incumbent is not None
            and not pending
            and not timed_out
            and not gap_stopped
            and nodes_explored > 0
            and nodes_cg_failed == 0
        ):
            status = "optimal"
            if incumbent is not None:
                tree_lb = incumbent
        elif gap_stopped:
            status = "gap_limit"
        elif timed_out:
            status = "time_limit"
        elif nodes_explored >= self.max_nodes:
            status = "max_nodes"
        elif incumbent is None:
            status = "no_feasible"  # 未找到可行解
        elif nodes_cg_failed > 0 and not pending:
            # 树已探完，但有节点 CG 失败，不能把上界当作已证最优
            status = "unproven_cg_fail"
        else:
            status = "stopped_early"

        return BranchAndPriceResult(
            status=status,
            objective=incumbent,
            lower_bound=tree_lb,
            root_lrmp=root_lrmp,
            root_valid_lb=root_valid_lb,
            root_cg_converged=root_cg_converged,
            root_cg_stop=root_cg_stop,
            best_node_lb=best_node_lb,
            incumbent=incumbent,
            nodes_explored=nodes_explored,
            nodes_pruned=nodes_pruned,
            nodes_infeasible=nodes_infeasible,
            nodes_cg_failed=nodes_cg_failed,
            max_depth=max_depth,
            runtime_sec=runtime,
            best_fixed_y=best_fixed_y,
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
    mip_gap: float = DEFAULT_BP_MIP_GAP,
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
            0.0 if result.status == "optimal" else
            (max(0.0, (result.objective - result.lower_bound) / result.objective * 100)
             if result.objective and result.lower_bound
             else None)
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
            "root_valid_lb": result.root_valid_lb,
            "root_cg_converged": result.root_cg_converged,
            "root_cg_stop": result.root_cg_stop,
            "best_node_lb": result.best_node_lb,
            "best_fixed_y_count": len(result.best_fixed_y),
            "y1_lazy_created": result.y1_lazy_created,
            "lazy_popped": result.lazy_popped,
            "dfs_popped": result.dfs_popped,
            "branch_strategy": branch_strategy,
        },
    )
