"""Branch-and-price 辅助：列兼容、分支策略与 setup/WIP/scrap 选点。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from common.data_models import ProcessedInstance, ScheduleColumn
from configuration import DEFAULT_BP_LB_STALL_EPS, DEFAULT_BP_Y0_DEPTH_BUDGET

if TYPE_CHECKING:
    from model_and_algo.column_generation import ColumnGenerationSolver

SetupFix = tuple[int, int, int]  # (u_idx, m_idx, t_idx) -> 0|1
WipFix = tuple[str, int]  # ('le'|'ge', bound) on implied window processing W_i
# Ryan–Foster pair (i,j) with i<j: 1 = same column, 0 = different columns
RfFix = int

FRAC_TOL = 1e-4
Y1_DEFER_PENALTY = 1e-4


@dataclass(frozen=True)
class BranchDecision:
    """单次分支决策。"""

    kind: str  # setup_y0 | wip | scrap | setup | ryan_foster
    key: SetupFix | int | tuple[int, int]
    value: float
    wip_fix: WipFix | None = None  # wip 分支：('le', k) 或 ('ge', k)


def machine_fixed_y(
    fixed_y: dict[SetupFix, int],
    machine_idx: int,
) -> dict[tuple[int, int], int]:
    out: dict[tuple[int, int], int] = {}
    for (u_idx, m_idx, t_idx), val in fixed_y.items():
        if m_idx == machine_idx:
            out[(u_idx, t_idx)] = val
    return out


def column_has_setup(col: ScheduleColumn, data: ProcessedInstance, u_idx: int, t_idx: int) -> bool:
    u_name = data.configs[u_idx]
    period = data.periods[t_idx]
    return col.y.get((u_name, period), 0) > 0.5


def column_compatible(
    col: ScheduleColumn,
    machine_idx: int,
    data: ProcessedInstance,
    fixed_y: dict[SetupFix, int],
) -> bool:
    if not fixed_y:
        return True
    is_empty = not col.y and not col.x
    for (u_idx, m_idx, t_idx), val in fixed_y.items():
        if m_idx != machine_idx:
            continue
        has_setup = column_has_setup(col, data, u_idx, t_idx)
        if val == 0 and has_setup:
            return False
        if val == 1 and not has_setup and not is_empty:
            return False
    return True


def column_item_indices(col: ScheduleColumn, data: ProcessedInstance) -> set[int]:
    items: set[int] = set()
    for (name, _period), qty in col.x.items():
        if qty > 0.5:
            try:
                items.add(data.cured_index[name])
            except KeyError:
                continue
    return items


def column_rf_compatible(
    col: ScheduleColumn,
    data: ProcessedInstance,
    fixed_rf: dict[tuple[int, int], int] | None,
) -> bool:
    """Ryan–Foster: same ⇒ 0 or 2 of {i,j}; differ ⇒ not both."""
    if not fixed_rf:
        return True
    items = column_item_indices(col, data)
    for (i, j), val in fixed_rf.items():
        has_i = i in items
        has_j = j in items
        if val == 0 and has_i and has_j:
            return False
        if val == 1 and has_i != has_j:
            return False
    return True


def filter_columns(
    columns: dict[str, list[ScheduleColumn]],
    data: ProcessedInstance,
    fixed_y: dict[SetupFix, int],
    fixed_rf: dict[tuple[int, int], int] | None = None,
) -> dict[str, list[ScheduleColumn]]:
    filtered: dict[str, list[ScheduleColumn]] = {}
    for m_idx, machine in enumerate(data.machines):
        filtered[machine] = [
            col
            for col in columns.get(machine, [])
            if column_compatible(col, m_idx, data, fixed_y)
            and column_rf_compatible(col, data, fixed_rf)
        ]
    return filtered


def pick_ryan_foster_branch(
    data: ProcessedInstance,
    cg: "ColumnGenerationSolver",
    fixed_rf: dict[tuple[int, int], int] | None = None,
) -> BranchDecision | None:
    """Pick the most fractional co-occurring item pair (Ryan–Foster)."""
    if cg._artifacts is None:
        return None
    fixes = fixed_rf or {}
    pair_score: dict[tuple[int, int], float] = {}
    try:
        q_vars = cg._artifacts["Q"]
    except Exception:
        return None
    for machine in data.machines:
        cols = cg.columns.get(machine, [])
        q_list = q_vars.get(machine, [])
        for col, qv in zip(cols, q_list):
            try:
                q = float(qv.X)
            except Exception:
                continue
            if q <= FRAC_TOL:
                continue
            items = sorted(column_item_indices(col, data))
            for a_idx in range(len(items)):
                for b_idx in range(a_idx + 1, len(items)):
                    pair = (items[a_idx], items[b_idx])
                    if pair in fixes:
                        continue
                    pair_score[pair] = pair_score.get(pair, 0.0) + q
    best: BranchDecision | None = None
    best_frac = 0.0
    for pair, cooccur in pair_score.items():
        # Closeness to 1/2: most fractional same/differ decision.
        frac = min(cooccur, max(0.0, 1.0 - cooccur))
        if frac <= FRAC_TOL:
            continue
        if frac > best_frac + 1e-12:
            best_frac = frac
            best = BranchDecision(kind="ryan_foster", key=pair, value=float(cooccur))
    return best


def copy_column_pool(columns: dict[str, list[ScheduleColumn]]) -> dict[str, list[ScheduleColumn]]:
    return {m: list(cols) for m, cols in columns.items()}


def prune_column_pool(
    columns: dict[str, list[ScheduleColumn]],
    q_vars: dict[str, list],
    tol: float = 1e-6,
) -> dict[str, list[ScheduleColumn]]:
    """裁剪列池：只保留权重 > tol 的列，减少子节点主问题规模。"""
    pruned: dict[str, list[ScheduleColumn]] = {}
    for m_idx, machine_cols in columns.items():
        q_list = q_vars.get(m_idx, [])
        keep: list[ScheduleColumn] = []
        for cidx, col in enumerate(machine_cols):
            try:
                w = q_list[cidx].X
            except Exception:
                w = 0.0
            if w > tol:
                keep.append(col)
        # 如果裁剪后为空，保留原列表（至少要有列才能做 LP）
        pruned[m_idx] = keep if keep else list(machine_cols)
    return pruned


def machine_fixed_y_feasible(
    data: ProcessedInstance,
    machine_idx: int,
    fixes: dict[tuple[int, int], int],
) -> bool:
    forced = sorted([(u, t) for (u, t), v in fixes.items() if v == 1], key=lambda x: x[1])
    last_end = -1
    num_t = len(data.periods)
    for u_idx, tau in forced:
        end = tau + data.l_u[u_idx] - 1
        if tau <= last_end or end >= num_t:
            return False
        last_end = end
    return True


def fixed_y_feasible(data: ProcessedInstance, fixed_y: dict[SetupFix, int]) -> bool:
    if not fixed_y:
        return True
    per_machine: dict[int, dict[tuple[int, int], int]] = {}
    for (u_idx, m_idx, t_idx), val in fixed_y.items():
        per_machine.setdefault(m_idx, {})[(u_idx, t_idx)] = val
    for m_idx, fixes in per_machine.items():
        if not machine_fixed_y_feasible(data, m_idx, fixes):
            return False
    return True


def scrap_fix_feasible(data: ProcessedInstance, fixed_scrap: dict[int, int]) -> bool:
    for i_idx, val in fixed_scrap.items():
        if val == 1 and data.w_i[i_idx] < 1:
            return False
    return True


def wip_fix_feasible(data: ProcessedInstance, fixed_wip: dict[int, WipFix]) -> bool:
    for i_idx, (sense, bound) in fixed_wip.items():
        w = data.w_i[i_idx]
        if sense == "le" and bound < 0:
            return False
        if sense == "ge" and bound > w:
            return False
        # 时间窗约束：每个 item 最多在 ceil(window_len / lead_time) 个批次中出现
        if sense == "ge" and bound > 0:
            e_i = data.earliest_starts[i_idx]
            d_i = data.deadlines[i_idx]
            l_i = max(1, int(data.l_ti[i_idx]))
            window_len = d_i - e_i + 1
            max_batches = int(math.ceil(window_len / float(l_i)))
            if bound > max_batches:
                return False
    return True


def _frac_part(v: float) -> float:
    return abs(v - round(v))


def get_rho_duals(cg: ColumnGenerationSolver) -> list[float]:
    """WIP dual magnitudes; safe if LP has no duals after cuts / status change."""
    n = len(cg.data.cured_items)
    if cg._artifacts is None:
        return [0.0] * n
    if hasattr(cg, "ensure_master_lp"):
        cg.ensure_master_lp()
    wip = cg._artifacts.get("wip_balance")
    if not wip:
        return [0.0] * n
    out: list[float] = []
    for i in range(n):
        try:
            out.append(abs(float(wip[i].Pi)))
        except Exception:
            out.append(0.0)
    return out


def _has_fractional_y_above_half(implied_y: dict[SetupFix, float]) -> bool:
    for val in implied_y.values():
        if val > 0.5 + FRAC_TOL and not _is_int(val):
            return True
    return False


def _has_fractional_y(implied_y: dict[SetupFix, float]) -> bool:
    for val in implied_y.values():
        if val > FRAC_TOL and not _is_int(val):
            return True
    return False


def _is_int(v: float, tol: float = 1e-5) -> bool:
    return abs(v - round(v)) <= tol


def pick_setup_y0_branch(
    data: ProcessedInstance,
    implied_y: dict[SetupFix, float],
    fixed_y: dict[SetupFix, int] | None,
    rho_duals: list[float] | None = None,
) -> BranchDecision | None:
    """Layer 1：仅选 Y=0 分支候选（implied Y in (0.5, 1)）。"""
    fixes = fixed_y or {}
    fixed_mt: dict[int, set[int]] = {}
    for _u, m_idx, t_idx in fixes:
        fixed_mt.setdefault(m_idx, set()).add(t_idx)
    rhos = rho_duals or [0.0] * len(data.cured_items)

    best: BranchDecision | None = None
    best_score = FRAC_TOL
    for (u_idx, m_idx, t_idx), val in implied_y.items():
        if (u_idx, m_idx, t_idx) in fixes:
            continue
        if val <= 0.5 + FRAC_TOL or _is_int(val):
            continue
        frac = _frac_part(val)

        penalty = 1.0
        for ft in fixed_mt.get(m_idx, ()):
            if ft == t_idx:
                penalty *= 0.15
            elif abs(ft - t_idx) < data.l_u[u_idx]:
                penalty *= 0.4

        period = data.periods[t_idx]
        rho_sum = 0.0
        window_items = 0
        urgency = 0.0
        for i_idx in data.items_by_config[u_idx]:
            e_i = data.earliest_starts[i_idx]
            d_i = data.deadlines[i_idx]
            if e_i <= period <= d_i:
                window_items += 1
                rho_sum += rhos[i_idx]
                urgency = max(urgency, 1.0 / max(1.0, float(d_i - period + 1)))

        window_factor = 1.0 + 0.06 * window_items + 0.02 * rho_sum
        setup_cost = data.p_cum[m_idx][u_idx]
        score = (
            frac
            * window_factor
            * (1.0 + urgency)
            * (1.0 + 1e-4 * setup_cost)
            * penalty
        )
        if score > best_score:
            best_score = score
            best = BranchDecision(kind="setup_y0", key=(u_idx, m_idx, t_idx), value=val)
    return best


def pick_wip_branch(
    data: ProcessedInstance,
    wip_values: dict[int, float],
    fixed_wip: dict[int, WipFix] | None = None,
    rho_duals: list[float] | None = None,
) -> BranchDecision | None:
    """Layer 2：窗内加工量 W_i 分支（W <= floor vs W >= ceil）。"""
    fixes = fixed_wip or {}
    rhos = rho_duals or [0.0] * len(data.cured_items)
    best: BranchDecision | None = None
    best_score = FRAC_TOL

    for i_idx, w_val in wip_values.items():
        if i_idx in fixes:
            continue
        if w_val <= FRAC_TOL and data.w_i[i_idx] <= FRAC_TOL:
            continue
        frac = _frac_part(w_val)
        if frac <= FRAC_TOL:
            continue
        e_i = data.earliest_starts[i_idx]
        d_i = data.deadlines[i_idx]
        window_len = max(1.0, float(d_i - e_i + 1))
        score = frac * (1.0 + 0.1 * rhos[i_idx]) * (1.0 + 0.12 / window_len)
        if score > best_score:
            best_score = score
            k = int(w_val // 1)
            if w_val - k > FRAC_TOL:
                best = BranchDecision(
                    kind="wip",
                    key=i_idx,
                    value=w_val,
                    wip_fix=("le", k),
                )
    return best


def pick_scrap_branch(
    data: ProcessedInstance,
    r_values: dict[int, float],
    fixed_scrap: dict[int, int] | None = None,
) -> BranchDecision | None:
    """Layer 2：分数 R_i，eager 侧仅 R>=1。"""
    fixes = fixed_scrap or {}
    best: BranchDecision | None = None
    best_score = FRAC_TOL
    for i_idx, rval in r_values.items():
        if i_idx in fixes:
            continue
        if rval <= FRAC_TOL:
            continue
        frac = _frac_part(rval)
        if frac <= FRAC_TOL:
            continue
        e_i = data.earliest_starts[i_idx]
        d_i = data.deadlines[i_idx]
        window_len = max(1.0, float(d_i - e_i + 1))
        score = frac * (1.0 + 1e-4 * data.sc_i[i_idx]) * (1.0 + 0.15 / window_len)
        if score > best_score:
            best_score = score
            best = BranchDecision(kind="scrap", key=i_idx, value=rval)
    return best


def pick_branch_old_hybrid(
    data: ProcessedInstance,
    implied_y: dict[SetupFix, float],
    r_values: dict[int, float],
    fixed_y: dict[SetupFix, int] | None = None,
    fixed_scrap: dict[int, int] | None = None,
) -> BranchDecision | None:
    """old_hybrid：scrap 优先 + 对称 setup Y。"""
    scrap = pick_scrap_branch(data, r_values, fixed_scrap)
    if scrap is not None:
        return scrap
    setup = pick_setup_y0_branch(data, implied_y, fixed_y)
    if setup is not None:
        return BranchDecision(kind="setup", key=setup.key, value=setup.value)
    return None


class BranchPolicy:
    """三层分支策略：Y=0 深搜 / WIP+scrap / lazy Y=1 & R=0；可选纯 ML 排序。"""

    def __init__(
        self,
        data: ProcessedInstance,
        strategy: str = "new_policy",
        y0_depth_budget: int = DEFAULT_BP_Y0_DEPTH_BUDGET,
        lb_stall_eps: float = DEFAULT_BP_LB_STALL_EPS,
    ) -> None:
        self.data = data
        self.strategy = strategy
        self.y0_depth_budget = y0_depth_budget
        self.lb_stall_eps = lb_stall_eps
        self._last_lb: float | None = None
        self._ml_selector: Any = None

    def note_lb(self, lb: float) -> None:
        self._last_lb = lb

    def lb_stalled(self, current_lb: float) -> bool:
        if self._last_lb is None:
            return False
        return current_lb - self._last_lb < self.lb_stall_eps

    def _get_ml_selector(self) -> Any:
        if self._ml_selector is None:
            from configuration import ML_BRANCH_MODEL_PATH
            from model_and_algo.ml_branch_selector import MLBranchSelector

            self._ml_selector = MLBranchSelector(model_path=ML_BRANCH_MODEL_PATH)
        return self._ml_selector

    def _pick_heuristic(
        self,
        depth: int,
        cg: ColumnGenerationSolver,
        fixed_y: dict[SetupFix, int],
        fixed_scrap: dict[int, int],
        fixed_wip: dict[int, WipFix],
    ) -> BranchDecision | None:
        implied = cg.get_implied_y()
        r_vals = cg.get_implied_r()
        wip_vals = cg.get_implied_wip()
        rhos = get_rho_duals(cg)

        if self.strategy == "baseline_y0_dfs":
            return pick_setup_y0_branch(self.data, implied, fixed_y, rhos)

        if self.strategy == "old_hybrid":
            return pick_branch_old_hybrid(
                self.data, implied, r_vals, fixed_y, fixed_scrap
            )

        if _has_fractional_y_above_half(implied) and depth < self.y0_depth_budget:
            dec = pick_setup_y0_branch(self.data, implied, fixed_y, rhos)
            if dec is not None:
                return dec

        current_lb = cg.lrmp_obj or 0.0
        if self.lb_stalled(current_lb) or depth >= self.y0_depth_budget:
            wip = pick_wip_branch(self.data, wip_vals, fixed_wip, rhos)
            if wip is not None:
                return wip
            scrap = pick_scrap_branch(self.data, r_vals, fixed_scrap)
            if scrap is not None:
                return scrap

        if _has_fractional_y(implied):
            return pick_setup_y0_branch(self.data, implied, fixed_y, rhos)
        return None

    def pick(
        self,
        depth: int,
        cg: ColumnGenerationSolver,
        fixed_y: dict[SetupFix, int],
        fixed_scrap: dict[int, int],
        fixed_wip: dict[int, WipFix],
    ) -> BranchDecision | None:
        if self.strategy == "ml":
            try:
                # Cuts / RMP-IP can leave the master without duals; repair first.
                if hasattr(cg, "ensure_master_lp"):
                    cg.ensure_master_lp()
                from model_and_algo.ml_branch_selector import pick_ml_decision

                sel = self._get_ml_selector()
                if sel.ready:
                    dec = pick_ml_decision(
                        self.data, cg, fixed_y, fixed_scrap, fixed_wip, sel
                    )
                    if dec is not None:
                        return dec
            except Exception as exc:
                print(f"    [ml-branch] fallback to heuristic ({exc})", flush=True)
            # fallback: same layering as best_bound
            return self._pick_heuristic(
                depth, cg, fixed_y, fixed_scrap, fixed_wip
            )

        if self.strategy == "frac_sb":
            try:
                if hasattr(cg, "ensure_master_lp"):
                    cg.ensure_master_lp()
                from model_and_algo.ml_branch_selector import pick_frac_decision

                dec = pick_frac_decision(
                    self.data, cg, fixed_y, fixed_scrap, fixed_wip
                )
                if dec is not None:
                    return dec
            except Exception as exc:
                print(f"    [frac-sb] fallback to heuristic ({exc})", flush=True)
            return self._pick_heuristic(
                depth, cg, fixed_y, fixed_scrap, fixed_wip
            )

        if self.strategy == "ryan_foster":
            try:
                if hasattr(cg, "ensure_master_lp"):
                    cg.ensure_master_lp()
                fixed_rf = getattr(cg, "fixed_rf", {}) or {}
                dec = pick_ryan_foster_branch(self.data, cg, fixed_rf)
                if dec is not None:
                    return dec
                from model_and_algo.ml_branch_selector import pick_frac_decision

                # No fractional pair left: fall back to setup/WIP/scrap.
                dec = pick_frac_decision(
                    self.data, cg, fixed_y, fixed_scrap, fixed_wip
                )
                if dec is not None:
                    return dec
            except Exception as exc:
                print(f"    [ryan-foster] fallback to heuristic ({exc})", flush=True)
            return self._pick_heuristic(
                depth, cg, fixed_y, fixed_scrap, fixed_wip
            )

        return self._pick_heuristic(depth, cg, fixed_y, fixed_scrap, fixed_wip)


def machine_needs_setup_seed(
    data: ProcessedInstance,
    machine_idx: int,
    columns: list[ScheduleColumn],
    fixed_y: dict[SetupFix, int],
) -> dict[tuple[int, int], int] | None:
    fixes = machine_fixed_y(fixed_y, machine_idx)
    forced = [(u, t) for (u, t), v in fixes.items() if v == 1]
    if not forced:
        return None
    for u_idx, t_idx in forced:
        if not any(column_has_setup(col, data, u_idx, t_idx) for col in columns):
            return fixes
    return None
