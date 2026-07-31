"""Pure-ML branch ranking: scale-free features + LightGBM LambdaRank.

Candidate pool: frac Top-K. Features (v5): frac, type one-hot, dual_share,
dens_rel, prox_pen.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from common.data_models import ProcessedInstance
from model_and_algo.branch_utils import (
    FRAC_TOL,
    BranchDecision,
    SetupFix,
    WipFix,
    _frac_part,
    _is_int,
)

FEATURE_NAMES: tuple[str, ...] = (
    "frac",
    "is_y",
    "is_wip",
    "is_scrap",
    "dual_share",
    "dens_rel",
    "prox_pen",
    "dual_rank",
    "frac_rank",
)
SCHEMA_VERSION = 6
TOP_K_DEFAULT = 20


def _frac_score(val: float) -> float:
    """How fractional: min(f, 1-f); larger ⇒ closer to 0.5."""
    if _is_int(val):
        return 0.0
    f = _frac_part(val)
    return float(min(f, 1.0 - f))


@dataclass
class MLBranchCandidate:
    decision: BranchDecision
    h_score: float
    dual: float
    dens: float
    urgency: float
    cost: float
    prox_pen: float
    t_pos: float
    win_frac: float
    features: list[float] | None = None


def _window_stats(
    data: ProcessedInstance,
    u_idx: int,
    t_idx: int,
    rhos: list[float],
) -> tuple[float, float, float]:
    period = data.periods[t_idx]
    dens = 0.0
    rho_sum = 0.0
    urgency = 0.0
    for i_idx in data.items_by_config[u_idx]:
        e_i = data.earliest_starts[i_idx]
        d_i = data.deadlines[i_idx]
        if e_i <= period <= d_i:
            dens += 1.0
            rho_sum += rhos[i_idx]
            urgency = max(urgency, 1.0 / max(1.0, float(d_i - period + 1)))
    return dens, rho_sum, urgency


def enumerate_setup_candidates(
    data: ProcessedInstance,
    implied_y: dict[SetupFix, float],
    fixed_y: dict[SetupFix, int] | None,
    rho_duals: list[float] | None = None,
) -> list[MLBranchCandidate]:
    fixes = fixed_y or {}
    fixed_mt: dict[int, set[int]] = {}
    for _u, m_idx, t_idx in fixes:
        fixed_mt.setdefault(m_idx, set()).add(t_idx)
    rhos = rho_duals or [0.0] * len(data.cured_items)
    T = max(1, len(data.periods))
    out: list[MLBranchCandidate] = []
    for (u_idx, m_idx, t_idx), val in implied_y.items():
        if (u_idx, m_idx, t_idx) in fixes:
            continue
        if _is_int(val) or val <= FRAC_TOL:
            continue
        frac = _frac_part(val)
        if frac <= FRAC_TOL:
            continue
        prox = 1.0
        for ft in fixed_mt.get(m_idx, ()):
            if ft == t_idx:
                prox *= 0.15
            elif abs(ft - t_idx) < data.l_u[u_idx]:
                prox *= 0.4
        dens, rho_sum, urgency = _window_stats(data, u_idx, t_idx, rhos)
        setup_cost = float(data.p_cum[m_idx][u_idx])
        win_len = float(data.l_u[u_idx])
        h = (
            frac
            * (1.0 + 0.06 * dens + 0.02 * rho_sum)
            * (1.0 + urgency)
            * (1.0 + 1e-4 * setup_cost)
            * prox
        )
        out.append(
            MLBranchCandidate(
                decision=BranchDecision(kind="setup_y0", key=(u_idx, m_idx, t_idx), value=val),
                h_score=h,
                dual=abs(rho_sum),
                dens=dens,
                urgency=urgency,
                cost=setup_cost,
                prox_pen=1.0 - min(1.0, prox),
                t_pos=t_idx / T,
                win_frac=min(1.0, win_len / T),
            )
        )
    return out


def enumerate_wip_candidates(
    data: ProcessedInstance,
    wip_values: dict[int, float],
    fixed_wip: dict[int, WipFix] | None = None,
    rho_duals: list[float] | None = None,
) -> list[MLBranchCandidate]:
    fixes = fixed_wip or {}
    rhos = rho_duals or [0.0] * len(data.cured_items)
    T = max(1, len(data.periods))
    out: list[MLBranchCandidate] = []
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
        mid = 0.5 * (float(e_i) + float(d_i))
        rho = abs(rhos[i_idx])
        h = frac * (1.0 + 0.1 * rho) * (1.0 + 0.12 / window_len)
        k = int(w_val // 1)
        if w_val - k <= FRAC_TOL:
            continue
        out.append(
            MLBranchCandidate(
                decision=BranchDecision(
                    kind="wip", key=i_idx, value=w_val, wip_fix=("le", k)
                ),
                h_score=h,
                dual=rho,
                dens=1.0 / window_len,
                urgency=1.0 / window_len,
                cost=0.0,
                prox_pen=0.0,
                t_pos=mid / T,
                win_frac=min(1.0, window_len / T),
            )
        )
    return out


def enumerate_scrap_candidates(
    data: ProcessedInstance,
    r_values: dict[int, float],
    fixed_scrap: dict[int, int] | None = None,
) -> list[MLBranchCandidate]:
    fixes = fixed_scrap or {}
    T = max(1, len(data.periods))
    out: list[MLBranchCandidate] = []
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
        mid = 0.5 * (float(e_i) + float(d_i))
        cost = float(data.sc_i[i_idx])
        h = frac * (1.0 + 1e-4 * cost) * (1.0 + 0.15 / window_len)
        out.append(
            MLBranchCandidate(
                decision=BranchDecision(kind="scrap", key=i_idx, value=rval),
                h_score=h,
                dual=0.0,
                dens=1.0 / window_len,
                urgency=1.0 / window_len,
                cost=cost,
                prox_pen=0.0,
                t_pos=mid / T,
                win_frac=min(1.0, window_len / T),
            )
        )
    return out


def enumerate_all_candidates(
    data: ProcessedInstance,
    cg: Any,
    fixed_y: dict[SetupFix, int],
    fixed_scrap: dict[int, int],
    fixed_wip: dict[int, WipFix],
    top_k: int = TOP_K_DEFAULT,
) -> list[MLBranchCandidate]:
    from model_and_algo.branch_utils import get_rho_duals

    rhos = get_rho_duals(cg)
    cands = (
        enumerate_setup_candidates(data, cg.get_implied_y(), fixed_y, rhos)
        + enumerate_wip_candidates(data, cg.get_implied_wip(), fixed_wip, rhos)
        + enumerate_scrap_candidates(data, cg.get_implied_r(), fixed_scrap)
    )
    if not cands:
        return []
    # Prefilter: Top-K by fractional distance (not composite heuristic h).
    cands.sort(key=lambda c: _frac_score(c.decision.value), reverse=True)
    cands = cands[: max(1, top_k)]
    _assign_features(cands)
    return cands


def _assign_features(cands: Sequence[MLBranchCandidate]) -> None:
    """Features v6: v5 + within-pool dual/frac ranks."""
    dual_sum = sum(c.dual for c in cands) + 1e-12
    dens_max = max((c.dens for c in cands), default=0.0) + 1e-12
    n = max(1, len(cands) - 1)
    dual_order = sorted(range(len(cands)), key=lambda i: cands[i].dual, reverse=True)
    frac_order = sorted(
        range(len(cands)),
        key=lambda i: _frac_score(cands[i].decision.value),
        reverse=True,
    )
    dual_rank = [0.0] * len(cands)
    frac_rank = [0.0] * len(cands)
    for r, i in enumerate(dual_order):
        dual_rank[i] = 1.0 - r / n
    for r, i in enumerate(frac_order):
        frac_rank[i] = 1.0 - r / n
    for i, c in enumerate(cands):
        kind = c.decision.kind
        c.features = [
            float(_frac_score(c.decision.value)),
            1.0 if kind == "setup_y0" else 0.0,
            1.0 if kind == "wip" else 0.0,
            1.0 if kind == "scrap" else 0.0,
            float(c.dual / dual_sum),
            float(c.dens / dens_max),
            float(min(1.0, max(0.0, c.prox_pen))),
            float(dual_rank[i]),
            float(frac_rank[i]),
        ]


def sb_label(
    parent_lb: float,
    lb_l: float | None,
    lb_r: float | None,
    incumbent: float | None,
    infeas_bonus: float = 1.0,
) -> float | None:
    if lb_l is None or lb_r is None:
        return None
    gap = max((incumbent or parent_lb * 1.05) - parent_lb, 1e-3)
    d_l = infeas_bonus * gap if lb_l == float("inf") else max(0.0, lb_l - parent_lb)
    d_r = infeas_bonus * gap if lb_r == float("inf") else max(0.0, lb_r - parent_lb)
    d_min = min(d_l, d_r)
    d_geo = math.sqrt(max(d_l, 0.0) * max(d_r, 0.0))
    return (0.6 * d_min + 0.4 * d_geo) / gap


def relevance_from_scores(scores: Sequence[float]) -> list[int]:
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    rel = [0] * len(scores)
    for rank, i in enumerate(order):
        rel[i] = len(scores) - 1 - rank
    return rel


class MLBranchSelector:
    def __init__(
        self,
        model_path: str | Path | None = None,
        collect_dir: str | Path | None = None,
        top_k: int = TOP_K_DEFAULT,
    ) -> None:
        self.model_path = Path(model_path) if model_path else None
        self.collect_dir = Path(collect_dir) if collect_dir else None
        self.top_k = top_k
        self._booster: Any = None
        if self.model_path and self.model_path.exists():
            self.load(self.model_path)

    def load(self, path: str | Path) -> None:
        import lightgbm as lgb

        self.model_path = Path(path)
        self._booster = lgb.Booster(model_file=str(path))

    @property
    def ready(self) -> bool:
        return self._booster is not None

    def collect_candidates(
        self,
        data: ProcessedInstance,
        cg: Any,
        fixed_y: dict[SetupFix, int],
        fixed_scrap: dict[int, int],
        fixed_wip: dict[int, WipFix],
    ) -> list[MLBranchCandidate]:
        return enumerate_all_candidates(
            data, cg, fixed_y, fixed_scrap, fixed_wip, top_k=self.top_k
        )

    def predict_ranked(self, cands: Sequence[MLBranchCandidate]) -> list[MLBranchCandidate]:
        """Return candidates sorted by ML score (desc); falls back to h_score."""
        if not cands:
            return []
        if self._booster is None:
            return sorted(cands, key=lambda c: c.h_score, reverse=True)
        X = np.asarray([c.features for c in cands], dtype=np.float64)
        scores = self._booster.predict(X)
        order = np.argsort(-scores)
        return [cands[int(i)] for i in order]

    def predict_best(self, cands: Sequence[MLBranchCandidate]) -> BranchDecision | None:
        ranked = self.predict_ranked(cands)
        return ranked[0].decision if ranked else None

    def append_group_csv(
        self,
        path: Path,
        query_id: str,
        cands: Sequence[MLBranchCandidate],
        labels: Sequence[float],
        extra: dict[str, Any] | None = None,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(
                    ["query_id", "kind", "label", *FEATURE_NAMES]
                    + (list(extra.keys()) if extra else [])
                )
            for c, y in zip(cands, labels):
                assert c.features is not None
                row = [query_id, c.decision.kind, float(y), *c.features]
                if extra:
                    row.extend(extra[k] for k in extra)
                w.writerow(row)

    @staticmethod
    def load_xy_groups(
        data_dir: str | Path,
    ) -> tuple[np.ndarray, np.ndarray, list[int]]:
        data_dir = Path(data_dir)
        rows: list[tuple[str, float, list[float]]] = []
        for fp in sorted(data_dir.glob("*.csv")):
            with fp.open(encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    feats = [float(r[n]) for n in FEATURE_NAMES]
                    rows.append((r["query_id"], float(r["label"]), feats))
        if not rows:
            raise FileNotFoundError(f"no CSV samples in {data_dir}")
        # group consecutive same query after sort
        rows.sort(key=lambda t: t[0])
        X_list: list[list[float]] = []
        y_cont: list[float] = []
        groups: list[int] = []
        i = 0
        while i < len(rows):
            q = rows[i][0]
            j = i
            while j < len(rows) and rows[j][0] == q:
                j += 1
            chunk = rows[i:j]
            if len(chunk) < 2:
                i = j
                continue
            scores = [c[1] for c in chunk]
            rel = relevance_from_scores(scores)
            for k, c in enumerate(chunk):
                X_list.append(c[2])
                y_cont.append(float(rel[k]))
            groups.append(len(chunk))
            i = j
        if not groups:
            raise RuntimeError("no valid listwise groups (>=2 cands)")
        return np.asarray(X_list, dtype=np.float64), np.asarray(y_cont, dtype=np.float64), groups

    def train_from_dir(
        self,
        data_dir: str | Path,
        out_path: str | Path | None = None,
        num_boost_round: int = 200,
        learning_rate: float = 0.05,
        init_model: Any | str | Path | None = None,
    ) -> Path:
        import lightgbm as lgb

        X, y, group = self.load_xy_groups(data_dir)
        out = Path(out_path or self.model_path or "ml_branch_model.txt")
        dtrain = lgb.Dataset(X, label=y, group=group, feature_name=list(FEATURE_NAMES))
        params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_at": [1, 3, 5],
            "learning_rate": learning_rate,
            "num_leaves": 31,
            "min_data_in_leaf": 20,
            "feature_fraction": 0.9,
            "verbosity": -1,
            "label_gain": list(range(0, max(32, int(y.max()) + 2))),
        }
        init = None
        if init_model is not None:
            if isinstance(init_model, (str, Path)):
                init = lgb.Booster(model_file=str(init_model))
            else:
                init = init_model
        booster = lgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            init_model=init,
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        booster.save_model(str(out))
        schema = {
            "version": SCHEMA_VERSION,
            "features": list(FEATURE_NAMES),
            "objective": "lambdarank",
        }
        schema_path = out.with_suffix(".schema.json")
        schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")
        self._booster = booster
        self.model_path = out
        return out


def apply_decision_side(
    dec: BranchDecision,
    fixed_y: dict[SetupFix, int],
    fixed_scrap: dict[int, int],
    fixed_wip: dict[int, WipFix],
    side: str,
) -> tuple[dict[SetupFix, int], dict[int, int], dict[int, WipFix]]:
    """Apply left (L) / right (R) child fixing for a branch decision."""
    fy = dict(fixed_y)
    fs = dict(fixed_scrap)
    fw = dict(fixed_wip)
    if dec.kind in ("setup_y0", "setup"):
        fy[dec.key] = 0 if side == "L" else 1  # type: ignore[index]
    elif dec.kind == "scrap":
        fs[int(dec.key)] = 1 if side == "L" else 0
    elif dec.kind == "wip":
        i = int(dec.key)
        k = int(dec.wip_fix[1]) if dec.wip_fix else int(dec.value // 1)
        fw[i] = ("le", k) if side == "L" else ("ge", k + 1)
    return fy, fs, fw


def sb_y_min(parent_lb: float, lb_l: float | None, lb_r: float | None) -> float:
    """Strong-branch score: min(ΔL, ΔR); missing side treated as large lift."""
    gap_floor = max(abs(parent_lb) * 0.05, 1.0)
    d_l = gap_floor if lb_l is None else max(0.0, float(lb_l) - parent_lb)
    d_r = gap_floor if lb_r is None else max(0.0, float(lb_r) - parent_lb)
    return min(d_l, d_r)


def pick_ml_decision(
    data: ProcessedInstance,
    cg: Any,
    fixed_y: dict[SetupFix, int],
    fixed_scrap: dict[int, int],
    fixed_wip: dict[int, WipFix],
    selector: MLBranchSelector,
) -> BranchDecision | None:
    cands = selector.collect_candidates(data, cg, fixed_y, fixed_scrap, fixed_wip)
    return selector.predict_best(cands)


def pick_ml_ranked(
    data: ProcessedInstance,
    cg: Any,
    fixed_y: dict[SetupFix, int],
    fixed_scrap: dict[int, int],
    fixed_wip: dict[int, WipFix],
    selector: MLBranchSelector,
) -> list[MLBranchCandidate]:
    cands = selector.collect_candidates(data, cg, fixed_y, fixed_scrap, fixed_wip)
    return selector.predict_ranked(cands)


def pick_frac_ranked(
    data: ProcessedInstance,
    cg: Any,
    fixed_y: dict[SetupFix, int],
    fixed_scrap: dict[int, int],
    fixed_wip: dict[int, WipFix],
    top_k: int = TOP_K_DEFAULT,
) -> list[MLBranchCandidate]:
    """Frac Top-K pool, already sorted by frac descending (no ML)."""
    cands = enumerate_all_candidates(
        data, cg, fixed_y, fixed_scrap, fixed_wip, top_k=top_k
    )
    cands.sort(key=lambda c: _frac_score(c.decision.value), reverse=True)
    return cands


def pick_frac_decision(
    data: ProcessedInstance,
    cg: Any,
    fixed_y: dict[SetupFix, int],
    fixed_scrap: dict[int, int],
    fixed_wip: dict[int, WipFix],
    top_k: int = TOP_K_DEFAULT,
) -> BranchDecision | None:
    ranked = pick_frac_ranked(data, cg, fixed_y, fixed_scrap, fixed_wip, top_k=top_k)
    return ranked[0].decision if ranked else None
