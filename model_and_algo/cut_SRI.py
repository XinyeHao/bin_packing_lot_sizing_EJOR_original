"""Subset-row inequalities (SRI) for the schedule-column master.

Odd-cardinality rank-1 cuts with k=2:

    sum_{m,k} floor(|C_mk ∩ S| / 2) Q_mk  <=  floor(|S| / 2).

Even |S| is redundant under unit WIP (see analysis); separation therefore
enumerates odd sets only.  Coefficients depend only on a column's processed
item set, so the same cut remains valid for every newly priced column.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING, Any, Iterable, Sequence

if TYPE_CHECKING:
    from common.data_models import ProcessedInstance, ScheduleColumn


@dataclass
class SRICut:
    """An odd-cardinality rank-1 subset-row inequality.

    ``constr`` is populated by the master and intentionally excluded from the
    mathematical definition of a cut.
    """

    items: tuple[int, ...]
    item_mask: int
    rhs: int
    name: str
    constr: Any | None = None


def item_mask_from_column(data: "ProcessedInstance", col: "ScheduleColumn") -> int:
    """Return the global item bit-mask processed by a schedule column."""
    mask = 0
    for (name, _period), qty in col.x.items():
        if qty <= 0:
            continue
        try:
            i_idx = data.cured_index[name]
        except KeyError:
            continue
        mask |= 1 << i_idx
    return mask


def coefficient_from_mask(column_mask: int, cut: SRICut) -> int:
    """Coefficient floor(|C intersect S| / 2) of one schedule column."""
    return ((column_mask & cut.item_mask).bit_count()) // 2


def coefficient(data: "ProcessedInstance", col: "ScheduleColumn", cut: SRICut) -> int:
    return coefficient_from_mask(item_mask_from_column(data, col), cut)


def transition_coefficient(used_mask: int, batch_mask: int, cut: SRICut) -> int:
    """Increase in the SRI coefficient after appending one EBRP batch."""
    before = ((used_mask & cut.item_mask).bit_count()) // 2
    after = (((used_mask | batch_mask) & cut.item_mask).bit_count()) // 2
    return after - before


def _normalize_cardinalities(cardinalities: Sequence[int] | None) -> tuple[int, ...]:
    if not cardinalities:
        return (3,)
    out = sorted({int(r) for r in cardinalities if int(r) >= 3 and int(r) % 2 == 1})
    if not out:
        raise ValueError(f"SRI cardinalities must contain an odd integer >= 3, got {cardinalities!r}")
    return tuple(out)


def separate_root_odd_sets(
    data: "ProcessedInstance",
    columns: dict[str, list["ScheduleColumn"]],
    q_values: dict[str, list[float]],
    *,
    existing_masks: set[int] | None = None,
    max_cuts: int = 8,
    violation_tol: float = 1e-6,
    cardinalities: Sequence[int] = (3, 5, 7),
    cand_max: int = 16,
) -> list[SRICut]:
    """Separate the most violated odd-cardinality SRIs at the root.

    Candidates are items that appear in fractional columns, truncated to the
    ``cand_max`` highest fractional scores so that C(cand,7) stays tractable.
    """
    existing_masks = existing_masks or set()
    ranks = _normalize_cardinalities(cardinalities)
    max_r = max(ranks)

    weighted_masks: list[tuple[int, float]] = []
    score: dict[int, float] = {}
    for machine, cols in columns.items():
        vals = q_values.get(machine, [])
        for col, q_val in zip(cols, vals):
            q = float(q_val)
            if q <= violation_tol or q >= 1.0 - violation_tol:
                continue
            col_mask = item_mask_from_column(data, col)
            if not col_mask:
                continue
            weighted_masks.append((col_mask, q))
            bit = col_mask
            while bit:
                low = bit & -bit
                i = low.bit_length() - 1
                score[i] = score.get(i, 0.0) + q
                bit ^= low

    if len(score) < min(ranks) or not weighted_masks:
        return []

    # Prefer items with coverage closest to 1/2 (most fractional), then raw score.
    def _item_key(i: int) -> tuple[float, float]:
        s = score[i]
        return (-min(s, 1.0 - s) if 0.0 < s < 1.0 else 0.0, -s)

    cands = sorted(score.keys(), key=_item_key)[: max(max_r, int(cand_max))]
    if len(cands) < min(ranks):
        return []

    violated: list[tuple[float, SRICut]] = []
    for r in ranks:
        if len(cands) < r:
            continue
        rhs = r // 2
        for items in combinations(sorted(cands), r):
            item_mask = 0
            for i in items:
                item_mask |= 1 << i
            if item_mask in existing_masks:
                continue
            lhs = 0.0
            for col_mask, q in weighted_masks:
                coeff = (col_mask & item_mask).bit_count() // 2
                if coeff:
                    lhs += coeff * q
            violation = lhs - rhs
            if violation > violation_tol:
                label = "_".join(map(str, items))
                violated.append(
                    (violation, SRICut(tuple(items), item_mask, rhs, f"sri{r}_{label}"))
                )

    violated.sort(key=lambda pair: pair[0], reverse=True)
    return [cut for _violation, cut in violated[:max_cuts]]


def separate_root_triples(
    data: "ProcessedInstance",
    columns: dict[str, list["ScheduleColumn"]],
    q_values: dict[str, list[float]],
    *,
    existing_masks: set[int] | None = None,
    max_cuts: int = 3,
    violation_tol: float = 1e-6,
) -> list[SRICut]:
    """Backward-compatible wrapper: odd triples only."""
    return separate_root_odd_sets(
        data,
        columns,
        q_values,
        existing_masks=existing_masks,
        max_cuts=max_cuts,
        violation_tol=violation_tol,
        cardinalities=(3,),
        cand_max=max(18, max_cuts + 3),
    )
