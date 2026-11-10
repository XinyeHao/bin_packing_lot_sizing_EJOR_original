"""Subset-row inequalities (SRI) for the schedule-column master.

The implementation deliberately starts with odd triples.  A triple is cheap
to separate at the root and its coefficient only depends on a column's set of
processed items, so it remains available when pricing generates a new column.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from common.data_models import ProcessedInstance, ScheduleColumn


@dataclass
class SRICut:
    """An odd-cardinality rank-1 subset-row inequality.

    For an odd item set S, the master row is

        sum_{m,k} floor(|C_mk intersect S| / 2) Q_mk <= floor(|S| / 2).

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


def separate_root_triples(
    data: "ProcessedInstance",
    columns: dict[str, list["ScheduleColumn"]],
    q_values: dict[str, list[float]],
    *,
    existing_masks: set[int] | None = None,
    max_cuts: int = 3,
    violation_tol: float = 1e-6,
) -> list[SRICut]:
    """Separate the most violated triple SRIs from a fractional RMP solution.

    We restrict candidates to items appearing in fractional columns.  This is
    complete for neither all odd subsets nor all rank-1 cuts; it is a safe,
    transparent root-node separation routine for the first ablation.
    """
    existing_masks = existing_masks or set()
    weighted_masks: list[tuple[int, float]] = []
    candidate_items: set[int] = set()
    for machine, cols in columns.items():
        vals = q_values.get(machine, [])
        for col, q_val in zip(cols, vals):
            if q_val <= violation_tol or q_val >= 1.0 - violation_tol:
                continue
            col_mask = item_mask_from_column(data, col)
            if not col_mask:
                continue
            weighted_masks.append((col_mask, float(q_val)))
            bit = col_mask
            while bit:
                low = bit & -bit
                candidate_items.add(low.bit_length() - 1)
                bit ^= low

    if len(candidate_items) < 3 or not weighted_masks:
        return []

    violated: list[tuple[float, SRICut]] = []
    for items in combinations(sorted(candidate_items), 3):
        item_mask = sum(1 << i for i in items)
        if item_mask in existing_masks:
            continue
        lhs = sum(((col_mask & item_mask).bit_count() // 2) * q for col_mask, q in weighted_masks)
        rhs = 1
        violation = lhs - rhs
        if violation > violation_tol:
            label = "_".join(map(str, items))
            violated.append((violation, SRICut(items, item_mask, rhs, f"sri3_{label}")))

    violated.sort(key=lambda pair: pair[0], reverse=True)
    return [cut for _violation, cut in violated[:max_cuts]]
