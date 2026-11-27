"""Report UB / global tree LB / gap for a running or finished DSSR BP log.

Unexplored RF children sit on the open heap with dual bound = parent LB.
Replaying the log restores that multiset and gives the same tree_LB the
solver uses for the gap stop (min open dual bound; lazy/dfs empty here).
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIAG = ROOT / "result" / "diagnostics"


def _tree_from_log(text: str) -> tuple[float | None, float | None, float | None, str]:
    ub = None
    open_bounds: Counter[float] = Counter()
    started = False
    for line in text.splitlines():
        if "incumbent=" in line:
            ub = float(line.split("incumbent=")[1].split()[0].rstrip(")"))
        if "BP stop: gap=" in line:
            gap = float(line.split("gap=")[1].split("%")[0])
            ub = float(line.split("UB=")[1].split(",")[0])
            lb = float(line.split("tree_LB=")[1].split(",")[0])
            return ub, lb, gap, "stop"
        if " LB=" in line and "branch" in line:
            lb = float([x for x in line.split() if x.startswith("LB=")][0][3:])
            if not started:
                # Root has been solved and is branching: two children inherit root LB.
                open_bounds[lb] += 2
                started = True
                continue
            if not open_bounds:
                return ub, None, None, "open_empty"
            # Best-bound pop: RF children use parent_lb as both priority and dual bound.
            key = min(open_bounds)
            open_bounds[key] -= 1
            if open_bounds[key] == 0:
                del open_bounds[key]
            open_bounds[lb] += 2
        elif "pruned" in line and "BP [" in line:
            if not started or not open_bounds:
                continue
            key = min(open_bounds)
            open_bounds[key] -= 1
            if open_bounds[key] == 0:
                del open_bounds[key]
    if ub is None or not open_bounds:
        return ub, None, None, "incomplete"
    tree_lb = min(open_bounds)
    gap = 100.0 * (ub - tree_lb) / abs(ub)
    return ub, tree_lb, gap, "open"


def report(seed: int) -> str:
    log = DIAG / f"ebrp_dssr_{seed}.log"
    js = DIAG / f"ebrp_dssr_{seed}.json"
    if not log.exists():
        return f"{seed}: not started"
    raw = log.read_bytes()
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        text = raw.decode("utf-16")
    else:
        text = raw.decode("utf-8", errors="replace")
    if js.exists():
        row = json.loads(js.read_text(encoding="utf-8"))
        if row.get("status") not in (None, "running") and row.get("ub") is not None:
            return (
                f"{seed}: runtime_sec={float(row['runtime_sec']):.1f} "
                f"ub={row['ub']} lb={row['lb']} gap_pct={float(row['gap_pct']):.4f}"
            )
        start = datetime.fromisoformat(row["started_at_utc"])
        runtime = (datetime.now(timezone.utc) - start).total_seconds()
    else:
        runtime = float("nan")
    ub, lb, gap, how = _tree_from_log(text)
    if ub is None or lb is None or gap is None:
        return f"{seed}: runtime_sec={runtime:.0f} bounds unavailable ({how})"
    return (
        f"{seed}: runtime_sec={runtime:.0f} ub={ub} lb={lb} gap_pct={gap:.4f}"
    )


print(report(40301))
print(report(40401))
