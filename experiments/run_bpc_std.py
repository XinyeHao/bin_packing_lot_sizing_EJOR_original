"""Ordinary BPC (best_bound) on small then medium instances."""
from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(r"C:\Users\haoxi\PycharmProjects\bin_packing_and_scheduling")
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ["GRB_CS_QUIET"] = "1"

from model_and_algo.branch_and_price import solve_branch_and_price
from preprocessor.instance_db import load_instance_from_db
from preprocessor.preprocess import preprocess

OUT_MD = ROOT / "supplement_materials" / "test_instance.md"
OUT_JSON = ROOT / "supplement_materials" / "bpc_std_results.json"
CUTS = "standard"

SMALL = {
    "names": [f"set_im_opt900_{i:02d}" for i in range(1, 6)],
    "limit": 900.0,
    "im_ref": {
        "set_im_opt900_01": {"ub": 15032.50, "time": 68.1},
        "set_im_opt900_02": {"ub": 14436.40, "time": 65.9},
        "set_im_opt900_03": {"ub": 17165.30, "time": 63.4},
        "set_im_opt900_04": {"ub": 13460.00, "time": 30.2},
        "set_im_opt900_05": {"ub": 18695.80, "time": 252.5},
    },
}
MEDIUM = {
    "names": [f"set_im_med_{i:02d}" for i in range(1, 6)],
    "limit": 1800.0,
    "im_ref": {
        "set_im_med_01": {"ub": 19812.50, "time": 1800.9, "gap": 1.86},
        "set_im_med_02": {"ub": 21561.40, "time": 1801.1, "gap": 5.60},
        "set_im_med_03": {"ub": 19428.90, "time": 1801.4, "gap": 5.51},
        "set_im_med_04": {"ub": 19400.00, "time": 1801.4, "gap": 8.77},
        "set_im_med_05": {"ub": 15964.20, "time": 1801.1, "gap": 3.20},
    },
}


def gap_pct(ub, lb):
    if ub is None or lb is None or ub <= 1e-9:
        return None
    return 100.0 * max(0.0, (ub - lb) / ub)


def is_proved(r) -> bool:
    if r.status == "optimal":
        return True
    if r.objective is not None and r.lower_bound is not None:
        return abs(r.objective - r.lower_bound) <= 1e-4
    return False


def run_one(name: str, limit: float) -> dict:
    raw = load_instance_from_db(name)
    data = preprocess(raw)
    print(
        f"\n=== {name} T={len(data.periods)} I={len(data.cured_items)} "
        f"M={len(data.machines)} limit={limit:.0f}s ===",
        flush=True,
    )
    cg_kwargs = {
        "pricing_mode": "ebrp",
        "ebrp_force": True,
        "use_gurobi_init": False,
        "skip_cross_check": True,
        "pricing_inprocess": True,
        "verbose": False,
        "progress_interval_sec": 30.0,
        "time_limit": limit,
        "lrmp_cuts": CUTS,
    }
    t0 = time.perf_counter()
    try:
        r = solve_branch_and_price(
            data,
            max_nodes=8000,
            im_time_limit=90.0,
            time_limit=limit,
            use_node_im=False,
            root_im=True,
            final_im=True,
            branch_strategy="best_bound",
            cg_kwargs=cg_kwargs,
        )
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        print(f"  ERROR: {exc}", flush=True)
        try:
            data.dispose()
        except Exception:
            pass
        return {
            "name": name,
            "status": "error",
            "ub": None,
            "lb": None,
            "gap_pct": None,
            "runtime_sec": elapsed,
            "proved": False,
            "nodes": None,
            "error": str(exc),
        }

    elapsed = time.perf_counter() - t0
    proved = is_proved(r)
    g = gap_pct(r.objective, r.lower_bound)
    nodes = r.extra.get("nodes_explored") if r.extra else None
    row = {
        "name": name,
        "status": "optimal" if proved else r.status,
        "ub": r.objective,
        "lb": r.lower_bound,
        "gap_pct": g,
        "runtime_sec": elapsed,
        "proved": proved,
        "nodes": nodes,
    }
    print(
        f"  BPC: status={row['status']} UB={r.objective} LB={r.lower_bound} "
        f"gap={None if g is None else f'{g:.2f}%'} t={elapsed:.1f}s nodes={nodes}",
        flush=True,
    )
    try:
        data.dispose()
    except Exception:
        pass
    gc.collect()
    time.sleep(0.4)
    return row


def write_md(small_rows: list[dict], med_rows: list[dict]) -> None:
    text = OUT_MD.read_text(encoding="utf-8")
    for marker in ("\n## BPC (", "\n## BPC+ML", "\n### Small set: BPC"):
        if marker in text:
            text = text.split(marker)[0].rstrip() + "\n"

    lines = [
        "",
        "## BPC (best_bound, no ML)",
        "",
        f"- `branch_strategy=best_bound`, `lrmp_cuts={CUTS}`, EBRP",
        "",
        "### Small set (limit 900s) vs IM",
        "",
        "| name | IM time | IM obj | BPC status | BPC UB | BPC LB | BPC gap % | BPC time | nodes | beat IM? |",
        "|------|---------|--------|------------|--------|--------|-----------|----------|-------|----------|",
    ]
    for r in small_rows:
        im = SMALL["im_ref"][r["name"]]
        g = "—" if r["gap_pct"] is None else f"{r['gap_pct']:.2f}"
        ub = "—" if r["ub"] is None else f"{r['ub']:.2f}"
        lb = "—" if r["lb"] is None else f"{r['lb']:.2f}"
        nodes = "—" if r.get("nodes") is None else str(r["nodes"])
        beat = "—"
        if r["proved"]:
            beat = "yes" if r["runtime_sec"] < im["time"] else "no"
        else:
            beat = "no"
        lines.append(
            f"| {r['name']} | {im['time']:.1f} | {im['ub']:.2f} | {r['status']} | {ub} | {lb} | "
            f"{g} | {r['runtime_sec']:.1f} | {nodes} | {beat} |"
        )

    lines += [
        "",
        "### Medium set (limit 1800s) vs IM",
        "",
        "| name | IM gap% | IM time | BPC status | BPC UB | BPC LB | BPC gap % | BPC time | nodes | better gap? |",
        "|------|---------|---------|------------|--------|--------|-----------|----------|-------|-------------|",
    ]
    for r in med_rows:
        im = MEDIUM["im_ref"][r["name"]]
        g = "—" if r["gap_pct"] is None else f"{r['gap_pct']:.2f}"
        ub = "—" if r["ub"] is None else f"{r['ub']:.2f}"
        lb = "—" if r["lb"] is None else f"{r['lb']:.2f}"
        nodes = "—" if r.get("nodes") is None else str(r["nodes"])
        better = "—"
        if r["proved"]:
            better = "proved"
        elif r["gap_pct"] is not None:
            better = "yes" if r["gap_pct"] < im["gap"] else "no"
        lines.append(
            f"| {r['name']} | {im['gap']:.2f} | {im['time']:.1f} | {r['status']} | {ub} | {lb} | "
            f"{g} | {r['runtime_sec']:.1f} | {nodes} | {better} |"
        )

    OUT_MD.write_text(text.rstrip() + "\n" + "\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    print(f"Ordinary BPC best_bound, cuts={CUTS}", flush=True)
    state = {"small": [], "medium": []}
    if OUT_JSON.exists():
        try:
            state = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        except Exception:
            pass

    done_s = {r["name"] for r in state.get("small", [])}
    print("\n======== SMALL ========", flush=True)
    for name in SMALL["names"]:
        if name in done_s:
            print(f"  skip cached {name}", flush=True)
            continue
        row = run_one(name, SMALL["limit"])
        state.setdefault("small", []).append(row)
        OUT_JSON.write_text(json.dumps(state, indent=2), encoding="utf-8")
        write_md(state["small"], state.get("medium", []))

    done_m = {r["name"] for r in state.get("medium", [])}
    print("\n======== MEDIUM ========", flush=True)
    for name in MEDIUM["names"]:
        if name in done_m:
            print(f"  skip cached {name}", flush=True)
            continue
        row = run_one(name, MEDIUM["limit"])
        state.setdefault("medium", []).append(row)
        OUT_JSON.write_text(json.dumps(state, indent=2), encoding="utf-8")
        write_md(state["small"], state["medium"])

    write_md(state["small"], state["medium"])
    n_beat = sum(
        1
        for r in state["small"]
        if r.get("proved") and r["runtime_sec"] < SMALL["im_ref"][r["name"]]["time"]
    )
    print(
        f"\nDONE: small proved {sum(1 for r in state['small'] if r.get('proved'))}/5 "
        f"beat_IM {n_beat}/5; medium proved "
        f"{sum(1 for r in state['medium'] if r.get('proved'))}/5",
        flush=True,
    )


if __name__ == "__main__":
    main()
