"""分析 CG 收敛后 LRMP 分数解结构。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from checkers.utils import load_processed_instance
from configuration import DEFAULT_CG_MAX_ITERATIONS, SET_EXPERIMENT_LIMITS
from model_and_algo.column_generation import ColumnGenerationSolver

INT_TOL = 1e-5
FRAC_TOL = 1e-4


def is_int(v: float, tol: float = INT_TOL) -> bool:
    return abs(v - round(v)) <= tol


def frac_part(v: float) -> float:
    return abs(v - round(v))


def analyze_instance(name: str) -> dict:
    data = load_processed_instance(name)
    cg = ColumnGenerationSolver(
        data,
        max_iterations=DEFAULT_CG_MAX_ITERATIONS,
        init_time_limit=SET_EXPERIMENT_LIMITS["B"]["cg_init"],
    )
    cg.run()
    assert cg.converged and cg._model and cg._artifacts

    q_stats = {"total": 0, "positive": 0, "integer": 0, "fractional": 0}
    machine_q: dict = {}
    for m_idx, machine in enumerate(data.machines):
        qvars = cg._artifacts["Q"][machine]
        cols = cg.columns[machine]
        entries = []
        qsum = 0.0
        for ci, qv in enumerate(qvars):
            val = qv.X
            q_stats["total"] += 1
            qsum += val
            if val > FRAC_TOL:
                q_stats["positive"] += 1
                if is_int(val):
                    q_stats["integer"] += 1
                else:
                    q_stats["fractional"] += 1
                entries.append(
                    {
                        "col_id": cols[ci].column_id,
                        "Q": round(val, 6),
                        "cost": round(cols[ci].production_cost, 2),
                        "setups": len(cols[ci].y),
                    }
                )
        machine_q[machine] = {
            "q_sum": round(qsum, 6),
            "num_active": len(entries),
            "active": entries[:8],
        }

    splus = {"total": 0, "positive": 0, "integer": 0, "fractional": 0, "max": 0.0}
    sminus = {"total": 0, "positive": 0, "integer": 0, "fractional": 0, "max": 0.0}
    for i in range(len(data.cured_items)):
        for t in range(len(data.periods)):
            sp = cg._artifacts["S_plus"][i, t].X
            sm = cg._artifacts["S_minus"][i, t].X
            splus["total"] += 1
            sminus["total"] += 1
            if sp > FRAC_TOL:
                splus["positive"] += 1
                splus["max"] = max(splus["max"], sp)
                splus["integer" if is_int(sp) else "fractional"] += 1
            if sm > FRAC_TOL:
                sminus["positive"] += 1
                sminus["max"] = max(sminus["max"], sm)
                sminus["integer" if is_int(sm) else "fractional"] += 1

    implied_y = cg.get_implied_y()
    y_stats = {
        "total_keys": len(implied_y),
        "integer": 0,
        "fractional": 0,
        "fractional_vals": [],
    }
    for k, v in implied_y.items():
        if is_int(v):
            y_stats["integer"] += 1
        else:
            y_stats["fractional"] += 1
            if v > FRAC_TOL:
                m, u, t = k
                y_stats["fractional_vals"].append(
                    {
                        "machine": data.machines[m],
                        "config": data.configs[u],
                        "period": data.periods[t],
                        "Y": round(v, 6),
                    }
                )
    y_stats["fractional_vals"].sort(key=lambda x: -frac_part(x["Y"]))
    y_stats["fractional_vals"] = y_stats["fractional_vals"][:15]

    machines_mixed = []
    for machine, info in machine_q.items():
        if info["num_active"] > 1:
            machines_mixed.append(
                {
                    "machine": machine,
                    "q_sum": info["q_sum"],
                    "num_active": info["num_active"],
                    "frac_cols": sum(1 for e in info["active"] if not is_int(e["Q"])),
                    "active": info["active"][:5],
                }
            )

    qsum_not_one = [
        {"machine": m, "q_sum": info["q_sum"], "num_active": info["num_active"]}
        for m, info in machine_q.items()
        if abs(info["q_sum"] - 1.0) > 0.01 and info["num_active"] > 0
    ]

    return {
        "instance": name,
        "lrmp": round(cg.lrmp_obj, 2),
        "Q": q_stats,
        "machine_q_summary": {
            m: {"q_sum": v["q_sum"], "num_active": v["num_active"]} for m, v in machine_q.items()
        },
        "machines_with_multiple_active_columns": machines_mixed,
        "machines_qsum_not_one": qsum_not_one,
        "S_plus": splus,
        "S_minus": sminus,
        "implied_Y": y_stats,
    }


def main() -> None:
    names = [f"set_b_{i:02d}" for i in (1, 4, 7, 10)]
    results = [analyze_instance(n) for n in names]
    out = PROJECT_ROOT / "result" / "lrmp_fractionality_analysis.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    for r in results:
        print(f"=== {r['instance']} LRMP={r['lrmp']} ===")
        q = r["Q"]
        print(
            f"  Q: active={q['positive']}/{q['total']}, "
            f"integer={q['integer']}, fractional={q['fractional']}"
        )
        n_multi = len(r["machines_with_multiple_active_columns"])
        print(f"  machines with convex combo (>=2 active cols): {n_multi}/7")
        for mm in r["machines_with_multiple_active_columns"][:2]:
            print(f"    {mm['machine']}: q_sum={mm['q_sum']}, active={mm['active']}")
        y = r["implied_Y"]
        print(
            f"  implied Y: {y['total_keys']} keys, "
            f"fractional={y['fractional']}, integer={y['integer']}"
        )
        if y["fractional_vals"]:
            print(f"    top fractional Y: {y['fractional_vals'][:3]}")
        sp, sm = r["S_plus"], r["S_minus"]
        print(
            f"  S+: pos={sp['positive']} frac={sp['fractional']}; "
            f"S-: pos={sm['positive']} frac={sm['fractional']}"
        )
        print(f"  q_sum!=1 machines: {r['machines_qsum_not_one']}")
        print()


if __name__ == "__main__":
    main()
