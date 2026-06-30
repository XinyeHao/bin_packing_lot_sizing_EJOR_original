"""分析 set_b_new_01/02/03 的 LRMP 分数结构，供 B&P 设计参考。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from checkers.utils import load_processed_instance
from configuration import DEFAULT_CG_MAX_ITERATIONS, DEFAULT_MIP_GAP, SET_EXPERIMENT_LIMITS
from model_and_algo.column_generation import ColumnGenerationSolver
from model_and_algo.im_model import solve_im

INT_TOL = 1e-5
FRAC_TOL = 1e-4
INSTANCES = ["set_b_new_01", "set_b_new_02", "set_b_new_03"]
OUTPUT = PROJECT_ROOT / "result" / "lrmp_bp_analysis_new3.json"


def is_int(v: float) -> bool:
    return abs(v - round(v)) <= INT_TOL


def frac_part(v: float) -> float:
    return abs(v - round(v))


def analyze(name: str) -> dict:
    data = load_processed_instance(name)
    cg = ColumnGenerationSolver(
        data,
        max_iterations=DEFAULT_CG_MAX_ITERATIONS,
        init_time_limit=SET_EXPERIMENT_LIMITS["B"]["cg_init"],
    )
    cg.run()
    art = cg._artifacts
    assert art is not None and cg.lrmp_obj is not None

    im = solve_im(data, time_limit=600, mip_gap=DEFAULT_MIP_GAP, method="im_gurobi")
    gap = (im.objective - cg.lrmp_obj) / im.objective * 100

    num_i = len(data.cured_items)
    hold = sum(data.h_c[i] * art["S_plus"][i, t].X for i in range(num_i) for t in range(len(data.periods)))
    back = sum(data.bc_j[i] * art["S_minus"][i, t].X for i in range(num_i) for t in range(len(data.periods)))
    scrap = sum(data.sc_i[i] * art["R"][i].X for i in range(num_i))
    prod = sum(
        col.production_cost * art["Q"][m][ci].X
        for m in data.machines
        for ci, col in enumerate(cg.columns[m])
    )

    q_pos = q_frac = 0
    per_m: list[dict] = []
    for machine in data.machines:
        active = []
        qsum = 0.0
        for ci, qv in enumerate(art["Q"][machine]):
            v = qv.X
            qsum += v
            if v <= FRAC_TOL:
                continue
            q_pos += 1
            if not is_int(v):
                q_frac += 1
            col = cg.columns[machine][ci]
            setups = sorted(col.y.keys(), key=lambda k: (k[1], k[0]))
            active.append(
                {
                    "col_id": col.column_id,
                    "Q": round(v, 4),
                    "cost": round(col.production_cost, 1),
                    "setups": [f"{u}@t{t}" for u, t in setups[:5]],
                }
            )
        per_m.append(
            {
                "machine": machine,
                "q_sum": round(qsum, 4),
                "n_active": len(active),
                "active": active,
            }
        )

    r_entries = []
    r_int = r_frac = r_pos = 0
    for i in range(num_i):
        v = art["R"][i].X
        if v <= FRAC_TOL:
            continue
        r_pos += 1
        entry = {
            "item": data.cured_items[i],
            "R": round(v, 4),
            "w_i": data.w_i[i],
            "D_i": data.deadlines[i],
            "frac_part": round(frac_part(v), 4),
        }
        if is_int(v):
            r_int += 1
        else:
            r_frac += 1
        r_entries.append(entry)
    r_entries.sort(key=lambda x: -x["frac_part"])

    implied = cg.get_implied_y()
    y_frac = [
        {
            "machine": data.machines[m],
            "config": data.configs[u],
            "period": data.periods[t],
            "Y": round(v, 4),
            "frac_part": round(frac_part(v), 4),
        }
        for (m, u, t), v in implied.items()
        if v > FRAC_TOL and not is_int(v)
    ]
    y_frac.sort(key=lambda x: -x["frac_part"])

    sp_frac = sm_frac = 0
    for i in range(num_i):
        for t in range(len(data.periods)):
            if art["S_plus"][i, t].X > FRAC_TOL and not is_int(art["S_plus"][i, t].X):
                sp_frac += 1
            if art["S_minus"][i, t].X > FRAC_TOL and not is_int(art["S_minus"][i, t].X):
                sm_frac += 1

    return {
        "instance": name,
        "lrmp": round(cg.lrmp_obj, 2),
        "im": round(im.objective, 2),
        "gap_pct": round(gap, 2),
        "cost": {"prod": round(prod, 2), "hold": round(hold, 2), "back": round(back, 2), "scrap": round(scrap, 2)},
        "Q": {
            "active": q_pos,
            "fractional": q_frac,
            "mixed_machines": sum(1 for pm in per_m if pm["n_active"] > 1),
        },
        "per_machine": per_m,
        "R": {"positive": r_pos, "integer": r_int, "fractional": r_frac, "top_frac": r_entries[:10]},
        "implied_Y": {"fractional_count": len(y_frac), "top": y_frac[:12]},
        "S_frac": {"S_plus": sp_frac, "S_minus": sm_frac},
        "num_columns": sum(len(cg.columns[m]) for m in data.machines),
        "iters": len(cg.iteration_times),
    }


def main() -> None:
    results = [analyze(n) for n in INSTANCES]
    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    for r in results:
        print(f"=== {r['instance']} LRMP={r['lrmp']} IM={r['im']} gap={r['gap_pct']}% ===")
        c = r["cost"]
        print(f"  cost: prod={c['prod']} hold={c['hold']} back={c['back']} scrap={c['scrap']}")
        print(
            f"  Q: active={r['Q']['active']} frac={r['Q']['fractional']} "
            f"mixed={r['Q']['mixed_machines']}/7 cols={r['num_columns']} iters={r['iters']}"
        )
        for pm in r["per_machine"]:
            if pm["n_active"] > 1:
                act = ", ".join(f"col{a['col_id']}={a['Q']}" for a in pm["active"])
                print(f"    {pm['machine']}: Q_sum={pm['q_sum']} [{act}]")
        print(f"  R: pos={r['R']['positive']} int={r['R']['integer']} frac={r['R']['fractional']}")
        if r["R"]["top_frac"]:
            print(f"    top R frac: {[(e['item'], e['R']) for e in r['R']['top_frac'][:4]]}")
        print(f"  implied Y fractional={r['implied_Y']['fractional_count']}")
        for y in r["implied_Y"]["top"][:5]:
            print(f"    {y['machine']} {y['config']} t{y['period']} Y={y['Y']}")
        print(f"  S+ frac={r['S_frac']['S_plus']} S- frac={r['S_frac']['S_minus']}")
        print()

    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
