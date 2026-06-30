"""输出含报废变量 LRMP 收敛后的分数解结构。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from checkers.utils import load_processed_instance
from configuration import DEFAULT_CG_MAX_ITERATIONS, SET_B, SET_EXPERIMENT_LIMITS
from instance_generator.generator import generate_instance
from model_and_algo.column_generation import ColumnGenerationSolver
from common.paths import INSTANCES_DIR

INT_TOL = 1e-5
FRAC_TOL = 1e-4
INSTANCES = [("set_b_scrap_01", 9201), ("set_b_scrap_02", 9202)]
OUTPUT = PROJECT_ROOT / "result" / "lrmp_scrap_fractionality.json"


def is_int(v: float) -> bool:
    return abs(v - round(v)) <= INT_TOL


def frac_part(v: float) -> float:
    return abs(v - round(v))


def ensure_instance(name: str, seed: int) -> None:
    path = INSTANCES_DIR / f"{name}.json"
    if path.exists():
        return
    inst = generate_instance(SET_B, 1, seed=seed)
    inst.name = name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(inst.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


def analyze(name: str) -> dict:
    data = load_processed_instance(name)
    cg = ColumnGenerationSolver(
        data,
        max_iterations=DEFAULT_CG_MAX_ITERATIONS,
        init_time_limit=SET_EXPERIMENT_LIMITS["B"]["cg_init"],
    )
    cg.run()
    if not cg.converged or not cg._artifacts:
        return {"instance": name, "error": "CG not converged"}

    art = cg._artifacts
    num_i = len(data.cured_items)

    # 成本分解
    hold = sum(data.h_c[i] * art["S_plus"][i, t].X for i in range(num_i) for t in range(len(data.periods)))
    back = sum(data.bc_j[i] * art["S_minus"][i, t].X for i in range(num_i) for t in range(len(data.periods)))
    scrap_cost = sum(data.sc_i[i] * art["R"][i].X for i in range(num_i))
    prod = sum(
        col.production_cost * art["Q"][m][ci].X
        for m in data.machines
        for ci, col in enumerate(cg.columns[m])
    )

    # Q
    q_pos, q_frac, q_int = 0, 0, 0
    mixed_machines = []
    for machine in data.machines:
        active = []
        qsum = 0.0
        for ci, qv in enumerate(art["Q"][machine]):
            v = qv.X
            qsum += v
            if v <= FRAC_TOL:
                continue
            q_pos += 1
            if is_int(v):
                q_int += 1
            else:
                q_frac += 1
            active.append((cg.columns[machine][ci].column_id, round(v, 4)))
        if len(active) > 1:
            mixed_machines.append({"machine": machine, "Q_sum": round(qsum, 4), "active": active[:6]})

    # R 报废
    r_entries = []
    r_int, r_frac, r_pos = 0, 0, 0
    for i in range(num_i):
        v = art["R"][i].X
        if v <= FRAC_TOL:
            continue
        r_pos += 1
        entry = {"item": data.cured_items[i], "R": round(v, 4), "w_i": data.w_i[i], "frac_part": round(frac_part(v), 4)}
        if is_int(v):
            r_int += 1
        else:
            r_frac += 1
        r_entries.append(entry)
    r_entries.sort(key=lambda x: -x["frac_part"])

    # implied Y
    implied_y = cg.get_implied_y()
    y_frac = [
        {
            "machine": data.machines[m],
            "config": data.configs[u],
            "period": data.periods[t],
            "Y": round(v, 4),
        }
        for (m, u, t), v in implied_y.items()
        if v > FRAC_TOL and not is_int(v)
    ]
    y_frac.sort(key=lambda x: -frac_part(x["Y"]))

    # S+/S-
    def scan_s(var_dict):
        pos, frac = 0, 0
        top = []
        for i in range(num_i):
            for t in range(len(data.periods)):
                v = var_dict[i, t].X
                if v <= FRAC_TOL:
                    continue
                pos += 1
                if not is_int(v):
                    frac += 1
                    top.append({"item": data.cured_items[i], "period": data.periods[t], "val": round(v, 4)})
        top.sort(key=lambda x: -frac_part(x["val"]))
        return {"positive": pos, "fractional": frac, "top": top[:5]}

    return {
        "instance": name,
        "lrmp": round(cg.lrmp_obj or 0, 2),
        "cost": {
            "production": round(prod, 2),
            "holding": round(hold, 2),
            "backorder": round(back, 2),
            "scrap": round(scrap_cost, 2),
            "total_check": round(prod + hold + back + scrap_cost, 2),
        },
        "Q": {"active": q_pos, "integer": q_int, "fractional": q_frac},
        "mixed_machines": mixed_machines,
        "R": {
            "positive": r_pos,
            "integer": r_int,
            "fractional": r_frac,
            "total_qty": round(sum(e["R"] for e in r_entries), 2),
            "entries": r_entries[:15],
        },
        "implied_Y": {"fractional_count": len(y_frac), "top": y_frac[:10]},
        "S_plus": scan_s(art["S_plus"]),
        "S_minus": scan_s(art["S_minus"]),
    }


def print_report(r: dict) -> None:
    if "error" in r:
        print(r)
        return
    print(f"=== {r['instance']}  LRMP={r['lrmp']} ===")
    c = r["cost"]
    print(f"  成本: 生产={c['production']} 库存={c['holding']} 缺货={c['backorder']} 报废={c['scrap']}")
    q = r["Q"]
    print(f"  Q: 活跃={q['active']}, 整数={q['integer']}, 分数={q['fractional']} (分数占比 100%)")
    print(f"  多列混合机器: {len(r['mixed_machines'])}/7")
    for mm in r["mixed_machines"][:3]:
        print(f"    {mm['machine']}: Q_sum={mm['Q_sum']}, {mm['active']}")
    ri = r["R"]
    print(f"  R(报废): 非零={ri['positive']}, 整数={ri['integer']}, 分数={ri['fractional']}, 总量={ri['total_qty']}")
    if ri["entries"]:
        print("    报废明细 (item, R, w_i):")
        for e in ri["entries"][:8]:
            print(f"      {e['item']}: R={e['R']} / w={e['w_i']}")
    y = r["implied_Y"]
    print(f"  implied Y 分数: {y['fractional_count']} 个")
    for e in y["top"][:5]:
        print(f"    {e['machine']} {e['config']} t{e['period']} Y={e['Y']}")
    sp, sm = r["S_plus"], r["S_minus"]
    print(f"  S+: 非零={sp['positive']}, 分数={sp['fractional']}")
    print(f"  S-: 非零={sm['positive']}, 分数={sm['fractional']}")
    print()


def main() -> None:
    results = []
    for name, seed in INSTANCES:
        ensure_instance(name, seed)
        results.append(analyze(name))
    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {OUTPUT}\n")
    for r in results:
        print_report(r)


if __name__ == "__main__":
    main()
