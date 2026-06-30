"""输出 CG 收敛后 LRMP 解的构成与分数变量，供分支设计参考。"""

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
INSTANCES = [f"set_b_rand_{i:02d}" for i in range(1, 6)]
OUTPUT_JSON = PROJECT_ROOT / "result" / "lrmp_solution_rand5_detail.json"


def is_int(v: float, tol: float = INT_TOL) -> bool:
    return abs(v - round(v)) <= tol


def frac_part(v: float) -> float:
    return abs(v - round(v))


def _column_summary(col, q_val: float) -> dict:
    setups = sorted(col.y.keys(), key=lambda k: (k[1], k[0]))
    prod = sorted((k, v) for k, v in col.x.items() if v > 0)
    return {
        "col_id": col.column_id,
        "Q": round(q_val, 6),
        "Q_int": is_int(q_val),
        "cost": round(col.production_cost, 2),
        "num_setups": len(setups),
        "setups": [f"{u}@t{t}" for u, t in setups[:6]] + (["..."] if len(setups) > 6 else []),
        "num_prod_entries": len(prod),
    }


def _cost_breakdown(cg: ColumnGenerationSolver) -> dict:
    data = cg.data
    art = cg._artifacts
    assert art is not None
    hold = sum(
        data.h_c[i] * art["S_plus"][i, t].X
        for i in range(len(data.cured_items))
        for t in range(len(data.periods))
    )
    back = sum(
        data.bc_j[i] * art["S_minus"][i, t].X
        for i in range(len(data.cured_items))
        for t in range(len(data.periods))
    )
    prod = 0.0
    for machine in data.machines:
        for ci, col in enumerate(cg.columns[machine]):
            prod += col.production_cost * art["Q"][machine][ci].X
    return {
        "production": round(prod, 2),
        "holding": round(hold, 2),
        "backorder": round(back, 2),
        "total": round(prod + hold + back, 2),
        "lrmp_obj": round(cg.lrmp_obj or 0, 2),
    }


def analyze_instance(name: str) -> dict:
    data = load_processed_instance(name)
    cg = ColumnGenerationSolver(
        data,
        max_iterations=DEFAULT_CG_MAX_ITERATIONS,
        init_time_limit=SET_EXPERIMENT_LIMITS["B"]["cg_init"],
    )
    cg.run()
    if not cg.converged or not cg._model or not cg._artifacts:
        return {"instance": name, "error": "CG did not converge"}

    art = cg._artifacts
    cost = _cost_breakdown(cg)

    # --- Q: 每台机器选哪些列、权重多少 ---
    q_global = {"total": 0, "positive": 0, "integer": 0, "fractional": 0}
    per_machine: list[dict] = []
    for machine in data.machines:
        qvars = art["Q"][machine]
        cols = cg.columns[machine]
        active = []
        qsum = 0.0
        for ci, qv in enumerate(qvars):
            val = qv.X
            q_global["total"] += 1
            qsum += val
            if val <= FRAC_TOL:
                continue
            q_global["positive"] += 1
            if is_int(val):
                q_global["integer"] += 1
            else:
                q_global["fractional"] += 1
            active.append(_column_summary(cols[ci], val))
        active.sort(key=lambda x: -x["Q"])
        per_machine.append(
            {
                "machine": machine,
                "Q_sum": round(qsum, 6),
                "num_active": len(active),
                "is_convex_combo": len(active) > 1,
                "has_fractional_Q": any(not e["Q_int"] for e in active),
                "active_columns": active,
            }
        )

    # --- implied Y: 配置-时段 setup 指示 ---
    implied_y = cg.get_implied_y()
    y_integer = []
    y_fractional = []
    for (m_idx, u_idx, t_idx), val in implied_y.items():
        entry = {
            "machine": data.machines[m_idx],
            "config": data.configs[u_idx],
            "period": data.periods[t_idx],
            "Y": round(val, 6),
            "frac_part": round(frac_part(val), 6),
        }
        if is_int(val):
            y_integer.append(entry)
        elif val > FRAC_TOL:
            y_fractional.append(entry)
    y_fractional.sort(key=lambda x: -x["frac_part"])

    # --- S+/S- 库存与缺货 ---
    def _scan_s(name: str, var_dict) -> dict:
        pos_int, pos_frac = [], []
        for i in range(len(data.cured_items)):
            for t in range(len(data.periods)):
                v = var_dict[i, t].X
                if v <= FRAC_TOL:
                    continue
                entry = {
                    "item": data.cured_items[i],
                    "period": data.periods[t],
                    "value": round(v, 6),
                    "frac_part": round(frac_part(v), 6),
                }
                if is_int(v):
                    pos_int.append(entry)
                else:
                    pos_frac.append(entry)
        pos_frac.sort(key=lambda x: -x["frac_part"])
        return {
            "positive_count": len(pos_int) + len(pos_frac),
            "integer_count": len(pos_int),
            "fractional_count": len(pos_frac),
            "max_value": round(max((e["value"] for e in pos_int + pos_frac), default=0.0), 6),
            "top_fractional": pos_frac[:8],
        }

    s_plus = _scan_s("S_plus", art["S_plus"])
    s_minus = _scan_s("S_minus", art["S_minus"])

    # --- 分支候选：最分数的 Y 与混合列机器 ---
    branch_candidates = {
        "fractional_Y_top10": y_fractional[:10],
        "machines_needing_column_branch": [
            {
                "machine": pm["machine"],
                "Q_sum": pm["Q_sum"],
                "num_active": pm["num_active"],
                "weights": [(c["col_id"], c["Q"]) for c in pm["active_columns"]],
            }
            for pm in per_machine
            if pm["is_convex_combo"] and pm["has_fractional_Q"]
        ],
    }

    return {
        "instance": name,
        "lrmp": cost["lrmp_obj"],
        "cost_breakdown": cost,
        "Q_summary": q_global,
        "per_machine": per_machine,
        "implied_Y": {
            "total_nonzero": len(y_integer) + len(y_fractional),
            "integer": len(y_integer),
            "fractional": len(y_fractional),
            "fractional_values": y_fractional[:20],
        },
        "S_plus": s_plus,
        "S_minus": s_minus,
        "branch_candidates": branch_candidates,
    }


def print_report(results: list[dict]) -> None:
    for r in results:
        if "error" in r:
            print(f"=== {r['instance']}: {r['error']} ===\n")
            continue
        print(f"=== {r['instance']}  LRMP={r['lrmp']} ===")
        cb = r["cost_breakdown"]
        print(
            f"  成本: 生产={cb['production']} 库存={cb['holding']} "
            f"缺货={cb['backorder']} (合计={cb['total']})"
        )
        q = r["Q_summary"]
        print(
            f"  Q变量: 活跃={q['positive']}/{q['total']}, "
            f"整数={q['integer']}, 分数={q['fractional']}"
        )
        mixed = [pm for pm in r["per_machine"] if pm["is_convex_combo"]]
        print(f"  多列混合机器: {len(mixed)}/7")
        for pm in mixed:
            weights = ", ".join(f"col{c['col_id']}={c['Q']:.3f}" for c in pm["active_columns"])
            print(f"    {pm['machine']}: Q_sum={pm['Q_sum']:.4f}, {weights}")
            for c in pm["active_columns"][:3]:
                print(f"      col{c['col_id']} Q={c['Q']:.4f} cost={c['cost']} setups={c['setups']}")

        y = r["implied_Y"]
        print(f"  implied Y: 非零={y['total_nonzero']}, 整数={y['integer']}, 分数={y['fractional']}")
        if y["fractional_values"]:
            print("    最分数 Y (machine, config, period, Y):")
            for e in y["fractional_values"][:8]:
                print(f"      {e['machine']} {e['config']} t{e['period']} Y={e['Y']:.4f}")

        sp, sm = r["S_plus"], r["S_minus"]
        print(
            f"  S+: 非零={sp['positive_count']} (整数{sp['integer_count']}/分数{sp['fractional_count']}) "
            f"max={sp['max_value']}"
        )
        print(
            f"  S-: 非零={sm['positive_count']} (整数{sm['integer_count']}/分数{sm['fractional_count']}) "
            f"max={sm['max_value']}"
        )
        if sp["top_fractional"]:
            print(f"    S+ 分数示例: {sp['top_fractional'][:3]}")
        if sm["top_fractional"]:
            print(f"    S- 分数示例: {sm['top_fractional'][:3]}")
        print()


def main() -> None:
    results = [analyze_instance(n) for n in INSTANCES]
    OUTPUT_JSON.parent.mkdir(exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {OUTPUT_JSON}\n")
    print_report(results)


if __name__ == "__main__":
    main()
