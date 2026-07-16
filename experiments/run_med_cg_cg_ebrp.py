"""med_cg 三算例：EBRP 精确子问题 CG，与 Gurobi 子问题结果对比。"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configuration import MED_CG_CG_TIME_LIMIT, MED_CG_INSTANCE_COUNT
from model_and_algo.column_generation import solve_column_generation
from preprocessor.preprocess import load_instance, preprocess

GUROBI_JSON = ROOT / "result" / "med_cg_im_cg_gurobi.json"
OUT_JSON = ROOT / "result" / "med_cg_cg_ebrp.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="med_cg CG with EBRP pricing")
    parser.add_argument("--ebrp-time-limit", type=float, default=None, help="单机 EBRP 限时（秒），默认不限")
    args = parser.parse_args()

    gurobi_rows: dict[str, dict] = {}
    if GUROBI_JSON.exists():
        for row in json.loads(GUROBI_JSON.read_text(encoding="utf-8")):
            gurobi_rows[row["instance"]] = row

    instances = [load_instance(f"set_med_cg_{idx:02d}") for idx in range(1, MED_CG_INSTANCE_COUNT + 1)]

    print("=== med_cg CG：EBRP 精确子问题 ===", flush=True)
    hdr = (
        f"\n{'instance':<18} {'iters':>5} {'cols':>5}  "
        f"{'LRMP':>10}  {'gap%':>6}  {'conv':>4}  {'CGs':>7}  "
        f"{'ΔLRMP':>8}  {'Δit':>5}  {'Δt':>7}"
    )
    print(hdr, flush=True)
    print("-" * 88, flush=True)

    rows: list[dict] = []
    for inst in instances:
        data = preprocess(inst)
        max_im = max(len(xs) for xs in data.items_by_machine)

        t0 = time.perf_counter()
        cg = solve_column_generation(
            data,
            time_limit=MED_CG_CG_TIME_LIMIT,
            max_iterations=5000,
            verbose=False,
            pricing_mode="ebrp",
            use_gurobi_init=False,
            ebrp_time_limit=args.ebrp_time_limit,
            ebrp_force=True,
            progress_interval_sec=0.0,
        )
        cg_sec = time.perf_counter() - t0

        ex = cg.extra or {}
        lrmp = ex.get("lrmp_obj") or cg.lower_bound
        lb_corr = ex.get("last_valid_lb")
        ref = gurobi_rows.get(inst.name, {})
        im_obj = ref.get("im_obj")
        gap_pct = (im_obj - lrmp) / im_obj * 100.0 if im_obj and lrmp else None

        d_lrmp = lrmp - ref["lrmp"] if ref.get("lrmp") is not None and lrmp else None
        d_iters = (ex.get("num_iterations") or 0) - (ref.get("cg_iters") or 0)
        d_time = cg_sec - (ref.get("cg_time_sec") or 0)

        row = {
            "instance": inst.name,
            "max_items_on_machine": max_im,
            "lrmp": lrmp,
            "last_valid_lb": lb_corr,
            "gap_pct": gap_pct,
            "cg_iters": ex.get("num_iterations"),
            "cg_converged": ex.get("converged"),
            "stop_reason": ex.get("stop_reason"),
            "min_rc": ex.get("last_min_rc"),
            "cg_time_sec": round(cg_sec, 1),
            "cols": sum((ex.get("num_columns") or {}).values()),
            "pricing_mode": "ebrp",
            "gurobi_ref": {
                "lrmp": ref.get("lrmp"),
                "cg_iters": ref.get("cg_iters"),
                "cg_time_sec": ref.get("cg_time_sec"),
                "cols": ref.get("cols"),
                "gap_pct": ref.get("gap_pct"),
            },
            "delta_lrmp": d_lrmp,
            "delta_iters": d_iters,
            "delta_time_sec": round(d_time, 1) if d_time is not None else None,
        }
        rows.append(row)

        conv = "Y" if row["cg_converged"] else "N"
        gap_s = f"{gap_pct:5.1f}%" if gap_pct is not None else "  n/a"
        d_lr_s = f"{d_lrmp:+8.2f}" if d_lrmp is not None else "     n/a"
        d_it_s = f"{d_iters:+5d}" if ref else "  n/a"
        d_t_s = f"{d_time:+7.1f}" if d_time is not None else "    n/a"
        print(
            f"{inst.name:<18} {row['cg_iters']:5d} {row['cols']:5d}  "
            f"{lrmp:10.2f}  {gap_s}  {conv:>4}  {cg_sec:7.1f}  "
            f"{d_lr_s}  {d_it_s}  {d_t_s}",
            flush=True,
        )

    OUT_JSON.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {OUT_JSON}", flush=True)

    n = len(rows)
    if n and gurobi_rows:
        print("\n=== EBRP vs Gurobi 子问题 CG 对比 ===", flush=True)
        avg_ebrp_t = sum(r["cg_time_sec"] for r in rows) / n
        avg_g_t = sum(r["gurobi_ref"]["cg_time_sec"] or 0 for r in rows) / n
        avg_ebrp_lr = sum(r["lrmp"] for r in rows) / n
        avg_g_lr = sum(r["gurobi_ref"]["lrmp"] or 0 for r in rows) / n
        print(f"  平均 CG 时间: EBRP {avg_ebrp_t:.1f}s  vs  Gurobi {avg_g_t:.1f}s  (Δ {avg_ebrp_t - avg_g_t:+.1f}s)", flush=True)
        print(f"  平均 LRMP:    EBRP {avg_ebrp_lr:.2f}  vs  Gurobi {avg_g_lr:.2f}  (Δ {avg_ebrp_lr - avg_g_lr:+.2f})", flush=True)
        print(f"  收敛: {sum(1 for r in rows if r['cg_converged'])}/{n}", flush=True)


if __name__ == "__main__":
    main()
