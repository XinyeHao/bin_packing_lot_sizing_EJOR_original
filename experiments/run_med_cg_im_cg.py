"""生成 med_cg 算例并跑 IM vs Gurobi 子问题 CG（每轮各机负 RC 列一并加入）。"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configuration import (
    MED_CG_CG_TIME_LIMIT,
    MED_CG_IM_TIME_LIMIT,
    MED_CG_INSTANCE_COUNT,
    MED_CG_PRICING_TIME_LIMIT,
)
from instance_generator.generator import generate_med_cg_instances
from model_and_algo.column_generation import solve_column_generation
from model_and_algo.im_model import solve_im
from preprocessor.preprocess import load_instance, preprocess


def _load_existing_med_cg(count: int) -> list:
    return [load_instance(f"set_med_cg_{idx:02d}") for idx in range(1, count + 1)]


def main() -> None:
    parser = argparse.ArgumentParser(description="med_cg IM vs Gurobi-CG benchmark")
    parser.add_argument(
        "--no-generate",
        action="store_true",
        help="跳过生成，直接加载 instances/set_med_cg_*.db",
    )
    args = parser.parse_args()

    if args.no_generate:
        print("=== 加载已有 med_cg 算例 ===", flush=True)
        instances = _load_existing_med_cg(MED_CG_INSTANCE_COUNT)
    else:
        print("=== 生成 med_cg 算例 ===", flush=True)
        instances = generate_med_cg_instances()

    for inst in instances:
        mi = [len(v) for v in inst.item_machines.values()]
        print(
            f"  {inst.name}: T={inst.num_periods} |I|={len(inst.cured_item_ids)} "
            f"|M|={len(inst.machine_ids)} |U|={len(inst.config_ids)} "
            f"|M_i|∈[{min(mi)},{max(mi)}]",
            flush=True,
        )

    rows: list[dict] = []
    hdr = (
        f"\n{'instance':<18} {'iters':>5} {'cols':>5} {'scr':>3}  "
        f"{'IM':>10}  {'LRMP':>10}  {'LB*':>10}  {'g_LR%':>6}  {'IMs':>6}  {'CGs':>6}"
    )
    print(hdr, flush=True)
    print("-" * 92, flush=True)

    for inst in instances:
        data = preprocess(inst)
        max_im = max(len(xs) for xs in data.items_by_machine)

        t0 = time.perf_counter()
        im = solve_im(data, time_limit=MED_CG_IM_TIME_LIMIT, mip_gap=0.001, verbose=False)
        im_sec = time.perf_counter() - t0

        t0 = time.perf_counter()
        cg = solve_column_generation(
            data,
            time_limit=MED_CG_CG_TIME_LIMIT,
            max_iterations=5000,
            verbose=False,
            pricing_mode="im_gurobi",
            use_gurobi_init=False,
            im_pricing_time_limit=MED_CG_PRICING_TIME_LIMIT,
        )
        cg_sec = time.perf_counter() - t0

        ex = cg.extra or {}
        lrmp = ex.get("lrmp_obj") or cg.lower_bound
        lb_corr = ex.get("last_valid_lb")
        im_obj = im.objective
        gap_pct = (im_obj - lrmp) / im_obj * 100.0 if im_obj and lrmp else None
        gap_lb_pct = (im_obj - lb_corr) / im_obj * 100.0 if im_obj and lb_corr else None

        row = {
            "instance": inst.name,
            "num_items": len(data.cured_items),
            "num_periods": len(data.periods),
            "max_items_on_machine": max_im,
            "im_obj": im_obj,
            "im_status": im.status,
            "im_time_sec": round(im_sec, 1),
            "im_scrap_qty": (im.extra or {}).get("total_scrap_qty"),
            "lrmp": lrmp,
            "last_valid_lb": lb_corr,
            "gap_lb_pct": gap_lb_pct,
            "lb_corr_valid": lb_corr <= im_obj + 1e-2 if im_obj and lb_corr else None,
            "cg_iters": ex.get("num_iterations"),
            "cg_converged": ex.get("converged"),
            "stop_reason": ex.get("stop_reason"),
            "min_rc": ex.get("last_min_rc"),
            "cg_time_sec": round(cg_sec, 1),
            "cols": sum((ex.get("num_columns") or {}).values()),
            "gap_pct": gap_pct,
            "lb_valid": lrmp <= im_obj + 1e-2 if im_obj and lrmp else None,
        }
        rows.append(row)
        gap_lr_s = f"{gap_pct:5.1f}%" if gap_pct is not None else "  n/a"
        lb_s = f"{lb_corr:10.2f}" if lb_corr is not None else "       n/a"
        scrap_q = row.get("im_scrap_qty")
        scrap_s = f"{scrap_q:3.0f}" if scrap_q is not None else "n/a"
        print(
            f"{inst.name:<18} {row['cg_iters']:5d} {row['cols']:5d} {scrap_s}  "
            f"{im_obj:10.2f}  {lrmp:10.2f}  {lb_s}  {gap_lr_s}  {im_sec:6.1f}  {cg_sec:6.1f}",
            flush=True,
        )

    out = ROOT / "result" / "med_cg_im_cg_gurobi.json"
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {out}", flush=True)

    n = len(rows)
    if n:
        print("\n=== 汇总（中规模 med_cg，Gurobi 子问题）===", flush=True)
        print(f"  规模: T=28 |I|=40 |M|=4", flush=True)
        print(f"  平均 CG 迭代: {sum(r['cg_iters'] for r in rows) / n:.0f}", flush=True)
        print(f"  平均 IM:   {sum(r['im_obj'] for r in rows) / n:10.2f}", flush=True)
        print(f"  平均 LRMP: {sum(r['lrmp'] for r in rows) / n:10.2f}", flush=True)
        avg_gap = sum(r["gap_pct"] for r in rows if r.get("gap_pct") is not None) / n
        print(f"  平均 (IM-LRMP)/IM: {avg_gap:+.2f}%", flush=True)
        print(f"  LRMP<=IM: {sum(1 for r in rows if r.get('lb_valid'))}/{n}", flush=True)
        print(f"  CG 收敛: {sum(1 for r in rows if r.get('cg_converged'))}/{n}", flush=True)


if __name__ == "__main__":
    main()
