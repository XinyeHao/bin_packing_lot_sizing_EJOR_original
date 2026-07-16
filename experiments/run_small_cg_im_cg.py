"""生成 small_cg 算例并跑 IM vs Gurobi 子问题 CG（空列起步，每轮各机负 RC 列一并加入）。"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configuration import SMALL_CG_INSTANCE_COUNT
from instance_generator.generator import generate_small_cg_instances
from model_and_algo.column_generation import solve_column_generation
from model_and_algo.im_model import solve_im
from preprocessor.preprocess import load_instance, preprocess


def _load_existing_small_cg(count: int) -> list:
    instances = []
    for idx in range(1, count + 1):
        name = f"set_small_cg_{idx:02d}"
        instances.append(load_instance(name))
    return instances


def main() -> None:
    parser = argparse.ArgumentParser(description="small_cg IM vs Gurobi-CG benchmark")
    parser.add_argument(
        "--no-generate",
        action="store_true",
        help="跳过生成，直接加载 instances/set_small_cg_*.db",
    )
    args = parser.parse_args()

    if args.no_generate:
        print("=== 加载已有 small_cg 算例 ===", flush=True)
        instances = _load_existing_small_cg(SMALL_CG_INSTANCE_COUNT)
    else:
        print("=== 生成 small_cg 算例 ===", flush=True)
        instances = generate_small_cg_instances()
    for inst in instances:
        per_u = {}
        for i, u in inst.item_config.items():
            per_u[u] = per_u.get(u, 0) + 1
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
        f"{'IM':>9}  {'LRMP':>9}  {'LB*':>9}  {'g_LR%':>6}  {'g_LB%':>6}  {'CGs':>5}"
    )
    print(hdr, flush=True)
    print("-" * 88, flush=True)

    for inst in instances:
        data = preprocess(inst)
        max_im = max(len(xs) for xs in data.items_by_machine)

        t0 = time.perf_counter()
        im = solve_im(data, time_limit=120.0, mip_gap=0.001, verbose=False)
        im_sec = time.perf_counter() - t0

        t0 = time.perf_counter()
        cg = solve_column_generation(
            data,
            time_limit=600.0,
            max_iterations=5000,
            verbose=False,
            pricing_mode="im_gurobi",
            use_gurobi_init=False,
            im_pricing_time_limit=30.0,
        )
        cg_sec = time.perf_counter() - t0

        ex = cg.extra or {}
        lrmp = ex.get("lrmp_obj") or cg.lower_bound
        lb_corr = ex.get("last_valid_lb")
        im_obj = im.objective
        gap_pct = (im_obj - lrmp) / im_obj * 100.0 if im_obj and lrmp else None
        gap_lb_pct = (im_obj - lb_corr) / im_obj * 100.0 if im_obj and lb_corr else None
        lb_ok = lrmp <= im_obj + 1e-2 if im_obj and lrmp else None
        lb_corr_ok = lb_corr <= im_obj + 1e-2 if im_obj and lb_corr else None

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
            "lb_corr_valid": lb_corr_ok,
            "cg_iters": ex.get("num_iterations"),
            "cg_converged": ex.get("converged"),
            "min_rc": ex.get("last_min_rc"),
            "cg_time_sec": round(cg_sec, 1),
            "cols": sum((ex.get("num_columns") or {}).values()),
            "gap_pct": gap_pct,
            "lb_valid": lb_ok,
        }
        rows.append(row)
        gap_lr_s = f"{gap_pct:5.1f}%" if gap_pct is not None else "  n/a"
        gap_lb_s = f"{gap_lb_pct:5.1f}%" if gap_lb_pct is not None else "  n/a"
        lb_s = f"{lb_corr:9.2f}" if lb_corr is not None else "      n/a"
        scrap_q = row.get("im_scrap_qty")
        scrap_s = f"{scrap_q:3.0f}" if scrap_q is not None else "n/a"
        print(
            f"{inst.name:<18} {row['cg_iters']:5d} {row['cols']:5d} {scrap_s}  "
            f"{im_obj:9.2f}  {lrmp:9.2f}  {lb_s}  {gap_lr_s}  {gap_lb_s}  {cg_sec:5.1f}",
            flush=True,
        )

    out = ROOT / "result" / "small_cg_im_cg_gurobi.json"
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {out}", flush=True)

    n = len(rows)
    if n:
        avg_im = sum(r["im_obj"] for r in rows) / n
        avg_lrmp = sum(r["lrmp"] for r in rows) / n
        lbs = [r["last_valid_lb"] for r in rows if r.get("last_valid_lb") is not None]
        avg_lb = sum(lbs) / len(lbs) if lbs else None
        avg_gap_lr = sum(r["gap_pct"] for r in rows if r.get("gap_pct") is not None) / n
        avg_gap_lb = sum(r["gap_lb_pct"] for r in rows if r.get("gap_lb_pct") is not None) / len(
            [r for r in rows if r.get("gap_lb_pct") is not None]
        )
        avg_iters = sum(r["cg_iters"] for r in rows) / n
        lb_ok_n = sum(1 for r in rows if r.get("lb_corr_valid"))
        print("\n=== 汇总（IM vs LRMP vs 修正下界 LB*）===", flush=True)
        print("LB* = LRMP + Σ_m min(RC_m, 0)，列池因重复列停住时仍有效的对偶下界。", flush=True)
        print(f"  算例数: {n}  平均 CG 迭代: {avg_iters:.0f}", flush=True)
        print(f"  平均 IM:   {avg_im:10.2f}", flush=True)
        print(f"  平均 LRMP: {avg_lrmp:10.2f}  (IM-LRMP)/IM = {avg_gap_lr:+.2f}%", flush=True)
        if avg_lb is not None:
            print(
                f"  平均 LB*:  {avg_lb:10.2f}  (IM-LB*)/IM = {avg_gap_lb:+.2f}%  "
                f"[LB*<=IM: {lb_ok_n}/{n}]",
                flush=True,
            )


if __name__ == "__main__":
    main()
