"""Gurobi 精确子问题 CG 收敛 vs IM：测 LRMP–IM gap。"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_and_algo.column_generation import solve_column_generation
from model_and_algo.im_model import solve_im
from preprocessor.preprocess import load_instance, preprocess

# 小规模：Gurobi 子问题 MIP 随 |I_m|·|T| 指数涨，避免 |I_m|>30 或 T>30
INSTANCES = [
    ("demo_small", 30.0, 60.0),       # |I|=4,  |T|=6
    ("large_01", 60.0, 180.0),        # |I|=28, |T|=20
    ("set_b_bench_01", 120.0, 600.0),  # |I|=45, |T|=25（上限，单轮定价仍较慢）
]


def main() -> None:
    rows: list[dict] = []
    print(f"{'instance':<18} {'|I|':>4} {'|T|':>4} {'maxIm':>5}  {'IM':>12}  {'LRMP':>12}  {'gap%':>7}  {'CG it':>5}  {'ok':>4}")
    print("-" * 95)

    for name, im_tl, cg_tl in INSTANCES:
        data = preprocess(load_instance(name))
        n_i, n_t = len(data.cured_items), len(data.periods)
        max_im = max(len(x) for x in data.items_by_machine)

        t0 = time.perf_counter()
        im = solve_im(data, time_limit=im_tl, mip_gap=0.001, verbose=False)
        im_sec = time.perf_counter() - t0

        t0 = time.perf_counter()
        cg = solve_column_generation(
            data,
            time_limit=cg_tl,
            max_iterations=100_000,
            verbose=False,
            pricing_mode="im_gurobi",
            im_pricing_time_limit=30.0,
        )
        cg_sec = time.perf_counter() - t0

        ex = cg.extra or {}
        lrmp = ex.get("lrmp_obj") or cg.lower_bound
        im_obj = im.objective
        gap_pct = None
        lb_ok = None
        if im_obj is not None and lrmp is not None and im_obj > 1e-9:
            gap_pct = (im_obj - lrmp) / im_obj * 100.0
            lb_ok = lrmp <= im_obj + 1e-2

        row = {
            "instance": name,
            "num_items": n_i,
            "num_periods": n_t,
            "max_items_on_machine": max_im,
            "im_obj": im_obj,
            "im_status": im.status,
            "im_mip_gap": im.mip_gap,
            "im_time_sec": round(im_sec, 1),
            "lrmp": lrmp,
            "cg_status": cg.status,
            "cg_iters": ex.get("num_iterations"),
            "cg_converged": ex.get("converged"),
            "min_rc": ex.get("last_min_rc"),
            "cg_time_sec": round(cg_sec, 1),
            "cols": sum((ex.get("num_columns") or {}).values()),
            "gap_im_minus_lrmp_pct": gap_pct,
            "lb_valid": lb_ok,
        }
        rows.append(row)

        gap_s = f"{gap_pct:6.2f}%" if gap_pct is not None else "   n/a"
        ok_s = "Y" if lb_ok else "N"
        print(
            f"{name:<18} {n_i:4d} {n_t:4d} {max_im:5d}  "
            f"{im_obj:12.2f}  {lrmp:12.2f}  {gap_s}  {ex.get('num_iterations') or 0:5d}  {ok_s:>4}"
        )

    out = ROOT / "result" / "cg_gurobi_lrmp_im_gap.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
