"""两个较大算例 IM vs CG 对比。"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configuration import SET_C_CG
from model_and_algo.column_generation import solve_column_generation
from model_and_algo.im_model import solve_im
from preprocessor.preprocess import load_instance, preprocess

RUNS = [
    ("bench_01", 600.0, 600.0, {"pricing_mode": "ebrp"}),
    ("set_c_01_elig", 1200.0, 600.0, dict(SET_C_CG)),
]


def main() -> None:
    results: list[dict] = []
    sep = "=" * 60

    for instance, im_tl, cg_tl, cg_extra in RUNS:
        data = preprocess(load_instance(instance))
        max_im = max(len(xs) for xs in data.items_by_machine)
        print(f"\n{sep}", flush=True)
        print(
            f"INSTANCE {instance} |I|={len(data.cured_items)} "
            f"|M|={len(data.machines)} |T|={len(data.periods)} max|I_m|={max_im}",
            flush=True,
        )
        print(sep, flush=True)

        print("--- IM ---", flush=True)
        t0 = time.perf_counter()
        im = solve_im(data, time_limit=im_tl, mip_gap=0.02, verbose=False)
        im_sec = time.perf_counter() - t0
        print(
            f"IM: status={im.status} obj={im.objective} lb={im.lower_bound} "
            f"gap={im.mip_gap} time={im_sec:.1f}s",
            flush=True,
        )

        print("--- CG ---", flush=True)
        t0 = time.perf_counter()
        cg_opts = {"time_limit": cg_tl, "max_iterations": 5000, "verbose": True, **cg_extra}
        cg = solve_column_generation(data, **cg_opts)
        cg_sec = time.perf_counter() - t0
        ex = cg.extra or {}
        lrmp = ex.get("lrmp_obj") or cg.lower_bound

        row: dict = {
            "instance": instance,
            "im_obj": im.objective,
            "im_status": im.status,
            "im_gap": im.mip_gap,
            "im_time": round(im_sec, 1),
            "lrmp": lrmp,
            "cg_status": cg.status,
            "cg_iters": ex.get("num_iterations"),
            "cg_converged": ex.get("converged"),
            "min_rc": ex.get("last_min_rc"),
            "cg_time": round(cg_sec, 1),
            "cols": sum((ex.get("num_columns") or {}).values()),
            "pricing_mode": cg_opts.get("pricing_mode"),
        }
        if im.objective is not None and lrmp is not None:
            row["im_minus_lrmp"] = im.objective - lrmp
            row["lb_valid"] = lrmp <= im.objective + 1e-2
        results.append(row)

        diff = row.get("im_minus_lrmp")
        print(
            f"\nSUMMARY {instance}: IM={im.objective:.2f} LRMP={lrmp:.2f} "
            f"diff={diff:.2f} lb_valid={row.get('lb_valid')}",
            flush=True,
        )

    out = ROOT / "result" / "im_cg_large_compare.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== ALL DONE ===", flush=True)
    for r in results:
        print(r, flush=True)
    print(f"Saved: {out}", flush=True)


if __name__ == "__main__":
    main()
