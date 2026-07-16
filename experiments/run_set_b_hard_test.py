"""当前 Set-B 专用参数下生成算例，运行 CG 与 IM 并对比。"""



from __future__ import annotations



import json

import sys

import time

from pathlib import Path



PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT))



from common.paths import INSTANCES_DIR, RESULT_DIR

from configuration import (

    DEFAULT_CG_MAX_ITERATIONS,

    DEFAULT_MIP_GAP,

    SET_B,

    SET_B_GENERATION,

    SET_EXPERIMENT_LIMITS,

)

from instance_generator.generator import generate_instance

from model_and_algo.column_generation import solve_column_generation

from model_and_algo.im_model import solve_im

from preprocessor.preprocess import load_instance, preprocess



IM_LIMIT = min(SET_EXPERIMENT_LIMITS["B"]["im_time"], 600.0)

CG_INIT = SET_EXPERIMENT_LIMITS["B"]["cg_init"]

SEEDS = (10001, 10002, 10003)

OUTPUT = RESULT_DIR / "set_b_hard_im_cg.json"





def main() -> None:

    print(

        f"Set-B gen: |I|={SET_B_GENERATION.get('num_cured_items')} "

        f"demand_periods={SET_B_GENERATION['demand_periods_per_item']}",

        flush=True,

    )

    rows: list[dict] = []

    for idx, seed in enumerate(SEEDS, 1):

        name = f"set_b_hard_{idx:02d}"

        inst = generate_instance(SET_B, idx, seed=seed)

        inst.name = name

        path = INSTANCES_DIR / f"{name}.json"

        path.write_text(json.dumps(inst.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

        data = preprocess(load_instance(path))

        num_i = len(data.cured_items)

        el, eh = min(data.earliest_starts), max(data.earliest_starts)

        dl, dh = min(data.deadlines), max(data.deadlines)



        t0 = time.perf_counter()

        cg = solve_column_generation(

            data,

            max_iterations=DEFAULT_CG_MAX_ITERATIONS,

            init_time_limit=CG_INIT,

            fast=True,

        )

        cg_sec = time.perf_counter() - t0



        t1 = time.perf_counter()

        im = solve_im(data, time_limit=IM_LIMIT, mip_gap=DEFAULT_MIP_GAP)

        im_sec = time.perf_counter() - t1



        lrmp = cg.lower_bound

        im_obj = im.objective

        gap = (im_obj - lrmp) / im_obj * 100 if im_obj and lrmp else None



        row = {
            "instance": name,
            "seed": seed,
            "|I|": num_i,
            "w_total": sum(data.w_i),
            "E_range": [el, eh],
            "D_range": [dl, dh],
            "lrmp": round(lrmp, 2) if lrmp is not None else None,
            "cg_sec": round(cg_sec, 2),
            "cg_iters": cg.extra.get("num_iterations"),
            "cg_converged": cg.extra.get("converged"),
            "cg_scrap_qty": cg.extra.get("total_scrap_qty"),
            "im_obj": round(im_obj, 2) if im_obj is not None else None,
            "im_lb": round(im.lower_bound, 2) if im.lower_bound is not None else None,
            "im_mip_gap": round(im.mip_gap, 6) if im.mip_gap is not None else None,
            "im_mip_gap_pct": round((im_obj - im.lower_bound) / im_obj * 100, 2)
            if im_obj and im.lower_bound is not None
            else None,
            "im_status": im.status,
            "im_sec": round(im_sec, 2),
            "im_scrap_qty": im.extra.get("total_scrap_qty"),
            "lrmp_gap_pct": round(gap, 2) if gap is not None else None,
        }

        rows.append(row)

        print(

            f"\n=== {name} (seed={seed}) |I|={num_i} w={row['w_total']} "

            f"E={row['E_range']} D={row['D_range']} ===",

            flush=True,

        )

        print(

            f"  CG: LRMP={row['lrmp']} scrap={row['cg_scrap_qty']} "

            f"iters={row['cg_iters']} time={row['cg_sec']}s converged={row['cg_converged']}",

            flush=True,

        )

        print(
            f"  IM: UB={row['im_obj']} LB={row['im_lb']} MIPgap={row['im_mip_gap_pct']}% "
            f"scrap={row['im_scrap_qty']} status={row['im_status']} time={row['im_sec']}s",
            flush=True,
        )

        print(f"  LP gap (IM vs LRMP): {row['lrmp_gap_pct']}%", flush=True)



    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    payload = {"generation": SET_B_GENERATION, "instances": rows}

    OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nSaved {OUTPUT}", flush=True)





if __name__ == "__main__":

    main()

