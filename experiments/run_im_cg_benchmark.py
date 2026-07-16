"""生成若干 Set-B 算例，运行 IM/CG，并对 IM 解做 checker 验证。"""



from __future__ import annotations



import argparse

import json

import sys

import time

from pathlib import Path



PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT))



from checkers.im_solution_checker import check_im_feasibility

from checkers.objective_checker import check_objective

from common.paths import INSTANCES_DIR, RESULT_DIR

from configuration import (

    DEFAULT_CG_MAX_ITERATIONS,

    DEFAULT_INSTANCE_TIME_LIMIT,

    DEFAULT_MIP_GAP,

    SET_B,

    SET_EXPERIMENT_LIMITS,

)

from instance_generator.generator import generate_instance

from model_and_algo.column_generation import solve_column_generation

from model_and_algo.im_model import solve_im

from preprocessor.preprocess import load_instance, preprocess



IM_TIME_LIMIT = min(SET_EXPERIMENT_LIMITS["B"]["im_time"], DEFAULT_INSTANCE_TIME_LIMIT)

CG_INIT = SET_EXPERIMENT_LIMITS["B"]["cg_init"]





def run_one(name: str, seed: int, regenerate: bool) -> dict:

    path = INSTANCES_DIR / f"{name}.json"

    if regenerate or not path.exists():

        inst = generate_instance(SET_B, int(name.split("_")[-1]), seed=seed)

        inst.name = name

        path.write_text(json.dumps(inst.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")



    data = preprocess(load_instance(path))

    el_min, el_max = min(data.earliest_starts), max(data.earliest_starts)

    dl_min, dl_max = min(data.deadlines), max(data.deadlines)

    print(

        f"\n=== {name} (seed={seed}) ===\n"

        f"  T={len(data.periods)} |I|={len(data.cured_items)} |M|={len(data.machines)} "

        f"w_total={sum(data.w_i)} E_i in [{el_min},{el_max}] D_i in [{dl_min},{dl_max}]",

        flush=True,

    )



    t0 = time.perf_counter()

    cg = solve_column_generation(data, max_iterations=DEFAULT_CG_MAX_ITERATIONS, init_time_limit=CG_INIT)

    cg_sec = time.perf_counter() - t0



    t1 = time.perf_counter()

    im = solve_im(data, time_limit=IM_TIME_LIMIT, mip_gap=DEFAULT_MIP_GAP, method="im_gurobi")

    im_sec = time.perf_counter() - t1



    feas = check_im_feasibility(data, im)

    obj = check_objective(data, im)

    checker_ok = feas.passed and obj.passed



    lrmp = cg.lower_bound

    im_obj = im.objective

    gap = (im_obj - lrmp) / im_obj * 100 if im_obj and lrmp else None



    row = {

        "instance": name,

        "seed": seed,

        "lrmp": round(lrmp, 2) if lrmp is not None else None,

        "cg_sec": round(cg_sec, 2),

        "cg_iters": cg.extra.get("num_iterations"),

        "cg_converged": cg.extra.get("converged"),

        "cg_scrap_qty": cg.extra.get("total_scrap_qty"),

        "im_obj": round(im_obj, 2) if im_obj is not None else None,

        "im_status": im.status,

        "im_sec": round(im_sec, 2),

        "im_scrap_qty": im.extra.get("total_scrap_qty"),

        "lrmp_gap_pct": round(gap, 2) if gap is not None else None,

        "checker_pass": checker_ok,

        "checker_violations": len(feas.violations) + len(obj.violations),

    }

    print(

        f"  CG: LRMP={row['lrmp']} scrap={row['cg_scrap_qty']} "

        f"iters={row['cg_iters']} time={row['cg_sec']}s",

        flush=True,

    )

    print(

        f"  IM: obj={row['im_obj']} scrap={row['im_scrap_qty']} "

        f"status={row['im_status']} time={row['im_sec']}s gap={row['lrmp_gap_pct']}% "

        f"checker={'PASS' if checker_ok else 'FAIL'}",

        flush=True,

    )

    if not checker_ok:

        for rep in (feas, obj):

            for v in rep.violations[:5]:

                print(f"    [{v.checker}/{v.code}] {v.message}", flush=True)

    return row





def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument("--count", type=int, default=8, help="算例数量")

    parser.add_argument("--base-seed", type=int, default=9400)

    parser.add_argument("--prefix", type=str, default="set_b_new")

    parser.add_argument("--start", type=int, default=1, help="算例编号起始")

    parser.add_argument("--regenerate", action="store_true", help="强制重新生成 JSON")

    parser.add_argument(

        "--output",

        type=str,

        default=str(RESULT_DIR / "set_b_im_cg_benchmark.json"),

    )

    args = parser.parse_args()



    INSTANCES_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []

    for idx in range(args.start, args.start + args.count):

        name = f"{args.prefix}_{idx:02d}"

        seed = args.base_seed + idx

        rows.append(run_one(name, seed, args.regenerate))



    out_path = Path(args.output)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {"instances": rows, "checker_failures": sum(1 for r in rows if not r["checker_pass"])}

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nSaved {out_path} ({payload['checker_failures']} checker failures)", flush=True)





if __name__ == "__main__":

    main()

