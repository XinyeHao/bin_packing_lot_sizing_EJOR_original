"""按当前生成规则随机生成 3 个 Set-B 算例，运行 IM 与 CG 对比。"""

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
    SET_EXPERIMENT_LIMITS,
)
from instance_generator.generator import generate_instance
from model_and_algo.column_generation import solve_column_generation
from model_and_algo.im_model import solve_im
from preprocessor.preprocess import load_instance, preprocess

COUNT = 3
BASE_SEED = 9300
NAME_PREFIX = "set_b_new"
IM_TIME_LIMIT = SET_EXPERIMENT_LIMITS["B"]["im_time"]
CG_INIT = SET_EXPERIMENT_LIMITS["B"]["cg_init"]
OUTPUT = RESULT_DIR / "set_b_im_cg_3.json"


def main() -> None:
    INSTANCES_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for idx in range(1, COUNT + 1):
        name = f"{NAME_PREFIX}_{idx:02d}"
        seed = BASE_SEED + idx
        inst = generate_instance(SET_B, idx, seed=seed)
        inst.name = name
        path = INSTANCES_DIR / f"{name}.json"
        path.write_text(json.dumps(inst.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

        data = preprocess(load_instance(path))
        dl_min = min(data.deadlines)
        dl_max = max(data.deadlines)
        print(f"\n=== {name} (seed={seed}) ===", flush=True)
        print(
            f"  T={len(data.periods)} |I|={len(data.cured_items)} |M|={len(data.machines)} "
            f"w_total={sum(data.w_i)} D_i in [{dl_min},{dl_max}]",
            flush=True,
        )

        t0 = time.perf_counter()
        cg = solve_column_generation(
            data,
            max_iterations=DEFAULT_CG_MAX_ITERATIONS,
            init_time_limit=CG_INIT,
        )
        cg_sec = time.perf_counter() - t0

        t1 = time.perf_counter()
        im = solve_im(data, time_limit=IM_TIME_LIMIT, mip_gap=DEFAULT_MIP_GAP, method="im_gurobi")
        im_sec = time.perf_counter() - t1

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
        }
        rows.append(row)
        print(
            f"  CG: LRMP={row['lrmp']} scrap={row['cg_scrap_qty']} "
            f"iters={row['cg_iters']} time={row['cg_sec']}s",
            flush=True,
        )
        print(
            f"  IM: obj={row['im_obj']} scrap={row['im_scrap_qty']} "
            f"status={row['im_status']} time={row['im_sec']}s gap={row['lrmp_gap_pct']}%",
            flush=True,
        )

    out = {"instances": rows}
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
