"""对已有 set_b_hard 算例重跑 IM，记录 Gurobi MIP 上下界。"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common.paths import INSTANCES_DIR, RESULT_DIR
from configuration import DEFAULT_MIP_GAP, SET_EXPERIMENT_LIMITS
from model_and_algo.im_model import solve_im
from preprocessor.preprocess import load_instance, preprocess

IM_LIMIT = min(SET_EXPERIMENT_LIMITS["B"]["im_time"], 600.0)


def main() -> None:
    rows: list[dict] = []
    for idx in range(1, 4):
        name = f"set_b_hard_{idx:02d}"
        data = preprocess(load_instance(INSTANCES_DIR / f"{name}.json"))
        t0 = time.perf_counter()
        im = solve_im(data, time_limit=IM_LIMIT, mip_gap=DEFAULT_MIP_GAP)
        sec = time.perf_counter() - t0
        ub = im.objective
        lb = im.lower_bound
        mip_gap_pct = (ub - lb) / ub * 100 if ub and lb is not None else None
        row = {
            "instance": name,
            "im_status": im.status,
            "im_sec": round(sec, 2),
            "im_ub": round(ub, 2) if ub is not None else None,
            "im_lb": round(lb, 2) if lb is not None else None,
            "im_mip_gap": round(im.mip_gap, 6) if im.mip_gap is not None else None,
            "im_mip_gap_pct": round(mip_gap_pct, 2) if mip_gap_pct is not None else None,
        }
        rows.append(row)
        print(
            f"{name}: status={row['im_status']} UB={row['im_ub']} LB={row['im_lb']} "
            f"MIPgap={row['im_mip_gap_pct']}% time={row['im_sec']}s",
            flush=True,
        )

    bounds_path = RESULT_DIR / "set_b_hard_im_bounds.json"
    bounds_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    cg_path = RESULT_DIR / "set_b_hard_im_cg.json"
    if cg_path.exists():
        payload = json.loads(cg_path.read_text(encoding="utf-8"))
        by_name = {r["instance"]: r for r in rows}
        for inst in payload.get("instances", []):
            b = by_name.get(inst["instance"])
            if b:
                inst["im_lb"] = b["im_lb"]
                inst["im_mip_gap"] = b["im_mip_gap"]
                inst["im_mip_gap_pct"] = b["im_mip_gap_pct"]
        payload["im_mip_bounds"] = rows
        cg_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved {bounds_path}", flush=True)


if __name__ == "__main__":
    main()
