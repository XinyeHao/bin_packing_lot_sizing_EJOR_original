"""分析 IM 解的热压罐工作率与空间利用率。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common.data_models import SolutionResult
from postprocessor.utilization import analyze_im_utilization
from preprocessor.preprocess import load_instance


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "set_c_01"
    inst = load_instance(name)
    path = PROJECT_ROOT / "result" / f"{name}_im_gurobi.json"
    d = json.loads(path.read_text(encoding="utf-8"))
    result = SolutionResult(**{k: d[k] for k in d if k in SolutionResult.__dataclass_fields__})
    result.extra = d.get("extra", {})
    u = analyze_im_utilization(inst, result)
    print(f"算例: {name}")
    print(f"状态: {result.status}  目标值: {result.objective:.2f}  MIP gap: {result.mip_gap}")
    print(f"耗时(s): {result.runtime_sec:.1f}")
    print(f"热压罐工作率(全机台): {u['machine_time_util']:.1%}  (目标>=60%)")
    print(f"开工装箱率均值: {u['avg_start_space_util']:.1%}  (目标>=50%)")
    print(f"占罐时段加权装箱率: {u['span_weighted_space_util']:.1%}")
    print(f"批次数: {u['num_batches']}  排产件数: {u['scheduled_qty']}  报废: {u['scrap_qty']}")
    if u.get("avg_items_per_batch") is not None:
        print(f"实际每批平均件数: {u['avg_items_per_batch']:.1f}")
    print("各机台工作率:")
    for m, v in sorted(u["per_machine_time_util"].items()):
        print(f"  {m}: {v['time_util']:.1%} ({v['busy_periods']}/{v['total_periods']})")


if __name__ == "__main__":
    main()
