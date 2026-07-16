"""生成新时间窗 Set-C 算例（写入 set_c_XX_new.db，避免旧 db 被锁）。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configuration import SET_C_INSTANCE_BASE_SEED, SET_C_INSTANCE_COUNT
from instance_generator.generator import SET_C, capacity_metrics, generate_instance, persist_instance
from preprocessor.instance_db import instance_db_path
from preprocessor.preprocess import load_instance, preprocess
from model_and_algo.im_model import _item_start_feasible
import statistics


def main() -> None:
    for idx in range(1, SET_C_INSTANCE_COUNT + 1):
        inst = generate_instance(SET_C, idx, seed=SET_C_INSTANCE_BASE_SEED + idx - 1)
        name = f"set_c_{idx:02d}_new"
        inst.name = name
        path = persist_instance(inst, db_path=instance_db_path(name))
        m = capacity_metrics(inst)
        data = preprocess(load_instance(name))
        raw = data.raw
        lens = [raw.deadlines[i] - raw.earliest_starts[i] + 1 for i in data.cured_items]
        num_i = len(data.cured_items)
        t_max = data.periods[-1]
        active = [
            sum(1 for ii in range(num_i) if _item_start_feasible(data, ii, t, t_max))
            for t in range(len(data.periods))
        ]
        print(
            f"{name} -> {path}\n"
            f"  |I|={int(m['num_items'])} win_med={statistics.median(lens):.0f}h "
            f"active_med={statistics.median(active):.0f}"
        )


if __name__ == "__main__":
    main()
