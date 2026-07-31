"""Replace set_im_med_05 with a harder mid-scale instance (IM 1800s)."""
from __future__ import annotations

import gc
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(r"C:\Users\haoxi\PycharmProjects\bin_packing_and_scheduling")
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ["GRB_CS_QUIET"] = "1"

import configuration as cfg
from configuration import SetConfig
from instance_generator import generator as genmod
from instance_generator.generator import generate_instance, persist_instance
from model_and_algo.im_model import solve_im
from preprocessor.instance_db import delete_instance
from preprocessor.preprocess import preprocess

OUT_MD = ROOT / "supplement_materials" / "test_instance.md"
OUT_JSON = ROOT / "supplement_materials" / "test_instance_medium.json"

T, U, I, M = 30, 5, 40, 5
IM_LIMIT = 1800.0
NAME = "set_im_med_05"
# Skip structurally easy seeds (large tanks), like old 05
MAX_Q_SUM = 28.0
# Also reject if IM proves too fast
MIN_HARD_TIME = 300.0
SEED_CANDIDATES = list(range(8106, 8200))


def apply_scale() -> None:
    cfg.apply_bench_generation()


def cleanup(*objs) -> None:
    for o in objs:
        try:
            if o is not None and hasattr(o, "dispose"):
                o.dispose()
        except Exception:
            pass
    gc.collect()
    time.sleep(0.3)


def gap_pct(ub, lb):
    if ub is None or lb is None or ub <= 1e-9:
        return None
    return 100.0 * max(0.0, (ub - lb) / ub)


def structural_ok(raw) -> tuple[bool, str]:
    qs = list(raw.machine_capacities.values())
    q_sum = sum(qs)
    lts = [raw.item_lead_times[iid] for iid in raw.cured_item_ids]
    if q_sum > MAX_Q_SUM:
        return False, f"q_sum={q_sum:.1f}> {MAX_Q_SUM}"
    return True, f"q_sum={q_sum:.1f} LT_hist={dict(Counter(lts))}"


def update_md(row: dict) -> None:
    text = OUT_MD.read_text(encoding="utf-8")
    lines = text.splitlines()
    out = []
    for line in lines:
        if line.startswith("| set_im_med_05 |"):
            g = "—" if row["gap_pct"] is None else f"{row['gap_pct']:.2f}"
            ub = "—" if row["ub"] is None else f"{row['ub']:.2f}"
            lb = "—" if row["lb"] is None else f"{row['lb']:.2f}"
            out.append(
                f"| {NAME} | {row['seed']} | {row['status']} | {ub} | {lb} | {g} | {row['runtime_sec']:.1f} |"
            )
        elif "seeds 8101–8105" in line:
            out.append(f"- Instances: `set_im_med_01`–`05` (seeds 8101–8104, {row['seed']})")
        else:
            out.append(line)
    OUT_MD.write_text("\n".join(out) + "\n", encoding="utf-8")


def update_json(row: dict) -> None:
    path = OUT_JSON
    rows = []
    if path.exists():
        rows = json.loads(path.read_text(encoding="utf-8"))
    replaced = False
    for i, r in enumerate(rows):
        if r.get("name") == NAME:
            rows[i] = row
            replaced = True
            break
    if not replaced:
        rows.append(row)
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def main() -> None:
    scfg = SetConfig("small_cg", T, U, 0, I, M)
    print(f"Replace {NAME}: avoid easy structure (q_sum<={MAX_Q_SUM})", flush=True)

    for seed in SEED_CANDIDATES:
        apply_scale()
        raw = generate_instance(scfg, instance_id=5, seed=seed)
        ok, note = structural_ok(raw)
        print(f"seed={seed}: {note} -> {'probe IM' if ok else 'skip'}", flush=True)
        if not ok:
            cleanup(raw)
            continue

        raw.name = NAME
        # overwrite db
        try:
            delete_instance(NAME)
        except Exception:
            pass
        persist_instance(raw)

        data = preprocess(raw)
        im = solve_im(data, time_limit=IM_LIMIT, mip_gap=0.0, verbose=False)
        g = gap_pct(im.objective, im.lower_bound)
        print(
            f"  IM: status={im.status} UB={im.objective} LB={im.lower_bound} "
            f"gap={None if g is None else f'{g:.2f}%'} t={im.runtime_sec:.1f}s",
            flush=True,
        )

        too_easy = im.status == "optimal" and im.runtime_sec < MIN_HARD_TIME
        if too_easy:
            print(f"  reject: proved too fast (<{MIN_HARD_TIME}s)", flush=True)
            cleanup(data, im, raw)
            continue

        row = {
            "name": NAME,
            "seed": seed,
            "T": T,
            "U": U,
            "I": I,
            "M": M,
            "LT": list(LT),
            "window": dict(WINDOW),
            "status": im.status,
            "ub": im.objective,
            "lb": im.lower_bound,
            "gap_pct": g,
            "runtime_sec": im.runtime_sec,
            "proved": im.status == "optimal",
            "note": note,
        }
        update_json(row)
        update_md(row)
        cleanup(data, im, raw)
        print(f"ACCEPTED {NAME} seed={seed}", flush=True)
        return

    print("FAILED: no suitable replacement found", flush=True)


if __name__ == "__main__":
    main()
