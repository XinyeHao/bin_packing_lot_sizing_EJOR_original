"""Generate 5 medium instances T=30,U=5,I=40,M=5 and run IM (1800s)."""
from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(r"C:\Users\haoxi\PycharmProjects\bin_packing_and_scheduling")
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ["GRB_CS_QUIET"] = "1"

import configuration as cfg
from configuration import BENCH_LEAD_TIME_CHOICES, BENCH_WINDOW, SetConfig
from instance_generator import generator as genmod
from instance_generator.generator import generate_instance, persist_instance
from model_and_algo.im_model import solve_im
from preprocessor.preprocess import preprocess

OUT_MD = ROOT / "supplement_materials" / "test_instance.md"
OUT_JSON = ROOT / "supplement_materials" / "test_instance_medium.json"

T, U, I, M = 30, 5, 40, 5
IM_LIMIT = 1800.0
SEEDS = [8101, 8102, 8103, 8104, 8105]


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


def update_md(rows: list[dict]) -> None:
    text = OUT_MD.read_text(encoding="utf-8")
    # drop previous medium section if re-run
    marker = "\n## Medium set"
    if marker in text:
        text = text.split(marker)[0].rstrip() + "\n"

    lines = [
        "",
        "## Medium set",
        "",
        "- Instances: `set_im_med_01`–`05` (seeds 8101–8105)",
        f"- Scale: `SetConfig('small_cg', {T}, {U}, 0, {I}, {M})` → **T={T}, U={U}, I={I}, M={M}**",
        f"- LT={list(BENCH_LEAD_TIME_CHOICES)}, window `length_range={BENCH_WINDOW['length_range']}`, "
        f"`rel_frac={BENCH_WINDOW['rel_frac']}`",
        f"- IM setting: time_limit={IM_LIMIT:.0f}s, mip_gap=0",
        "",
        "| name | seed | status | UB | LB | MIP gap % | time (s) |",
        "|------|------|--------|----|----|-----------|----------|",
    ]
    for r in rows:
        g = "—" if r["gap_pct"] is None else f"{r['gap_pct']:.2f}"
        ub = "—" if r["ub"] is None else f"{r['ub']:.2f}"
        lb = "—" if r["lb"] is None else f"{r['lb']:.2f}"
        lines.append(
            f"| {r['name']} | {r['seed']} | {r['status']} | {ub} | {lb} | {g} | {r['runtime_sec']:.1f} |"
        )
    OUT_MD.write_text(text.rstrip() + "\n" + "\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    scfg = SetConfig("small_cg", T, U, 0, I, M)
    rows: list[dict] = []
    print(
        f"Medium IM test: T={T},U={U},I={I},M={M}, LT={BENCH_LEAD_TIME_CHOICES}, "
        f"limit={IM_LIMIT:.0f}s",
        flush=True,
    )

    for idx, seed in enumerate(SEEDS, start=1):
        name = f"set_im_med_{idx:02d}"
        print(f"\n=== {name} seed={seed} ===", flush=True)
        apply_scale()
        raw = generate_instance(scfg, instance_id=idx, seed=seed)
        raw.name = name
        persist_instance(raw)

        data = preprocess(raw)
        im = solve_im(data, time_limit=IM_LIMIT, mip_gap=0.0, verbose=False)
        g = gap_pct(im.objective, im.lower_bound)
        row = {
            "name": name,
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
        }
        rows.append(row)
        print(
            f"  status={im.status} UB={im.objective} LB={im.lower_bound} "
            f"gap={None if g is None else f'{g:.2f}%'} t={im.runtime_sec:.1f}s",
            flush=True,
        )
        OUT_JSON.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        update_md(rows)
        cleanup(data, im, raw)

    print("\nDONE", flush=True)
    for r in rows:
        print(
            f"  {r['name']}: {r['status']} gap={r['gap_pct']} t={r['runtime_sec']:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
