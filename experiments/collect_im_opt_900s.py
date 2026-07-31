"""Probe then collect instances at T=25,U=5,I=35,M=5 (IM opt <=900s)."""
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
from instance_generator.generator import capacity_metrics, generate_instance, persist_instance
from model_and_algo.im_model import solve_im
from postprocessor.utilization import analyze_im_utilization
from preprocessor.preprocess import preprocess

OUT_MD = ROOT / "supplement_materials" / "im_opt_900s_scale.md"
OUT_JSON = ROOT / "supplement_materials" / "im_opt_900s_scale.json"

T, U, I, M = 25, 5, 35, 5
LT_CANDIDATES = [
    (BENCH_LEAD_TIME_CHOICES, dict(BENCH_WINDOW), "bench"),
]

IM_LIMIT = 900.0
TARGET = 5
MIN_TIME_UTIL = 0.55
MIN_SPACE_UTIL = 0.70
PROBE_SEEDS = [6101, 6202, 6303]
COLLECT_SEEDS = list(range(7101, 7300))


def apply_scale(lt, window) -> None:
    cfg.apply_bench_generation(lead_times=tuple(lt), window=window)


def cleanup(*objs) -> None:
    for o in objs:
        try:
            if o is not None and hasattr(o, "dispose"):
                o.dispose()
        except Exception:
            pass
    gc.collect()
    time.sleep(0.3)


def pct(x):
    return "—" if x is None else f"{100.0 * x:.1f}%"


def run_im(scfg, seed, lt, window, time_limit=IM_LIMIT) -> dict:
    apply_scale(lt, window)
    raw = generate_instance(scfg, instance_id=seed % 1000, seed=seed)
    cm = capacity_metrics(raw)
    data = preprocess(raw)
    im = solve_im(data, time_limit=time_limit, mip_gap=0.0, verbose=False)
    util = analyze_im_utilization(data.raw, im) if im.objective is not None else {}
    gap = None
    if im.objective is not None and im.lower_bound is not None and im.objective > 1e-9:
        gap = max(0.0, (im.objective - im.lower_bound) / im.objective)
    row = {
        "seed": seed,
        "T": T,
        "U": U,
        "I": I,
        "M": M,
        "LT": list(lt),
        "window": dict(window),
        "status": im.status,
        "objective": im.objective,
        "lower_bound": im.lower_bound,
        "gap": gap,
        "runtime_sec": im.runtime_sec,
        "proved": im.status == "optimal",
        "tray_load_util": cm["tray_load_util"],
        "slot_density": cm["slot_density"],
        "machine_time_util": util.get("machine_time_util"),
        "avg_start_space_util": util.get("avg_start_space_util"),
        "span_weighted_space_util": util.get("span_weighted_space_util"),
        "num_batches": util.get("num_batches"),
        "scrap_qty": util.get("scrap_qty"),
    }
    cleanup(data, im, raw)
    return row


def util_ok(row: dict) -> bool:
    tu, su = row.get("machine_time_util"), row.get("avg_start_space_util")
    return tu is not None and su is not None and tu >= MIN_TIME_UTIL and su >= MIN_SPACE_UTIL


def write_doc(accepted, probe, lt, window, tag) -> None:
    lines = [
        "# IM opt ≤900s scale (regular T/I/M)",
        "",
        "## Goal",
        "",
        "- Hard baseline: `SetConfig('small_cg', 35, 5, 0, 50, 5)`, LT=(8,9,10)",
        "- Target regular scale: **T=25, U=5, I=35, M=5**",
        "- IM proves optimal within 900s; time_util ≥ 55%, space_util ≥ 70%",
        "- IM only (no BPC)",
        "",
        "## Chosen scale",
        "",
        f"- Tag: `{tag}`",
        f"- Config: `SetConfig('small_cg', {T}, {U}, 0, {I}, {M})` → **T={T}, U={U}, I={I}, M={M}**",
        f"- LT={list(lt)}, window={window}",
        "",
        "## Accepted instances",
        "",
        "| name | seed | T | U | I | M | LT | tray_load | time_util | space_util | obj | IM time (s) |",
        "|------|------|---|---|---|---|----|-----------|-----------|------------|-----|-------------|",
    ]
    for e in accepted:
        lines.append(
            f"| {e['name']} | {e['seed']} | {e['T']} | {e['U']} | {e['I']} | {e['M']} | {e['LT']} | "
            f"{pct(e['tray_load_util'])} | {pct(e['machine_time_util'])} | "
            f"{pct(e['avg_start_space_util'])} | {e['objective']:.2f} | {e['runtime_sec']:.1f} |"
        )
    lines += ["", "### Generation recipe", ""]
    for e in accepted:
        lines.append(
            f"- `{e['name']}`: `SetConfig('small_cg', {e['T']}, {e['U']}, 0, {e['I']}, {e['M']})`, "
            f"seed={e['seed']}, LT={e['LT']}, window={e['window']}"
        )
    lines += [
        "",
        "## Probe log",
        "",
        "| tag | seed | proved | IM time | gap | time_util | space_util | note |",
        "|-----|------|---------|---------|-----|-----------|------------|------|",
    ]
    for p in probe:
        g = "—" if p.get("gap") is None else f"{100 * p['gap']:.1f}%"
        lines.append(
            f"| {p.get('tag')} | {p['seed']} | {p['proved']} | {p['runtime_sec']:.1f} | {g} | "
            f"{pct(p.get('machine_time_util'))} | {pct(p.get('avg_start_space_util'))} | {p.get('note','')} |"
        )
    lines += ["", f"_Updated {time.strftime('%Y-%m-%d %H:%M:%S')}_", ""]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    OUT_JSON.write_text(
        json.dumps(
            {
                "chosen": {"T": T, "U": U, "I": I, "M": M, "LT": list(lt), "window": window, "tag": tag},
                "accepted": accepted,
                "probe": probe,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    scfg = SetConfig("small_cg", T, U, 0, I, M)
    probe: list[dict] = []
    chosen = None

    print(f"=== Probe T={T},U={U},I={I},M={M} ===", flush=True)
    for lt, win, tag in LT_CANDIDATES:
        print(f"\n-- LT={lt} ({tag}) --", flush=True)
        ok = []
        hard_fail = False
        for seed in PROBE_SEEDS:
            print(f"  probe seed={seed}...", flush=True)
            row = run_im(scfg, seed, lt, win)
            note = "ok" if row["proved"] and util_ok(row) else (
                "not_proved" if not row["proved"] else "util_low"
            )
            row["tag"] = tag
            row["note"] = note
            probe.append(row)
            print(
                f"    {row['status']} proved={row['proved']} t={row['runtime_sec']:.1f}s "
                f"gap={'—' if row['gap'] is None else f'{100*row['gap']:.1f}%'} "
                f"time_u={pct(row['machine_time_util'])} space_u={pct(row['avg_start_space_util'])} "
                f"note={note}",
                flush=True,
            )
            if not row["proved"]:
                # if clearly hard, skip remaining seeds for this LT
                if row["gap"] is not None and row["gap"] > 0.05:
                    hard_fail = True
                    break
            elif util_ok(row):
                ok.append(row)
        if hard_fail:
            print(f"  -> {tag} too hard", flush=True)
            continue
        if len(ok) >= 2:
            chosen = (lt, win, tag)
            print(f"  -> SELECTED {tag}", flush=True)
            break
        print(f"  -> {tag} not enough ok probes ({len(ok)})", flush=True)

    if chosen is None:
        write_doc([], probe, LT_CANDIDATES[0][0], LT_CANDIDATES[0][1], "none")
        print("No LT setting worked for T=25,U=5,I=35,M=5", flush=True)
        return

    lt, win, tag = chosen
    accepted: list[dict] = []
    print(f"\n=== Collect {TARGET} at {tag} ===", flush=True)
    # remove old mid26 dbs with same names
    for i in range(1, TARGET + 1):
        p = ROOT / "instances" / f"set_im_opt900_{i:02d}.db"
        if p.exists():
            p.unlink()

    for seed in COLLECT_SEEDS:
        if len(accepted) >= TARGET:
            break
        print(f"  collect seed={seed}...", flush=True)
        row = run_im(scfg, seed, lt, win)
        if not row["proved"] or not util_ok(row):
            print(
                f"    reject: proved={row['proved']} "
                f"time_u={pct(row.get('machine_time_util'))} "
                f"space_u={pct(row.get('avg_start_space_util'))}",
                flush=True,
            )
            continue
        idx = len(accepted) + 1
        name = f"set_im_opt900_{idx:02d}"
        apply_scale(lt, win)
        save_raw = generate_instance(scfg, instance_id=idx, seed=seed)
        save_raw.name = name
        persist_instance(save_raw)
        entry = {"name": name, **row}
        accepted.append(entry)
        write_doc(accepted, probe, lt, win, tag)
        print(
            f"    ACCEPT {name}: obj={row['objective']:.2f} t={row['runtime_sec']:.1f}s "
            f"time_u={pct(row['machine_time_util'])} space_u={pct(row['avg_start_space_util'])}",
            flush=True,
        )
        cleanup(save_raw)

    write_doc(accepted, probe, lt, win, tag)
    print(f"\nDone: {len(accepted)}/{TARGET}. See {OUT_MD}", flush=True)


if __name__ == "__main__":
    main()
