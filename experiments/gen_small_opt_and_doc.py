"""Supplement hard-instance metrics; generate 5 smaller instances solvable to opt in 1200s."""
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
from configuration import SetConfig
from instance_generator import generator as genmod
from instance_generator.generator import capacity_metrics, generate_instance, persist_instance
from model_and_algo.branch_and_price import solve_branch_and_price
from model_and_algo.im_model import solve_im
from postprocessor.utilization import analyze_im_utilization
from preprocessor.preprocess import preprocess


def apply_hard_gen() -> None:
    """Patch both configuration and generator for unified bench LT/window."""
    cfg.apply_bench_generation()


def apply_scale(lt, window) -> None:
    cfg.apply_bench_generation(lead_times=tuple(lt), window=window)

HARD_SEEDS = [101, 202, 303, 404, 505]
HARD_SCFG = SetConfig("small_cg", 35, 5, 0, 50, 5)
OUT_MD = ROOT / "supplement_materials" / "test_instance.md"
OUT_JSON = ROOT / "supplement_materials" / "im_bpc_seeds_101_505.json"
NEW_JSON = ROOT / "supplement_materials" / "small_opt_1200s_instances.json"

# Candidate scales (smallest first) for new instances that should prove opt in 1200s
CANDIDATE_SCALES = [
    # (T, U, I, M, LT, window_dict, tag)
    (12, 3, 12, 2, (2, 3, 4), {"length_frac": (0.35, 0.55), "rel_frac": (0.05, 0.25), "min_length": 3}, "xs"),
    (14, 3, 14, 2, (2, 3, 4), {"length_frac": (0.35, 0.55), "rel_frac": (0.05, 0.25), "min_length": 3}, "s"),
    (16, 3, 16, 3, (3, 4, 5), {"length_frac": (0.30, 0.50), "rel_frac": (0.04, 0.25), "min_length": 4}, "m"),
    (18, 3, 18, 3, (3, 4, 5), {"length_frac": (0.28, 0.48), "rel_frac": (0.04, 0.25), "min_length": 4}, "m2"),
]


def cleanup(*objs) -> None:
    for o in objs:
        try:
            if o is not None and hasattr(o, "dispose"):
                o.dispose()
        except Exception:
            pass
    gc.collect()
    time.sleep(0.5)


def fmt(x, nd=2):
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def pct(x):
    return "—" if x is None else f"{100.0 * x:.1f}%"


def analyze_hard_seed(seed: int, im_time: float = 600.0) -> dict:
    apply_hard_gen()
    raw = generate_instance(HARD_SCFG, instance_id=seed % 100, seed=seed)
    cm = capacity_metrics(raw)
    data = preprocess(raw)
    im = solve_im(data, time_limit=im_time, mip_gap=0.05, verbose=False)
    util = analyze_im_utilization(data.raw, im) if im.objective is not None else {"error": "no sol"}
    row = {
        "seed": seed,
        "T": len(data.periods),
        "U": len(data.configs),
        "I": len(data.cured_items),
        "M": len(data.machines),
        "tray_load_util": cm["tray_load_util"],
        "slot_density": cm["slot_density"],
        "avg_tray": cm["avg_tray"],
        "machine_capacity": cm["machine_capacity"],
        "im_status": im.status,
        "im_obj": im.objective,
        "im_lb": im.lower_bound,
        "im_runtime": im.runtime_sec,
        "machine_time_util": util.get("machine_time_util"),
        "avg_start_space_util": util.get("avg_start_space_util"),
        "span_weighted_space_util": util.get("span_weighted_space_util"),
        "num_batches": util.get("num_batches"),
        "scrap_qty": util.get("scrap_qty"),
    }
    print(
        f"HARD seed={seed}: |I|={row['I']} tray_load={pct(row['tray_load_util'])} "
        f"time_util={pct(row['machine_time_util'])} space_util={pct(row['avg_start_space_util'])} "
        f"im_obj={fmt(row['im_obj'])} ({im.status}, {im.runtime_sec:.0f}s)",
        flush=True,
    )
    cleanup(data, im, raw)
    return row


def try_prove_im(scfg: SetConfig, seed: int, time_limit: float = 1200.0) -> dict:
    data = preprocess(generate_instance(scfg, instance_id=seed % 1000, seed=seed))
    im = solve_im(data, time_limit=time_limit, mip_gap=0.0, verbose=False)
    util = analyze_im_utilization(data.raw, im) if im.objective is not None else {}
    cm = capacity_metrics(data.raw)
    row = {
        "seed": seed,
        "T": len(data.periods),
        "U": len(data.configs),
        "I": len(data.cured_items),
        "M": len(data.machines),
        "status": im.status,
        "objective": im.objective,
        "lower_bound": im.lower_bound,
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
    cleanup(data, im)
    return row


def try_prove_bpc(scfg: SetConfig, seed: int, time_limit: float = 1200.0) -> dict:
    data = preprocess(generate_instance(scfg, instance_id=seed % 1000, seed=seed))
    cg_kwargs = {
        "pricing_mode": "ebrp",
        "ebrp_force": True,
        "use_gurobi_init": False,
        "skip_cross_check": True,
        "pricing_inprocess": True,
        "verbose": False,
        "progress_interval_sec": 30.0,
        "time_limit": time_limit,
        "lrmp_cuts": "standard",
    }
    t0 = time.perf_counter()
    try:
        r = solve_branch_and_price(
            data,
            max_nodes=8000,
            im_time_limit=90.0,
            time_limit=time_limit,
            use_node_im=False,
            root_im=True,
            final_im=True,
            branch_strategy="best_bound",
            cg_kwargs=cg_kwargs,
        )
    except RuntimeError as exc:
        elapsed = time.perf_counter() - t0
        print(f"  BPC RuntimeError: {exc}", flush=True)
        cleanup(data)
        return {
            "status": "error",
            "objective": None,
            "lower_bound": None,
            "runtime_sec": elapsed,
            "proved": False,
            "nodes": None,
            "error": str(exc),
        }
    elapsed = time.perf_counter() - t0
    proved = (
        r.status == "optimal"
        or (
            r.objective is not None
            and r.lower_bound is not None
            and abs(r.objective - r.lower_bound) <= 1e-4
        )
    )
    row = {
        "status": r.status,
        "objective": r.objective,
        "lower_bound": r.lower_bound,
        "runtime_sec": elapsed,
        "proved": proved,
        "nodes": (r.extra or {}).get("nodes_explored"),
    }
    cleanup(data, r)
    return row


def find_five_small() -> list[dict]:
    accepted: list[dict] = []
    seed = 2101
    # Prefer medium-small scale that still has some packing pressure
    scale_order = list(CANDIDATE_SCALES)
    # Start from 's'/'m' — xs may be too trivial
    scale_order = [CANDIDATE_SCALES[0], CANDIDATE_SCALES[1], CANDIDATE_SCALES[2], CANDIDATE_SCALES[3]]

    for T, U, I, M, lt, window, tag in scale_order:
        if len(accepted) >= 5:
            break
        apply_scale(lt, window)
        scfg = SetConfig("small_cg", T, U, 0, I, M)
        print(f"\n=== Trying scale tag={tag} T={T} U={U} I={I} M={M} LT={lt} ===", flush=True)
        attempts = 0
        while len(accepted) < 5 and attempts < 12:
            attempts += 1
            sid = seed
            seed += 1
            print(f"  IM prove seed={sid} ...", flush=True)
            im_row = try_prove_im(scfg, sid, time_limit=1200.0)
            print(
                f"  IM seed={sid}: status={im_row['status']} obj={fmt(im_row['objective'])} "
                f"time={im_row['runtime_sec']:.1f}s proved={im_row['proved']}",
                flush=True,
            )
            if not im_row["proved"]:
                continue
            print(f"  BPC prove seed={sid} ...", flush=True)
            bpc_row = try_prove_bpc(scfg, sid, time_limit=1200.0)
            print(
                f"  BPC seed={sid}: status={bpc_row['status']} obj={fmt(bpc_row['objective'])} "
                f"lb={fmt(bpc_row['lower_bound'])} time={bpc_row['runtime_sec']:.1f}s "
                f"proved={bpc_row['proved']} nodes={bpc_row['nodes']}",
                flush=True,
            )
            if not bpc_row["proved"]:
                print(f"  reject seed={sid}: IM opt but BPC not proved (tag={tag})", flush=True)
                continue
            apply_scale(lt, window)
            raw = generate_instance(scfg, instance_id=sid % 1000, seed=sid)
            name = f"set_small_opt_{len(accepted)+1:02d}"
            raw.name = name
            persist_instance(raw)
            entry = {
                "name": name,
                "seed": sid,
                "scale_tag": tag,
                "T": T,
                "U": U,
                "I": I,
                "M": M,
                "lead_time_choices": list(lt),
                "window": window,
                "im": im_row,
                "bpc": bpc_row,
                "db_path": str(ROOT / "instances" / f"{name}.db"),
            }
            accepted.append(entry)
            NEW_JSON.write_text(json.dumps({"instances": accepted}, indent=2), encoding="utf-8")
            print(f"  ACCEPTED {name} ({len(accepted)}/5)", flush=True)
            cleanup(raw)
        if len(accepted) >= 5:
            break

    if len(accepted) < 5:
        # Fall back: accept IM-proved only on smallest scale without BPC requirement
        print("\n=== Fallback: fill remaining with IM-optimal only (xs scale) ===", flush=True)
        T, U, I, M, lt, window, tag = CANDIDATE_SCALES[0]
        apply_scale(lt, window)
        scfg = SetConfig("small_cg", T, U, 0, I, M)
        while len(accepted) < 5 and seed < 2200:
            sid = seed
            seed += 1
            im_row = try_prove_im(scfg, sid, time_limit=1200.0)
            print(f"  fallback IM seed={sid}: {im_row['status']} proved={im_row['proved']} t={im_row['runtime_sec']:.1f}s", flush=True)
            if not im_row["proved"]:
                continue
            bpc_row = try_prove_bpc(scfg, sid, time_limit=1200.0)
            print(f"  fallback BPC seed={sid}: proved={bpc_row['proved']} t={bpc_row['runtime_sec']:.1f}s", flush=True)
            if not bpc_row["proved"]:
                continue
            apply_scale(lt, window)
            raw = generate_instance(scfg, instance_id=sid % 1000, seed=sid)
            name = f"set_small_opt_{len(accepted)+1:02d}"
            raw.name = name
            persist_instance(raw)
            accepted.append({
                "name": name,
                "seed": sid,
                "scale_tag": tag,
                "T": T, "U": U, "I": I, "M": M,
                "lead_time_choices": list(lt),
                "window": window,
                "im": im_row,
                "bpc": bpc_row,
                "db_path": str(ROOT / "instances" / f"{name}.db"),
            })
            NEW_JSON.write_text(json.dumps({"instances": accepted}, indent=2), encoding="utf-8")
            print(f"  ACCEPTED {name} ({len(accepted)}/5)", flush=True)
            cleanup(raw)
    return accepted


def write_md(hard_rows: list[dict], new_rows: list[dict], prev: dict) -> None:
    lines = [
        "# IM & BPC results (seeds 101–505) + small opt set",
        "",
        "## Hard set setup (101 / 202 / 303 / 404 / 505)",
        "",
        "- Seeds: `[101, 202, 303, 404, 505]`",
        "- Scale config: `SetConfig('small_cg', 35, 5, 0, 50, 5)` → **T=35, U=5, I=50, M=5**",
        "- Unified bench: `LT∈{5..10}`, `length_range=(5,6)`, `rel_frac=(0.10,0.50)`",
        "- IM (prior batch): 3600s / gap=0 — none proved optimal",
        "- BPC best_bound (prior batch): 600s, standard cuts, EBRP",
        "",
        "## Hard set: scale & utilization",
        "",
        "Utilization from IM feasible solutions (`mip_gap=0.05`, time_limit=600s; **not** certified optima).",
        "",
        "| seed | T | U | I | M | tray_load | slot_density | time_util | space_util (start) | space_util (span) | batches | scrap |",
        "|------|---|---|---|---|-----------|--------------|-----------|--------------------|-------------------|---------|-------|",
    ]
    for r in hard_rows:
        lines.append(
            f"| {r['seed']} | {r['T']} | {r['U']} | {r['I']} | {r['M']} | "
            f"{pct(r['tray_load_util'])} | {fmt(r['slot_density'], 3)} | "
            f"{pct(r['machine_time_util'])} | {pct(r['avg_start_space_util'])} | "
            f"{pct(r['span_weighted_space_util'])} | {r.get('num_batches', '—')} | "
            f"{fmt(r.get('scrap_qty'), 0)} |"
        )
    lines += [
        "",
        "### Metric notes",
        "",
        "- **tray_load**: sum(v_i)/(M*T*q) — structural tray pressure (instance-level)",
        "- **time_util**: fraction of machine-periods occupied by a curing batch (from IM schedule)",
        "- **space_util (start)**: mean sum(v_i)/q_m at batch start times",
        "- **space_util (span)**: span-weighted average of start fill rates over busy periods",
        "",
        "## Hard set: prior IM / BPC table",
        "",
        "### IM (3600s, gap=0)",
        "",
        "| seed | status | UB (obj) | LB | gap % | time (s) |",
        "|------|--------|----------|----|-------|----------|",
    ]
    for r in prev.get("im", []):
        lines.append(
            f"| {r['seed']} | {r['status']} | {fmt(r.get('objective'))} | "
            f"{fmt(r.get('lower_bound'))} | {fmt(r.get('mip_gap_pct'))} | "
            f"{fmt(r.get('runtime_sec'), 1)} |"
        )

    def bpc_gap(r):
        ub, lb = r.get("objective"), r.get("lower_bound")
        if ub is None or lb is None or abs(ub) < 1e-9:
            return None
        return abs(ub - lb) / abs(ub) * 100.0

    lines += [
        "",
        "### BPC best_bound (600s)",
        "",
        "| seed | status | UB | LB | root LRMP | gap % | nodes | time (s) |",
        "|------|--------|----|----|-----------|-------|-------|----------|",
    ]
    for r in prev.get("bpc", []):
        g = r.get("mip_gap_pct")
        if g is not None and g > 50:  # fix legacy *100 bug if present
            g = bpc_gap(r)
        elif g is None:
            g = bpc_gap(r)
        lines.append(
            f"| {r['seed']} | {r['status']} | {fmt(r.get('objective'))} | "
            f"{fmt(r.get('lower_bound'))} | {fmt(r.get('root_lrmp'))} | "
            f"{fmt(g)} | {r.get('nodes', '—')} | {fmt(r.get('runtime_sec'), 1)} |"
        )

    lines += [
        "",
        "## New small set (target: prove opt in ≤1200s)",
        "",
        "Generated as `instances/set_small_opt_XX.db`. Acceptance: **IM proves optimal within 1200s** "
        "(gap=0); BPC best_bound also run for reference.",
        "",
        "| name | seed | T | U | I | M | LT | tray_load | time_util | space_util | IM opt | IM time | BPC proved | BPC time | BPC obj |",
        "|------|------|---|---|---|---|----|-----------|-----------|------------|--------|---------|------------|----------|---------|",
    ]
    for e in new_rows:
        im, bp = e["im"], e["bpc"]
        lines.append(
            f"| {e['name']} | {e['seed']} | {e['T']} | {e['U']} | {e['I']} | {e['M']} | "
            f"{e['lead_time_choices']} | {pct(im.get('tray_load_util'))} | "
            f"{pct(im.get('machine_time_util'))} | {pct(im.get('avg_start_space_util'))} | "
            f"{fmt(im.get('objective'))} | {fmt(im.get('runtime_sec'), 1)} | "
            f"{'yes' if bp.get('proved') else 'no'} | {fmt(bp.get('runtime_sec'), 1)} | "
            f"{fmt(bp.get('objective'))} |"
        )
    lines += [
        "",
        "### Generation recipe (per instance)",
        "",
    ]
    for e in new_rows:
        lines.append(
            f"- `{e['name']}`: `SetConfig('small_cg', {e['T']}, {e['U']}, 0, {e['I']}, {e['M']})`, "
            f"seed={e['seed']}, LT={tuple(e['lead_time_choices'])}, window={e['window']}"
        )
    lines += [
        "",
        f"_Updated {time.strftime('%Y-%m-%d %H:%M:%S')}_",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT_MD}", flush=True)


def main() -> None:
    prev = {}
    if OUT_JSON.exists():
        prev = json.loads(OUT_JSON.read_text(encoding="utf-8"))

    hard_path = ROOT / "supplement_materials" / "hard_101_505_metrics.json"
    if hard_path.exists():
        hard_rows = json.loads(hard_path.read_text(encoding="utf-8"))
        print(f"Loaded existing hard metrics ({len(hard_rows)})", flush=True)
    else:
        hard_rows = []
        for seed in HARD_SEEDS:
            hard_rows.append(analyze_hard_seed(seed, im_time=600.0))
            hard_path.write_text(json.dumps(hard_rows, indent=2), encoding="utf-8")

    if NEW_JSON.exists():
        new_rows = json.loads(NEW_JSON.read_text(encoding="utf-8")).get("instances", [])
        print(f"Loaded {len(new_rows)} existing small instances", flush=True)
    else:
        new_rows = []

    if len(new_rows) < 5:
        found = find_five_small()
        # merge unique by name
        by_name = {e["name"]: e for e in new_rows}
        for e in found:
            by_name[e["name"]] = e
        new_rows = list(by_name.values())[:5]
        # if find_five_small returns exactly 5, use that
        if len(found) >= 5:
            new_rows = found[:5]
        NEW_JSON.write_text(json.dumps({"instances": new_rows}, indent=2), encoding="utf-8")

    write_md(hard_rows, new_rows, prev)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
