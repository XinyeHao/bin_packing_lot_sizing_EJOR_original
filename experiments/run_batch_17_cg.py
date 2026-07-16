"""17 instances CG+Gurobi vs CG+EBRP benchmark (excludes intractable bench_01-03)."""
from __future__ import annotations

import json, sys, time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocessor.preprocess import load_instance, preprocess
from model_and_algo.column_generation import solve_column_generation

INSTANCE_NAMES = [
    "demo_small", "large_01",
    "set_b_bench_01",
    "set_small_cg_01", "set_small_cg_02", "set_small_cg_03",
    "set_small_cg_04", "set_small_cg_05", "set_small_cg_06",
    "set_med_cg_01", "set_med_cg_02", "set_med_cg_03",
    "set_a_01", "set_a_02", "set_a_03", "set_a_04", "set_a_05",
]

MODE_SPEC: dict[str, dict[str, Any]] = {
    "im_gurobi": {"pricing_mode": "im_gurobi", "use_gurobi_init": False},
    "ebrp": {"pricing_mode": "ebrp", "use_gurobi_init": False, "ebrp_force": True},
}

BASE_KWARGS: dict[str, Any] = {
    "max_iterations": 5000,
    "verbose": False,
    "progress_interval_sec": 0.0,
    "ebrp_time_limit": None,
    "im_pricing_time_limit": 10.0,
    "pricing_workers": 0,
}


def cg_time_limit(name: str) -> float:
    if any(p in name for p in ("small_cg", "demo", "large", "set_a")):
        return 300.0
    if "med_cg" in name:
        return 300.0
    return 180.0


def run_one(name: str, mode: str) -> dict:
    label = f"{name}/{mode}"
    print(f"  {label} ...", end=" ", flush=True)
    t0 = time.perf_counter()
    data = preprocess(load_instance(name))
    try:
        result = solve_column_generation(
            data,
            **{**BASE_KWARGS, **MODE_SPEC[mode], "time_limit": cg_time_limit(name)}
        )
    except Exception as exc:
        return {"instance": name, "pricing": mode, "error": str(exc)}
    elapsed = time.perf_counter() - t0
    ex = result.extra or {}
    conv = ex.get("converged")
    iters = ex.get("num_iterations")
    cols = sum((ex.get("num_columns") or {}).values())
    lrmp = result.lower_bound
    print(f"LRMP={lrmp:.1f} it={iters} cols={cols} conv={conv} {elapsed:.1f}s", flush=True)
    return {
        "instance": name, "pricing": mode,
        "T": len(data.periods), "I": len(data.cured_items), "M": len(data.machines),
        "LRMP": lrmp, "iters": iters, "cols": cols, "converged": conv,
        "stop_reason": ex.get("stop_reason"), "last_min_rc": ex.get("last_min_rc"),
        "time_sec": round(elapsed, 1), "status": result.status,
    }


def main() -> None:
    rows: list[dict] = []
    out_path = ROOT / "result" / "batch_17_cg_compare.json"

    for name in INSTANCE_NAMES:
        for mode in ("im_gurobi", "ebrp"):
            rows.append(run_one(name, mode))
            out_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    # ── Summary ──
    print()
    print("=" * 90)
    hdr = f"{'instance':<20} {'G_LRMP':>10} {'E_LRMP':>10} {'delta':>8} {'G_t':>6} {'E_t':>6} {'spd':>7}"
    print(hdr)
    print("-" * 85)

    seen: dict[str, dict[str, dict]] = {}
    for r in rows:
        if r.get("error"):
            continue
        seen.setdefault(r["instance"], {})[r["pricing"]] = r

    for name in INSTANCE_NAMES:
        g = seen.get(name, {}).get("im_gurobi")
        e = seen.get(name, {}).get("ebrp")
        if g and e:
            delta = e["LRMP"] - g["LRMP"]
            spd = f"{g['time_sec']/e['time_sec']:.1f}x" if e["time_sec"] > 0 else "N/A"
            mark = "  !!!" if abs(delta) > 1.0 else ""
            print(f"{name:<20} {g['LRMP']:10.2f} {e['LRMP']:10.2f} {delta:+8.2f} {g['time_sec']:5.1f}s {e['time_sec']:5.1f}s {spd:>7}{mark}")
        elif g:
            print(f"{name:<20} {g['LRMP']:10.2f} {'FAILED':>10}")
        elif e:
            print(f"{name:<20} {'FAILED':>10} {e['LRMP']:10.2f}")

    pairs = [(g, e) for name in INSTANCE_NAMES
             if (g := seen.get(name, {}).get("im_gurobi"))
             and (e := seen.get(name, {}).get("ebrp"))]
    if pairs:
        avg_spd = sum(g["time_sec"] / e["time_sec"] for g, e in pairs if e["time_sec"] > 0) / max(1, len(pairs))
        matches = sum(1 for g, e in pairs if abs(e["LRMP"] - g["LRMP"]) < 1.0)
        print(f"\n  LRMP matches (<1.0): {matches}/{len(pairs)}")
        print(f"  Average speedup: {avg_spd:.1f}x")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
