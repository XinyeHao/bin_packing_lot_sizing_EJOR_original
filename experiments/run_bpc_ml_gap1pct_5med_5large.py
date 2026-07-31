"""BPC+ML: 5 medium + 5 large, mip_gap=1%; one subprocess per instance; compare vs frac_sb after each."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from configuration import BENCH_LARGE, BENCH_MEDIUM, ML_SB_REFINE_K  # noqa: E402

JOBS = [
    ("medium", 30101, 1800.0),
    ("medium", 30201, 1800.0),
    ("medium", 30301, 1800.0),
    ("medium", 30401, 1800.0),
    ("medium", 30501, 1800.0),
    ("large", 40101, 3600.0),
    ("large", 40201, 3600.0),
    ("large", 40301, 3600.0),
    ("large", 40401, 3600.0),
    ("large", 40501, 3600.0),
]
MIP_GAP = 0.01
WORKER = ROOT / "experiments" / "_bpc_ml_gap1pct_one.py"
FRAC_JSON = ROOT / "supplement_materials" / "bpc_frac_gap1pct_5med_5large.json"
OUT_JSON = ROOT / "supplement_materials" / "bpc_ml_gap1pct_5med_5large.json"
OUT_MD = ROOT / "supplement_materials" / "bpc_ml_gap1pct_5med_5large.md"
TMP_DIR = ROOT / "supplement_materials" / "_bpc_ml_gap1pct_tmp"


def _fmt(v, nd=2):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _load_frac() -> dict[tuple[str, int], dict]:
    if not FRAC_JSON.exists():
        return {}
    data = json.loads(FRAC_JSON.read_text(encoding="utf-8"))
    out: dict[tuple[str, int], dict] = {}
    for r in data.get("rows", []):
        if "error" in r:
            continue
        out[(r["scale"], int(r["seed"]))] = r
    return out


def _print_compare(ml: dict, frac: dict | None, i: int, n: int) -> None:
    seed = ml.get("seed")
    scale = ml.get("scale")
    print(f"\n===== [{i}/{n}] {scale} seed={seed} 汇总 vs frac_sb =====", flush=True)
    if "error" in ml:
        print(f"  ML ERROR: {ml['error']}", flush=True)
        if frac:
            print(
                f"  frac_sb: status={frac.get('status')} gap={frac.get('gap_pct')}% "
                f"sec={frac.get('runtime_sec')}",
                flush=True,
            )
        return
    print(
        f"  ML     : status={ml.get('status')} UB={_fmt(ml.get('ub'))} LB={_fmt(ml.get('lb'))} "
        f"gap={_fmt(ml.get('gap_pct'))}% time={_fmt(ml.get('runtime_sec'), 1)}s "
        f"nodes={ml.get('nodes_explored')} cg_fail={ml.get('nodes_cg_failed')}",
        flush=True,
    )
    if not frac:
        print("  frac_sb: (no baseline row)", flush=True)
        return
    print(
        f"  frac_sb: status={frac.get('status')} UB={_fmt(frac.get('ub'))} LB={_fmt(frac.get('lb'))} "
        f"gap={_fmt(frac.get('gap_pct'))}% time={_fmt(frac.get('runtime_sec'), 1)}s "
        f"nodes={frac.get('nodes_explored')} cg_fail={frac.get('nodes_cg_failed')}",
        flush=True,
    )
    mt, ft = ml.get("runtime_sec"), frac.get("runtime_sec")
    mg, fg = ml.get("gap_pct"), frac.get("gap_pct")
    if mt is not None and ft is not None:
        d = mt - ft
        print(f"  Δtime   : {d:+.1f}s  (ML − frac)", flush=True)
    if mg is not None and fg is not None:
        print(f"  Δgap%   : {mg - fg:+.4f}  (ML − frac)", flush=True)


def write_md(meta: dict, frac_map: dict[tuple[str, int], dict]) -> None:
    lines = [
        "# BPC+ML：5 medium + 5 large（gap≤1% 停）vs frac_sb",
        "",
        f"- 策略：`ml`（粗排 Top-{ML_SB_REFINE_K} + 短 SB），模型 `ml_branch_model.txt`",
        f"- 停止：`(UB−tree_LB)/UB ≤ {100 * MIP_GAP:.0f}%` 或时限",
        f"- medium `{BENCH_MEDIUM}` @1800s；large `{BENCH_LARGE}` @3600s；M=U=5",
        "- 每例独立子进程，结束后 `disposeDefaultEnv` + 进程退出以释放内存",
        f"- 对照：`{FRAC_JSON.name}`",
        "",
        "| scale | seed | ML status | ML gap% | ML time | frac status | frac gap% | frac time | Δtime | Δgap |",
        "|-------|------|-----------|---------|---------|-------------|-----------|-----------|-------|------|",
    ]
    for r in meta["rows"]:
        key = (r.get("scale"), int(r["seed"])) if r.get("seed") is not None else None
        f = frac_map.get(key) if key else None
        if "error" in r:
            lines.append(
                f"| {r.get('scale', '')} | {r['seed']} | ERROR | — | — | "
                f"{f.get('status') if f else '—'} | {_fmt(f.get('gap_pct') if f else None)} | "
                f"{_fmt(f.get('runtime_sec') if f else None, 1)} | — | — |"
            )
            continue
        mt, ft = r.get("runtime_sec"), (f or {}).get("runtime_sec")
        mg, fg = r.get("gap_pct"), (f or {}).get("gap_pct")
        dtime = "—" if mt is None or ft is None else f"{mt - ft:+.1f}"
        dgap = "—" if mg is None or fg is None else f"{mg - fg:+.3f}"
        lines.append(
            f"| {r['scale']} | {r['seed']} | {r.get('status')} | {_fmt(mg)} | {_fmt(mt, 1)} | "
            f"{(f or {}).get('status', '—')} | {_fmt(fg)} | {_fmt(ft, 1)} | {dtime} | {dgap} |"
        )

    ok = [r for r in meta["rows"] if "error" not in r]
    lines += ["", "## 小结", ""]
    if ok:
        gaps = [r["gap_pct"] for r in ok if r.get("gap_pct") is not None]
        times = [r["runtime_sec"] for r in ok if r.get("runtime_sec") is not None]
        n_gap = sum(1 for r in ok if r.get("status") == "gap_limit")
        n_opt = sum(1 for r in ok if r.get("status") == "optimal")
        n_tl = sum(1 for r in ok if r.get("status") == "time_limit")
        lines.append(
            f"- ML 完成 {len(ok)}/{len(JOBS)}：`gap_limit`={n_gap}，`optimal`={n_opt}，`time_limit`={n_tl}"
        )
        if gaps:
            lines.append(
                f"- ML gap%：min={min(gaps):.3f}，max={max(gaps):.3f}，mean={sum(gaps)/len(gaps):.3f}"
            )
        if times:
            lines.append(
                f"- ML 时间(s)：min={min(times):.1f}，max={max(times):.1f}，mean={sum(times)/len(times):.1f}"
            )
        # paired means vs frac
        paired = []
        for r in ok:
            f = frac_map.get((r["scale"], int(r["seed"])))
            if f and r.get("runtime_sec") is not None and f.get("runtime_sec") is not None:
                paired.append((r, f))
        if paired:
            ml_t = sum(r["runtime_sec"] for r, _ in paired) / len(paired)
            fr_t = sum(f["runtime_sec"] for _, f in paired) / len(paired)
            ml_g = sum(r["gap_pct"] for r, _ in paired if r.get("gap_pct") is not None)
            fr_g = sum(f["gap_pct"] for _, f in paired if f.get("gap_pct") is not None)
            n_g = sum(1 for r, _ in paired if r.get("gap_pct") is not None)
            lines.append(
                f"- 已完成对照 {len(paired)} 例：mean time ML={ml_t:.1f}s vs frac={fr_t:.1f}s "
                f"(Δ={ml_t - fr_t:+.1f}s)"
            )
            if n_g:
                lines.append(
                    f"- 已完成对照 mean gap% ML={ml_g/n_g:.3f} vs frac={fr_g/n_g:.3f} "
                    f"(Δ={ml_g/n_g - fr_g/n_g:+.3f})"
                )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    model = ROOT / "ml_branch_model.txt"
    assert model.exists(), f"missing {model}"
    assert WORKER.exists(), f"missing {WORKER}"
    frac_map = _load_frac()
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    meta = {
        "note": "BPC ml mip_gap=1%: 5 medium (30101–30501) + 5 large (40101–40501)",
        "branch_strategy": "ml",
        "ml_sb_refine_k": ML_SB_REFINE_K,
        "model": str(model),
        "model_mtime": model.stat().st_mtime,
        "mip_gap": MIP_GAP,
        "frac_baseline": str(FRAC_JSON),
        "memory": "one subprocess per instance + gp.disposeDefaultEnv + os._exit",
        "jobs": [{"scale": s, "seed": seed, "limit": lim} for s, seed, lim in JOBS],
        "rows": rows,
    }
    OUT_JSON.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    write_md(meta, frac_map)

    py = sys.executable
    for i, (scale, seed, limit) in enumerate(JOBS, 1):
        one_json = TMP_DIR / f"{scale}_{seed}.json"
        if one_json.exists():
            one_json.unlink()
        log_path = TMP_DIR / f"{scale}_{seed}.log"
        print(
            f"\n[{i}/{len(JOBS)}] spawn ML {scale} seed={seed} limit={limit:.0f}s ...",
            flush=True,
        )
        t0 = time.perf_counter()
        with open(log_path, "w", encoding="utf-8") as logf:
            proc = subprocess.run(
                [
                    py,
                    "-u",
                    str(WORKER),
                    scale,
                    str(seed),
                    str(limit),
                    str(MIP_GAP),
                    str(one_json),
                ],
                cwd=str(ROOT),
                stdout=logf,
                stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONUNBUFFERED": "1", "GRB_CS_QUIET": "1"},
            )
        wall = time.perf_counter() - t0
        if one_json.exists():
            row = json.loads(one_json.read_text(encoding="utf-8"))
        else:
            row = {
                "scale": scale,
                "seed": seed,
                "error": f"worker rc={proc.returncode}, no output json (see {log_path.name})",
            }
        rows.append(row)
        OUT_JSON.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        write_md(meta, frac_map)
        print(
            f"  worker wall={wall:.1f}s rc={proc.returncode}",
            flush=True,
        )
        _print_compare(row, frac_map.get((scale, seed)), i, len(JOBS))
        time.sleep(2.0)

    print("\n=== ALL DONE ===", flush=True)
    print(f"JSON: {OUT_JSON}\nMD:   {OUT_MD}", flush=True)


if __name__ == "__main__":
    main()
