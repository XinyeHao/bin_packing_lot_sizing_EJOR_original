"""BPC frac_sb: 5 medium + 5 large, mip_gap=1%, one subprocess per instance (memory isolation)."""
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

from configuration import BENCH_LARGE, BENCH_MEDIUM, FRAC_SB_REFINE_K  # noqa: E402

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
WORKER = ROOT / "experiments" / "_bpc_frac_gap1pct_one.py"
OUT_JSON = ROOT / "supplement_materials" / "bpc_frac_gap1pct_5med_5large.json"
OUT_MD = ROOT / "supplement_materials" / "bpc_frac_gap1pct_5med_5large.md"
TMP_DIR = ROOT / "supplement_materials" / "_bpc_frac_gap1pct_tmp"


def _fmt(v, nd=2):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def write_md(meta: dict) -> None:
    lines = [
        "# BPC frac_sb：5 medium + 5 large（gap≤1% 停）",
        "",
        f"- 策略：`frac_sb`（k={FRAC_SB_REFINE_K}），无 ML",
        f"- 停止：`(UB−tree_LB)/UB ≤ {100 * MIP_GAP:.0f}%` 或时限",
        f"- medium `{BENCH_MEDIUM}` @1800s；large `{BENCH_LARGE}` @3600s；M=U=5",
        "- 每例独立子进程，结束后 `disposeDefaultEnv` + 进程退出以释放内存",
        "",
        "| scale | seed | T | I | status | UB | LB | gap% | time(s) | nodes | cg_fail |",
        "|-------|------|---|---|--------|----|----|------|---------|-------|---------|",
    ]
    for r in meta["rows"]:
        if "error" in r:
            lines.append(
                f"| {r.get('scale', '')} | {r['seed']} | — | — | ERROR | — | — | — | — | — | "
                f"{str(r['error'])[:48]} |"
            )
            continue
        lines.append(
            f"| {r['scale']} | {r['seed']} | {r['T']} | {r['I']} | {r['status']} | "
            f"{_fmt(r.get('ub'))} | {_fmt(r.get('lb'))} | {_fmt(r.get('gap_pct'))} | "
            f"{_fmt(r.get('runtime_sec'), 1)} | {r.get('nodes_explored')} | "
            f"{r.get('nodes_cg_failed')} |"
        )
    ok = [r for r in meta["rows"] if "error" not in r]
    n_gap = sum(1 for r in ok if r.get("status") == "gap_limit")
    n_opt = sum(1 for r in ok if r.get("status") == "optimal")
    n_tl = sum(1 for r in ok if r.get("status") == "time_limit")
    lines += [
        "",
        "## 小结",
        "",
        f"- 完成 {len(ok)}/{len(JOBS)}：`gap_limit`={n_gap}，`optimal`={n_opt}，`time_limit`={n_tl}",
    ]
    if ok:
        gaps = [r["gap_pct"] for r in ok if r.get("gap_pct") is not None]
        times = [r["runtime_sec"] for r in ok if r.get("runtime_sec") is not None]
        if gaps:
            lines.append(
                f"- gap%：min={min(gaps):.3f}，max={max(gaps):.3f}，mean={sum(gaps)/len(gaps):.3f}"
            )
        if times:
            lines.append(
                f"- 时间(s)：min={min(times):.1f}，max={max(times):.1f}，mean={sum(times)/len(times):.1f}"
            )
        lines.append(
            f"- 合计 `nodes_cg_failed`={sum(int(r.get('nodes_cg_failed') or 0) for r in ok)}"
        )
        med = [r for r in ok if r.get("scale") == "medium"]
        large = [r for r in ok if r.get("scale") == "large"]
        if med:
            mg = [r["gap_pct"] for r in med if r.get("gap_pct") is not None]
            mt = [r["runtime_sec"] for r in med if r.get("runtime_sec") is not None]
            lines.append(
                f"- medium：n={len(med)}，mean gap%={sum(mg)/len(mg):.3f}，mean time={sum(mt)/len(mt):.1f}s"
            )
        if large:
            lg = [r["gap_pct"] for r in large if r.get("gap_pct") is not None]
            lt = [r["runtime_sec"] for r in large if r.get("runtime_sec") is not None]
            lines.append(
                f"- large：n={len(large)}，mean gap%={sum(lg)/len(lg):.3f}，mean time={sum(lt)/len(lt):.1f}s"
            )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    meta = {
        "note": "BPC frac_sb mip_gap=1%: 5 medium (30101–30501) + 5 large (40101–40501)",
        "frac_sb_refine_k": FRAC_SB_REFINE_K,
        "mip_gap": MIP_GAP,
        "memory": "one subprocess per instance + gp.disposeDefaultEnv + os._exit",
        "jobs": [{"scale": s, "seed": seed, "limit": lim} for s, seed, lim in JOBS],
        "rows": rows,
    }
    OUT_JSON.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    write_md(meta)

    py = sys.executable
    for i, (scale, seed, limit) in enumerate(JOBS, 1):
        one_json = TMP_DIR / f"{scale}_{seed}.json"
        if one_json.exists():
            one_json.unlink()
        log_path = TMP_DIR / f"{scale}_{seed}.log"
        print(
            f"\n[{i}/{len(JOBS)}] spawn {scale} seed={seed} limit={limit:.0f}s ...",
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
        write_md(meta)
        status = row.get("status") or row.get("error", "?")
        print(
            f"  done wall={wall:.1f}s rc={proc.returncode} -> {status} "
            f"gap={row.get('gap_pct')} sec={row.get('runtime_sec')}",
            flush=True,
        )
        # give OS a moment after child exit
        time.sleep(2.0)

    print("\n=== DONE ===", flush=True)
    for r in rows:
        if "error" in r:
            print(f"{r.get('scale')} seed={r['seed']}: ERROR {r['error']}", flush=True)
        else:
            print(
                f"{r['scale']} seed={r['seed']}: status={r['status']} "
                f"gap={r['gap_pct']} sec={r['runtime_sec']} cg_fail={r['nodes_cg_failed']}",
                flush=True,
            )
    print(f"\nJSON: {OUT_JSON}\nMD:   {OUT_MD}", flush=True)


if __name__ == "__main__":
    main()
