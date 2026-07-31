"""IM: 5 medium + 5 large, mip_gap=1%, one subprocess per instance (memory isolation).

Same seeds/limits as BPC batch in run_bpc_frac_gap1pct_5med_5large.py.
"""
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

from configuration import BENCH_LARGE, BENCH_MEDIUM  # noqa: E402

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
WORKER = ROOT / "experiments" / "_im_gap1pct_one.py"
OUT_JSON = ROOT / "supplement_materials" / "im_gap1pct_5med_5large.json"
OUT_MD = ROOT / "supplement_materials" / "im_gap1pct_5med_5large.md"
CMP_MD = ROOT / "supplement_materials" / "bpc_vs_im_gap1pct_5med_5large.md"
BPC_JSON = ROOT / "supplement_materials" / "bpc_frac_gap1pct_5med_5large.json"
TMP_DIR = ROOT / "supplement_materials" / "_im_gap1pct_tmp"


def _fmt(v, nd=2):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def write_md(meta: dict) -> None:
    lines = [
        "# IM：5 medium + 5 large（gap≤1% 停）",
        "",
        f"- 停止：Gurobi `MIPGap={MIP_GAP}` 或时限",
        f"- medium `{BENCH_MEDIUM}` @1800s；large `{BENCH_LARGE}` @3600s；M=U=5",
        "- 每例独立子进程 + `disposeDefaultEnv` + 进程退出",
        "",
        "| scale | seed | T | I | status | UB | LB | gap% | time(s) |",
        "|-------|------|---|---|--------|----|----|------|---------|",
    ]
    for r in meta["rows"]:
        if "error" in r:
            lines.append(
                f"| {r.get('scale', '')} | {r['seed']} | — | — | ERROR | — | — | — | "
                f"{str(r['error'])[:40]} |"
            )
            continue
        lines.append(
            f"| {r['scale']} | {r['seed']} | {r['T']} | {r['I']} | {r['status']} | "
            f"{_fmt(r.get('ub'))} | {_fmt(r.get('lb'))} | {_fmt(r.get('gap_pct'))} | "
            f"{_fmt(r.get('runtime_sec'), 1)} |"
        )
    ok = [r for r in meta["rows"] if "error" not in r]
    lines += ["", "## 小结", ""]
    if ok:
        gaps = [r["gap_pct"] for r in ok if r.get("gap_pct") is not None]
        times = [r["runtime_sec"] for r in ok if r.get("runtime_sec") is not None]
        n_opt = sum(1 for r in ok if r.get("status") == "optimal")
        n_tl = sum(1 for r in ok if r.get("status") == "time_limit")
        lines.append(f"- 完成 {len(ok)}/{len(JOBS)}：optimal={n_opt}，time_limit={n_tl}")
        if gaps:
            lines.append(
                f"- gap%：min={min(gaps):.3f}，max={max(gaps):.3f}，mean={sum(gaps)/len(gaps):.3f}"
            )
        if times:
            lines.append(
                f"- 时间(s)：min={min(times):.1f}，max={max(times):.1f}，mean={sum(times)/len(times):.1f}"
            )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_compare() -> None:
    if not (OUT_JSON.exists() and BPC_JSON.exists()):
        return
    im = json.loads(OUT_JSON.read_text(encoding="utf-8"))
    bpc = json.loads(BPC_JSON.read_text(encoding="utf-8"))
    im_map = {(r.get("scale"), r.get("seed")): r for r in im.get("rows", []) if "error" not in r}
    bpc_map = {(r.get("scale"), r.get("seed")): r for r in bpc.get("rows", []) if "error" not in r}
    lines = [
        "# BPC frac_sb vs IM（同种子，gap≤1% / 时限）",
        "",
        "| scale | seed | BPC status | BPC gap% | BPC s | IM status | IM gap% | IM s |",
        "|-------|------|------------|----------|-------|-----------|---------|------|",
    ]
    for scale, seed, _ in JOBS:
        b = bpc_map.get((scale, seed), {})
        i = im_map.get((scale, seed), {})
        lines.append(
            f"| {scale} | {seed} | {b.get('status', '—')} | {_fmt(b.get('gap_pct'))} | "
            f"{_fmt(b.get('runtime_sec'), 1)} | {i.get('status', '—')} | "
            f"{_fmt(i.get('gap_pct'))} | {_fmt(i.get('runtime_sec'), 1)} |"
        )
    CMP_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    meta = {
        "note": "IM mip_gap=1%: 5 medium (30101–30501) + 5 large (40101–40501)",
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
            f"\n[{i}/{len(JOBS)}] spawn IM {scale} seed={seed} limit={limit:.0f}s ...",
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
        write_compare()
        print(
            f"  done wall={wall:.1f}s rc={proc.returncode} -> "
            f"{row.get('status') or row.get('error')} gap={row.get('gap_pct')} "
            f"sec={row.get('runtime_sec')}",
            flush=True,
        )
        time.sleep(2.0)

    write_compare()
    print("\n=== DONE ===", flush=True)
    for r in rows:
        if "error" in r:
            print(f"{r.get('scale')} seed={r['seed']}: ERROR {r['error']}", flush=True)
        else:
            print(
                f"{r['scale']} seed={r['seed']}: status={r['status']} "
                f"gap={r['gap_pct']} sec={r['runtime_sec']}",
                flush=True,
            )
    print(f"\nJSON: {OUT_JSON}\nMD: {OUT_MD}\nCMP: {CMP_MD}", flush=True)


if __name__ == "__main__":
    main()
