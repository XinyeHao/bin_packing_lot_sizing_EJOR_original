"""10 large instances × three methods under the common-unload assumption.

Seeds 41101–42001 (avoid 40101–40501). mip_gap=1%, limit=3600s.
One subprocess per solve + disposeDefaultEnv + os._exit for memory isolation.
After each solve: update JSON/MD and print running three-way summary.
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

from configuration import BENCH_LARGE  # noqa: E402

# 10 new large seeds (step 100), disjoint from 40101–40501
SEEDS = [41101 + 100 * i for i in range(10)]  # 41101 … 42001
LIMIT = 3600.0
MIP_GAP = 0.01
SCALE = "large"

WORKERS = {
    "im": ROOT / "experiments" / "_im_gap1pct_one.py",
    "frac": ROOT / "experiments" / "_bpc_frac_gap1pct_one.py",
    "ml": ROOT / "experiments" / "_bpc_ml_gap1pct_one.py",
}
METHOD_ORDER = ("im", "frac", "ml")

OUT_JSON = ROOT / "supplement_materials" / "three_method_10large_common_unload_v2.json"
OUT_MD = ROOT / "supplement_materials" / "three_method_10large_common_unload_v2.md"
TMP_DIR = ROOT / "supplement_materials" / "_three_method_10large_common_unload_v2_tmp"
PROCESS_ASSUMPTION = (
    "All items in a batch leave the autoclave and enter inventory together "
    "at the batch maximum curing duration."
)


def _fmt(v, nd=2):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def write_outputs(meta: dict) -> None:
    by_seed: dict[int, dict] = {}
    for r in meta["rows"]:
        seed = int(r["seed"])
        method = r.get("method", "?")
        by_seed.setdefault(seed, {})[method] = r

    lines = [
        "# 大规模 10 例：IM · BPC-frac · BPC-ML（共同出罐）",
        "",
        f"- 种子：`{SEEDS[0]}`–`{SEEDS[-1]}`（步长 100），与旧 `40101–40501` 不重叠",
        f"- 规模：large `{BENCH_LARGE}`，时限 {LIMIT:.0f}s，mip_gap={100 * MIP_GAP:.0f}%",
        "- 工艺假设：同一批次工件按该批次最长固化时间共同出罐并进入库存。",
        "- 每例每次求解独立子进程；结束后 disposeDefaultEnv + 进程退出",
        "",
        "| seed | IM status | IM gap% | IM s | frac status | frac gap% | frac s | ML status | ML gap% | ML s |",
        "|------|-----------|---------|------|-------------|-----------|--------|-----------|---------|------|",
    ]
    for seed in SEEDS:
        m = by_seed.get(seed, {})
        cells = [str(seed)]
        for meth in METHOD_ORDER:
            r = m.get(meth)
            if not r:
                cells += ["—", "—", "—"]
            elif "error" in r:
                cells += ["ERROR", "—", "—"]
            else:
                cells += [
                    str(r.get("status", "—")),
                    _fmt(r.get("gap_pct")),
                    _fmt(r.get("runtime_sec"), 1),
                ]
        lines.append("| " + " | ".join(cells) + " |")

    done_seeds = [s for s in SEEDS if all(m in by_seed.get(s, {}) for m in METHOD_ORDER)]
    lines += ["", "## 小结（已完成三方法的种子）", "", f"- 完成三方法：{len(done_seeds)}/{len(SEEDS)}"]
    if done_seeds:
        for meth, label in (("im", "IM"), ("frac", "frac"), ("ml", "ML")):
            rows = [by_seed[s][meth] for s in done_seeds if "error" not in by_seed[s][meth]]
            gaps = [r["gap_pct"] for r in rows if r.get("gap_pct") is not None]
            times = [r["runtime_sec"] for r in rows if r.get("runtime_sec") is not None]
            n_gap1 = sum(
                1 for r in rows if r.get("gap_pct") is not None and r["gap_pct"] <= 1.0 + 1e-9
            )
            if gaps and times:
                lines.append(
                    f"- {label}：mean gap%={_mean(gaps):.3f}，mean time={_mean(times):.1f}s，"
                    f"gap≤1% = {n_gap1}/{len(rows)}"
                )

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    OUT_JSON.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def run_one(method: str, seed: int) -> dict:
    worker = WORKERS[method]
    assert worker.exists(), f"missing {worker}"
    one_json = TMP_DIR / f"{method}_{seed}.json"
    log_path = TMP_DIR / f"{method}_{seed}.log"
    if one_json.exists():
        one_json.unlink()
    t0 = time.perf_counter()
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.run(
            [
                sys.executable,
                "-u",
                str(worker),
                SCALE,
                str(seed),
                str(LIMIT),
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
            "scale": SCALE,
            "seed": seed,
            "error": f"worker rc={proc.returncode}, no json ({log_path.name})",
        }
    row["method"] = method
    row["wall_sec"] = round(wall, 2)
    row["worker_rc"] = proc.returncode
    return row


def main() -> None:
    model = ROOT / "ml_branch_model.txt"
    assert model.exists(), f"missing {model} (needed for ML)"
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    meta = {
        "note": "10 large seeds × IM / BPC-frac / BPC-ML, mip_gap=1%, common unload",
        "process_assumption": PROCESS_ASSUMPTION,
        "model_version": "common_unload_v2_span_link",
        "seeds": SEEDS,
        "scale": SCALE,
        "T_I": list(BENCH_LARGE),
        "time_limit": LIMIT,
        "mip_gap": MIP_GAP,
        "memory": "one subprocess per solve + gp.disposeDefaultEnv + os._exit",
        "methods": list(METHOD_ORDER),
        "rows": [],
    }
    write_outputs(meta)

    total = len(SEEDS) * len(METHOD_ORDER)
    k = 0
    for seed in SEEDS:
        print(f"\n######## seed={seed} large {BENCH_LARGE} ########", flush=True)
        for method in METHOD_ORDER:
            k += 1
            print(f"\n[{k}/{total}] {method} seed={seed} ...", flush=True)
            row = run_one(method, seed)
            meta["rows"].append(row)
            write_outputs(meta)
            if "error" in row:
                print(f"  FAIL {row['error']}", flush=True)
            else:
                print(
                    f"  -> {method}: status={row.get('status')} "
                    f"gap={row.get('gap_pct')}% time={row.get('runtime_sec')}s "
                    f"UB={row.get('ub')} LB={row.get('lb')}",
                    flush=True,
                )
            time.sleep(2.0)

        by_m = {r["method"]: r for r in meta["rows"] if r.get("seed") == seed}
        print(f"\n===== seed={seed} 三方小结 =====", flush=True)
        for method in METHOD_ORDER:
            r = by_m.get(method, {})
            if "error" in r:
                print(f"  {method}: ERROR {r['error']}", flush=True)
            else:
                print(
                    f"  {method:4s}: status={r.get('status')} gap={_fmt(r.get('gap_pct'))}% "
                    f"time={_fmt(r.get('runtime_sec'), 1)}s",
                    flush=True,
                )

    print(f"\n=== ALL DONE ===\nJSON: {OUT_JSON}\nMD:   {OUT_MD}", flush=True)


if __name__ == "__main__":
    main()
