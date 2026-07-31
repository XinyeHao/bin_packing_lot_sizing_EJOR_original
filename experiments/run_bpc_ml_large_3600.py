"""BPC+ML 3600s on large seeds 101–505, one subprocess per seed (memory isolation)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [101, 202, 303, 404, 505]
LIMIT = 3600.0
OUT_JSON = ROOT / "supplement_materials" / "bpc_ml_large_101_505_3600.json"
OUT_MD = ROOT / "supplement_materials" / "test_instance.md"
WORKER = ROOT / "experiments" / "_bpc_ml_large_one.py"


def load_payload() -> dict:
    if OUT_JSON.exists():
        return json.loads(OUT_JSON.read_text(encoding="utf-8"))
    return {
        "seeds": SEEDS,
        "bpc_time": LIMIT,
        "strategy": "ml",
        "scale": {"T": 35, "U": 5, "I": 50, "M": 5, "LT": [5, 6, 7, 8, 9, 10]},
        "rows": [],
    }


def save_payload(payload: dict) -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def update_test_instance_md(payload: dict) -> None:
    if not OUT_MD.exists():
        return
    text = OUT_MD.read_text(encoding="utf-8")
    marker_start = "### BPC+ML (600s)"
    marker_end = "\n来源：`im_bpc_seeds_101_505.json`"
    if marker_start not in text:
        # already updated or structure changed
        marker_start = "### BPC+ML (3600s)"
        marker_end = "\n来源：`bpc_ml_large_101_505_3600.json`"
        if marker_start not in text:
            print("WARN: could not find BPC section in test_instance.md", flush=True)
            return

    rows = {r["seed"]: r for r in payload.get("rows", [])}
    lines = [
        "### BPC+ML (3600s)",
        "",
        "| seed | status | UB | LB | MIP gap % | nodes | time (s) |",
        "|------|--------|----|----|-----------|-------|----------|",
    ]
    for seed in SEEDS:
        r = rows.get(seed)
        if not r:
            lines.append(f"| {seed} | — | — | — | — | — | — |")
            continue
        ub = "—" if r.get("ub") is None else f"{r['ub']:.2f}"
        lb = "—" if r.get("lb") is None else f"{r['lb']:.2f}"
        gap = "—" if r.get("gap") is None else f"{r['gap']:.2f}"
        nodes = "—" if r.get("nodes") is None else str(r["nodes"])
        t = "—" if r.get("time") is None else f"{r['time']:.1f}"
        lines.append(
            f"| {seed} | {r.get('status', '—')} | {ub} | {lb} | {gap} | {nodes} | {t} |"
        )
    lines += [
        "",
        "来源：`bpc_ml_large_101_505_3600.json`（此前 600s `best_bound` 见 `im_bpc_seeds_101_505.json`）；利用率见 `hard_101_505_metrics.json`。",
    ]
    # Replace from heading through the full 来源 line (incl. any duplicated tails).
    import re

    pattern = re.compile(
        r"### BPC\+ML \((?:600|3600)s\).*?来源：`[^`]+`[^\n]*",
        re.DOTALL,
    )
    new_block = "\n".join(lines)
    new_text, n = pattern.subn(new_block, text, count=1)
    if n == 0:
        print("WARN: regex replace failed", flush=True)
        return
    # Also fix the setup bullet that says 600s
    new_text = new_text.replace(
        "- BPC+ML: time_limit=600s（历史短时扫描；均未证优）",
        "- BPC+ML: time_limit=3600s, mip_gap=0, `branch_strategy=ml`",
    )
    new_text = new_text.replace(
        "| Large | 36, 50, 5 | 3600s | 0/5 | 0/5（仅 600s） |",
        "| Large | 36, 50, 5 | 3600s | 0/5 | "
        + (
            f"{sum(1 for r in payload.get('rows', []) if r.get('proved'))}/5"
            if len(payload.get("rows", [])) >= 5
            else f"{sum(1 for r in payload.get('rows', []) if r.get('proved'))}/5（进行中）"
        )
        + " |",
    )
    OUT_MD.write_text(new_text, encoding="utf-8")
    print(f"Updated {OUT_MD}", flush=True)


def main() -> None:
    assert (ROOT / "ml_branch_model.txt").exists(), "missing ml_branch_model.txt"
    assert WORKER.exists(), f"missing worker {WORKER}"
    payload = load_payload()
    # Successful rows skip; error/incomplete rows are retried.
    done = {
        r["seed"]
        for r in payload.get("rows", [])
        if r.get("status") not in (None, "error") and "error" not in r
    }
    print(f"Resume: done={sorted(done)} limit={LIMIT:.0f}s", flush=True)

    for seed in SEEDS:
        if seed in done:
            print(f"skip seed={seed} (already done)", flush=True)
            continue
        one_json = ROOT / "supplement_materials" / f"_bpc_ml_large_{seed}.json"
        if one_json.exists():
            one_json.unlink()
        print(f"\n>>> launch seed={seed} BPC+ML {LIMIT:.0f}s (subprocess)", flush=True)
        t0 = time.perf_counter()
        log_path = ROOT / "supplement_materials" / f"_bpc_ml_large_{seed}.log"
        with open(log_path, "w", encoding="utf-8", errors="replace") as logf:
            proc = subprocess.run(
                [sys.executable, "-u", str(WORKER), str(seed), str(LIMIT), str(one_json)],
                cwd=str(ROOT),
                stdout=logf,
                stderr=subprocess.STDOUT,
            )
        wall = time.perf_counter() - t0
        try:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
            sys.stdout.buffer.write(tail.encode("utf-8", "replace"))
            sys.stdout.buffer.write(b"\n")
            sys.stdout.buffer.flush()
        except Exception:
            pass
        row = None
        if one_json.exists():
            row = json.loads(one_json.read_text(encoding="utf-8"))
        if row is None:
            print(f"FAIL seed={seed} rc={proc.returncode} wall={wall:.1f}s", flush=True)
            row = {
                "seed": seed,
                "status": "error",
                "proved": False,
                "ub": None,
                "lb": None,
                "gap": None,
                "nodes": None,
                "time": wall,
                "error": f"rc={proc.returncode}; see {log_path.name}",
            }
        else:
            print(
                f"OK seed={seed} proved={row.get('proved')} gap={row.get('gap')} "
                f"t={row.get('time')} rc={proc.returncode}",
                flush=True,
            )
        payload.setdefault("rows", []).append(row)
        # keep one row per seed
        by = {r["seed"]: r for r in payload["rows"]}
        payload["rows"] = [by[s] for s in SEEDS if s in by]
        save_payload(payload)
        update_test_instance_md(payload)
        print(f"<<< finished seed={seed} wall={wall:.1f}s", flush=True)
        time.sleep(2.0)

    save_payload(payload)
    update_test_instance_md(payload)
    print("\nDONE all seeds", flush=True)
    for r in payload["rows"]:
        print(
            f"  seed={r['seed']} proved={r.get('proved')} "
            f"gap={r.get('gap')} t={r.get('time')}",
            flush=True,
        )


if __name__ == "__main__":
    main()
