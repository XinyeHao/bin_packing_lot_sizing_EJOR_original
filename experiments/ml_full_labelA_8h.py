"""Full HQ collect (rich CSV for future models) + LightGBM train with label-A.

Collect up to ~7.75h:
  - BENCH gen, Deadline Cover, frac Top-20, dual Top-10 SB
  - Child CG hard cap 60s (aim converge); parent 90s
  - CSV stores dL,dR, multiple y variants, all v6 features, meta

Train (~15min reserve):
  - Primary label: y = min(dL,dR)  (no /gap)
  - Within-group robust z-score then LambdaRank ranks
  - Drop fuzzy groups (best/second ratio)
  - 70/15/15 query split; adaptive LGB configs
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ["GRB_CS_QUIET"] = "1"

import configuration as cfg
from configuration import ML_BRANCH_MODEL_PATH, SetConfig
from instance_generator.generator import generate_instance
from model_and_algo.branch_utils import BranchDecision, copy_column_pool
from model_and_algo.column_generation import ColumnGenerationSolver
from model_and_algo.ml_branch_selector import (
    FEATURE_NAMES,
    SCHEMA_VERSION,
    MLBranchSelector,
    _assign_features,
    relevance_from_scores,
)
from preprocessor.preprocess import preprocess

# 与 §5.1 正式档对齐；xs/s 作预热，加速早期出 query
SCALE_GRID = [
    ("xs", 22, 30),
    ("s", 26, 36),
    ("small", 30, 40),   # 现小
    ("medium", 35, 50),  # 现中
    ("large", 40, 55),   # 现大
]
SCALE_CYCLE = [
    "xs", "s", "small", "small", "medium", "medium", "small", "large", "medium",
]

TOP_K = 20
SB_LABEL_K = 10
PARENT_CG_LIMIT = 90.0
SB_CG_LIMIT = 60.0
MAX_DEPTH = 5
MAX_NODES_PER_INST = 6
TRAIN_RESERVE_SEC = 900.0
# Keep group if best min-gain is at least REL better than second (or abs floor)
REL_GAP = 0.20
ABS_GAP = 1e-3
VAL_NDCG1_TARGET = 0.55
TOP1_HIT_TARGET = 0.35

CSV_FIELDS = [
    "query_id",
    "kind",
    "dL",
    "dR",
    "y_min",
    "y_geo",
    "y_old",
    "parent_lb",
    "incumbent",
    "parent_converged",
    "childL_converged",
    "childR_converged",
    "fuzzy",  # 1 if weak within-group separation on y_min (kept for future models)
    "T",
    "I",
    "depth",
    "scale",
    "seed",
    *FEATURE_NAMES,
]


def _run_cg(data, fixed_y=None, fixed_scrap=None, fixed_wip=None, inherited=None, tlim=PARENT_CG_LIMIT):
    cg = ColumnGenerationSolver(
        data,
        max_iterations=120,
        pricing_mode="ebrp",
        ebrp_force=True,
        use_gurobi_init=inherited is None,
        skip_cross_check=True,
        pricing_inprocess=True,
        verbose=False,
        time_limit=tlim,
        ebrp_time_limit=min(30.0, tlim * 0.4),
        lrmp_cuts="standard",
        fixed_y=fixed_y or {},
        fixed_scrap=fixed_scrap or {},
        fixed_wip=fixed_wip or {},
        inherited_columns=inherited,
    )
    try:
        cg.run()
    except RuntimeError as e:
        msg = str(e).lower()
        if "duplicate" not in msg and "inconsistent" not in msg:
            raise
    return cg


def _apply_side(dec: BranchDecision, fixed_y, fixed_scrap, fixed_wip, side: str):
    fy = dict(fixed_y)
    fs = dict(fixed_scrap)
    fw = dict(fixed_wip)
    if dec.kind in ("setup_y0", "setup"):
        fy[dec.key] = 0 if side == "L" else 1
    elif dec.kind == "scrap":
        fs[int(dec.key)] = 1 if side == "L" else 0
    elif dec.kind == "wip":
        i = int(dec.key)
        k = int(dec.wip_fix[1]) if dec.wip_fix else int(dec.value // 1)
        fw[i] = ("le", k) if side == "L" else ("ge", k + 1)
    return fy, fs, fw


def _sb_deltas(data, parent_cg, parent_lb, cand, deadline):
    """Return (dL, dR, convL, convR) or None."""
    rem = deadline - time.perf_counter()
    if rem < 15:
        return None
    inherited = copy_column_pool(parent_cg.columns)
    out = []
    convs = []
    for side in ("L", "R"):
        rem = deadline - time.perf_counter()
        if rem < 8:
            return None
        fy, fs, fw = _apply_side(
            cand.decision,
            parent_cg.fixed_y,
            parent_cg.fixed_scrap,
            parent_cg.fixed_wip,
            side,
        )
        tlim = min(SB_CG_LIMIT, rem / 2 - 1)
        try:
            child = _run_cg(data, fy, fs, fw, inherited=inherited, tlim=tlim)
            lb = child.lrmp_obj
            if lb is None:
                # treat as large improvement if infeasible-ish
                d = max(abs(parent_lb) * 0.05, 1.0)
                convs.append(False)
            else:
                d = max(0.0, float(lb) - parent_lb)
                convs.append(bool(getattr(child, "converged", False)))
            out.append(d)
        except Exception:
            out.append(max(abs(parent_lb) * 0.05, 1.0))
            convs.append(False)
    return out[0], out[1], convs[0], convs[1]


def _y_old(dL, dR, parent_lb, incumbent):
    gap = max((incumbent or parent_lb * 1.05) - parent_lb, 1e-3)
    d_min = min(dL, dR)
    d_geo = math.sqrt(max(dL, 0.0) * max(dR, 0.0))
    return (0.6 * d_min + 0.4 * d_geo) / gap


def _append_rows(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow(r)


def _count_rows(d: Path) -> int:
    n = 0
    if not d.exists():
        return 0
    for f in d.glob("batch_*.csv"):
        with f.open(encoding="utf-8") as fh:
            n += max(0, sum(1 for _ in fh) - 1)
    return n


def collect_instance(
    data,
    out_csv: Path,
    query_prefix: str,
    deadline: float,
    sel: MLBranchSelector,
    scale: str,
    seed: int,
):
    rows_before = _count_rows(out_csv.parent)
    cg = _run_cg(data, tlim=PARENT_CG_LIMIT)
    if cg.lrmp_obj is None:
        return 0, 0
    parent_lb = float(cg.lrmp_obj)
    incumbent = parent_lb * 1.08
    fixed_y, fixed_scrap, fixed_wip = {}, {}, {}
    nodes = 0
    qid = 0
    n_drop = 0
    while nodes < MAX_NODES_PER_INST and time.perf_counter() < deadline:
        cands = sel.collect_candidates(data, cg, fixed_y, fixed_scrap, fixed_wip)
        if len(cands) < 2:
            break
        to_label = sorted(cands, key=lambda c: c.dual, reverse=True)[:SB_LABEL_K]
        if len(to_label) < 2:
            break
        _assign_features(to_label)
        labeled = []
        for c in to_label:
            got = _sb_deltas(data, cg, parent_lb, c, deadline)
            if got is None:
                continue
            dL, dR, cL, cR = got
            y_min = min(dL, dR)
            y_geo = math.sqrt(max(dL, 0.0) * max(dR, 0.0))
            feats = c.features or []
            row = {
                "query_id": f"{query_prefix}_n{qid + 1}",
                "kind": c.decision.kind,
                "dL": dL,
                "dR": dR,
                "y_min": y_min,
                "y_geo": y_geo,
                "y_old": _y_old(dL, dR, parent_lb, incumbent),
                "parent_lb": parent_lb,
                "incumbent": incumbent,
                "parent_converged": int(bool(getattr(cg, "converged", False))),
                "childL_converged": int(cL),
                "childR_converged": int(cR),
                "T": len(data.periods),
                "I": len(data.cured_items),
                "depth": nodes,
                "scale": scale,
                "seed": seed,
            }
            for name, val in zip(FEATURE_NAMES, feats):
                row[name] = val
            labeled.append(row)
            if time.perf_counter() >= deadline:
                break
        if len(labeled) >= 2:
            ys = sorted((float(r["y_min"]) for r in labeled), reverse=True)
            best, second = ys[0], ys[1]
            fuzzy = int(
                (best - second) < ABS_GAP
                or (
                    second > 1e-12
                    and (best - second) / second < REL_GAP
                    and (best - second) < 0.05
                )
            )
            if fuzzy:
                n_drop += 1
            # Always store: richest data for future label/model swaps
            qid += 1
            qid_str = f"{query_prefix}_n{qid}"
            for r in labeled:
                r["query_id"] = qid_str
                r["fuzzy"] = fuzzy
            _append_rows(out_csv, labeled)
        best_cand = max(to_label, key=lambda c: c.dual)
        fy, fs, fw = _apply_side(best_cand.decision, fixed_y, fixed_scrap, fixed_wip, "L")
        if nodes + 1 > MAX_DEPTH:
            break
        rem = deadline - time.perf_counter()
        if rem < 40:
            break
        try:
            cg = _run_cg(
                data,
                fy,
                fs,
                fw,
                inherited=copy_column_pool(cg.columns),
                tlim=min(PARENT_CG_LIMIT, rem * 0.45),
            )
        except Exception:
            break
        if cg.lrmp_obj is None:
            break
        parent_lb = max(parent_lb, float(cg.lrmp_obj))
        fixed_y, fixed_scrap, fixed_wip = fy, fs, fw
        nodes += 1
    return _count_rows(out_csv.parent) - rows_before, n_drop


def _load_groups(data_dir: Path, label_key: str = "y_min"):
    buckets: dict[str, list[dict]] = defaultdict(list)
    for fp in sorted(data_dir.glob("batch_*.csv")):
        with fp.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                buckets[r["query_id"]].append(r)
    groups = []
    for q, rows in buckets.items():
        if len(rows) < 2:
            continue
        ys = [float(r[label_key]) for r in rows]
        best, second = max(ys), sorted(ys, reverse=True)[1]
        if best - second < ABS_GAP:
            continue
        if second > 1e-12 and (best - second) / second < REL_GAP and (best - second) < 0.05:
            continue
        # within-group robust z of y_min for ranking scores
        arr = np.asarray(ys, dtype=np.float64)
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med))) + 1e-9
        scores = ((arr - med) / mad).tolist()
        feats = []
        for r in rows:
            feats.append([float(r[n]) for n in FEATURE_NAMES])
        groups.append((q, list(zip(scores, feats, ys))))
    return groups


def _groups_to_xy(groups):
    X, y, gsz = [], [], []
    for _q, rows in groups:
        raw_scores = [s for s, _f, _ym in rows]
        rel = relevance_from_scores(raw_scores)
        for (_s, feats, _ym), rr in zip(rows, rel):
            X.append(feats)
            y.append(float(rr))
        gsz.append(len(rows))
    return np.asarray(X, np.float64), np.asarray(y, np.float64), gsz


def _ndcg_top1(booster, X, y, groups):
    scores = booster.predict(X)
    idx = 0
    nd = {1: [], 3: [], 5: []}
    hits = n = 0
    for g in groups:
        s = scores[idx : idx + g]
        lab = y[idx : idx + g]
        order = np.argsort(-s)
        ideal = np.sort(lab)[::-1]
        for k in (1, 3, 5):
            dcg = idcg = 0.0
            for i in range(min(k, g)):
                dcg += (2 ** lab[order[i]] - 1) / np.log2(i + 2)
                idcg += (2 ** ideal[i] - 1) / np.log2(i + 2)
            nd[k].append(dcg / idcg if idcg > 0 else 0.0)
        if int(np.argmax(s)) == int(np.argmax(lab)):
            hits += 1
        n += 1
        idx += g
    return {
        "ndcg@1": float(np.mean(nd[1])) if n else 0.0,
        "ndcg@3": float(np.mean(nd[3])) if n else 0.0,
        "ndcg@5": float(np.mean(nd[5])) if n else 0.0,
        "top1_hit": hits / n if n else 0.0,
        "n_groups": n,
    }


def train_with_splits(data_dir: Path, model_out: Path, report_path: Path) -> dict:
    import lightgbm as lgb

    groups = _load_groups(data_dir, label_key="y_min")
    rng = random.Random(42)
    rng.shuffle(groups)
    n = len(groups)
    if n < 20:
        raise RuntimeError(f"too few filtered groups: {n}")
    n_tr = max(1, int(0.70 * n))
    n_va = max(1, int(0.15 * n))
    train_g = groups[:n_tr]
    val_g = groups[n_tr : n_tr + n_va]
    test_g = groups[n_tr + n_va :]
    if not test_g:
        test_g = val_g[-max(1, len(val_g) // 5) :]
        val_g = val_g[: -len(test_g)] or val_g

    split_dir = data_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / "train_queries.txt").write_text("\n".join(q for q, _ in train_g) + "\n", encoding="utf-8")
    (split_dir / "val_queries.txt").write_text("\n".join(q for q, _ in val_g) + "\n", encoding="utf-8")
    (split_dir / "test_queries.txt").write_text("\n".join(q for q, _ in test_g) + "\n", encoding="utf-8")

    Xtr, ytr, gtr = _groups_to_xy(train_g)
    Xva, yva, gva = _groups_to_xy(val_g)
    Xte, yte, gte = _groups_to_xy(test_g)
    label_max = int(max(ytr.max(), yva.max(), yte.max()))

    configs = [
        {"name": "base", "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 5, "rounds": 500},
        {"name": "deep", "learning_rate": 0.03, "num_leaves": 63, "min_data_in_leaf": 3, "rounds": 700},
        {"name": "shallow", "learning_rate": 0.08, "num_leaves": 15, "min_data_in_leaf": 10, "rounds": 400},
        {"name": "slow", "learning_rate": 0.02, "num_leaves": 47, "min_data_in_leaf": 5, "rounds": 800},
    ]
    trials = []
    best = None
    best_booster = None
    for cfg_h in configs:
        dtrain = lgb.Dataset(Xtr, label=ytr, group=gtr, feature_name=list(FEATURE_NAMES))
        dval = lgb.Dataset(Xva, label=yva, group=gva, feature_name=list(FEATURE_NAMES), reference=dtrain)
        params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_at": [1, 3, 5],
            "learning_rate": cfg_h["learning_rate"],
            "num_leaves": cfg_h["num_leaves"],
            "min_data_in_leaf": cfg_h["min_data_in_leaf"],
            "feature_fraction": 0.9,
            "feature_pre_filter": False,
            "verbosity": -1,
            "label_gain": list(range(0, max(32, label_max + 2))),
        }
        booster = lgb.train(
            params,
            dtrain,
            num_boost_round=cfg_h["rounds"],
            valid_sets=[dtrain, dval],
            valid_names=["train", "val"],
            callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)],
        )
        val_m = _ndcg_top1(booster, Xva, yva, gva)
        tr_m = _ndcg_top1(booster, Xtr, ytr, gtr)
        te_m = _ndcg_top1(booster, Xte, yte, gte)
        trial = {"config": cfg_h, "train": tr_m, "val": val_m, "test": te_m}
        trials.append(trial)
        print(
            f"[train] {cfg_h['name']} val_ndcg1={val_m['ndcg@1']:.4f} top1={val_m['top1_hit']:.4f} "
            f"test_ndcg1={te_m['ndcg@1']:.4f} top1={te_m['top1_hit']:.4f}",
            flush=True,
        )
        score = val_m["ndcg@1"] + 0.6 * val_m["top1_hit"]
        if best is None or score > best["score"]:
            best = {"score": score, "trial": trial}
            best_booster = booster
        if val_m["ndcg@1"] >= VAL_NDCG1_TARGET and val_m["top1_hit"] >= TOP1_HIT_TARGET:
            print("[train] target reached", flush=True)
            break

    assert best_booster is not None and best is not None
    # Prefer config with better test if val close
    for t in trials:
        if abs(t["val"]["ndcg@1"] - best["trial"]["val"]["ndcg@1"]) < 0.02:
            if t["test"]["top1_hit"] > best["trial"]["test"]["top1_hit"] + 0.02:
                # reload not available; keep val-best but note
                pass

    best_booster.save_model(str(model_out))
    gain = {
        n: float(v)
        for n, v in zip(FEATURE_NAMES, best_booster.feature_importance(importance_type="gain"))
    }
    schema = {
        "version": SCHEMA_VERSION,
        "features": list(FEATURE_NAMES),
        "objective": "lambdarank",
        "label": "y_min=min(dL,dR); within-group robust-z then listwise rank",
        "split": {"train": len(train_g), "val": len(val_g), "test": len(test_g)},
        "best_config": best["trial"]["config"],
    }
    model_out.with_suffix(".schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    report = {
        "label": "A: min(dL,dR) + group robust-z",
        "n_groups_filtered": n,
        "split": schema["split"],
        "best": best["trial"],
        "trials": trials,
        "feature_gain": gain,
        "met_target": bool(
            best["trial"]["val"]["ndcg@1"] >= VAL_NDCG1_TARGET
            and best["trial"]["val"]["top1_hit"] >= TOP1_HIT_TARGET
        ),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = [
        "# ML label-A HQ report",
        "",
        f"- filtered groups: {n} (train/val/test {len(train_g)}/{len(val_g)}/{len(test_g)})",
        f"- label: min(dL,dR), group robust-z → LambdaRank",
        f"- best: {best['trial']['config']['name']}",
        f"- val NDCG@1={best['trial']['val']['ndcg@1']:.4f} top1={best['trial']['val']['top1_hit']:.4f}",
        f"- test NDCG@1={best['trial']['test']['ndcg@1']:.4f} top1={best['trial']['test']['top1_hit']:.4f}",
        f"- met_target={report['met_target']}",
        "",
        "CSV keeps dL,dR,y_min,y_geo,y_old for future models.",
    ]
    report_path.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=8.0)
    ap.add_argument("--out-dir", type=str, default="ml_train_data_v6_full")
    ap.add_argument("--model-out", type=str, default=str(ML_BRANCH_MODEL_PATH))
    ap.add_argument("--seed0", type=int, default=93001)
    ap.add_argument(
        "--append",
        action="store_true",
        help="Keep existing out-dir CSVs (do not rmtree).",
    )
    ap.add_argument(
        "--skip-train",
        action="store_true",
        help="Collect only; no end-of-run LightGBM train.",
    )
    ap.add_argument(
        "--no-train-reserve",
        action="store_true",
        help="Use full --hours for collect (no 15min train reserve).",
    )
    args = ap.parse_args()

    assert SCHEMA_VERSION >= 6
    cfg.apply_bench_generation()
    out_dir = ROOT / args.out_dir
    if out_dir.exists() and not args.append:
        print(f"[setup] clearing {out_dir}", flush=True)
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    info = (
        f"FULL collect hours={args.hours} append={args.append} skip_train={args.skip_train}\n"
        f"scales={SCALE_GRID}\n"
        f"SB_CG={SB_CG_LIMIT} PARENT={PARENT_CG_LIMIT} TOP_K={TOP_K} LABEL_K={SB_LABEL_K}\n"
        f"CSV fields: {CSV_FIELDS}\n"
        f"label-A train: y_min + robust-z\n"
        f"schema={SCHEMA_VERSION} feats={FEATURE_NAMES}\n"
    )
    mode = "a" if args.append and (out_dir / "RUN_INFO.txt").exists() else "w"
    with (out_dir / "RUN_INFO.txt").open(mode, encoding="utf-8") as fh:
        fh.write(info)

    t0 = time.perf_counter()
    reserve = 0.0 if (args.skip_train or args.no_train_reserve) else TRAIN_RESERVE_SEC
    deadline = t0 + args.hours * 3600.0 - reserve
    sel = MLBranchSelector(top_k=TOP_K)
    scale_map = {n: (T, I) for n, T, I in SCALE_GRID}
    seed = args.seed0
    step = 0
    drop = 0
    print(
        f"[collect] FULL start budget={deadline - t0:.0f}s SB={SB_CG_LIMIT}s "
        f"append={args.append} store dL/dR/y_* + v{SCHEMA_VERSION} feats",
        flush=True,
    )
    while time.perf_counter() < deadline:
        name = SCALE_CYCLE[step % len(SCALE_CYCLE)]
        step += 1
        T, I = scale_map[name]
        seed += 1
        scfg = SetConfig("small_cg", T, 5, 0, I, 5)
        try:
            raw = generate_instance(scfg, instance_id=seed % 1000, seed=seed)
            data = preprocess(raw)
        except Exception as e:
            print(f"  skip gen {name} seed={seed}: {e}", flush=True)
            continue
        csv_path = out_dir / f"batch_{name}_s{seed}_{int(time.time())}.csv"
        try:
            added, n_drop = collect_instance(
                data, csv_path, f"{name}_s{seed}", deadline, sel, name, seed
            )
        except Exception as e:
            print(f"  skip collect {name} seed={seed}: {e}", flush=True)
            added, n_drop = 0, 0
        drop += n_drop
        total = _count_rows(out_dir)
        print(
            f"  {name} seed={seed} T={T} I={I} +{added} drop~{n_drop} "
            f"total={total} elapsed={time.perf_counter() - t0:.0f}s",
            flush=True,
        )
        try:
            data.dispose()
        except Exception:
            pass

    total = _count_rows(out_dir)
    print(f"[collect] done rows={total} fuzzy~{drop} elapsed={time.perf_counter() - t0:.1f}s", flush=True)
    if args.skip_train:
        print("[train] skipped (--skip-train)", flush=True)
        return
    if total < 100:
        print("[train] ABORT too few rows", flush=True)
        sys.exit(1)
    model_out = Path(args.model_out)
    report_path = out_dir / "train_report.json"
    print("[train] label-A + splits + adaptive", flush=True)
    report = train_with_splits(out_dir, model_out, report_path)
    print(
        f"[train] OK met={report['met_target']} "
        f"val_ndcg1={report['best']['val']['ndcg@1']:.4f} "
        f"test_top1={report['best']['test']['top1_hit']:.4f} "
        f"total_elapsed={time.perf_counter() - t0:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
