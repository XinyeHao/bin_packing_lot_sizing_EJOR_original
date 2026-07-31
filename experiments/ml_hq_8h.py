"""High-quality 8h ML collect + train/val/test + adaptive retrain (schema v5).

- CG toward convergence with hard cap 60s (SB children) / 90s (parent)
- frac Top-20 pool; dual_share Top-10 labeled
- Drop groups where best-second y gap < MIN_LABEL_GAP
- Split by query_id: 70% train / 15% val / 15% test
- Early-stopping on val; if weak, auto-adjust hyperparams and retry
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
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
    sb_label,
)
from preprocessor.preprocess import preprocess

SCALE_GRID = [
    ("xs", 18, 24),
    ("s", 20, 28),
    ("s2", 22, 32),
    ("small", 25, 35),
    ("medium", 30, 40),
    ("large", 35, 50),
]
SCALE_CYCLE = ["xs", "s", "small", "small", "medium", "medium", "medium", "large", "s2"]

TOP_K = 20
SB_LABEL_K = 10
PARENT_CG_LIMIT = 90.0
SB_CG_LIMIT = 60.0  # hard cap; aim for convergence within this
MAX_DEPTH = 5
MAX_NODES_PER_INST = 6
MIN_LABEL_GAP = 0.01  # drop fuzzy groups (continuous y)
TRAIN_RESERVE_SEC = 900.0  # 15 min for split/train/adjust
VAL_NDCG1_TARGET = 0.70
TOP1_HIT_TARGET = 0.40


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


def _sb_one(data, parent_cg, parent_lb, cand, incumbent, deadline):
    rem = deadline - time.perf_counter()
    need = 2 * min(SB_CG_LIMIT, 20.0) + 2
    if rem < need:
        return None
    inherited = copy_column_pool(parent_cg.columns)
    lbs = []
    for side in ("L", "R"):
        rem = deadline - time.perf_counter()
        if rem < 5:
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
                return None
            lbs.append(max(float(lb), parent_lb))
        except Exception:
            lbs.append(float("inf"))
    return sb_label(parent_lb, lbs[0], lbs[1], incumbent)


def _count_rows(d: Path) -> int:
    n = 0
    if not d.exists():
        return 0
    for f in d.glob("*.csv"):
        with f.open(encoding="utf-8") as fh:
            n += max(0, sum(1 for _ in fh) - 1)
    return n


def collect_instance(data, out_csv: Path, query_prefix: str, deadline: float, sel: MLBranchSelector):
    rows_before = _count_rows(out_csv.parent)
    cg = _run_cg(data, tlim=PARENT_CG_LIMIT)
    if cg.lrmp_obj is None:
        return 0, 0
    parent_lb = float(cg.lrmp_obj)
    incumbent = parent_lb * 1.08
    fixed_y, fixed_scrap, fixed_wip = {}, {}, {}
    nodes = 0
    qid = 0
    n_drop_fuzzy = 0
    while nodes < MAX_NODES_PER_INST and time.perf_counter() < deadline:
        cands = sel.collect_candidates(data, cg, fixed_y, fixed_scrap, fixed_wip)
        if len(cands) < 2:
            break
        to_label = sorted(cands, key=lambda c: c.dual, reverse=True)[:SB_LABEL_K]
        if len(to_label) < 2:
            break
        _assign_features(to_label)
        labels = []
        keep = []
        for c in to_label:
            y = _sb_one(data, cg, parent_lb, c, incumbent, deadline)
            if y is None:
                continue
            keep.append(c)
            labels.append(y)
            if time.perf_counter() >= deadline:
                break
        if len(keep) >= 2:
            gap = max(labels) - sorted(labels, reverse=True)[1]
            if gap < MIN_LABEL_GAP:
                n_drop_fuzzy += 1
            else:
                qid += 1
                sel.append_group_csv(
                    out_csv,
                    f"{query_prefix}_n{qid}",
                    keep,
                    labels,
                    extra={
                        "T": len(data.periods),
                        "I": len(data.cured_items),
                        "y_gap": gap,
                        "parent_converged": int(bool(getattr(cg, "converged", False))),
                    },
                )
        best = max(keep or cands, key=lambda c: c.dual)
        fy, fs, fw = _apply_side(best.decision, fixed_y, fixed_scrap, fixed_wip, "L")
        if nodes + 1 > MAX_DEPTH:
            break
        rem = deadline - time.perf_counter()
        if rem < 30:
            break
        try:
            cg = _run_cg(
                data, fy, fs, fw,
                inherited=copy_column_pool(cg.columns),
                tlim=min(PARENT_CG_LIMIT, rem * 0.5),
            )
        except Exception:
            break
        if cg.lrmp_obj is None:
            break
        parent_lb = max(parent_lb, float(cg.lrmp_obj))
        fixed_y, fixed_scrap, fixed_wip = fy, fs, fw
        nodes += 1
    added = _count_rows(out_csv.parent) - rows_before
    return added, n_drop_fuzzy


def _load_groups(data_dir: Path):
    """Return list of (query_id, list of (label, features))."""
    import csv

    buckets: dict[str, list[tuple[float, list[float]]]] = {}
    for fp in sorted(data_dir.glob("*.csv")):
        with fp.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                feats = [float(r[n]) for n in FEATURE_NAMES]
                buckets.setdefault(r["query_id"], []).append((float(r["label"]), feats))
    groups = [(q, rows) for q, rows in buckets.items() if len(rows) >= 2]
    return groups


def _groups_to_xy(groups: list):
    X, y, gsize = [], [], []
    for _q, rows in groups:
        scores = [r[0] for r in rows]
        rel = relevance_from_scores(scores)
        for (_lab, feats), r in zip(rows, rel):
            X.append(feats)
            y.append(float(r))
        gsize.append(len(rows))
    return (
        np.asarray(X, dtype=np.float64),
        np.asarray(y, dtype=np.float64),
        gsize,
    )


def _ndcg_and_top1(booster, X, y, groups):
    scores = booster.predict(X)
    idx = 0
    ndcgs = {1: [], 3: [], 5: []}
    hits = 0
    n = 0
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
            ndcgs[k].append(dcg / idcg if idcg > 0 else 0.0)
        if int(np.argmax(s)) == int(np.argmax(lab)):
            hits += 1
        n += 1
        idx += g
    return {
        "ndcg@1": float(np.mean(ndcgs[1])) if n else 0.0,
        "ndcg@3": float(np.mean(ndcgs[3])) if n else 0.0,
        "ndcg@5": float(np.mean(ndcgs[5])) if n else 0.0,
        "top1_hit": hits / n if n else 0.0,
        "n_groups": n,
    }


def train_with_splits(data_dir: Path, model_out: Path, report_path: Path) -> dict:
    import lightgbm as lgb

    groups = _load_groups(data_dir)
    rng = random.Random(42)
    rng.shuffle(groups)
    n = len(groups)
    if n < 10:
        raise RuntimeError(f"too few groups for split: {n}")

    n_train = max(1, int(0.70 * n))
    n_val = max(1, int(0.15 * n))
    # remainder test
    train_g = groups[:n_train]
    val_g = groups[n_train : n_train + n_val]
    test_g = groups[n_train + n_val :]
    if not test_g:
        test_g = val_g[-max(1, len(val_g) // 5) :]
        val_g = val_g[: -len(test_g)] or val_g

    # persist split ids
    split_dir = data_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / "train_queries.txt").write_text(
        "\n".join(q for q, _ in train_g) + "\n", encoding="utf-8"
    )
    (split_dir / "val_queries.txt").write_text(
        "\n".join(q for q, _ in val_g) + "\n", encoding="utf-8"
    )
    (split_dir / "test_queries.txt").write_text(
        "\n".join(q for q, _ in test_g) + "\n", encoding="utf-8"
    )

    Xtr, ytr, gtr = _groups_to_xy(train_g)
    Xva, yva, gva = _groups_to_xy(val_g)
    Xte, yte, gte = _groups_to_xy(test_g)

    label_max = int(max(ytr.max(), yva.max(), yte.max()))
    configs = [
        {"name": "base", "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 10, "rounds": 400},
        {"name": "deeper", "learning_rate": 0.03, "num_leaves": 63, "min_data_in_leaf": 5, "rounds": 600},
        {"name": "wide_slow", "learning_rate": 0.02, "num_leaves": 47, "min_data_in_leaf": 8, "rounds": 800},
        {"name": "shallow", "learning_rate": 0.08, "num_leaves": 15, "min_data_in_leaf": 20, "rounds": 300},
    ]

    # Recreate datasets per trial so min_data_in_leaf can vary safely.
    trials = []
    best = None
    best_booster = None

    for cfg_h in configs:
        dtrain = lgb.Dataset(Xtr, label=ytr, group=gtr, feature_name=list(FEATURE_NAMES))
        dval = lgb.Dataset(
            Xva, label=yva, group=gva, feature_name=list(FEATURE_NAMES), reference=dtrain
        )
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
        evals: dict = {}
        booster = lgb.train(
            params,
            dtrain,
            num_boost_round=cfg_h["rounds"],
            valid_sets=[dtrain, dval],
            valid_names=["train", "val"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.record_evaluation(evals),
                lgb.log_evaluation(period=0),
            ],
        )
        val_m = _ndcg_and_top1(booster, Xva, yva, gva)
        tr_m = _ndcg_and_top1(booster, Xtr, ytr, gtr)
        te_m = _ndcg_and_top1(booster, Xte, yte, gte)
        trial = {
            "config": cfg_h,
            "best_iteration": getattr(booster, "best_iteration", None),
            "train": tr_m,
            "val": val_m,
            "test": te_m,
        }
        trials.append(trial)
        print(
            f"[train] cfg={cfg_h['name']} val_ndcg1={val_m['ndcg@1']:.4f} "
            f"val_top1={val_m['top1_hit']:.4f} test_ndcg1={te_m['ndcg@1']:.4f} "
            f"test_top1={te_m['top1_hit']:.4f}",
            flush=True,
        )
        score = val_m["ndcg@1"] + 0.5 * val_m["top1_hit"]
        if best is None or score > best["score"]:
            best = {"score": score, "trial": trial}
            best_booster = booster

        # stop early if already good
        if val_m["ndcg@1"] >= VAL_NDCG1_TARGET and val_m["top1_hit"] >= TOP1_HIT_TARGET:
            print("[train] target reached; skip remaining configs", flush=True)
            break

    assert best_booster is not None and best is not None
    model_out.parent.mkdir(parents=True, exist_ok=True)
    best_booster.save_model(str(model_out))
    schema = {
        "version": SCHEMA_VERSION,
        "features": list(FEATURE_NAMES),
        "objective": "lambdarank",
        "split": {"train": len(train_g), "val": len(val_g), "test": len(test_g)},
        "best_config": best["trial"]["config"],
    }
    model_out.with_suffix(".schema.json").write_text(
        json.dumps(schema, indent=2), encoding="utf-8"
    )

    # feature gain
    imp = best_booster.feature_importance(importance_type="gain")
    feat_gain = {n: float(v) for n, v in zip(FEATURE_NAMES, imp)}

    report = {
        "schema_version": SCHEMA_VERSION,
        "n_groups_total": n,
        "n_rows_approx": int(sum(len(r) for _, r in groups)),
        "split": {
            "train_groups": len(train_g),
            "val_groups": len(val_g),
            "test_groups": len(test_g),
            "train_rows": int(len(ytr)),
            "val_rows": int(len(yva)),
            "test_rows": int(len(yte)),
        },
        "targets": {"val_ndcg@1": VAL_NDCG1_TARGET, "val_top1_hit": TOP1_HIT_TARGET},
        "best": best["trial"],
        "all_trials": trials,
        "feature_gain": feat_gain,
        "model_path": str(model_out),
        "met_target": bool(
            best["trial"]["val"]["ndcg@1"] >= VAL_NDCG1_TARGET
            and best["trial"]["val"]["top1_hit"] >= TOP1_HIT_TARGET
        ),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md = [
        "# ML HQ train report (schema v5)",
        "",
        f"- groups: train {len(train_g)} / val {len(val_g)} / test {len(test_g)}",
        f"- best config: `{best['trial']['config']['name']}`",
        f"- val NDCG@1={best['trial']['val']['ndcg@1']:.4f}, top1={best['trial']['val']['top1_hit']:.4f}",
        f"- test NDCG@1={best['trial']['test']['ndcg@1']:.4f}, top1={best['trial']['test']['top1_hit']:.4f}",
        f"- met_target={report['met_target']}",
        "",
        "## Feature gain",
    ]
    for k, v in sorted(feat_gain.items(), key=lambda t: -t[1]):
        md.append(f"- {k}: {v:.1f}")
    (report_path.with_suffix(".md")).write_text("\n".join(md) + "\n", encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=8.0)
    ap.add_argument("--out-dir", type=str, default="ml_train_data_v5")
    ap.add_argument("--model-out", type=str, default=str(ML_BRANCH_MODEL_PATH))
    ap.add_argument("--seed0", type=int, default=92001)
    ap.add_argument("--fresh", action="store_true", default=True)
    args = ap.parse_args()

    cfg.apply_bench_generation()
    out_dir = ROOT / args.out_dir
    if args.fresh and out_dir.exists():
        print(f"[setup] clearing {out_dir}", flush=True)
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "RUN_INFO.txt").write_text(
        f"HQ collect hours={args.hours} SB_CG={SB_CG_LIMIT}s PARENT_CG={PARENT_CG_LIMIT}s\n"
        f"TOP_K={TOP_K} SB_LABEL_K={SB_LABEL_K} MIN_LABEL_GAP={MIN_LABEL_GAP}\n"
        f"schema={SCHEMA_VERSION} feats={FEATURE_NAMES}\n",
        encoding="utf-8",
    )

    t0 = time.perf_counter()
    wall = args.hours * 3600.0
    collect_deadline = t0 + wall - TRAIN_RESERVE_SEC
    sel = MLBranchSelector(top_k=TOP_K)
    scale_map = {name: (T, I) for name, T, I in SCALE_GRID}

    seed = args.seed0
    total = 0
    drop_fuzzy = 0
    step = 0
    print(
        f"[collect] HQ start deadline={collect_deadline - t0:.0f}s "
        f"SB_CG={SB_CG_LIMIT}s pool={TOP_K} label_k={SB_LABEL_K} gap>={MIN_LABEL_GAP}",
        flush=True,
    )

    while time.perf_counter() < collect_deadline:
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
            added, n_fuzzy = collect_instance(
                data, csv_path, f"{name}_s{seed}", collect_deadline, sel
            )
        except Exception as e:
            print(f"  skip collect {name} seed={seed}: {e}", flush=True)
            added, n_fuzzy = 0, 0
        drop_fuzzy += n_fuzzy
        total = _count_rows(out_dir)
        print(
            f"  {name} seed={seed} T={T} I={I} +{added} fuzzy_drop={n_fuzzy} "
            f"total_rows={total} elapsed={time.perf_counter() - t0:.0f}s",
            flush=True,
        )
        try:
            data.dispose()
        except Exception:
            pass

    print(
        f"[collect] done rows={total} fuzzy_dropped_groups~={drop_fuzzy} "
        f"elapsed={time.perf_counter() - t0:.1f}s",
        flush=True,
    )

    model_out = Path(args.model_out)
    report_path = out_dir / "train_report.json"
    if total < 80:
        print(f"[train] ABORT too few rows={total}", flush=True)
        sys.exit(1)

    print("[train] split 70/15/15 by query + adaptive configs", flush=True)
    report = train_with_splits(out_dir, model_out, report_path)
    print(
        f"[train] OK model={model_out} met_target={report['met_target']} "
        f"val_ndcg1={report['best']['val']['ndcg@1']:.4f} "
        f"test_ndcg1={report['best']['test']['ndcg@1']:.4f} "
        f"total_elapsed={time.perf_counter() - t0:.1f}s",
        flush=True,
    )
    print(f"[train] report: {report_path} / {report_path.with_suffix('.md')}", flush=True)


if __name__ == "__main__":
    main()
