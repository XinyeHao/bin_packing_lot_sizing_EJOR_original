"""Retrain for coarse-ranker use (optimize NDCG@5) + report Recall@K for N=5.

Uses ml_train_data_v7_w1216, label y_min by default. Writes ml_branch_model.txt.
N itself is online (ML_SB_REFINE_K); this aligns training metric with Top-5 refine.
"""
from __future__ import annotations

import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from configuration import ML_SB_REFINE_K
from model_and_algo.ml_branch_selector import FEATURE_NAMES, relevance_from_scores

DATA_DIR = ROOT / "ml_train_data_v7_w1216"
MODEL_OUT = ROOT / "ml_branch_model.txt"
REPORT_OUT = DATA_DIR / "retrain_top5_coarse.json"
ABS_GAP = 1e-3
REL_GAP = 0.20
SEED = 42
LABEL = "y_min"  # label-A; matches ml_full_labelA_8h primary label


def _load(data_dir: Path):
    buckets: dict[str, list[dict]] = defaultdict(list)
    for fp in sorted(data_dir.glob("batch_*.csv")):
        with fp.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                buckets[r["query_id"]].append(r)
    groups = []
    for q, rows in buckets.items():
        if len(rows) < 2:
            continue
        ys = np.asarray([float(r[LABEL]) for r in rows], dtype=np.float64)
        best, second = float(ys.max()), float(np.sort(ys)[::-1][1])
        if best - second < ABS_GAP:
            continue
        if second > 1e-12 and (best - second) / second < REL_GAP and (best - second) < 0.05:
            continue
        med = float(np.median(ys))
        mad = float(np.median(np.abs(ys - med))) + 1e-9
        scores = ((ys - med) / mad).tolist()
        feats = [[float(r[n]) for n in FEATURE_NAMES] for r in rows]
        fracs = [float(r["frac"]) for r in rows]
        duals = [float(r["dual_share"]) for r in rows]
        groups.append(
            {"q": q, "scores": scores, "ys": ys.tolist(), "feats": feats, "frac": fracs, "dual": duals}
        )
    return groups


def _split(groups):
    rng = random.Random(SEED)
    g = list(groups)
    rng.shuffle(g)
    n = len(g)
    n_tr = max(1, int(0.70 * n))
    n_va = max(1, int(0.15 * n))
    train, val, test = g[:n_tr], g[n_tr : n_tr + n_va], g[n_tr + n_va :]
    if not test:
        test = val[-max(1, len(val) // 5) :]
        val = val[: -len(test)] or val
    return train, val, test


def _xy(groups):
    X, y, gsz = [], [], []
    for g in groups:
        rel = relevance_from_scores(g["scores"])
        for feats, rr in zip(g["feats"], rel):
            X.append(feats)
            y.append(float(rr))
        gsz.append(len(g["feats"]))
    return np.asarray(X, np.float64), np.asarray(y, np.float64), gsz


def _recall_at_k(score_lists, y_lists, k: int) -> float:
    hits = n = 0
    for s, y in zip(score_lists, y_lists):
        w = int(np.argmax(y))
        order = np.argsort(-np.asarray(s))
        if w in set(order[:k].tolist()):
            hits += 1
        n += 1
    return hits / n if n else 0.0


def _ndcg_at(booster, X, y, groups, ks=(1, 3, 5)):
    scores = booster.predict(X)
    idx = 0
    nd = {k: [] for k in ks}
    score_lists, y_lists = [], []
    for g in groups:
        s = scores[idx : idx + g]
        lab = y[idx : idx + g]
        score_lists.append(s)
        y_lists.append(lab)
        order = np.argsort(-s)
        ideal = np.sort(lab)[::-1]
        for k in ks:
            dcg = idcg = 0.0
            for i in range(min(k, g)):
                dcg += (2 ** lab[order[i]] - 1) / np.log2(i + 2)
                idcg += (2 ** ideal[i] - 1) / np.log2(i + 2)
            nd[k].append(dcg / idcg if idcg > 0 else 0.0)
        idx += g
    out = {f"ndcg@{k}": float(np.mean(nd[k])) for k in ks}
    for k in (1, 2, 3, 5, 10):
        out[f"R@{k}"] = _recall_at_k(score_lists, y_lists, k)
    out["top1_hit"] = out["R@1"]
    out["n_groups"] = len(groups)
    return out


def _baseline_recalls(groups):
    out = {}
    for name, key in (("frac", "frac"), ("dual", "dual")):
        rec = {}
        for k in (1, 2, 3, 5, 10):
            rec[k] = _recall_at_k([g[key] for g in groups], [g["ys"] for g in groups], k)
        out[name] = rec
    m = float(np.mean([len(g["ys"]) for g in groups]))
    out["random"] = {k: min(1.0, k / m) for k in (1, 2, 3, 5, 10)}
    return out


def main():
    import argparse
    import lightgbm as lgb

    global DATA_DIR, MODEL_OUT, REPORT_OUT, LABEL

    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    ap.add_argument("--model-out", type=str, default=str(MODEL_OUT))
    ap.add_argument("--label", type=str, default=LABEL, choices=("y_old", "y_min", "y_geo"))
    ap.add_argument("--min-groups", type=int, default=20)
    args = ap.parse_args()
    DATA_DIR = Path(args.data_dir)
    MODEL_OUT = Path(args.model_out)
    REPORT_OUT = DATA_DIR / "retrain_top5_coarse.json"
    LABEL = args.label

    print(
        f"[setup] online N=ML_SB_REFINE_K={ML_SB_REFINE_K} label={LABEL} data={DATA_DIR}",
        flush=True,
    )
    groups = _load(DATA_DIR)
    if len(groups) < args.min_groups:
        print(f"[train] ABORT groups={len(groups)} < {args.min_groups}", flush=True)
        sys.exit(1)
    train_g, val_g, test_g = _split(groups)
    print(
        f"[data] groups={len(groups)} split={len(train_g)}/{len(val_g)}/{len(test_g)} "
        f"mean_cands={np.mean([len(g['ys']) for g in groups]):.2f}",
        flush=True,
    )

    Xtr, ytr, gtr = _xy(train_g)
    Xva, yva, gva = _xy(val_g)
    Xte, yte, gte = _xy(test_g)
    label_max = int(max(ytr.max(), yva.max(), yte.max()))

    configs = [
        {"name": "ndcg5_deep", "learning_rate": 0.03, "num_leaves": 63, "min_data_in_leaf": 3, "rounds": 800},
        {"name": "ndcg5_base", "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 5, "rounds": 600},
        {"name": "ndcg5_shallow", "learning_rate": 0.08, "num_leaves": 15, "min_data_in_leaf": 8, "rounds": 500},
    ]
    trials = []
    best = None
    best_booster = None
    for cfg in configs:
        dtrain = lgb.Dataset(Xtr, label=ytr, group=gtr, feature_name=list(FEATURE_NAMES))
        dval = lgb.Dataset(Xva, label=yva, group=gva, feature_name=list(FEATURE_NAMES), reference=dtrain)
        params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_at": [1, 3, 5],  # emphasize @5 for coarse Top-5
            "learning_rate": cfg["learning_rate"],
            "num_leaves": cfg["num_leaves"],
            "min_data_in_leaf": cfg["min_data_in_leaf"],
            "feature_fraction": 0.9,
            "feature_pre_filter": False,
            "verbosity": -1,
            "label_gain": list(range(0, max(32, label_max + 2))),
        }
        booster = lgb.train(
            params,
            dtrain,
            num_boost_round=cfg["rounds"],
            valid_sets=[dtrain, dval],
            valid_names=["train", "val"],
            callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)],
        )
        val_m = _ndcg_at(booster, Xva, yva, gva)
        te_m = _ndcg_at(booster, Xte, yte, gte)
        trial = {"config": cfg, "val": val_m, "test": te_m}
        trials.append(trial)
        print(
            f"[{cfg['name']}] val R@3={val_m['R@3']:.3f} R@5={val_m['R@5']:.3f} "
            f"ndcg5={val_m['ndcg@5']:.3f} | test R@3={te_m['R@3']:.3f} R@5={te_m['R@5']:.3f}",
            flush=True,
        )
        # primary: val R@5 (coarse ranker), tie-break ndcg@5
        score = val_m["R@5"] + 0.3 * val_m["ndcg@5"] + 0.1 * val_m["R@3"]
        if best is None or score > best["score"]:
            best = {"score": score, "trial": trial}
            best_booster = booster

    assert best is not None and best_booster is not None
    best_booster.save_model(str(MODEL_OUT))
    schema = {
        "version": 6,
        "features": list(FEATURE_NAMES),
        "objective": "lambdarank",
        "label": LABEL,
        "online_refine_k": ML_SB_REFINE_K,
        "train_focus": "NDCG@5 / Recall@5 for coarse Top-N",
        "split": {"train": len(train_g), "val": len(val_g), "test": len(test_g)},
        "best_config": best["trial"]["config"]["name"],
        "data_dir": str(DATA_DIR),
    }
    MODEL_OUT.with_suffix(".schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")

    base_val = _baseline_recalls(val_g)
    base_te = _baseline_recalls(test_g)
    bv, bt = best["trial"]["val"], best["trial"]["test"]
    report = {
        "online_N": ML_SB_REFINE_K,
        "data_dir": str(DATA_DIR),
        "best": best["trial"],
        "baselines_val": base_val,
        "baselines_test": base_te,
        "delta_test_R5_vs_frac": bt["R@5"] - base_te["frac"][5],
        "delta_test_R3_vs_frac": bt["R@3"] - base_te["frac"][3],
        "trials": [{"config": t["config"]["name"], "val": t["val"], "test": t["test"]} for t in trials],
    }
    REPORT_OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = [
        f"# Coarse ranker retrain (online N={ML_SB_REFINE_K})",
        "",
        f"- data: `{DATA_DIR}`",
        f"- best: `{best['trial']['config']['name']}`",
        f"- test R@3={bt['R@3']:.3f} (frac {base_te['frac'][3]:.3f}, Δ={bt['R@3']-base_te['frac'][3]:+.3f})",
        f"- test R@5={bt['R@5']:.3f} (frac {base_te['frac'][5]:.3f}, Δ={bt['R@5']-base_te['frac'][5]:+.3f})",
        f"- test NDCG@5={bt['ndcg@5']:.3f}",
        "",
        "Online: ML Top-N then short SB refine.",
    ]
    REPORT_OUT.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n=== BEST vs baselines (test) ===", flush=True)
    print(
        f"ML   R@3={bt['R@3']:.3f} R@5={bt['R@5']:.3f} | "
        f"frac R@3={base_te['frac'][3]:.3f} R@5={base_te['frac'][5]:.3f} | "
        f"rand R@5≈{base_te['random'][5]:.3f}",
        flush=True,
    )
    print(f"wrote {MODEL_OUT} and {REPORT_OUT}", flush=True)


if __name__ == "__main__":
    main()
