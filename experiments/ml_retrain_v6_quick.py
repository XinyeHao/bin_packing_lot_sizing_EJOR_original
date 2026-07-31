"""Quick retrain v6 from existing v5 CSVs (no recollect): ranks + gap filter."""
from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np

from model_and_algo.ml_branch_selector import FEATURE_NAMES, SCHEMA_VERSION, relevance_from_scores

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "ml_train_data_v5"
MODEL = ROOT / "ml_branch_model.txt"
GAP_MIN = 0.05


def _ndcg_top1(booster, X, y, groups):
    scores = booster.predict(X)
    idx = 0
    nd1, nd3, nd5, hits, n = [], [], [], 0, 0
    for g in groups:
        s = scores[idx : idx + g]
        lab = y[idx : idx + g]
        order = np.argsort(-s)
        ideal = np.sort(lab)[::-1]
        for k, bucket in ((1, nd1), (3, nd3), (5, nd5)):
            dcg = idcg = 0.0
            for i in range(min(k, g)):
                dcg += (2 ** lab[order[i]] - 1) / np.log2(i + 2)
                idcg += (2 ** ideal[i] - 1) / np.log2(i + 2)
            bucket.append(dcg / idcg if idcg > 0 else 0.0)
        if int(np.argmax(s)) == int(np.argmax(lab)):
            hits += 1
        n += 1
        idx += g
    return {
        "ndcg@1": float(np.mean(nd1)),
        "ndcg@3": float(np.mean(nd3)),
        "ndcg@5": float(np.mean(nd5)),
        "top1_hit": hits / n if n else 0.0,
        "n_groups": n,
    }


def build_features_row(base: dict, dual_rank: float, frac_rank: float) -> list[float]:
    return [
        float(base["frac"]),
        float(base["is_y"]),
        float(base["is_wip"]),
        float(base["is_scrap"]),
        float(base["dual_share"]),
        float(base["dens_rel"]),
        float(base["prox_pen"]),
        dual_rank,
        frac_rank,
    ]


def load_groups(gap_min: float):
    buckets: dict[str, list[dict]] = defaultdict(list)
    for fp in sorted(DATA.glob("*.csv")):
        with fp.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                buckets[r["query_id"]].append(r)
    groups = []
    for q, rows in buckets.items():
        if len(rows) < 2:
            continue
        labs = sorted((float(r["label"]) for r in rows), reverse=True)
        gap = labs[0] - labs[1]
        if gap < gap_min:
            continue
        # ranks within group
        n = max(1, len(rows) - 1)
        dual_order = sorted(range(len(rows)), key=lambda i: float(rows[i]["dual_share"]), reverse=True)
        frac_order = sorted(range(len(rows)), key=lambda i: float(rows[i]["frac"]), reverse=True)
        dual_rank = [0.0] * len(rows)
        frac_rank = [0.0] * len(rows)
        for rnk, i in enumerate(dual_order):
            dual_rank[i] = 1.0 - rnk / n
        for rnk, i in enumerate(frac_order):
            frac_rank[i] = 1.0 - rnk / n
        packed = []
        for i, r in enumerate(rows):
            feats = build_features_row(r, dual_rank[i], frac_rank[i])
            packed.append((float(r["label"]), feats))
        groups.append((q, packed, gap))
    return groups


def to_xy(gs):
    X, y, gsz = [], [], []
    for _q, rows, _gap in gs:
        scores = [a for a, _ in rows]
        rel = relevance_from_scores(scores)
        for (_lab, feats), rr in zip(rows, rel):
            X.append(feats)
            y.append(float(rr))
        gsz.append(len(rows))
    return np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64), gsz


def main():
    assert len(FEATURE_NAMES) == 9 and SCHEMA_VERSION == 6
    best_report = None
    best_booster = None
    best_score = -1.0

    for gap_min in (0.05, 0.08, 0.10):
        groups = load_groups(gap_min)
        print(f"[gap>={gap_min}] groups={len(groups)}", flush=True)
        if len(groups) < 40:
            print("  skip: too few groups", flush=True)
            continue
        rng = random.Random(42)
        rng.shuffle(groups)
        n = len(groups)
        n_tr = int(0.70 * n)
        n_va = int(0.15 * n)
        tr, va, te = groups[:n_tr], groups[n_tr : n_tr + n_va], groups[n_tr + n_va :]
        Xtr, ytr, gtr = to_xy(tr)
        Xva, yva, gva = to_xy(va)
        Xte, yte, gte = to_xy(te)

        for cfg in (
            {"name": "base", "lr": 0.05, "leaves": 31, "min_leaf": 5, "rounds": 500},
            {"name": "deep", "lr": 0.03, "leaves": 63, "min_leaf": 3, "rounds": 700},
            {"name": "shallow", "lr": 0.08, "leaves": 15, "min_leaf": 10, "rounds": 400},
        ):
            params = {
                "objective": "lambdarank",
                "metric": "ndcg",
                "ndcg_at": [1, 3, 5],
                "learning_rate": cfg["lr"],
                "num_leaves": cfg["leaves"],
                "min_data_in_leaf": cfg["min_leaf"],
                "feature_fraction": 0.9,
                "feature_pre_filter": False,
                "verbosity": -1,
                "label_gain": list(range(0, max(32, int(ytr.max()) + 2))),
            }
            dtr = lgb.Dataset(Xtr, label=ytr, group=gtr, feature_name=list(FEATURE_NAMES))
            dva = lgb.Dataset(Xva, label=yva, group=gva, feature_name=list(FEATURE_NAMES), reference=dtr)
            booster = lgb.train(
                params,
                dtr,
                num_boost_round=cfg["rounds"],
                valid_sets=[dtr, dva],
                valid_names=["train", "val"],
                callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)],
            )
            val = _ndcg_top1(booster, Xva, yva, gva)
            test = _ndcg_top1(booster, Xte, yte, gte)
            score = val["ndcg@1"] + 0.5 * val["top1_hit"]
            print(
                f"  cfg={cfg['name']} val_ndcg1={val['ndcg@1']:.4f} top1={val['top1_hit']:.4f} "
                f"test_ndcg1={test['ndcg@1']:.4f} top1={test['top1_hit']:.4f}",
                flush=True,
            )
            if score > best_score:
                best_score = score
                best_booster = booster
                best_report = {
                    "gap_min": gap_min,
                    "config": cfg,
                    "val": val,
                    "test": test,
                    "n_train": len(tr),
                    "n_val": len(va),
                    "n_test": len(te),
                    "note": "v6: +dual_rank +frac_rank; gap-filtered; from existing v5 CSV",
                }

    assert best_booster is not None and best_report is not None
    best_booster.save_model(str(MODEL))
    schema = {
        "version": SCHEMA_VERSION,
        "features": list(FEATURE_NAMES),
        "objective": "lambdarank",
        "best": best_report,
    }
    MODEL.with_suffix(".schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    imp = best_booster.feature_importance(importance_type="gain")
    feat_gain = {n: float(v) for n, v in zip(FEATURE_NAMES, imp)}
    best_report["feature_gain"] = feat_gain
    out = DATA / "train_report_v6_quick.json"
    out.write_text(json.dumps(best_report, indent=2), encoding="utf-8")
    print("BEST", json.dumps({k: best_report[k] for k in ("gap_min", "config", "val", "test")}, indent=2))
    print("feature_gain", feat_gain)
    print("saved", MODEL)


if __name__ == "__main__":
    main()
