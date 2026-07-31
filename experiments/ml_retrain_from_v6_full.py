"""Offline label scan + retrain on ml_train_data_v6_full (no recollect).

Compares y_min / y_geo / y_prod / y_old vs frac & dual baselines,
picks best by val top1 (then ndcg@1), trains and writes ml_branch_model.txt.
"""
from __future__ import annotations

import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model_and_algo.ml_branch_selector import (  # noqa: E402
    FEATURE_NAMES,
    SCHEMA_VERSION,
    relevance_from_scores,
)

DATA_DIR = ROOT / "ml_train_data_v6_full"
MODEL_OUT = ROOT / "ml_branch_model.txt"
REPORT_OUT = DATA_DIR / "retrain_label_scan.json"
ABS_GAP = 1e-3
REL_GAP = 0.20
SEED = 42


def _load_raw_groups():
    buckets: dict[str, list[dict]] = defaultdict(list)
    for fp in sorted(DATA_DIR.glob("batch_*.csv")):
        with fp.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                buckets[r["query_id"]].append(r)
    return buckets


def _label_of(r: dict, key: str) -> float:
    if key == "y_min":
        return float(r["y_min"])
    if key == "y_geo":
        return float(r["y_geo"])
    if key == "y_old":
        return float(r["y_old"])
    if key == "y_prod":
        return float(r["dL"]) * float(r["dR"])
    raise KeyError(key)


def _keep_group(ys: np.ndarray) -> bool:
    if len(ys) < 2:
        return False
    best, second = float(ys.max()), float(np.sort(ys)[::-1][1])
    if best - second < ABS_GAP:
        return False
    if second > 1e-12 and (best - second) / second < REL_GAP and (best - second) < 0.05:
        return False
    return True


def _robust_z(ys: np.ndarray) -> list[float]:
    med = float(np.median(ys))
    mad = float(np.median(np.abs(ys - med))) + 1e-9
    return ((ys - med) / mad).tolist()


def _build_groups(buckets, label_key: str):
    groups = []
    for q, rows in buckets.items():
        ys = np.asarray([_label_of(r, label_key) for r in rows], dtype=np.float64)
        if not _keep_group(ys):
            continue
        scores = _robust_z(ys)
        feats = [[float(r[n]) for n in FEATURE_NAMES] for r in rows]
        fracs = [float(r["frac"]) for r in rows]
        duals = [float(r["dual_share"]) for r in rows]
        groups.append(
            {
                "q": q,
                "scores": scores,
                "ys": ys.tolist(),
                "feats": feats,
                "frac": fracs,
                "dual": duals,
            }
        )
    return groups


def _groups_to_xy(groups):
    X, y, gsz = [], [], []
    for g in groups:
        rel = relevance_from_scores(g["scores"])
        for feats, rr in zip(g["feats"], rel):
            X.append(feats)
            y.append(float(rr))
        gsz.append(len(g["feats"]))
    return np.asarray(X, np.float64), np.asarray(y, np.float64), gsz


def _top1_by_score(score_lists, y_lists):
    hits = n = 0
    for s, y in zip(score_lists, y_lists):
        if int(np.argmax(s)) == int(np.argmax(y)):
            hits += 1
        n += 1
    return hits / n if n else 0.0


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


def _train_one(train_g, val_g, test_g, cfg):
    import lightgbm as lgb

    Xtr, ytr, gtr = _groups_to_xy(train_g)
    Xva, yva, gva = _groups_to_xy(val_g)
    Xte, yte, gte = _groups_to_xy(test_g)
    label_max = int(max(ytr.max(), yva.max(), yte.max()))
    dtrain = lgb.Dataset(Xtr, label=ytr, group=gtr, feature_name=list(FEATURE_NAMES))
    dval = lgb.Dataset(Xva, label=yva, group=gva, feature_name=list(FEATURE_NAMES), reference=dtrain)
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_at": [1, 3, 5],
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
        callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)],
    )
    return {
        "booster": booster,
        "train": _ndcg_top1(booster, Xtr, ytr, gtr),
        "val": _ndcg_top1(booster, Xva, yva, gva),
        "test": _ndcg_top1(booster, Xte, yte, gte),
        "config": cfg,
    }


def main():
    assert SCHEMA_VERSION >= 6
    print(f"[load] {DATA_DIR}", flush=True)
    buckets = _load_raw_groups()
    print(f"[load] raw queries={len(buckets)}", flush=True)

    label_keys = ["y_min", "y_geo", "y_prod", "y_old"]
    configs = [
        {"name": "base", "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 5, "rounds": 500},
        {"name": "deep", "learning_rate": 0.03, "num_leaves": 63, "min_data_in_leaf": 3, "rounds": 700},
        {"name": "shallow", "learning_rate": 0.08, "num_leaves": 15, "min_data_in_leaf": 10, "rounds": 400},
    ]

    results = []
    best = None
    best_booster = None

    for lk in label_keys:
        groups = _build_groups(buckets, lk)
        train_g, val_g, test_g = _split(groups)
        # baselines on val/test using same label ys for top1
        def baseline(split, key):
            return _top1_by_score([g[key] for g in split], [g["ys"] for g in split])

        base = {
            "label": lk,
            "n_groups": len(groups),
            "split": {"train": len(train_g), "val": len(val_g), "test": len(test_g)},
            "baseline_val": {
                "frac": baseline(val_g, "frac"),
                "dual": baseline(val_g, "dual"),
                "random": 1.0 / max(1e-9, np.mean([len(g["ys"]) for g in val_g])),
            },
            "baseline_test": {
                "frac": baseline(test_g, "frac"),
                "dual": baseline(test_g, "dual"),
                "random": 1.0 / max(1e-9, np.mean([len(g["ys"]) for g in test_g])),
            },
            "trials": [],
        }
        print(
            f"\n[label={lk}] groups={len(groups)} "
            f"val_frac={base['baseline_val']['frac']:.4f} dual={base['baseline_val']['dual']:.4f}",
            flush=True,
        )
        local_best = None
        local_booster = None
        for cfg in configs:
            out = _train_one(train_g, val_g, test_g, cfg)
            trial = {
                "config": cfg["name"],
                "train": out["train"],
                "val": out["val"],
                "test": out["test"],
            }
            base["trials"].append(trial)
            print(
                f"  {cfg['name']}: val_top1={out['val']['top1_hit']:.4f} "
                f"ndcg1={out['val']['ndcg@1']:.4f} | "
                f"test_top1={out['test']['top1_hit']:.4f} ndcg1={out['test']['ndcg@1']:.4f}",
                flush=True,
            )
            score = out["val"]["top1_hit"] + 0.5 * out["val"]["ndcg@1"]
            if local_best is None or score > local_best["score"]:
                local_best = {"score": score, "trial": trial, "label": lk, "base": base}
                local_booster = out["booster"]
        assert local_best is not None
        base["best"] = local_best["trial"]
        delta = local_best["trial"]["val"]["top1_hit"] - base["baseline_val"]["frac"]
        base["val_top1_minus_frac"] = delta
        results.append(base)
        print(f"  >> best {local_best['trial']['config']} Δtop1_vs_frac={delta:+.4f}", flush=True)
        # global pick: prefer beating frac, then absolute val top1
        gscore = (
            local_best["trial"]["val"]["top1_hit"]
            + 0.3 * max(0.0, delta)
            + 0.2 * local_best["trial"]["val"]["ndcg@1"]
        )
        if best is None or gscore > best["gscore"]:
            best = {
                "gscore": gscore,
                "label": lk,
                "trial": local_best["trial"],
                "base": base,
            }
            best_booster = local_booster

    assert best is not None and best_booster is not None
    best_booster.save_model(str(MODEL_OUT))
    gain = {
        n: float(v)
        for n, v in zip(FEATURE_NAMES, best_booster.feature_importance(importance_type="gain"))
    }
    schema = {
        "version": SCHEMA_VERSION,
        "features": list(FEATURE_NAMES),
        "objective": "lambdarank",
        "label": f"{best['label']} + group robust-z (retrain from v6_full)",
        "split": best["base"]["split"],
        "best_config": best["trial"]["config"],
        "source": "ml_train_data_v6_full",
    }
    MODEL_OUT.with_suffix(".schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")

    report = {
        "chosen_label": best["label"],
        "chosen": best["trial"],
        "baselines": {
            "val": best["base"]["baseline_val"],
            "test": best["base"]["baseline_test"],
        },
        "val_top1_minus_frac": best["base"]["val_top1_minus_frac"],
        "feature_gain": gain,
        "all_labels": [
            {
                "label": r["label"],
                "n_groups": r["n_groups"],
                "best_val_top1": r["best"]["val"]["top1_hit"],
                "best_test_top1": r["best"]["test"]["top1_hit"],
                "frac_val": r["baseline_val"]["frac"],
                "delta_val": r["val_top1_minus_frac"],
            }
            for r in results
        ],
        "details": results,
    }
    REPORT_OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = [
        "# Retrain from v6_full (no recollect)",
        "",
        f"- chosen label: **{best['label']}** ({best['trial']['config']})",
        f"- val top1={best['trial']['val']['top1_hit']:.4f} (frac={best['base']['baseline_val']['frac']:.4f}, "
        f"Δ={best['base']['val_top1_minus_frac']:+.4f})",
        f"- test top1={best['trial']['test']['top1_hit']:.4f} (frac={best['base']['baseline_test']['frac']:.4f})",
        f"- val NDCG@1={best['trial']['val']['ndcg@1']:.4f} test NDCG@1={best['trial']['test']['ndcg@1']:.4f}",
        "",
        "## All labels",
    ]
    for r in results:
        md.append(
            f"- `{r['label']}`: val_top1={r['best']['val']['top1_hit']:.4f} "
            f"(frac={r['baseline_val']['frac']:.4f}, Δ={r['val_top1_minus_frac']:+.4f}); "
            f"test_top1={r['best']['test']['top1_hit']:.4f}"
        )
    REPORT_OUT.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n=== CHOSEN ===", flush=True)
    print(
        f"label={best['label']} cfg={best['trial']['config']} "
        f"val_top1={best['trial']['val']['top1_hit']:.4f} "
        f"test_top1={best['trial']['test']['top1_hit']:.4f} "
        f"Δvs_frac_val={best['base']['val_top1_minus_frac']:+.4f}",
        flush=True,
    )
    print(f"wrote {MODEL_OUT} and {REPORT_OUT}", flush=True)


if __name__ == "__main__":
    main()
