"""Offline: add pseudocost (replay) + gap/depth features + depth-weighted LTR.

Uses ml_train_data_v6_full only (no recollect). Compares:
  - v6 9 feats (baseline retrain)
  - v6 + gap/depth
  - v6 + gap/depth + replay PC
  - same with depth-weighted samples

Writes best model to ml_branch_model.txt if it beats frac baseline and prior.
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

from model_and_algo.ml_branch_selector import FEATURE_NAMES, relevance_from_scores

DATA_DIR = ROOT / "ml_train_data_v6_full"
MODEL_OUT = ROOT / "ml_branch_model.txt"
REPORT_OUT = DATA_DIR / "retrain_pc_depth.json"
ABS_GAP = 1e-3
REL_GAP = 0.20
SEED = 42
LABEL = "y_old"  # previously best offline

# Extended feature names for this experiment (not necessarily deployed schema)
EXT_FEATS = list(FEATURE_NAMES) + [
    "gap_rel",
    "depth_norm",
    "pc_up",
    "pc_down",
    "pc_prod",
    "pc_min",
]


def _label(r: dict) -> float:
    return float(r[LABEL])


def _keep(ys: np.ndarray) -> bool:
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


def _load_raw():
    buckets: dict[str, list[dict]] = defaultdict(list)
    for fp in sorted(DATA_DIR.glob("batch_*.csv")):
        with fp.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                buckets[r["query_id"]].append(r)
    return buckets


def _sort_key(qid: str, rows: list[dict]):
    r0 = rows[0]
    # chronological within instance: seed, depth, query index
    seed = int(float(r0.get("seed", 0)))
    depth = int(float(r0.get("depth", 0)))
    # query_id like small_s93004_n2
    n = 0
    if "_n" in qid:
        try:
            n = int(qid.rsplit("_n", 1)[1])
        except Exception:
            n = 0
    return (seed, depth, n, qid)


def _pc_score(frac: float, psi_dn: float, psi_up: float) -> tuple[float, float, float, float]:
    """Classic product / min style pseudocost scores."""
    f = max(0.0, min(1.0, frac))
    # downward uses fractional part toward floor; upward toward ceil
    # frac feature is min(f,1-f); recover f from val if possible — use frac as proxy mass
    a = f * psi_dn
    b = f * psi_up
    prod = math.sqrt(max(a, 0.0) * max(b, 0.0))
    return psi_up, psi_dn, prod, min(a, b)


def build_groups(buckets, use_pc: bool, use_ctx: bool):
    """Build groups with extended features; PC via chronological replay per seed."""
    items = sorted(buckets.items(), key=lambda kv: _sort_key(kv[0], kv[1]))
    # psi[kind] = [down, up] averages; count
    psi: dict[tuple[int, str], list[float]] = {}
    cnt: dict[tuple[int, str], int] = {}
    groups = []
    for qid, rows in items:
        ys = np.asarray([_label(r) for r in rows], dtype=np.float64)
        if not _keep(ys):
            continue
        r0 = rows[0]
        seed = int(float(r0.get("seed", 0)))
        depth = int(float(r0.get("depth", 0)))
        parent_lb = float(r0.get("parent_lb", 0.0))
        incumbent = float(r0.get("incumbent", parent_lb * 1.05))
        gap_rel = max(0.0, incumbent - parent_lb) / max(abs(parent_lb), 1.0)
        depth_norm = min(1.0, depth / 10.0)

        scores = _robust_z(ys)
        feats = []
        fracs = []
        duals = []
        for r in rows:
            base = [float(r[n]) for n in FEATURE_NAMES]
            frac = float(r["frac"])
            kind = r.get("kind", "setup_y0")
            key = (seed, kind)
            if use_pc:
                pu, pd = psi.get(key, [1.0, 1.0])  # neutral prior
                _, _, pc_prod, pc_min = _pc_score(frac, pd, pu)
                # normalize lightly later within group
                pc_up, pc_dn = pu, pd
            else:
                pc_up = pc_dn = pc_prod = pc_min = 0.0
            extra = []
            if use_ctx:
                extra.extend([gap_rel, depth_norm])
            else:
                extra.extend([0.0, 0.0])
            if use_pc:
                extra.extend([pc_up, pc_dn, pc_prod, pc_min])
            else:
                extra.extend([0.0, 0.0, 0.0, 0.0])
            feats.append(base + extra)
            fracs.append(frac)
            duals.append(float(r["dual_share"]))

        # within-group normalize PC columns if used
        if use_pc and feats:
            arr = np.asarray(feats, dtype=np.float64)
            for j in range(len(FEATURE_NAMES) + 2, len(FEATURE_NAMES) + 6):
                mx = float(arr[:, j].max()) + 1e-12
                arr[:, j] = arr[:, j] / mx
            feats = arr.tolist()

        # sample weight by depth (Hybrid-style): shallower → larger weight
        w = 1.0 / (1.0 + 0.35 * depth)

        groups.append(
            {
                "q": qid,
                "scores": scores,
                "ys": ys.tolist(),
                "feats": feats,
                "frac": fracs,
                "dual": duals,
                "depth": depth,
                "weight": w,
            }
        )

        # update PC only from the SB winner of this node (closer to real branching)
        if use_pc:
            win = max(rows, key=lambda r: _label(r))
            kind = win.get("kind", "setup_y0")
            key = (seed, kind)
            dL, dR = float(win["dL"]), float(win["dR"])
            old = psi.get(key, [1.0, 1.0])
            n = cnt.get(key, 0)
            n2 = n + 1
            psi[key] = [
                (old[0] * n + dL) / n2,
                (old[1] * n + dR) / n2,
            ]
            cnt[key] = n2
    return groups


def _groups_to_xy(groups, feat_dim: int, use_weight: bool):
    X, y, gsz, w = [], [], [], []
    for g in groups:
        rel = relevance_from_scores(g["scores"])
        for feats, rr in zip(g["feats"], rel):
            X.append(feats[:feat_dim])
            y.append(float(rr))
            w.append(g["weight"] if use_weight else 1.0)
        gsz.append(len(g["feats"]))
    return (
        np.asarray(X, np.float64),
        np.asarray(y, np.float64),
        gsz,
        np.asarray(w, np.float64),
    )


def _top1(score_lists, y_lists):
    h = n = 0
    for s, y in zip(score_lists, y_lists):
        if int(np.argmax(s)) == int(np.argmax(y)):
            h += 1
        n += 1
    return h / n if n else 0.0


def _ndcg_top1(booster, X, y, groups):
    scores = booster.predict(X)
    idx = 0
    nd1 = []
    hits = n = 0
    for g in groups:
        s = scores[idx : idx + g]
        lab = y[idx : idx + g]
        order = np.argsort(-s)
        ideal = np.sort(lab)[::-1]
        dcg = idcg = 0.0
        for i in range(min(1, g)):
            dcg += (2 ** lab[order[i]] - 1) / np.log2(i + 2)
            idcg += (2 ** ideal[i] - 1) / np.log2(i + 2)
        nd1.append(dcg / idcg if idcg > 0 else 0.0)
        if int(np.argmax(s)) == int(np.argmax(lab)):
            hits += 1
        n += 1
        idx += g
    return {
        "ndcg@1": float(np.mean(nd1)) if n else 0.0,
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


def train_eval(name, groups, feat_names, use_weight: bool):
    import lightgbm as lgb

    train_g, val_g, test_g = _split(groups)
    dim = len(feat_names)
    Xtr, ytr, gtr, wtr = _groups_to_xy(train_g, dim, use_weight)
    Xva, yva, gva, _ = _groups_to_xy(val_g, dim, False)
    Xte, yte, gte, _ = _groups_to_xy(test_g, dim, False)

    frac_val = _top1([g["frac"] for g in val_g], [g["ys"] for g in val_g])
    frac_te = _top1([g["frac"] for g in test_g], [g["ys"] for g in test_g])
    dual_val = _top1([g["dual"] for g in val_g], [g["ys"] for g in val_g])

    label_max = int(max(ytr.max(), yva.max(), yte.max()))
    dtrain = lgb.Dataset(
        Xtr, label=ytr, group=gtr, weight=wtr, feature_name=list(feat_names)
    )
    dval = lgb.Dataset(
        Xva, label=yva, group=gva, feature_name=list(feat_names), reference=dtrain
    )
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_at": [1, 3, 5],
        "learning_rate": 0.03,
        "num_leaves": 63,
        "min_data_in_leaf": 3,
        "feature_fraction": 0.9,
        "feature_pre_filter": False,
        "verbosity": -1,
        "label_gain": list(range(0, max(32, label_max + 2))),
    }
    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=700,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)],
    )
    val_m = _ndcg_top1(booster, Xva, yva, gva)
    te_m = _ndcg_top1(booster, Xte, yte, gte)
    tr_m = _ndcg_top1(booster, Xtr, ytr, gtr)
    gain = {
        n: float(v)
        for n, v in zip(feat_names, booster.feature_importance(importance_type="gain"))
    }
    out = {
        "name": name,
        "n_groups": len(groups),
        "feat_names": feat_names,
        "use_weight": use_weight,
        "baseline_val_frac": frac_val,
        "baseline_val_dual": dual_val,
        "baseline_test_frac": frac_te,
        "train": tr_m,
        "val": val_m,
        "test": te_m,
        "delta_val_vs_frac": val_m["top1_hit"] - frac_val,
        "delta_test_vs_frac": te_m["top1_hit"] - frac_te,
        "feature_gain": gain,
        "booster": booster,
        "split": {"train": len(train_g), "val": len(val_g), "test": len(test_g)},
    }
    print(
        f"[{name}] val_top1={val_m['top1_hit']:.4f} (frac={frac_val:.4f}, "
        f"Δ={out['delta_val_vs_frac']:+.4f}) | "
        f"test_top1={te_m['top1_hit']:.4f} (frac={frac_te:.4f}, "
        f"Δ={out['delta_test_vs_frac']:+.4f}) ndcg1_val={val_m['ndcg@1']:.4f}",
        flush=True,
    )
    top_g = sorted(gain.items(), key=lambda x: -x[1])[:8]
    print(f"  top gain: {', '.join(f'{k}={v:.0f}' for k,v in top_g)}", flush=True)
    return out


def main():
    print(f"[load] {DATA_DIR} label={LABEL}", flush=True)
    buckets = _load_raw()
    print(f"[load] queries={len(buckets)}", flush=True)

    variants = []
    # 1) plain v6
    g0 = build_groups(buckets, use_pc=False, use_ctx=False)
    variants.append(
        train_eval("v6_only", g0, list(FEATURE_NAMES), use_weight=False)
    )
    # 2) + context
    g1 = build_groups(buckets, use_pc=False, use_ctx=True)
    variants.append(
        train_eval(
            "v6+ctx",
            g1,
            list(FEATURE_NAMES) + ["gap_rel", "depth_norm"],
            use_weight=False,
        )
    )
    # 3) + ctx + PC
    g2 = build_groups(buckets, use_pc=True, use_ctx=True)
    variants.append(
        train_eval("v6+ctx+pc", g2, EXT_FEATS, use_weight=False)
    )
    # 4) + depth weight
    variants.append(
        train_eval("v6+ctx+pc+depthW", g2, EXT_FEATS, use_weight=True)
    )
    # 5) ctx + depthW without PC
    variants.append(
        train_eval(
            "v6+ctx+depthW",
            g1,
            list(FEATURE_NAMES) + ["gap_rel", "depth_norm"],
            use_weight=True,
        )
    )

    # pick by val top1 then delta vs frac
    best = max(
        variants,
        key=lambda v: (v["val"]["top1_hit"], v["delta_val_vs_frac"], v["val"]["ndcg@1"]),
    )
    print("\n=== BEST ===", best["name"], flush=True)

    # Only overwrite deployed model if feats ⊆ deployable AND name is v6_only
    # Extended feats need schema bump — save side model + report; deploy v6-compatible if best is v6
    side = DATA_DIR / f"model_{best['name']}.txt"
    best["booster"].save_model(str(side))

    deployable = best["feat_names"] == list(FEATURE_NAMES)
    if deployable:
        best["booster"].save_model(str(MODEL_OUT))
        schema = {
            "version": 6,
            "features": list(FEATURE_NAMES),
            "label": LABEL,
            "note": f"retrain {best['name']}",
        }
        MODEL_OUT.with_suffix(".schema.json").write_text(
            json.dumps(schema, indent=2), encoding="utf-8"
        )
        deployed = str(MODEL_OUT)
    else:
        # Keep previous deploy; save extended schema beside
        schema = {
            "version": "6+ext",
            "features": best["feat_names"],
            "label": LABEL,
            "note": "extended PC/depth — needs online wiring before deploy",
        }
        side.with_suffix(".schema.json").write_text(
            json.dumps(schema, indent=2), encoding="utf-8"
        )
        deployed = f"(not deployed; saved {side.name})"

    report = {
        "label": LABEL,
        "best": best["name"],
        "deployed": deployed,
        "variants": [
            {
                k: v[k]
                for k in (
                    "name",
                    "n_groups",
                    "use_weight",
                    "feat_names",
                    "baseline_val_frac",
                    "baseline_test_frac",
                    "train",
                    "val",
                    "test",
                    "delta_val_vs_frac",
                    "delta_test_vs_frac",
                    "feature_gain",
                    "split",
                )
            }
            for v in variants
        ],
    }
    # drop boosters
    REPORT_OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# PC + depth retrain (offline)",
        "",
        f"- label: `{LABEL}`",
        f"- best: **{best['name']}**",
        f"- val top1={best['val']['top1_hit']:.4f} (frac={best['baseline_val_frac']:.4f}, "
        f"Δ={best['delta_val_vs_frac']:+.4f})",
        f"- test top1={best['test']['top1_hit']:.4f} (frac={best['baseline_test_frac']:.4f}, "
        f"Δ={best['delta_test_vs_frac']:+.4f})",
        f"- deploy: {deployed}",
        "",
        "## Variants",
    ]
    for v in variants:
        lines.append(
            f"- `{v['name']}`: val={v['val']['top1_hit']:.4f} "
            f"(Δfrac {v['delta_val_vs_frac']:+.4f}); "
            f"test={v['test']['top1_hit']:.4f} (Δfrac {v['delta_test_vs_frac']:+.4f})"
        )
    REPORT_OUT.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {REPORT_OUT}", flush=True)


if __name__ == "__main__":
    main()
