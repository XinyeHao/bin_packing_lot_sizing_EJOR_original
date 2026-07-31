"""Paired stats: BPC-ML vs BPC-frac on medium / large."""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

SUP = Path(__file__).resolve().parents[1] / "supplement_materials"


def load():
    frac, ml = {}, {}
    for name, dest in [
        ("bpc_frac_gap1pct_5med_5large.json", "frac"),
        ("bpc_ml_gap1pct_5med_5large.json", "ml"),
    ]:
        for r in json.loads((SUP / name).read_text(encoding="utf-8"))["rows"]:
            if "error" in r:
                continue
            key = (r["scale"], int(r["seed"]))
            if dest == "frac":
                frac[key] = r
            else:
                ml[key] = r
    for fname, scale in [
        ("three_method_10med_new.json", "medium"),
        ("three_method_10large_new.json", "large"),
    ]:
        for r in json.loads((SUP / fname).read_text(encoding="utf-8"))["rows"]:
            if "error" in r:
                continue
            key = (scale, int(r["seed"]))
            if r["method"] == "frac":
                frac[key] = r
            elif r["method"] == "ml":
                ml[key] = r
    return frac, ml


def summarize(xs):
    xs = list(xs)
    sd = st.stdev(xs) if len(xs) > 1 else 0.0
    mu = st.mean(xs)
    return {
        "n": len(xs),
        "mean": mu,
        "median": st.median(xs),
        "stdev": sd,
        "min": min(xs),
        "max": max(xs),
        "range": max(xs) - min(xs),
        "cv": (sd / mu) if mu != 0 else 0.0,
    }


def fmt(d, nd=2):
    return (
        f"n={d['n']} mean={d['mean']:.{nd}f} med={d['median']:.{nd}f} "
        f"sd={d['stdev']:.{nd}f} min={d['min']:.{nd}f} max={d['max']:.{nd}f} "
        f"range={d['range']:.{nd}f} cv={d['cv']:.2f}"
    )


def analyze(label, keys, frac, ml):
    print(f"\n======== {label} n={len(keys)} ========")
    fg = [frac[k]["gap_pct"] for k in keys]
    mg = [ml[k]["gap_pct"] for k in keys]
    ft = [frac[k]["runtime_sec"] for k in keys]
    mt = [ml[k]["runtime_sec"] for k in keys]
    dg = [ml[k]["gap_pct"] - frac[k]["gap_pct"] for k in keys]
    dt = [ml[k]["runtime_sec"] - frac[k]["runtime_sec"] for k in keys]
    tr = [ml[k]["runtime_sec"] / frac[k]["runtime_sec"] for k in keys]

    print("【效果 gap%】")
    print("  frac:", fmt(summarize(fg), 3))
    print("  ML  :", fmt(summarize(mg), 3))
    print("  Δ(ML−frac):", fmt(summarize(dg), 3))
    print(
        f"  ML更好(gap更小): {sum(1 for x in dg if x < -1e-9)}/{len(dg)}; "
        f"更差: {sum(1 for x in dg if x > 1e-9)}/{len(dg)}; "
        f"持平: {sum(1 for x in dg if abs(x) <= 1e-9)}/{len(dg)}"
    )
    print(
        f"  达1% frac={sum(1 for g in fg if g <= 1 + 1e-9)}/{len(fg)}; "
        f"ML={sum(1 for g in mg if g <= 1 + 1e-9)}/{len(mg)}"
    )

    print("【效率 time s】")
    print("  frac:", fmt(summarize(ft), 1))
    print("  ML  :", fmt(summarize(mt), 1))
    print("  Δ(ML−frac):", fmt(summarize(dt), 1))
    print("  时间比 ML/frac:", fmt(summarize(tr), 3))
    print(
        f"  ML更快: {sum(1 for x in dt if x < -1e-6)}/{len(dt)}; "
        f"更慢: {sum(1 for x in dt if x > 1e-6)}/{len(dt)}"
    )

    best_g = min(keys, key=lambda k: ml[k]["gap_pct"] - frac[k]["gap_pct"])
    worst_g = max(keys, key=lambda k: ml[k]["gap_pct"] - frac[k]["gap_pct"])
    best_t = min(keys, key=lambda k: ml[k]["runtime_sec"] - frac[k]["runtime_sec"])
    worst_t = max(keys, key=lambda k: ml[k]["runtime_sec"] - frac[k]["runtime_sec"])
    print("【配对极值】")
    k = best_g
    print(
        f"  gap改进最大 seed={k[1]}: {frac[k]['gap_pct']:.3f}% → {ml[k]['gap_pct']:.3f}% "
        f"(Δ={ml[k]['gap_pct']-frac[k]['gap_pct']:+.3f})"
    )
    k = worst_g
    print(
        f"  gap变差最大 seed={k[1]}: {frac[k]['gap_pct']:.3f}% → {ml[k]['gap_pct']:.3f}% "
        f"(Δ={ml[k]['gap_pct']-frac[k]['gap_pct']:+.3f})"
    )
    k = best_t
    print(
        f"  时间节省最多 seed={k[1]}: {frac[k]['runtime_sec']:.1f}s → {ml[k]['runtime_sec']:.1f}s "
        f"(Δ={ml[k]['runtime_sec']-frac[k]['runtime_sec']:+.1f})"
    )
    k = worst_t
    print(
        f"  时间多花最多 seed={k[1]}: {frac[k]['runtime_sec']:.1f}s → {ml[k]['runtime_sec']:.1f}s "
        f"(Δ={ml[k]['runtime_sec']-frac[k]['runtime_sec']:+.1f})"
    )

    win_both = sum(
        1
        for k in keys
        if ml[k]["gap_pct"] <= frac[k]["gap_pct"] + 1e-9
        and ml[k]["runtime_sec"] <= frac[k]["runtime_sec"] + 1e-6
    )
    print(f"  双赢(gap≤且time≤): {win_both}/{len(keys)}")


def main():
    frac, ml = load()
    med = sorted(k for k in frac if k[0] == "medium" and k in ml)
    large = sorted(k for k in frac if k[0] == "large" and k in ml)
    analyze("MEDIUM", med, frac, ml)
    analyze("LARGE", large, frac, ml)
    analyze("ALL", med + large, frac, ml)


if __name__ == "__main__":
    main()
