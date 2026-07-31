"""Generate supplement_materials/all_results_im_frac_ml.md from JSON artifacts."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUP = ROOT / "supplement_materials"


def _load_rows(name: str) -> dict[tuple[str, int], dict]:
    data = json.loads((SUP / name).read_text(encoding="utf-8"))
    out: dict[tuple[str, int], dict] = {}
    for r in data.get("rows", []):
        if "error" in r:
            continue
        out[(r["scale"], int(r["seed"]))] = r
    return out


def _merge_three_method(path: str, scale: str, im, frac, ml) -> None:
    data = json.loads((SUP / path).read_text(encoding="utf-8"))
    for r in data["rows"]:
        if "error" in r:
            continue
        key = (scale, int(r["seed"]))
        meth = r["method"]
        if meth == "im":
            im[key] = r
        elif meth == "frac":
            frac[key] = r
        elif meth == "ml":
            ml[key] = r


def fmt(v, nd=2):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def mean(xs):
    return sum(xs) / len(xs)


def block(title: str, keys, im_m, frac_m, ml_m):
    lines = [
        f"### {title}",
        "",
        "| seed | IM gap% | IM s | frac gap% | frac s | ML gap% | ML s | IM status | frac status | ML status |",
        "|------|--------:|-----:|----------:|-------:|--------:|------:|-----------|-------------|-----------|",
    ]
    gaps = {"im": [], "frac": [], "ml": []}
    times = {"im": [], "frac": [], "ml": []}
    hits = {"im": 0, "frac": 0, "ml": 0}
    n = 0
    for key in keys:
        n += 1
        if isinstance(key, tuple):
            _, seed = key
            rim, rfr, rml = im_m.get(key), frac_m.get(key), ml_m.get(key)
        else:
            seed = key
            rim = im_m.get(("medium", seed))
            rfr = frac_m.get(("medium", seed))
            rml = ml_m.get(("medium", seed))
        for name, r in (("im", rim), ("frac", rfr), ("ml", rml)):
            if r and r.get("gap_pct") is not None:
                gaps[name].append(r["gap_pct"])
                times[name].append(r["runtime_sec"])
                if r["gap_pct"] <= 1.0 + 1e-9:
                    hits[name] += 1
        lines.append(
            f"| {seed} | {fmt(rim.get('gap_pct') if rim else None)} | "
            f"{fmt(rim.get('runtime_sec') if rim else None, 1)} | "
            f"{fmt(rfr.get('gap_pct') if rfr else None)} | "
            f"{fmt(rfr.get('runtime_sec') if rfr else None, 1)} | "
            f"{fmt(rml.get('gap_pct') if rml else None)} | "
            f"{fmt(rml.get('runtime_sec') if rml else None, 1)} | "
            f"{rim.get('status') if rim else '—'} | "
            f"{rfr.get('status') if rfr else '—'} | "
            f"{rml.get('status') if rml else '—'} |"
        )
    lines.append(
        f"| **均值** | **{mean(gaps['im']):.2f}** | **{mean(times['im']):.1f}** | "
        f"**{mean(gaps['frac']):.2f}** | **{mean(times['frac']):.1f}** | "
        f"**{mean(gaps['ml']):.2f}** | **{mean(times['ml']):.1f}** | "
        f"gap≤1%: {hits['im']}/{n} | {hits['frac']}/{n} | {hits['ml']}/{n} |"
    )
    lines.append("")
    stats = {
        k: (mean(gaps[k]), mean(times[k]), hits[k], n) for k in ("im", "frac", "ml")
    }
    return "\n".join(lines), stats


def row_sum(label: str, s: dict) -> str:
    g_im, t_im, h_im, n = s["im"]
    g_f, t_f, h_f, _ = s["frac"]
    g_m, t_m, h_m, _ = s["ml"]
    return (
        f"| {label} | {g_im:.2f} | {g_f:.2f} | {g_m:.2f} | "
        f"{t_im:.1f} | {t_f:.1f} | {t_m:.1f} | "
        f"{h_im}/{n} · {h_f}/{n} · {h_m}/{n} |"
    )


def main() -> None:
    im = _load_rows("im_gap1pct_5med_5large.json")
    frac = _load_rows("bpc_frac_gap1pct_5med_5large.json")
    ml = _load_rows("bpc_ml_gap1pct_5med_5large.json")
    _merge_three_method("three_method_10med_new.json", "medium", im, frac, ml)
    _merge_three_method("three_method_10large_new.json", "large", im, frac, ml)

    med5 = [("medium", s) for s in range(30101, 30601, 100)]
    large5 = [("large", s) for s in range(40101, 40601, 100)]
    med10 = [("medium", s) for s in range(31101, 32101, 100)]
    large10 = [("large", s) for s in range(41101, 42101, 100)]
    med15 = med5 + med10
    large15 = large5 + large10
    all30 = med15 + large15

    b_med, s_med = block("中规模：15 例（30101–30501 + 31101–32001）", med15, im, frac, ml)
    b_large, s_large = block("大规模：15 例（40101–40501 + 41101–42001）", large15, im, frac, ml)
    b_all, s_all = block("全部算例：中15 + 大15 = 30 例", all30, im, frac, ml)

    header = """# 全量算例结果：IM · BPC-frac · BPC-ML

统一设定：窗长 `length_range=[12,16]`（§5.1），LT∈{5…10}，M=U=5；停机 `(UB−LB)/UB ≤ 1%` 或时限。

| 档 | 种子 | (T,I) | 时限 | n |
|----|------|-------|------|---|
| 中 | 30101–30501，31101–32001 | (35,50) | 1800s | 15 |
| 大 | 40101–40501，41101–42001 | (40,55) | 3600s | 15 |

方法：IM（`MIPGap=0.01`）；BPC-frac（`frac_sb`）；BPC-ML（`ml`，`ml_branch_model.txt` v7）。

原始 JSON：`im_gap1pct_5med_5large.json`，`bpc_frac_gap1pct_5med_5large.json`，`bpc_ml_gap1pct_5med_5large.json`，`three_method_10med_new.json`，`three_method_10large_new.json`。

## 总览

| 集合 | IM mean gap% | frac mean gap% | ML mean gap% | IM mean s | frac mean s | ML mean s | 达1% IM/frac/ML |
|------|-------------:|---------------:|-------------:|----------:|------------:|----------:|-----------------|
"""
    overview = "\n".join(
        [
            row_sum("中（15）", s_med),
            row_sum("大（15）", s_large),
            row_sum("全部（30）", s_all),
        ]
    )
    footer = """
## 简要结论

1. **IM** 几乎全程满时限，gap 远大于 1%。
2. **BPC-frac / BPC-ML** 将 gap 压到约 1% 量级；优势主要来自更紧的 tree LB。
3. **BPC-ML** 相对 frac 为**小幅改进**（全量约 gap −0.1pp、均时 −10%、达 1% 21→23/30），不是质变；中规模更稳更快，大规模期望略优但波动更大。分规模均值/中位数/sd/极差/CV 与配对胜负见 `test_instance.md` **§5.7**。

## 文档索引

| 文件 | 内容 |
|------|------|
| `all_results_im_frac_ml.md` | 本文件（逐例全表） |
| `test_instance.md` §5.7 | ML vs frac 效率/效果/方差分析 |
| `three_method_10med_new.md` | 新中 10 |
| `three_method_10large_new.md` | 新大 10 |
"""
    out = header + overview + "\n\n" + b_med + "\n" + b_large + "\n" + b_all + footer
    path = SUP / "all_results_im_frac_ml.md"
    path.write_text(out, encoding="utf-8")
    print("wrote", path)
    print(overview)

    (SUP / "im_frac_ml_three_way.md").write_text(
        out.replace(
            "# 全量算例结果：IM · BPC-frac · BPC-ML",
            "# 三方对比（全量刷新）：IM · BPC-frac · BPC-ML",
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
