# 中 / 大规模对比汇总（指向全量三方表）

全量 **30 例**（中15 + 大15）三方结果见：

- [`all_results_im_frac_ml.md`](all_results_im_frac_ml.md)
- [`test_instance.md`](test_instance.md) §5
- 新大 10：[`three_method_10large_new.md`](three_method_10large_new.md)
- 新中 10：[`three_method_10med_new.md`](three_method_10med_new.md)

## 总览（摘录）

| 集合 | IM mean gap% | frac mean gap% | ML mean gap% | IM mean s | frac mean s | ML mean s | 达1% IM/frac/ML |
|------|-------------:|---------------:|-------------:|----------:|------------:|----------:|-----------------|
| 中（15） | 7.97 | 1.10 | 1.04 | 1723.8 | 898.2 | 692.2 | 1/15 · 12/15 · 13/15 |
| 大（15） | 11.51 | 1.28 | 1.14 | 3602.2 | 1813.8 | 1736.8 | 0/15 · 9/15 · 10/15 |
| 全部（30） | 9.74 | 1.19 | 1.09 | 2663.0 | 1356.0 | 1214.5 | 1/30 · 21/30 · 23/30 |

ML vs frac 的分规模均值/方差/极差/配对胜负见 [`test_instance.md`](test_instance.md) **§5.7**。
