# 全量算例结果：IM · BPC-frac · BPC-ML

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
| 中（15） | 7.97 | 1.10 | 1.04 | 1723.8 | 898.2 | 692.2 | 1/15 · 12/15 · 13/15 |
| 大（15） | 11.51 | 1.28 | 1.14 | 3602.2 | 1813.8 | 1736.8 | 0/15 · 9/15 · 10/15 |
| 全部（30） | 9.74 | 1.19 | 1.09 | 2663.0 | 1356.0 | 1214.5 | 1/30 · 21/30 · 23/30 |

### 中规模：15 例（30101–30501 + 31101–32001）

| seed | IM gap% | IM s | frac gap% | frac s | ML gap% | ML s | IM status | frac status | ML status |
|------|--------:|-----:|----------:|-------:|--------:|------:|-----------|-------------|-----------|
| 30101 | 13.58 | 1801.3 | 0.77 | 26.7 | 0.75 | 55.0 | time_limit | gap_limit | gap_limit |
| 30201 | 7.24 | 1801.3 | 0.91 | 26.9 | 0.91 | 37.2 | time_limit | gap_limit | gap_limit |
| 30301 | 5.34 | 1801.8 | 0.84 | 47.0 | 0.79 | 37.6 | time_limit | gap_limit | gap_limit |
| 30401 | 7.48 | 1801.4 | 0.98 | 1153.5 | 0.92 | 471.6 | time_limit | gap_limit | gap_limit |
| 30501 | 7.44 | 1801.6 | 1.00 | 429.1 | 1.00 | 405.2 | time_limit | gap_limit | gap_limit |
| 31101 | 6.22 | 1801.7 | 1.04 | 1800.1 | 0.99 | 1572.4 | time_limit | time_limit | gap_limit |
| 31201 | 10.41 | 1801.4 | 0.99 | 1748.9 | 1.00 | 856.0 | time_limit | gap_limit | gap_limit |
| 31301 | 13.41 | 1801.5 | 2.55 | 1799.2 | 2.35 | 1800.3 | time_limit | time_limit | time_limit |
| 31401 | 11.83 | 1801.3 | 1.00 | 440.3 | 1.00 | 408.8 | time_limit | gap_limit | gap_limit |
| 31501 | 8.22 | 1801.3 | 1.00 | 1557.9 | 1.18 | 1799.2 | time_limit | gap_limit | time_limit |
| 31601 | 0.98 | 634.0 | 0.92 | 152.4 | 0.72 | 45.3 | optimal | gap_limit | gap_limit |
| 31701 | 5.83 | 1801.4 | 0.99 | 586.2 | 1.00 | 1178.1 | time_limit | gap_limit | gap_limit |
| 31801 | 7.70 | 1801.2 | 1.58 | 1800.0 | 0.98 | 566.4 | time_limit | time_limit | gap_limit |
| 31901 | 8.88 | 1801.3 | 0.98 | 103.1 | 0.96 | 135.6 | time_limit | gap_limit | gap_limit |
| 32001 | 5.06 | 1803.8 | 1.00 | 1801.2 | 1.00 | 1014.5 | time_limit | time_limit | gap_limit |
| **均值** | **7.97** | **1723.8** | **1.10** | **898.2** | **1.04** | **692.2** | gap≤1%: 1/15 | 12/15 | 13/15 |

### 大规模：15 例（40101–40501 + 41101–42001）

| seed | IM gap% | IM s | frac gap% | frac s | ML gap% | ML s | IM status | frac status | ML status |
|------|--------:|-----:|----------:|-------:|--------:|------:|-----------|-------------|-----------|
| 40101 | 14.53 | 3602.3 | 1.31 | 3600.5 | 0.96 | 1115.8 | time_limit | time_limit | gap_limit |
| 40201 | 10.85 | 3602.6 | 0.99 | 1839.0 | 1.00 | 2593.6 | time_limit | gap_limit | gap_limit |
| 40301 | 15.43 | 3601.9 | 2.06 | 3602.6 | 1.12 | 3602.0 | time_limit | time_limit | time_limit |
| 40401 | 8.28 | 3601.8 | 1.58 | 3602.5 | 1.31 | 3599.6 | time_limit | time_limit | time_limit |
| 40501 | 7.34 | 3602.4 | 0.99 | 604.0 | 0.97 | 569.2 | time_limit | gap_limit | gap_limit |
| 41101 | 13.41 | 3602.1 | 0.99 | 556.8 | 0.98 | 371.2 | time_limit | gap_limit | gap_limit |
| 41201 | 14.88 | 3602.2 | 0.91 | 22.6 | 0.47 | 32.4 | time_limit | gap_limit | gap_limit |
| 41301 | 9.58 | 3602.1 | 2.21 | 3600.9 | 1.38 | 3600.3 | time_limit | time_limit | time_limit |
| 41401 | 6.95 | 3601.9 | 0.98 | 1045.6 | 0.99 | 721.7 | time_limit | gap_limit | gap_limit |
| 41501 | 11.60 | 3602.2 | 1.81 | 3601.6 | 2.72 | 3600.7 | time_limit | time_limit | time_limit |
| 41601 | 7.20 | 3601.8 | 1.42 | 3602.2 | 1.39 | 3601.3 | time_limit | time_limit | time_limit |
| 41701 | 10.53 | 3602.4 | 0.99 | 395.4 | 0.98 | 541.1 | time_limit | gap_limit | gap_limit |
| 41801 | 14.09 | 3602.6 | 1.00 | 204.8 | 0.91 | 220.4 | time_limit | gap_limit | gap_limit |
| 41901 | 12.83 | 3602.9 | 0.96 | 65.2 | 0.94 | 96.4 | time_limit | gap_limit | gap_limit |
| 42001 | 15.13 | 3602.3 | 0.99 | 863.8 | 0.98 | 1786.8 | time_limit | gap_limit | gap_limit |
| **均值** | **11.51** | **3602.2** | **1.28** | **1813.8** | **1.14** | **1736.8** | gap≤1%: 0/15 | 9/15 | 10/15 |

### 全部算例：中15 + 大15 = 30 例

| seed | IM gap% | IM s | frac gap% | frac s | ML gap% | ML s | IM status | frac status | ML status |
|------|--------:|-----:|----------:|-------:|--------:|------:|-----------|-------------|-----------|
| 30101 | 13.58 | 1801.3 | 0.77 | 26.7 | 0.75 | 55.0 | time_limit | gap_limit | gap_limit |
| 30201 | 7.24 | 1801.3 | 0.91 | 26.9 | 0.91 | 37.2 | time_limit | gap_limit | gap_limit |
| 30301 | 5.34 | 1801.8 | 0.84 | 47.0 | 0.79 | 37.6 | time_limit | gap_limit | gap_limit |
| 30401 | 7.48 | 1801.4 | 0.98 | 1153.5 | 0.92 | 471.6 | time_limit | gap_limit | gap_limit |
| 30501 | 7.44 | 1801.6 | 1.00 | 429.1 | 1.00 | 405.2 | time_limit | gap_limit | gap_limit |
| 31101 | 6.22 | 1801.7 | 1.04 | 1800.1 | 0.99 | 1572.4 | time_limit | time_limit | gap_limit |
| 31201 | 10.41 | 1801.4 | 0.99 | 1748.9 | 1.00 | 856.0 | time_limit | gap_limit | gap_limit |
| 31301 | 13.41 | 1801.5 | 2.55 | 1799.2 | 2.35 | 1800.3 | time_limit | time_limit | time_limit |
| 31401 | 11.83 | 1801.3 | 1.00 | 440.3 | 1.00 | 408.8 | time_limit | gap_limit | gap_limit |
| 31501 | 8.22 | 1801.3 | 1.00 | 1557.9 | 1.18 | 1799.2 | time_limit | gap_limit | time_limit |
| 31601 | 0.98 | 634.0 | 0.92 | 152.4 | 0.72 | 45.3 | optimal | gap_limit | gap_limit |
| 31701 | 5.83 | 1801.4 | 0.99 | 586.2 | 1.00 | 1178.1 | time_limit | gap_limit | gap_limit |
| 31801 | 7.70 | 1801.2 | 1.58 | 1800.0 | 0.98 | 566.4 | time_limit | time_limit | gap_limit |
| 31901 | 8.88 | 1801.3 | 0.98 | 103.1 | 0.96 | 135.6 | time_limit | gap_limit | gap_limit |
| 32001 | 5.06 | 1803.8 | 1.00 | 1801.2 | 1.00 | 1014.5 | time_limit | time_limit | gap_limit |
| 40101 | 14.53 | 3602.3 | 1.31 | 3600.5 | 0.96 | 1115.8 | time_limit | time_limit | gap_limit |
| 40201 | 10.85 | 3602.6 | 0.99 | 1839.0 | 1.00 | 2593.6 | time_limit | gap_limit | gap_limit |
| 40301 | 15.43 | 3601.9 | 2.06 | 3602.6 | 1.12 | 3602.0 | time_limit | time_limit | time_limit |
| 40401 | 8.28 | 3601.8 | 1.58 | 3602.5 | 1.31 | 3599.6 | time_limit | time_limit | time_limit |
| 40501 | 7.34 | 3602.4 | 0.99 | 604.0 | 0.97 | 569.2 | time_limit | gap_limit | gap_limit |
| 41101 | 13.41 | 3602.1 | 0.99 | 556.8 | 0.98 | 371.2 | time_limit | gap_limit | gap_limit |
| 41201 | 14.88 | 3602.2 | 0.91 | 22.6 | 0.47 | 32.4 | time_limit | gap_limit | gap_limit |
| 41301 | 9.58 | 3602.1 | 2.21 | 3600.9 | 1.38 | 3600.3 | time_limit | time_limit | time_limit |
| 41401 | 6.95 | 3601.9 | 0.98 | 1045.6 | 0.99 | 721.7 | time_limit | gap_limit | gap_limit |
| 41501 | 11.60 | 3602.2 | 1.81 | 3601.6 | 2.72 | 3600.7 | time_limit | time_limit | time_limit |
| 41601 | 7.20 | 3601.8 | 1.42 | 3602.2 | 1.39 | 3601.3 | time_limit | time_limit | time_limit |
| 41701 | 10.53 | 3602.4 | 0.99 | 395.4 | 0.98 | 541.1 | time_limit | gap_limit | gap_limit |
| 41801 | 14.09 | 3602.6 | 1.00 | 204.8 | 0.91 | 220.4 | time_limit | gap_limit | gap_limit |
| 41901 | 12.83 | 3602.9 | 0.96 | 65.2 | 0.94 | 96.4 | time_limit | gap_limit | gap_limit |
| 42001 | 15.13 | 3602.3 | 0.99 | 863.8 | 0.98 | 1786.8 | time_limit | gap_limit | gap_limit |
| **均值** | **9.74** | **2663.0** | **1.19** | **1356.0** | **1.09** | **1214.5** | gap≤1%: 1/30 | 21/30 | 23/30 |

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
