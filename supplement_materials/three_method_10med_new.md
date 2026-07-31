# 新中规模 10 例：IM · BPC-frac · BPC-ML

- 种子：`31101`–`32001`（步长 100），与旧 `30101–30501` 不重叠
- 规模：medium `(35, 50)`，时限 1800s，mip_gap=1%
- 每例每次求解独立子进程；结束后 disposeDefaultEnv + 进程退出

| seed | IM status | IM gap% | IM s | frac status | frac gap% | frac s | ML status | ML gap% | ML s |
|------|-----------|---------|------|-------------|-----------|--------|-----------|---------|------|
| 31101 | time_limit | 6.22 | 1801.7 | time_limit | 1.04 | 1800.1 | gap_limit | 0.99 | 1572.4 |
| 31201 | time_limit | 10.41 | 1801.4 | gap_limit | 0.99 | 1748.9 | gap_limit | 1.00 | 856.0 |
| 31301 | time_limit | 13.41 | 1801.5 | time_limit | 2.55 | 1799.2 | time_limit | 2.35 | 1800.3 |
| 31401 | time_limit | 11.83 | 1801.3 | gap_limit | 1.00 | 440.3 | gap_limit | 1.00 | 408.8 |
| 31501 | time_limit | 8.22 | 1801.3 | gap_limit | 1.00 | 1557.9 | time_limit | 1.18 | 1799.2 |
| 31601 | optimal | 0.98 | 634.0 | gap_limit | 0.92 | 152.4 | gap_limit | 0.72 | 45.3 |
| 31701 | time_limit | 5.83 | 1801.4 | gap_limit | 0.99 | 586.2 | gap_limit | 1.00 | 1178.1 |
| 31801 | time_limit | 7.70 | 1801.2 | time_limit | 1.58 | 1800.0 | gap_limit | 0.98 | 566.4 |
| 31901 | time_limit | 8.88 | 1801.3 | gap_limit | 0.98 | 103.1 | gap_limit | 0.96 | 135.6 |
| 32001 | time_limit | 5.06 | 1803.8 | time_limit | 1.00 | 1801.2 | gap_limit | 1.00 | 1014.5 |

## 小结（已完成三方法的种子）

- 完成三方法：10/10
- IM：mean gap%=7.853，mean time=1684.9s，gap≤1% = 1/10
- frac：mean gap%=1.204，mean time=1178.9s，gap≤1% = 7/10
- ML：mean gap%=1.116，mean time=937.7s，gap≤1% = 8/10
