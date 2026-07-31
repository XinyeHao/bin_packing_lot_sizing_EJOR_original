# 新大规模 10 例：IM · BPC-frac · BPC-ML

- 种子：`41101`–`42001`（步长 100），与旧 `40101–40501` 不重叠
- 规模：large `(40, 55)`，时限 3600s，mip_gap=1%
- 每例每次求解独立子进程；结束后 disposeDefaultEnv + 进程退出

| seed | IM status | IM gap% | IM s | frac status | frac gap% | frac s | ML status | ML gap% | ML s |
|------|-----------|---------|------|-------------|-----------|--------|-----------|---------|------|
| 41101 | time_limit | 13.41 | 3602.1 | gap_limit | 0.99 | 556.8 | gap_limit | 0.98 | 371.2 |
| 41201 | time_limit | 14.88 | 3602.2 | gap_limit | 0.91 | 22.6 | gap_limit | 0.47 | 32.4 |
| 41301 | time_limit | 9.58 | 3602.1 | time_limit | 2.21 | 3600.9 | time_limit | 1.38 | 3600.3 |
| 41401 | time_limit | 6.95 | 3601.9 | gap_limit | 0.98 | 1045.6 | gap_limit | 0.99 | 721.7 |
| 41501 | time_limit | 11.60 | 3602.2 | time_limit | 1.81 | 3601.6 | time_limit | 2.72 | 3600.7 |
| 41601 | time_limit | 7.20 | 3601.8 | time_limit | 1.42 | 3602.2 | time_limit | 1.39 | 3601.3 |
| 41701 | time_limit | 10.53 | 3602.4 | gap_limit | 0.99 | 395.4 | gap_limit | 0.98 | 541.1 |
| 41801 | time_limit | 14.09 | 3602.6 | gap_limit | 1.00 | 204.8 | gap_limit | 0.91 | 220.4 |
| 41901 | time_limit | 12.83 | 3602.9 | gap_limit | 0.96 | 65.2 | gap_limit | 0.94 | 96.4 |
| 42001 | time_limit | 15.13 | 3602.3 | gap_limit | 0.99 | 863.8 | gap_limit | 0.98 | 1786.8 |

## 小结（已完成三方法的种子）

- 完成三方法：10/10
- IM：mean gap%=11.620，mean time=3602.3s，gap≤1% = 0/10
- frac：mean gap%=1.225，mean time=1395.9s，gap≤1% = 7/10
- ML：mean gap%=1.175，mean time=1457.2s，gap≤1% = 7/10
