# 新中规模 10 例：IM · BPC-frac · BPC-ML

- 种子：`31101`–`32001`（步长 100），与旧 `30101–30501` 不重叠
- 规模：medium `(35, 50)`，时限 1800s，mip_gap=1%
- 每例每次求解独立子进程；结束后 disposeDefaultEnv + 进程退出

| seed | IM status | IM gap% | IM s | frac status | frac gap% | frac s | ML status | ML gap% | ML s |
|------|-----------|---------|------|-------------|-----------|--------|-----------|---------|------|
| 31101 | time_limit | 5.08 | 1802.1 | gap_limit | 0.99 | 1173.7 | gap_limit | 0.99 | 1338.3 |
| 31201 | time_limit | 6.11 | 1802.2 | gap_limit | 0.99 | 434.8 | gap_limit | 0.88 | 225.5 |
| 31301 | time_limit | 5.86 | 1802.0 | time_limit | 3.01 | 1801.9 | time_limit | 3.30 | 1802.2 |
| 31401 | time_limit | 10.31 | 1802.0 | gap_limit | 0.88 | 174.8 | gap_limit | 1.00 | 208.4 |
| 31501 | time_limit | 9.38 | 1802.1 | gap_limit | 0.98 | 1608.1 | time_limit | 1.56 | 1800.5 |
| 31601 | optimal | 0.86 | 758.4 | optimal | 0.00 | 96.5 | gap_limit | 0.30 | 104.2 |
| 31701 | time_limit | 7.41 | 1802.0 | gap_limit | 0.96 | 1270.5 | gap_limit | 1.00 | 1668.2 |
| 31801 | time_limit | 9.04 | 1802.0 | time_limit | 3.54 | 3132.6 | gap_limit | 1.00 | 1612.3 |
| 31901 | time_limit | 5.66 | 1802.0 | gap_limit | 0.96 | 43.2 | gap_limit | 0.91 | 47.9 |
| 32001 | time_limit | 4.16 | 1802.0 | gap_limit | 0.99 | 927.8 | gap_limit | 0.92 | 381.7 |

## 小结（已完成三方法的种子）

- 完成三方法：10/10
- IM：mean gap%=6.387，mean time=1697.7s，gap≤1% = 1/10
- frac：mean gap%=1.331，mean time=1066.4s，gap≤1% = 8/10
- ML：mean gap%=1.184，mean time=918.9s，gap≤1% = 8/10
