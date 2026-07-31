# BPC frac_sb：5 medium + 5 large（gap≤1% 停）

- 策略：`frac_sb`（k=5），无 ML
- 停止：`(UB−tree_LB)/UB ≤ 1%` 或时限
- medium `(35, 50)` @1800s；large `(40, 55)` @3600s；M=U=5
- 每例独立子进程，结束后 `disposeDefaultEnv` + 进程退出以释放内存

| scale | seed | T | I | status | UB | LB | gap% | time(s) | nodes | cg_fail |
|-------|------|---|---|--------|----|----|------|---------|-------|---------|
| medium | 30101 | 35 | 50 | gap_limit | 40226.30 | 39915.42 | 0.77 | 26.7 | 9 | 0 |
| medium | 30201 | 35 | 50 | gap_limit | 32784.30 | 32486.36 | 0.91 | 26.9 | 11 | 0 |
| medium | 30301 | 35 | 50 | gap_limit | 34325.50 | 34035.75 | 0.84 | 47.0 | 15 | 0 |
| medium | 30401 | 35 | 50 | gap_limit | 25559.00 | 25307.52 | 0.98 | 1153.5 | 237 | 0 |
| medium | 30501 | 35 | 50 | gap_limit | 33217.80 | 32886.07 | 1.00 | 429.1 | 174 | 0 |
| large | 40101 | 40 | 55 | time_limit | 25902.80 | 25562.93 | 1.31 | 3600.5 | 372 | 0 |
| large | 40201 | 40 | 55 | gap_limit | 38310.90 | 37932.72 | 0.99 | 1839.0 | 269 | 0 |
| large | 40301 | 40 | 55 | time_limit | 27464.50 | 26899.19 | 2.06 | 3602.6 | 575 | 0 |
| large | 40401 | 40 | 55 | time_limit | 23594.60 | 23221.56 | 1.58 | 3602.5 | 286 | 0 |
| large | 40501 | 40 | 55 | gap_limit | 34759.10 | 34415.33 | 0.99 | 604.0 | 60 | 0 |

## 小结

- 完成 10/10：`gap_limit`=7，`optimal`=0，`time_limit`=3
- gap%：min=0.773，max=2.058，mean=1.144
- 时间(s)：min=26.7，max=3602.6，mean=1493.2
- 合计 `nodes_cg_failed`=0
- medium：n=5，mean gap%=0.902，mean time=336.6s
- large：n=5，mean gap%=1.385，mean time=2649.7s
