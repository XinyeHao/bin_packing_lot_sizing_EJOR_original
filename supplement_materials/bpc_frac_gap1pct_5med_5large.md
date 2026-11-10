# BPC frac_sb：5 medium + 5 large（gap≤1% 停）

- 策略：`frac_sb`（k=5），无 ML
- 停止：`(UB−tree_LB)/UB ≤ 1%` 或时限
- medium `(35, 50)` @1800s；large `(40, 55)` @3600s；M=U=5
- 每例独立子进程，结束后 `disposeDefaultEnv` + 进程退出以释放内存

| scale | seed | T | I | status | UB | LB | gap% | time(s) | nodes | cg_fail |
|-------|------|---|---|--------|----|----|------|---------|-------|---------|
| medium | 30101 | 35 | 50 | gap_limit | 40425.70 | 40382.00 | 0.11 | 19.2 | 2 | 0 |
| medium | 30201 | 35 | 50 | gap_limit | 33587.10 | 33311.73 | 0.82 | 74.4 | 12 | 0 |
| medium | 30301 | 35 | 50 | gap_limit | 34952.30 | 34645.59 | 0.88 | 80.1 | 9 | 0 |
| medium | 30401 | 35 | 50 | gap_limit | 25790.50 | 25548.04 | 0.94 | 909.3 | 71 | 0 |
| medium | 30501 | 35 | 50 | gap_limit | 33658.80 | 33327.90 | 0.98 | 239.3 | 36 | 0 |
| large | 40101 | 40 | 55 | time_limit | 26199.70 | 25886.41 | 1.20 | 3600.9 | 243 | 2 |
| large | 40201 | 40 | 55 | time_limit | 38838.60 | 38173.24 | 1.71 | 3603.3 | 237 | 6 |
| large | 40301 | 40 | 55 | time_limit | 27478.90 | 27092.32 | 1.41 | 3601.9 | 257 | 2 |

## 小结

- 完成 8/10：`gap_limit`=5，`optimal`=0，`time_limit`=3
- gap%：min=0.108，max=1.713，mean=1.006
- 时间(s)：min=19.2，max=3603.3，mean=1516.1
- 合计 `nodes_cg_failed`=10
- medium：n=5，mean gap%=0.746，mean time=264.5s
- large：n=3，mean gap%=1.439，mean time=3602.1s
