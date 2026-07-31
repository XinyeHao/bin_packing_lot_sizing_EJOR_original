# BPC+ML：5 medium + 5 large（gap≤1% 停）vs frac_sb

- 策略：`ml`（粗排 Top-5 + 短 SB），模型 `ml_branch_model.txt`
- 停止：`(UB−tree_LB)/UB ≤ 1%` 或时限
- medium `(35, 50)` @1800s；large `(40, 55)` @3600s；M=U=5
- 每例独立子进程，结束后 `disposeDefaultEnv` + 进程退出以释放内存
- 对照：`bpc_frac_gap1pct_5med_5large.json`

| scale | seed | ML status | ML gap% | ML time | frac status | frac gap% | frac time | Δtime | Δgap |
|-------|------|-----------|---------|---------|-------------|-----------|-----------|-------|------|
| medium | 30101 | gap_limit | 0.75 | 55.0 | gap_limit | 0.77 | 26.7 | +28.3 | -0.025 |
| medium | 30201 | gap_limit | 0.91 | 37.2 | gap_limit | 0.91 | 26.9 | +10.3 | +0.005 |
| medium | 30301 | gap_limit | 0.79 | 37.6 | gap_limit | 0.84 | 47.0 | -9.4 | -0.054 |
| medium | 30401 | gap_limit | 0.92 | 471.6 | gap_limit | 0.98 | 1153.5 | -681.9 | -0.061 |
| medium | 30501 | gap_limit | 1.00 | 405.2 | gap_limit | 1.00 | 429.1 | -23.9 | +0.001 |
| large | 40101 | gap_limit | 0.96 | 1115.8 | time_limit | 1.31 | 3600.5 | -2484.6 | -0.350 |
| large | 40201 | gap_limit | 1.00 | 2593.6 | gap_limit | 0.99 | 1839.0 | +754.6 | +0.011 |
| large | 40301 | time_limit | 1.12 | 3602.0 | time_limit | 2.06 | 3602.6 | -0.6 | -0.935 |
| large | 40401 | time_limit | 1.31 | 3599.6 | time_limit | 1.58 | 3602.5 | -2.8 | -0.274 |
| large | 40501 | gap_limit | 0.97 | 569.2 | gap_limit | 0.99 | 604.0 | -34.7 | -0.016 |

## 小结

- ML 完成 10/10：`gap_limit`=8，`optimal`=0，`time_limit`=2
- ML gap%：min=0.748，max=1.306，mean=0.974
- ML 时间(s)：min=37.2，max=3602.0，mean=1248.7
- 已完成对照 10 例：mean time ML=1248.7s vs frac=1493.2s (Δ=-244.5s)
- 已完成对照 mean gap% ML=0.974 vs frac=1.144 (Δ=-0.170)
