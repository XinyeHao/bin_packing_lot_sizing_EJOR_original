# IM opt ≤900s scale (regular T/I/M)

## Goal

- Hard baseline: `SetConfig('small_cg', 35, 5, 0, 50, 5)`, LT=(8,9,10)
- Target regular scale: **T=25, U=5, I=35, M=5**
- IM proves optimal within 900s; time_util ≥ 55%, space_util ≥ 70%
- IM only (no BPC)

## Chosen scale

- Tag: `lt567`
- Config: `SetConfig('small_cg', 25, 5, 0, 35, 5)` → **T=25, U=5, I=35, M=5**
- LT=[5, 6, 7], window={'length_frac': (0.38, 0.58), 'rel_frac': (0.04, 0.25), 'min_length': 5}

## Accepted instances

| name | seed | T | U | I | M | LT | tray_load | time_util | space_util | obj | IM time (s) |
|------|------|---|---|---|---|----|-----------|-----------|------------|-----|-------------|
| set_im_opt900_01 | 7101 | 25 | 5 | 35 | 5 | [5, 6, 7] | 9.8% | 64.0% | 84.0% | 15032.50 | 68.1 |
| set_im_opt900_02 | 7102 | 25 | 5 | 35 | 5 | [5, 6, 7] | 7.7% | 60.8% | 85.9% | 14436.40 | 65.9 |
| set_im_opt900_03 | 7103 | 25 | 5 | 35 | 5 | [5, 6, 7] | 7.9% | 69.6% | 75.7% | 17165.30 | 63.4 |
| set_im_opt900_04 | 7104 | 25 | 5 | 35 | 5 | [5, 6, 7] | 6.0% | 64.0% | 78.7% | 13460.00 | 30.2 |
| set_im_opt900_05 | 7105 | 25 | 5 | 35 | 5 | [5, 6, 7] | 9.2% | 68.0% | 82.6% | 18695.80 | 252.5 |

## BPC+ML (900s, same instances)

| name | status | UB | LB | gap % | nodes | time (s) |
|------|--------|----|----|-------|-------|----------|
| set_im_opt900_01 | optimal | 15032.50 | 15032.50 | 0.00 | 27 | 461.4 |
| set_im_opt900_02 | optimal | 14436.40 | 14436.40 | 0.00 | 3 | 40.0 |
| set_im_opt900_03 | optimal | 17165.30 | 17165.30 | 0.00 | 49 | 795.4 |
| set_im_opt900_04 | optimal | 13460.00 | 13460.00 | 0.00 | 27 | 336.8 |
| set_im_opt900_05 | time_limit | 18695.80 | 18320.21 | 2.01 | 35 | 906.8 |

完整三档汇总见 [`test_instance.md`](test_instance.md)。

### Generation recipe

- `set_im_opt900_01`: `SetConfig('small_cg', 25, 5, 0, 35, 5)`, seed=7101, LT=[5, 6, 7], window={'length_frac': (0.38, 0.58), 'rel_frac': (0.04, 0.25), 'min_length': 5}
- `set_im_opt900_02`: `SetConfig('small_cg', 25, 5, 0, 35, 5)`, seed=7102, LT=[5, 6, 7], window={'length_frac': (0.38, 0.58), 'rel_frac': (0.04, 0.25), 'min_length': 5}
- `set_im_opt900_03`: `SetConfig('small_cg', 25, 5, 0, 35, 5)`, seed=7103, LT=[5, 6, 7], window={'length_frac': (0.38, 0.58), 'rel_frac': (0.04, 0.25), 'min_length': 5}
- `set_im_opt900_04`: `SetConfig('small_cg', 25, 5, 0, 35, 5)`, seed=7104, LT=[5, 6, 7], window={'length_frac': (0.38, 0.58), 'rel_frac': (0.04, 0.25), 'min_length': 5}
- `set_im_opt900_05`: `SetConfig('small_cg', 25, 5, 0, 35, 5)`, seed=7105, LT=[5, 6, 7], window={'length_frac': (0.38, 0.58), 'rel_frac': (0.04, 0.25), 'min_length': 5}

## Probe log

| tag | seed | proved | IM time | gap | time_util | space_util | note |
|-----|------|---------|---------|-----|-----------|------------|------|
| lt567 | 6101 | True | 46.8 | 0.0% | 52.8% | 80.7% | util_low |
| lt567 | 6202 | True | 75.8 | 0.0% | 59.2% | 88.7% | ok |
| lt567 | 6303 | True | 73.7 | 0.0% | 64.0% | 85.9% | ok |

_Updated 2026-07-23 11:52:41_
