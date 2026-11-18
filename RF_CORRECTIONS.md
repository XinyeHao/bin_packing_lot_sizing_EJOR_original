# Ryan–Foster pricing corrections

Based on teacher commit d7709f3.

## Changes

1. Convert global item indices to machine-local bit positions before enforcing Ryan–Foster constraints in EBRP. Missing items use zero bits.
2. Carry fixed_rf in shared pricing state and pass it through both worker pricing entry points.
3. Use the in-process shared-dual route at RF-constrained nodes, so process-pool workers cannot use stale node constraints.
4. Add experiments/run_rf_small.py with --scale, --seeds and --all.

## Run

    python -u experiments/run_rf_small.py --scale small --all

Gurobi 13.0.3 was used for diagnostic reruns. Small instances use a 900 s budget, zero target gap, strict CG certification, no cuts, and 20 s EBRP calls.

## Diagnostic results

Results are in result/rf_corrected_diagnostic. Seeds 10102 and 10107 used the corrected direct-dual diagnostic route. Other seeds used the corrected shared-dual diagnostic route. Seed 10102 matches the Gurobi optimum 28227.10. Seeds 10106 and 10112 reached the time limit while finding the known optimal objective; retain their nonzero reported gaps.

These JSON files were produced by temporary diagnostic patches before this source integration. They are not fresh runs of this integrated checkout.

## Remaining audit issues

Queue truncation above 300 nodes, batch dominance and nonnegative-batch deletion under RF constraints, and feasibility recovery after column filtering still require validation. Reported optimal statuses remain subject to this audit. Medium/large RF results and case-study outputs have not been rerun after these corrections.
