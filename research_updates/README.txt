Research updates, 2026-11-25

Latest large-instance table: large_gurobi_bp_latest.tex/csv.
Large BP uses 100 s pricing retries for 40401, 41301, 41601.
All three retry runs reached the time limit and each records one CG-failed node.
Their bounds/gaps are reported diagnostics, not independently certified.

Case comparison: case1_comparison_tables_figures.tex, with figures under case1_comparison/.
RF-BP and Gurobi have equal cost 32300.70; FCFS cost is 52128.70.
Reported BP certification remains subject to the audit in RF_CORRECTIONS.md.

Run from this folder using the project's configured Python environment:
python run_case1_comparison.py --reuse-gurobi
python run_case1_fcfs.py
python run_bp_corrected_large_ebrp100.py

These files preserve diagnostic result provenance. Original historical result folders remain available.
