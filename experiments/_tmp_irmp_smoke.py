"""Smoke: root IRMP produces a UB, LP duals remain usable, branching continues."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ["GRB_CS_QUIET"] = "1"

import configuration as cfg
from experiments.paper_experiment_config import make_set_config, scale_spec
from instance_generator.generator import generate_instance
from model_and_algo.branch_and_price import BranchAndPriceSolver
from preprocessor.preprocess import preprocess

cfg.apply_bench_generation()
spec = scale_spec("small")
data = preprocess(generate_instance(make_set_config(spec), 10101 % 1000, 10101))
solver = BranchAndPriceSolver(
    data,
    max_nodes=5,
    time_limit=120.0,
    mip_gap=0.01,
    branch_strategy="ryan_foster",
    use_node_im=False,
    root_im=False,
    final_im=False,
    strict_cg_certification=True,
    cg_kwargs={
        "pricing_mode": "ebrp",
        "ebrp_force": True,
        "use_gurobi_init": True,
        "pricing_inprocess": True,
        "skip_cross_check": True,
        "lrmp_cuts": "none",
        "max_iterations": 120,
        "ebrp_time_limit": 20.0,
        "verbose": False,
    },
)
result = solver.solve()
print(
    f"SMOKE status={result.status} ub={result.incumbent} "
    f"nodes={result.nodes_explored} irmp={solver._time_irmp:.2f}s "
    f"root_lb={result.root_valid_lb}",
    flush=True,
)
if result.incumbent is None:
    raise SystemExit("IRMP produced no incumbent")
if result.nodes_explored < 2:
    raise SystemExit("did not branch after root")
if result.root_valid_lb is None:
    raise SystemExit("root dual bound missing after IRMP")
print("SMOKE_OK")
