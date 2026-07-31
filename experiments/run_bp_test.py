"""Branch-and-price with 900s limit, no node IM, target optimal."""
import sys, time, gc, os, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_original_stderr = sys.stderr
os.environ["GRB_CS_QUIET"] = "1"
sys.stderr = open(os.devnull, "w")

import configuration as cfg
cfg.apply_bench_generation()

from configuration import SetConfig
from instance_generator.generator import generate_instance
from preprocessor.preprocess import preprocess
from model_and_algo.branch_and_price import BranchAndPriceSolver, BranchAndPriceResult

SEED = 101
scfg = SetConfig("small_cg", 35, 5, 0, 50, 5)

gc.collect(); time.sleep(1.0)
print(f"Generating instance seed={SEED}...", flush=True)
inst = generate_instance(scfg, 1, seed=SEED)
data = preprocess(inst)
T = len(data.periods); I = len(data.cured_items); M = len(data.machines)
print(f"T={T} I={I} M={M}", flush=True)

BP_TIME_LIMIT = 900
BP_MAX_NODES  = 50       # many nodes; time is the real limit
BP_Y0_DEPTH   = 2        # 2 Y=0 branches then switch to WIP/scrap

cg_kwargs = {
    "pricing_mode": "ebrp",
    "pricing_inprocess": True,
    "use_gurobi_init": False,
    "skip_cross_check": True,
}   # no per-CG time limit; B&P handles time via node_cg_limit

print(f"B&P: max_nodes={BP_MAX_NODES}, time_limit={BP_TIME_LIMIT}s, y0_depth={BP_Y0_DEPTH}")
print(f"     use_node_im=False, final_im=True, im_time_limit=120s")
print(flush=True)

t0 = time.perf_counter()
bp = BranchAndPriceSolver(
    data,
    max_nodes=BP_MAX_NODES,
    time_limit=BP_TIME_LIMIT,
    y0_depth_budget=BP_Y0_DEPTH,
    use_node_im=False,         # ← off
    root_im=True,              # ← get initial UB for pruning
    final_im=True,             # ← run IM at end
    im_time_limit=120.0,       # ← 2 min for final IM
    cg_kwargs=cg_kwargs,
)

result = None
try:
    result: BranchAndPriceResult = bp.solve()
except Exception:
    sys.stderr = _original_stderr
    traceback.print_exc()
    sys.exit(1)
runtime = time.perf_counter() - t0

print(f"\n{'='*60}")
print(f"BRANCH-AND-PRICE RESULT")
print(f"{'='*60}")
print(f"  Status:           {result.status}")
print(f"  Incumbent (UB):   {result.incumbent}")
print(f"  Root LRMP (LB):   {result.root_lrmp}")
print(f"  Global LB:        {result.best_node_lb}")
if result.incumbent and result.best_node_lb and result.incumbent > 0:
    gap = (result.incumbent - result.best_node_lb) / result.incumbent * 100
    print(f"  Gap:              {gap:.2f}% (UB={result.incumbent:.1f}, LB={result.best_node_lb:.1f})")
print(f"  Runtime:          {runtime:.1f}s ({runtime/60:.1f}min)")
print(f"  Nodes explored:   {result.nodes_explored} / {result.nodes_created} created")
print(f"  Nodes pruned:     {result.nodes_pruned}")
print(f"  Max depth:        {result.max_depth}")
print(f"  Y1 lazy created:  {result.y1_lazy_created}")
print(f"  Lazy popped:      {result.lazy_popped}")
print(f"  DFS popped:       {result.dfs_popped}")
print(f"  Best fixed Y:     {len(result.best_fixed_y)} assignments")
print(f"\nDone.", flush=True)
