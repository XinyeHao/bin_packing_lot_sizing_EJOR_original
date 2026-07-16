"""Quick test: run_big5_03 EBRP with verbose exception logging."""
import sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocessor.preprocess import load_instance, preprocess
from model_and_algo.column_generation import ColumnGenerationSolver

name = "run_big5_03"
data = preprocess(load_instance(name))

cg = ColumnGenerationSolver(
    data,
    max_iterations=5000,
    verbose=True,
    progress_interval_sec=3.0,
    time_limit=600.0,
    pricing_mode="ebrp",
    use_gurobi_init=False,
    ebrp_force=True,
    ebrp_time_limit=30.0,
    pricing_workers=0,
    pricing_inprocess=True,
)

t0 = time.perf_counter()
lrmp, rcs, cols = cg.run()
elapsed = time.perf_counter() - t0
print(f"\nEBRP: LRMP={lrmp:.4f}  iter={len(rcs) if rcs else 0}  elapsed={elapsed:.1f}s")
