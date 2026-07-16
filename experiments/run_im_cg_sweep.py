"""Round 3: M=5 sweet spot refinement (8.14x at T=25,I=45,M=5,U=5)."""
import sys, time, gc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from configuration import SetConfig
from instance_generator.generator import generate_instance
from preprocessor.preprocess import preprocess

SWEEPS = [
    (25, 48, 5, 5),
    (26, 45, 5, 5),
    (28, 45, 5, 5),
    (25, 50, 5, 4),
]

IM_LIMIT = 180.0
CG_LIMIT = 120.0
EBRP_LIMIT = 60.0


def run_one(T, I, M, U, seed):
    label = f"T{T}_I{I}_M{M}_U{U}"
    config = SetConfig("small_cg", T, U, 0, I, M)
    inst = generate_instance(config, instance_id=1, seed=seed)
    data = preprocess(inst)
    impm = [len(data.items_by_machine[m]) for m in range(M)]

    print(f"\n{'='*65}")
    print(f"  {label}  |I_m|={impm}  max={max(impm)}")
    print(f"{'='*65}")

    print(f"  [CG] start ...", flush=True)
    t0 = time.perf_counter()
    try:
        from model_and_algo.column_generation import ColumnGenerationSolver
        cg = ColumnGenerationSolver(
            data, pricing_mode="ebrp", ebrp_force=True,
            use_gurobi_init=False, max_iterations=5000, verbose=False,
            progress_interval_sec=0.0, time_limit=CG_LIMIT,
            ebrp_time_limit=EBRP_LIMIT, pricing_workers=0,
            pricing_inprocess=True, skip_cross_check=True,
        )
        lrmp, rcs, _ = cg.run()
        cg_t = time.perf_counter() - t0
        cg_iter = len(rcs) if rcs else 0
        cg_conv = cg.converged
        print(f"  [CG] LRMP={lrmp:.1f}  iter={cg_iter}  conv={cg_conv}  {cg_t:.1f}s", flush=True)
    except Exception as e:
        print(f"  [CG] ERROR: {e}", flush=True)
        lrmp, cg_iter, cg_t, cg_conv = None, 0, time.perf_counter() - t0, False

    gc.collect()

    print(f"  [IM] start ...", flush=True)
    t0 = time.perf_counter()
    try:
        from model_and_algo.im_model import solve_im
        im = solve_im(data, time_limit=IM_LIMIT, mip_gap=0.01, verbose=False)
        im_t = time.perf_counter() - t0
        print(f"  [IM] obj={im.objective}  lb={im.lower_bound}  "
              f"gap={im.mip_gap:.2%}  st={im.status}  {im_t:.1f}s", flush=True)
    except Exception as e:
        print(f"  [IM] ERROR: {e}", flush=True)
        im, im_t = None, time.perf_counter() - t0

    gc.collect()

    sp = im_t / cg_t if cg_t > 0 else 0
    print(f"  >>> IM={im_t:.0f}s  CG={cg_t:.0f}s  speedup={sp:.2f}x  conv={cg_conv}", flush=True)
    return sp, cg_conv


def main():
    print(f"{'Config':<22} {'|I_m|':>28} {'IM_s':>7} {'CG_s':>7} {'Spd':>7} {'IM_gap':>9} {'Conv':>5}")
    print("-"*85)
    for idx, (T, I, M, U) in enumerate(SWEEPS):
        sp, conv = run_one(T, I, M, U, seed=42 + idx)
    print("\nDone.")


if __name__ == "__main__":
    main()
