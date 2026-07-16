"""Generate 5 new run_big5-v2 instances then benchmark CG+Gurobi vs CG+EBRP."""
import sys, time, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "result" / "big5v2_cg_compare.json"


def gen_instances():
    from instance_generator.generator import generate_instance
    from preprocessor.instance_db import save_instance
    from preprocessor.preprocess import load_instance, preprocess
    from configuration import SetConfig

    config = SetConfig("small_cg", 22, 4, 0, 30, 4)
    seeds = [996, 997, 998, 999, 1000]
    names = [f"run_big5v2_{i:02d}" for i in range(1, 6)]

    for name, seed in zip(names, seeds):
        db_path = ROOT / "instances" / f"{name}.db"
        if db_path.exists():
            print(f"[gen] Replacing existing {name}")
            db_path.unlink()
        inst = generate_instance(config, 1, seed=seed)
        inst.name = name
        save_instance(inst, db_path=db_path)
        d = preprocess(load_instance(name))
        print(f"[gen] {name}: I={len(d.cured_items)} T={len(d.periods)} M={len(d.machines)}")
    return names


def run_one(name, pricing_mode):
    from preprocessor.preprocess import load_instance, preprocess
    from model_and_algo.column_generation import ColumnGenerationSolver

    data = preprocess(load_instance(name))
    t0 = time.perf_counter()

    if pricing_mode == "im_gurobi":
        kwargs = {
            "pricing_mode": "im_gurobi",
            "use_gurobi_init": False,
            "im_pricing_time_limit": 30.0,
        }
    else:
        kwargs = {
            "pricing_mode": "ebrp",
            "use_gurobi_init": False,
            "ebrp_force": True,
            "ebrp_time_limit": 30.0,
        }

    cg = ColumnGenerationSolver(
        data,
        max_iterations=5000,
        verbose=False,
        progress_interval_sec=0.0,
        time_limit=600.0,
        pricing_workers=0,
        pricing_inprocess=True,
        **kwargs,
    )
    lrmp, rcs, cols = cg.run()
    elapsed = time.perf_counter() - t0
    iterations = len(rcs) if rcs else 0
    return {
        "name": name,
        "mode": pricing_mode,
        "lrmp": round(lrmp, 2),
        "iterations": iterations,
        "elapsed_s": round(elapsed, 1),
        "ni": len(data.cured_items),
        "nt": len(data.periods),
        "nm": len(data.machines),
    }


def main():
    print("=== Generating 5 new big5v2 instances ===\n")
    names = gen_instances()

    print("\n=== Benchmarking CG+Gurobi vs CG+EBRP ===\n")
    results = []

    for name in names:
        print(f"--- {name} ---", flush=True)
        r_g = run_one(name, "im_gurobi")
        print(f"  Gurobi: LRMP={r_g['lrmp']:.2f}  iter={r_g['iterations']}  {r_g['elapsed_s']:.1f}s", flush=True)
        results.append(r_g)

        r_e = run_one(name, "ebrp")
        print(f"  EBRP:   LRMP={r_e['lrmp']:.2f}  iter={r_e['iterations']}  {r_e['elapsed_s']:.1f}s", flush=True)
        results.append(r_e)

        delta = r_g["lrmp"] - r_e["lrmp"]
        speedup = r_g["elapsed_s"] / r_e["elapsed_s"] if r_e["elapsed_s"] > 0 else 0
        print(f"  Delta={delta:+.2f}  speedup={speedup:.1f}x\n", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"Results saved to {OUT}")

    print(f"\n{'='*70}")
    print(f"{'Instance':<18} {'G_LRMP':>10} {'E_LRMP':>10} {'D':>8} {'G_s':>7} {'E_s':>7} {'Spd':>6}")
    print(f"{'-'*70}")
    for name in names:
        r_g = next(r for r in results if r["name"] == name and r["mode"] == "im_gurobi")
        r_e = next(r for r in results if r["name"] == name and r["mode"] == "ebrp")
        d = r_g["lrmp"] - r_e["lrmp"]
        sp = r_g["elapsed_s"] / r_e["elapsed_s"] if r_e["elapsed_s"] > 0 else 0
        print(f"{name:<18} {r_g['lrmp']:>10.2f} {r_e['lrmp']:>10.2f} {d:>+8.2f} {r_g['elapsed_s']:>7.1f} {r_e['elapsed_s']:>7.1f} {sp:>6.1f}x")

    g_total = sum(r["elapsed_s"] for r in results if r["mode"] == "im_gurobi")
    e_total = sum(r["elapsed_s"] for r in results if r["mode"] == "ebrp")
    if e_total > 0:
        avg_sp = g_total / e_total
        print(f"{'-'*70}")
        print(f"Total:   Gurobi={g_total:.1f}s  EBRP={e_total:.1f}s  Speedup={avg_sp:.1f}x")


if __name__ == "__main__":
    main()
