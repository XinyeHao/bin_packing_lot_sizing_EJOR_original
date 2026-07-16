"""One larger instance CG Gurobi vs EBRP (between small_cg & med_cg scales)."""
import json, time, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    from instance_generator.generator import generate_instance, _window_params, _generation_profile
    from preprocessor.preprocess import load_instance, preprocess
    from preprocessor.instance_db import save_instance
    from model_and_algo.column_generation import solve_column_generation
    
    # Custom SetConfig between small_cg and med_cg
    from configuration import SetConfig
    config = SetConfig("small_cg", 22, 4, 0, 30, 4)  # T=22, I=30, M=4

    name = "run_big_single_01"
    seed = 990
    inst = generate_instance(config, 1, seed=seed)
    inst.name = name
    db_path = ROOT / "instances" / f"{name}.db"
    if db_path.exists():
        db_path.unlink()
    save_instance(inst, db_path=db_path)
    data = preprocess(load_instance(name))
    ni = len(data.cured_items)
    nt = len(data.periods)
    nm = len(data.machines)
    print(f"Instance {name}: I={ni} T={nt} M={nm}")

    rows = []

    for mode_key, mode_opts in [
        ("Gurobi", {"pricing_mode": "im_gurobi", "use_gurobi_init": False, "im_pricing_time_limit": 30.0}),
        ("EBRP", {"pricing_mode": "ebrp", "use_gurobi_init": False, "ebrp_force": True}),
    ]:
        print(f"\n{name}/{mode_key} ...", flush=True)
        t0 = time.perf_counter()
        try:
            r = solve_column_generation(
                data,
                max_iterations=5000,
                verbose=False,
                progress_interval_sec=0.0,
                time_limit=900.0,
                ebrp_time_limit=None,
                pricing_workers=0,
                pricing_inprocess=True,
                **mode_opts,
            )
        except Exception as exc:
            rows.append({"instance": name, "pricing": mode_key, "error": str(exc)[:120]})
            print(f"ERROR: {exc}", flush=True)
            continue
        t = time.perf_counter() - t0
        ex = r.extra or {}
        conv = ex.get("converged")
        iters = ex.get("num_iterations")
        cols = sum((ex.get("num_columns") or {}).values())
        lrmp = r.lower_bound
        print(f"  LRMP={lrmp:.2f} it={iters} cols={cols} conv={conv} {t:.1f}s", flush=True)
        rows.append({
            "instance": name, "pricing": mode_key,
            "T": nt, "I": ni, "M": nm,
            "LRMP": lrmp, "iters": iters, "cols": cols, "converged": conv,
            "time_sec": round(t, 1), "status": r.status,
        })

    out = ROOT / "result" / "run_big_single_compare.json"
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    g = rows[0] if rows and "LRMP" in rows[0] else None
    e = rows[1] if len(rows) > 1 and "LRMP" in rows[1] else None
    if g and e:
        d = e["LRMP"] - g["LRMP"]
        spd = g["time_sec"] / e["time_sec"] if e["time_sec"] > 0 else 0
        flag = "OK" if abs(d) < 1.0 else "!!!"
        print(f"\nGurobi LRMP={g['LRMP']:.2f} {g['time_sec']:.1f}s  EBRP LRMP={e['LRMP']:.2f} {e['time_sec']:.1f}s  Delta={d:+.2f}  Speedup={spd:.1f}x  {flag}")

    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
