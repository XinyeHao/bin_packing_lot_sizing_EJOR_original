"""5 new instances CG Gurobi vs EBRP."""
import json, time, sys, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    from instance_generator.generator import generate_instance
    from instance_generator.generator import SET_CONFIGS
    from preprocessor.preprocess import load_instance, preprocess
    from preprocessor.instance_db import save_instance
    from model_and_algo.column_generation import solve_column_generation

    # Generate 5 instances
    names = []
    config = SET_CONFIGS["A"]
    for i, seed in enumerate([800, 801, 802, 803, 804], 1):
        name = f"run5_test_{i:02d}"
        inst = generate_instance(config, i, seed=seed)
        inst.name = name
        db_path = ROOT / "instances" / f"{name}.db"
        # Delete existing first to avoid permission error
        if db_path.exists():
            db_path.unlink()
        save_instance(inst, db_path=db_path)
        names.append(name)
        print(f"Generated: {name}")

    print()

    rows = []
    out = ROOT / "result" / "run5_compare.json"

    for name in names:
        data = preprocess(load_instance(name))
        ni = len(data.cured_items)
        nt = len(data.periods)
        nm = len(data.machines)
        print(f"{name}: I={ni} T={nt} M={nm}", flush=True)

        for mode_key, mode_opts in [
            ("Gurobi", {"pricing_mode": "im_gurobi", "use_gurobi_init": False, "im_pricing_time_limit": 10.0}),
            ("EBRP", {"pricing_mode": "ebrp", "use_gurobi_init": False, "ebrp_force": True}),
        ]:
            print(f"  {name}/{mode_key} ...", end=" ", flush=True)
            t0 = time.perf_counter()
            try:
                r = solve_column_generation(
                    data,
                    max_iterations=5000,
                    verbose=False,
                    progress_interval_sec=0.0,
                    time_limit=300.0,
                    ebrp_time_limit=None,
                    pricing_workers=0,
                    pricing_inprocess=True,
                    **mode_opts,
                )
            except Exception as e:
                rows.append({"instance": name, "pricing": mode_key, "error": str(e)[:120]})
                out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
                print(f"ERROR: {str(e)[:60]}", flush=True)
                continue
            t = time.perf_counter() - t0
            ex = r.extra or {}
            conv = ex.get("converged")
            iters = ex.get("num_iterations")
            cols = sum((ex.get("num_columns") or {}).values())
            lrmp = r.lower_bound
            print(f"LRMP={lrmp:.2f} it={iters} conv={conv} {t:.1f}s", flush=True)
            rows.append({
                "instance": name, "pricing": mode_key,
                "T": nt, "I": ni, "M": nm,
                "LRMP": lrmp, "iters": iters, "cols": cols, "converged": conv,
                "time_sec": round(t, 1), "status": r.status,
            })
            out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    # Summary
    print("\n" + "=" * 90)
    print(f"{'Instance':<22} {'Gurobi':>12} {'EBRP':>12} {'Delta':>8} {'G_s':>6} {'E_s':>6} {'Spd':>7}")
    print("-" * 85)

    lookup = {}
    for r in rows:
        if r.get("error"):
            continue
        lookup.setdefault(r["instance"], {})[r["pricing"]] = r

    total_g = 0.0
    total_e = 0.0
    ok = 0
    for name in names:
        g = lookup.get(name, {}).get("Gurobi")
        e = lookup.get(name, {}).get("EBRP")
        if g and e:
            d = e["LRMP"] - g["LRMP"]
            spd = g["time_sec"] / e["time_sec"] if e["time_sec"] > 0 else 0
            total_g += g["time_sec"]
            total_e += e["time_sec"]
            flag = "OK" if abs(d) < 1.0 else "!!!"
            if abs(d) < 1.0:
                ok += 1
            print(f"{name:<22} {g['LRMP']:12.2f} {e['LRMP']:12.2f} {d:+8.2f} {g['time_sec']:5.1f}s {e['time_sec']:5.1f}s {spd:6.1f}x  {flag}")

    if total_e > 0:
        print("-" * 85)
        print(f"{'TOTAL':>22}                                {total_g:5.1f}s {total_e:5.1f}s {total_g/total_e:6.1f}x")
    print(f"\nLRMP match: {ok}/{len(names)}")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
