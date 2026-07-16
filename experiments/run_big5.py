"""5 larger instances (I=30 T=22 M=4) CG Gurobi vs EBRP benchmark."""
import json, time, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    from instance_generator.generator import generate_instance
    from preprocessor.preprocess import load_instance, preprocess
    from preprocessor.instance_db import save_instance
    from model_and_algo.column_generation import solve_column_generation
    from configuration import SetConfig

    config = SetConfig("small_cg", 22, 4, 0, 30, 4)
    seeds = [991, 992, 993, 994, 995]
    names = [f"run_big5_{i:02d}" for i in range(1, 6)]

    for name, seed in zip(names, seeds):
        inst = generate_instance(config, 1, seed=seed)
        inst.name = name
        db_path = ROOT / "instances" / f"{name}.db"
        if db_path.exists():
            db_path.unlink()
        save_instance(inst, db_path=db_path)
        d = preprocess(load_instance(name))
        print(f"Generated: {name}  I={len(d.cured_items)} T={len(d.periods)} M={len(d.machines)}")

    out = ROOT / "result" / "run_big5_compare.json"
    rows = []

    for name in names:
        data = preprocess(load_instance(name))
        ni = len(data.cured_items)
        nt = len(data.periods)
        nm = len(data.machines)
        print(f"\n{name}: I={ni} T={nt} M={nm}", flush=True)

        for mode_key, mode_opts in [
            ("Gurobi", {"pricing_mode": "im_gurobi", "use_gurobi_init": False, "im_pricing_time_limit": 30.0}),
            ("EBRP", {"pricing_mode": "ebrp", "use_gurobi_init": False, "ebrp_force": True}),
        ]:
            print(f"  {mode_key} ...", end=" ", flush=True)
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
                out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
                print(f"ERROR: {str(exc)[:60]}", flush=True)
                continue
            t = time.perf_counter() - t0
            ex = r.extra or {}
            conv = ex.get("converged")
            iters = ex.get("num_iterations")
            cols = sum((ex.get("num_columns") or {}).values())
            lrmp = r.lower_bound
            print(f"LRMP={lrmp:.2f} it={iters} cols={cols} conv={conv} {t:.1f}s", flush=True)
            rows.append({
                "instance": name, "pricing": mode_key,
                "T": nt, "I": ni, "M": nm,
                "LRMP": lrmp, "iters": iters, "cols": cols, "converged": conv,
                "time_sec": round(t, 1), "status": r.status,
            })
            out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    # Summary
    print("\n" + "=" * 95)
    print(f"{'Instance':<18} {'I':>3} {'T':>3} {'M':>2} {'Gurobi':>12} {'EBRP':>12} {'Delta':>8} {'Gurobi_s':>8} {'EBRP_s':>7} {'It_G':>5} {'It_E':>5} {'Speedup':>7}")
    print("-" * 95)

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
            print(f"{name:<18} {g['I']:3d} {g['T']:3d} {g['M']:2d} {g['LRMP']:12.2f} {e['LRMP']:12.2f} {d:+8.2f} {g['time_sec']:7.1f}s {e['time_sec']:6.1f}s {g.get('iters','?'):>5} {e.get('iters','?'):>5} {spd:6.1f}x  {flag}")

    if total_e > 0:
        print("-" * 95)
        print(f"{'TOTAL':>18}                                {total_g:7.1f}s {total_e:6.1f}s                       {total_g/total_e:6.1f}x")
    print(f"\nLRMP match: {ok}/{len(names)}")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
