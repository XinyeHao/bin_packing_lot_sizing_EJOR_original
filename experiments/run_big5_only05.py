"""Run instance 05 only with both Gurobi and EBRP."""
import json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    from preprocessor.preprocess import load_instance, preprocess
    from model_and_algo.column_generation import solve_column_generation

    res_path = ROOT / "result" / "run_big5_compare.json"
    existing = json.loads(res_path.read_text(encoding="utf-8"))

    name = "run_big5_05"
    data = preprocess(load_instance(name))
    ni, nt, nm = len(data.cured_items), len(data.periods), len(data.machines)
    print(f"{name}: I={ni} T={nt} M={nm}", flush=True)

    for mode_key, mode_opts in [
        ("Gurobi", {"pricing_mode": "im_gurobi", "use_gurobi_init": False, "im_pricing_time_limit": 30.0}),
        ("EBRP", {"pricing_mode": "ebrp", "use_gurobi_init": False, "ebrp_force": True, "ebrp_time_limit": 300.0}),
    ]:
        print(f"  {name}/{mode_key} ...", end=" ", flush=True)
        t0 = time.perf_counter()
        try:
            r = solve_column_generation(
                data, max_iterations=5000, verbose=False,
                progress_interval_sec=0.0, time_limit=900.0,
                ebrp_time_limit=(300.0 if mode_key == "EBRP" else None),
                pricing_workers=0, pricing_inprocess=True,
                **mode_opts,
            )
        except Exception as exc:
            existing.append({"instance": name, "pricing": mode_key, "error": str(exc)[:120]})
            res_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"ERROR: {str(exc)[:60]}", flush=True)
            continue
        t = time.perf_counter() - t0
        ex = r.extra or {}
        conv = ex.get("converged")
        iters = ex.get("num_iterations")
        cols = sum((ex.get("num_columns") or {}).values())
        lrmp = r.lower_bound
        print(f"LRMP={lrmp:.2f} it={iters} cols={cols} conv={conv} {t:.1f}s", flush=True)
        existing.append({
            "instance": name, "pricing": mode_key,
            "T": nt, "I": ni, "M": nm,
            "LRMP": lrmp, "iters": iters, "cols": cols, "converged": conv,
            "time_sec": round(t, 1), "status": r.status,
        })
        res_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")

    # Summary
    names = [f"run_big5_{i:02d}" for i in range(1, 6)]
    lookup = {}
    for r in existing:
        if r.get("error"):
            continue
        lookup.setdefault(r["instance"], {})[r["pricing"]] = r

    print("\n" + "=" * 105)
    print(f"{'Instance':<18} {'I':>3} {'T':>3} {'M':>2} {'Gurobi LRMP':>14} {'EBRP LRMP':>14} {'Delta':>9} {'Gurobi_s':>8} {'EBRP_s':>7} {'It_G':>5} {'It_E':>5} {'Speedup':>8}")
    print("-" * 105)

    total_g = total_e = ok = 0
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
            print(f"{name:<18} {g['I']:3d} {g['T']:3d} {g['M']:2d} {g['LRMP']:14.2f} {e['LRMP']:14.2f} {d:+9.4f} {g['time_sec']:7.1f}s {e['time_sec']:6.1f}s {g.get('iters','?'):>5} {e.get('iters','?'):>5} {spd:7.1f}x  {flag}")
        elif g:
            print(f"{name:<18} {g['I']:3d} {g['T']:3d} {g['M']:2d} {g['LRMP']:14.2f} {'N/A':>14} {'N/A':>9} {g['time_sec']:7.1f}s {'N/A':>6} {g.get('iters','?'):>5} {'N/A':>5} {'  (no EBRP)':>8}")

    if total_e > 0:
        print("-" * 105)
        avg = total_g / total_e
        print(f"{'TOTAL':>18}                                {total_g:7.1f}s {total_e:6.1f}s                       {avg:7.1f}x")
    print(f"\nLRMP match: {ok}/{sum(1 for n in names if n in lookup and 'Gurobi' in lookup[n] and 'EBRP' in lookup[n])}")
    print(f"Saved: {res_path}")


if __name__ == "__main__":
    main()
