import argparse
import json
import time

from run_rf_hard_med import (
    ROOT, BranchAndPriceSolver, cfg, gap, generate_instance,
    make_set_config, preprocess, scale_spec,
)


def main():
    scales = ("small", "medium", "large")
    parser = argparse.ArgumentParser(description="Run Ryan-Foster BP on benchmark instances.")
    parser.add_argument("--scale", choices=scales, default="small")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    spec = scale_spec(args.scale)
    if args.all and args.seeds:
        parser.error("--all and --seeds cannot be combined")
    seeds = spec.seeds if args.all else (args.seeds or spec.seeds[:1])
    if any(seed not in spec.seeds for seed in seeds):
        parser.error(f"Seeds must belong to the {spec.name} benchmark set")
    cfg.apply_bench_generation()

    for seed in seeds:
        print(f"Ryan-Foster BP | {spec.name} | seed={seed}", flush=True)
        data = preprocess(generate_instance(make_set_config(spec), seed % 1000, seed))
        start = time.perf_counter()
        solver = BranchAndPriceSolver(
            data,
            max_nodes=8000,
            time_limit=spec.time_limit_sec,
            mip_gap=0.0 if spec.name == "small" else 0.01,
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
        row = {
            "scale": spec.name,
            "seed": seed,
            "branch_strategy": "ryan_foster",
            "cuts": "none",
            "status": result.status,
            "ub": result.objective,
            "lb": result.lower_bound,
            "gap_pct": gap(result.objective, result.lower_bound),
            "runtime_sec": round(time.perf_counter() - start, 3),
            "nodes": result.nodes_explored,
            "nodes_cg_failed": result.nodes_cg_failed,
            "root_converged": result.root_cg_converged,
        }
        output = ROOT / "result" / f"rf_{spec.name}_{seed}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(row, indent=2), encoding="utf-8")
        print(f"Saved to {output}", flush=True)


if __name__ == "__main__":
    main()


