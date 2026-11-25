"""Case 1 reporting pipeline. RF solver supplies a documented JSON result.

Example: python -m experiments.run_case1_study --result result/case1/rf_result.json
The current repository does not contain a Ryan--Foster implementation. Never
substitute another branching rule for this study.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def validate_result(data: dict, result: dict) -> None:
    if result.get("branch_strategy") != "ryan_foster":
        raise ValueError("Result must come from Ryan--Foster BP")
    for key in ("status", "ub", "lb", "runtime_sec", "bound_history", "batches"):
        if key not in result:
            raise ValueError(f"RF result missing {key}")
    ub, lb = result["ub"], result["lb"]
    if ub is not None and lb is not None and lb > ub + 1e-5:
        raise ValueError("Certified lower bound exceeds upper bound")
    previous_time = -1.0
    for event in result["bound_history"]:
        elapsed = float(event["elapsed_sec"])
        if elapsed < previous_time:
            raise ValueError("bound_history must be in elapsed-time order")
        previous_time = elapsed
        if event.get("lb") is not None and not event.get("lb_certified", False):
            raise ValueError("Uncertified node bound cannot appear as global LB")
    seen = set()
    for batch in result["batches"]:
        machine, config = batch["machine"], batch["config"]
        start, end = int(batch["start"]), int(batch["end"])
        if machine not in data["machine_ids"] or config not in data["config_ids"]:
            raise ValueError("Unknown machine or configuration")
        if not (1 <= start < end <= data["num_periods"] + 1):
            raise ValueError("Batch outside planning horizon")
        load = 0.0
        for item in batch["items"]:
            if item in seen or item not in data["cured_item_ids"]:
                raise ValueError(f"Repeated or unknown item: {item}")
            seen.add(item)
            if data["item_config"][item] != config or machine not in data["item_machines"][item]:
                raise ValueError(f"Invalid configuration or machine for {item}")
            if not (data["earliest_starts"][item] <= start <= data["deadlines"][item]):
                raise ValueError(f"Start outside process window for {item}")
            load += data["tray_lengths"][item]
        if load > data["machine_capacities"][machine] + 1e-7:
            raise ValueError("Machine spatial capacity exceeded")
    for machine in data["machine_ids"]:
        blocks = sorted((int(b["start"]), int(b["end"])) for b in result["batches"] if b["machine"] == machine)
        if any(left[1] > right[0] for left, right in zip(blocks, blocks[1:])):
            raise ValueError(f"Overlapping batches on {machine}")


def analyse(data: dict, result: dict, out: Path) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T = data["num_periods"]
    batches = result["batches"]
    rows = []
    completion = {}
    by_machine = defaultdict(list)
    for batch in batches:
        m, u = batch["machine"], batch["config"]
        load = sum(data["tray_lengths"][i] for i in batch["items"])
        row = {"machine": m, "config": u, "start": batch["start"], "end": batch["end"],
               "duration": batch["end"] - batch["start"], "items": ",".join(batch["items"]),
               "load": round(load, 3), "space_utilization": load / data["machine_capacities"][m]}
        rows.append(row)
        by_machine[m].append(row)
        for item in batch["items"]:
            completion[item] = int(batch["end"]) - 1
    write_csv(out / "batches.csv", rows, ["machine", "config", "start", "end", "duration", "items", "load", "space_utilization"])

    fig, ax = plt.subplots(figsize=(11, 4))
    colors = {u: plt.get_cmap("tab10")(k) for k, u in enumerate(data["config_ids"])}
    for j, m in enumerate(data["machine_ids"]):
        for b in by_machine[m]:
            ax.barh(j, b["duration"], left=b["start"], color=colors[b["config"]], edgecolor="black", height=.65)
    ax.set(yticks=range(len(data["machine_ids"])), yticklabels=data["machine_ids"], xlabel="Period (hour)", title="Case 1: machine schedule", xlim=(1, T + 1))
    ax.invert_yaxis()
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=colors[u]) for u in colors], list(colors), ncol=5, loc="upper center", bbox_to_anchor=(.5, -.16))
    fig.tight_layout(); fig.savefig(out / "gantt.png", dpi=200); plt.close(fig)

    util = []
    for m in data["machine_ids"]:
        bb = by_machine[m]
        busy = sum(b["duration"] for b in bb)
        util.append({"machine": m, "busy_hours": busy, "available_hours": T,
                     "time_utilization": busy / T, "mean_batch_space_utilization": sum(b["space_utilization"] for b in bb) / len(bb) if bb else 0,
                     "time_weighted_space_utilization": sum(b["space_utilization"] * b["duration"] for b in bb) / busy if busy else 0,
                     "batch_count": len(bb)})
    write_csv(out / "machine_utilization.csv", util, list(util[0]))

    orders = []
    inventory = []
    deliveries = defaultdict(int)
    due = defaultdict(list)
    for item, periods in data["demand"].items():
        for t, q in periods.items():
            due[int(t)].extend([item] * int(q))
    stock = defaultdict(int)
    backlog = []
    pending = []
    for t in range(1, T + 1):
        for item, finish in completion.items():
            if finish == t:
                stock[item] += 1
        pending.extend((item, t) for item in due[t])
        next_pending = []
        for item, due_t in pending:
            if stock[item]:
                stock[item] -= 1
                deliveries[t] += 1
                orders.append({"item": item, "due": due_t, "delivered": t, "delay": max(0, t - due_t), "status": "on_time" if t <= due_t else "late"})
            else:
                next_pending.append((item, due_t))
        pending = next_pending
        inventory.append({"period": t, "production": sum(f == t for f in completion.values()), "delivery": deliveries[t], "ending_inventory": sum(stock.values()), "backlog": len(pending)})
    for item, due_t in pending:
        orders.append({"item": item, "due": due_t, "delivered": "", "delay": "", "status": "undelivered"})
    write_csv(out / "orders.csv", orders, ["item", "due", "delivered", "delay", "status"])
    write_csv(out / "inventory.csv", inventory, ["period", "production", "delivery", "ending_inventory", "backlog"])
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.step([r["period"] for r in inventory], [r["ending_inventory"] for r in inventory], where="post", label="Finished-goods inventory")
    ax.step([r["period"] for r in inventory], [r["backlog"] for r in inventory], where="post", label="Backlog")
    ax.set(xlabel="Period (hour)", ylabel="Units", title="Case 1: inventory and backlog"); ax.legend(); fig.tight_layout(); fig.savefig(out / "inventory.png", dpi=200); plt.close(fig)

    history = result["bound_history"]
    write_csv(out / "bound_history.csv", history, ["elapsed_sec", "ub", "lb", "lb_certified"])
    fig, ax = plt.subplots(figsize=(10, 4))
    for key, label in (("ub", "Upper bound"), ("lb", "Certified lower bound")):
        xs, ys = [], []
        for event in history:
            if event.get(key) is not None:
                xs.append(event["elapsed_sec"]); ys.append(event[key])
        if xs:
            ax.step(xs, ys, where="post", label=label)
    ax.set(xlabel="Elapsed time (s)", ylabel="Objective", title="Case 1: Ryan-Foster BP bounds"); ax.legend(); fig.tight_layout(); fig.savefig(out / "bounds.png", dpi=200); plt.close(fig)

    ub, lb = result["ub"], result["lb"]
    summary = {"status": result["status"], "ub": ub, "lb": lb,
               "gap_pct": 100 * (ub - lb) / abs(ub) if ub not in (None, 0) and lb is not None else None,
               "runtime_sec": result["runtime_sec"], "nodes": result.get("nodes"),
               "on_time": sum(x["status"] == "on_time" for x in orders),
               "late": sum(x["status"] == "late" for x in orders),
               "undelivered": sum(x["status"] == "undelivered" for x in orders),
               "total_delay_hours": sum(x["delay"] for x in orders if isinstance(x["delay"], int)),
               "max_delay_hours": max((x["delay"] for x in orders if isinstance(x["delay"], int)), default=0)}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse a real Ryan--Foster BP result for case1")
    parser.add_argument("--instance", type=Path, default=ROOT / "instances" / "case1.json")
    parser.add_argument("--result", type=Path, required=True, help="RF solver JSON result")
    parser.add_argument("--out", type=Path, default=ROOT / "result" / "case1")
    args = parser.parse_args()
    data, result = read_json(args.instance), read_json(args.result)
    validate_result(data, result)
    args.out.mkdir(parents=True, exist_ok=True)
    print(json.dumps(analyse(data, result, args.out), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
