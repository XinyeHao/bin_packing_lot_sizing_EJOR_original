"""Deterministic FCFS batch scheduling for unit-demand autoclave instances."""
import argparse
import csv
import json
import sys
import time
from pathlib import Path


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument("--instance", type=Path, default=root / "bin_packing_lot_sizing_EJOR_original-main/instances/case1.json")
    parser.add_argument("--solver-dir", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--out", type=Path, default=root / "case1_fcfs")
    args = parser.parse_args()
    sys.path.insert(0, str(args.solver_dir))
    from common.data_models import InstanceData
    from preprocessor.preprocess import preprocess

    raw = json.loads(args.instance.read_text(encoding="utf-8-sig"))
    data = preprocess(InstanceData.from_dict(raw))
    items = raw["cured_item_ids"]
    T = raw["num_periods"]
    # This entry point implements the paper's single-unit, single-demand case.
    for i in items:
        if raw["wip_quantities"][i] != 1 or sum(raw["demand"][i].values()) != 1:
            raise ValueError("FCFS requires one initial WIP unit and one demand per item")
    order = {i: j for j, i in enumerate(items)}
    queue = sorted(items, key=lambda i: (raw["earliest_starts"][i], order[i]))
    remaining = set(items)
    available = {m: 1 for m in raw["machine_ids"]}
    completion = {}
    batches = []
    started = time.perf_counter()
    for t in range(1, T + 1):
        for m in raw["machine_ids"]:
            if available[m] > t:
                continue
            eligible = [
                i for i in queue if i in remaining
                and raw["earliest_starts"][i] <= t <= raw["deadlines"][i]
                and m in raw["item_machines"][i]
                and raw["tray_lengths"][i] <= raw["machine_capacities"][m] + 1e-8
                and t + data.lt_it[data.cured_index[i]][t - 1] - 1 <= T
            ]
            if not eligible:
                continue
            config = raw["item_config"][eligible[0]]
            selected, load, span = [], 0.0, 0
            for i in eligible:
                duration = data.lt_it[data.cured_index[i]][t - 1]
                new_span = max(span, duration)
                if (raw["item_config"][i] == config
                    and load + raw["tray_lengths"][i] <= raw["machine_capacities"][m] + 1e-8
                    and t + new_span - 1 <= T):
                    selected.append(i)
                    load += raw["tray_lengths"][i]
                    span = new_span
            for i in selected:
                remaining.remove(i)
                completion[i] = t + span - 1
            available[m] = t + span
            batches.append(dict(batch=f"B{len(batches)+1:02}", machine=m, config=config,
                                start=t, end=t+span, duration=span,
                                items=",".join(selected), item_count=len(selected),
                                load=load, space_utilization=load/raw["machine_capacities"][m]))
    schedule_runtime = time.perf_counter() - started
    inventory = []
    costs = dict(holding=0.0, backorder=0.0, configuration_setup=0.0, scrap=0.0)
    for b in batches:
        costs["configuration_setup"] += raw["production_costs"][b["machine"]][b["config"]]
    orders = []
    for i in items:
        due = next(int(t) for t, q in raw["demand"][i].items() if q)
        finish = completion.get(i)
        if finish is None:
            costs["scrap"] += data.sc_i[data.cured_index[i]]
            orders.append(dict(item=i, due=due, completion="", delivery="", delay="", status="scrapped"))
        else:
            delay = max(0, finish-due)
            orders.append(dict(item=i, due=due, completion=finish, delivery=max(due, finish),
                               delay=delay, status="late" if delay else "on_time"))
    for t in range(1, T+1):
        stock, backlog, unmet_scrap = 0, 0, 0
        for row in orders:
            i, due = row["item"], row["due"]
            if row["status"] == "scrapped":
                unmet_scrap += int(due <= t)
                continue
            inv = int(row["completion"] <= t < due)
            bo = int(due <= t < row["completion"])
            stock += inv
            backlog += bo
            costs["holding"] += inv*raw["holding_costs"][i]
            costs["backorder"] += bo*raw["backorder_costs"][i]
        inventory.append(dict(period=t, finished_inventory=stock, model_backlog=backlog,
                              net_inventory=stock-backlog, unmet_scrapped_demand=unmet_scrap))
    # Independent feasibility checks for the exported batch schedule.
    seen = set()
    for b in batches:
        for i in b["items"].split(","):
            assert i not in seen
            seen.add(i)
            assert b["machine"] in raw["item_machines"][i]
            assert raw["item_config"][i] == b["config"]
            assert raw["earliest_starts"][i] <= b["start"] <= raw["deadlines"][i]
        assert b["space_utilization"] <= 1+1e-8 and b["end"] <= T+1
    for m in available:
        blocks = sorted((b["start"], b["end"]) for b in batches if b["machine"] == m)
        assert all(a[1] <= b[0] for a, b in zip(blocks, blocks[1:]))
    costs["total"] = sum(costs.values())
    summary = dict(method="FCFS", instance=raw["name"], costs=costs,
                   schedule_runtime_sec=schedule_runtime, batch_count=len(batches),
                   on_time=sum(r["status"]=="on_time" for r in orders),
                   late=sum(r["status"]=="late" for r in orders),
                   scrapped=len(remaining),
                   total_delay_item_hours=sum(r["delay"] for r in orders if r["delay"] != ""),
                   rule="release-time FCFS; original item order breaks ties; machine ID order; immediate compatible batching; final unprocessed items scrapped")
    args.out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("batches", batches), ("orders", orders), ("inventory", inventory)):
        if rows:
            write_csv(args.out / f"{name}.csv", rows)
    (args.out/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
