"""Compare case1 RF-BP, Gurobi IM and release-time FCFS."""
import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(raw, data, batches, expected_cost=None):
    """Reconstruct all methods with the model's common-unloading convention."""
    finish, seen = {}, set()
    costs = dict(holding=0.0, backorder=0.0, configuration_setup=0.0, scrap=0.0)
    for b in batches:
        machine, config = b["machine"], b["config"]
        start, end = int(b["start"]), int(b["end"])
        ids = b["items"].split(",") if isinstance(b["items"], str) else b["items"]
        assert ids and 1 <= start < end <= raw["num_periods"]+1
        span = max(data.lt_it[data.cured_index[i]][start-1] for i in ids)
        assert end-start == span
        assert sum(raw["tray_lengths"][i] for i in ids) <= raw["machine_capacities"][machine]+1e-7
        for i in ids:
            assert i not in seen
            assert machine in raw["item_machines"][i] and raw["item_config"][i] == config
            assert raw["earliest_starts"][i] <= start <= raw["deadlines"][i]
            seen.add(i)
            finish[i] = end-1
        costs["configuration_setup"] += raw["production_costs"][machine][config]
    for machine in raw["machine_ids"]:
        blocks = sorted((int(b["start"]), int(b["end"])) for b in batches if b["machine"]==machine)
        assert all(a[1] <= b[0] for a,b in zip(blocks,blocks[1:]))
    orders = []
    for i in raw["cured_item_ids"]:
        due = next(int(t) for t,q in raw["demand"][i].items() if q)
        if i not in finish:
            costs["scrap"] += data.sc_i[data.cured_index[i]]
            orders.append(dict(item=i,due=due,completion=None,status="scrapped",delay=None))
        else:
            delay = max(0,finish[i]-due)
            orders.append(dict(item=i,due=due,completion=finish[i],
                               status="late" if delay else "on_time",delay=delay))
    trajectory = []
    for t in range(1,raw["num_periods"]+1):
        inv,bo = 0,0
        for row in orders:
            if row["status"]=="scrapped":
                continue
            i=row["item"]
            stock=int(row["completion"] <= t < row["due"])
            backlog=int(row["due"] <= t < row["completion"])
            inv+=stock
            bo+=backlog
            costs["holding"]+=stock*raw["holding_costs"][i]
            costs["backorder"]+=backlog*raw["backorder_costs"][i]
        trajectory.append(dict(period=t,inventory=inv,backlog=bo,net_inventory=inv-bo))
    costs["total"]=sum(costs.values())
    if expected_cost is not None and abs(costs["total"]-expected_cost)>0.02:
        raise ValueError(f"Schedule cost {costs['total']} differs from objective {expected_cost}")
    delivery=dict(on_time=sum(r["status"]=="on_time" for r in orders),
                  late=sum(r["status"]=="late" for r in orders),
                  scrapped=sum(r["status"]=="scrapped" for r in orders),
                  total_delay=sum(r["delay"] or 0 for r in orders))
    return costs,trajectory,orders,delivery


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance",type=Path,default=ROOT/"bin_packing_lot_sizing_EJOR_original-main/instances/case1.json")
    parser.add_argument("--solver-dir",type=Path,default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--bp-result",type=Path,default=ROOT/"case1_rf_corrected/rf_result.json")
    parser.add_argument("--out",type=Path,default=ROOT/"case1_comparison")
    parser.add_argument("--time-limit",type=float,default=3600)
    parser.add_argument("--mip-gap",type=float,default=0.01)
    parser.add_argument("--reuse-gurobi",action="store_true",
                        help="Reuse a matching cached Gurobi result.")
    args=parser.parse_args()
    if args.time_limit<=0 or not 0<=args.mip_gap<1:
        parser.error("Invalid time limit or gap")
    sys.path.insert(0,str(args.solver_dir.resolve()))
    from common.data_models import InstanceData
    from preprocessor.preprocess import preprocess
    from model_and_algo.im_model import solve_im
    raw=read(args.instance)
    if any(raw["wip_quantities"][i]!=1 or sum(raw["demand"][i].values())!=1
           for i in raw["cured_item_ids"]):
        raise ValueError("Only unit-WIP, single-demand instances are supported")
    data=preprocess(InstanceData.from_dict(raw))
    args.out.mkdir(parents=True,exist_ok=True)
    fingerprint=hashlib.sha256(json.dumps(raw,sort_keys=True).encode()).hexdigest()
    cache=args.out/"gurobi_result.json"
    if args.reuse_gurobi:
        saved=read(cache)
        if saved["instance_sha256"]!=fingerprint:
            raise ValueError("Cached Gurobi instance does not match")
        im=saved["solution"]
    else:
        print("Solving Gurobi integrated model...",flush=True)
        result=solve_im(data,time_limit=args.time_limit,mip_gap=args.mip_gap,verbose=True)
        im=result.to_dict()
        cache.write_text(json.dumps(dict(instance_sha256=fingerprint,solution=im,
                                        time_limit=args.time_limit,target_gap=args.mip_gap),indent=2),encoding="utf-8")
    if im["objective"] is None:
        raise ValueError("Gurobi found no feasible schedule; saved result for inspection")
    grouped={}
    for m,items in im["packing"].items():
        for i,periods in items.items():
            for t,value in periods.items():
                if value>.5:
                    grouped.setdefault((m,raw["item_config"][i],int(t)),[]).append(i)
    im_batches=[]
    for (m,u,t),ids in sorted(grouped.items()):
        span=max(data.lt_it[data.cured_index[i]][t-1] for i in ids)
        im_batches.append(dict(machine=m,config=u,start=t,end=t+span,items=ids))
    subprocess.run([sys.executable,str(ROOT/"run_case1_fcfs.py"),
                    "--instance",str(args.instance.resolve()),"--solver-dir",str(args.solver_dir.resolve()),
                    "--out",str((args.out/"fcfs").resolve())],check=True)
    with (args.out/"fcfs/batches.csv").open(encoding="utf-8-sig") as f:
        fcfs=list(csv.DictReader(f))
    bp=read(args.bp_result)
    if bp.get("branch_strategy")!="ryan_foster" or bp.get("ub") is None:
        raise ValueError("A feasible Ryan-Foster BP result is required")
    print("Validating schedules and reconciling costs...",flush=True)
    summaries=[]
    trajectories={}
    methods=[("RF-BP",bp["batches"],bp["ub"],bp.get("lb"),bp["runtime_sec"],bp["status"]),
             ("Gurobi",im_batches,im["objective"],im.get("lower_bound"),im["runtime_sec"],im["status"]),
             ("FCFS",fcfs,None,None,read(args.out/"fcfs/summary.json")["schedule_runtime_sec"],"heuristic")]
    for method,batches,ub,lb,runtime,status in methods:
        costs,traj,orders,delivery=evaluate(raw,data,batches,ub)
        trajectories[method]=traj
        save_csv(args.out/(method+"_inventory.csv"),traj)
        save_csv(args.out/(method+"_orders.csv"),orders)
        summaries.append(dict(method=method,status=status,**costs,**delivery,
                              lb=lb,runtime_sec=runtime))
    save_csv(args.out/"comparison.csv",summaries)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    plt.rcParams.update({"font.size":9,"pdf.fonttype":42})
    fig,axes=plt.subplots(3,1,sharex=True,figsize=(6.5,5),layout="constrained")
    for ax,key,label in zip(axes,("inventory","backlog","net_inventory"),
                            ("Finished inventory (items)","Model backlog (items)","Net inventory (items)")):
        for (method,traj),color,style in zip(trajectories.items(),
                                           ("#176188","#ce651f","#547569"),("-","--","-.")):
            ax.step([0]+[r["period"] for r in traj],[0]+[r[key] for r in traj],
                    where="post",label=method,color=color,ls=style,lw=1.4)
        ax.set_ylabel(label)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y",alpha=.15)
    axes[0].legend(frameon=False,ncol=3,loc="upper left")
    axes[2].axhline(0,color=".5",lw=.6)
    axes[2].set(xlabel="End of planning period (h)",xlim=(0,raw["num_periods"]))
    for ext in ("pdf","png"):
        fig.savefig(args.out/f"inventory_comparison.{ext}",dpi=300,bbox_inches="tight")
    lines=[r"% Requires \usepackage{booktabs}",r"\begin{table}[htbp]",r"\centering",
           r"\caption{Cost comparison for the illustrative instance}",r"\begin{tabular}{lrrr}",
           r"\toprule",r"Cost component & RF--BP & Gurobi & FCFS \\",r"\midrule"]
    for key,label in (("holding","Inventory holding"),("backorder","Backorder"),
                      ("configuration_setup","Configuration setup"),("scrap","Scrap"),("total","Total")):
        lines.append(label+" & "+" & ".join(f"{r[key]:,.2f}" for r in summaries)+r" \\")
    lines += [r"\bottomrule",r"\end{tabular}",r"\end{table}"]
    (args.out/"cost_comparison.tex").write_text("\n".join(lines),encoding="utf-8")
    print(f"Saved comparison to {args.out.resolve()}",flush=True)


if __name__=="__main__":
    main()
