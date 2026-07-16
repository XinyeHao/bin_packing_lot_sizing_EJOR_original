"""Analyze instance business reasonableness:
1. Item volumes vs machine capacities (space utilization)
2. Lead times vs horizon (time utilization)
3. End-of-horizon items
4. Feasible windows tightness
5. EBRP CG columns: batch-level volume & time utilization
"""
import sys, time, gc, statistics
from pathlib import Path
from collections import defaultdict
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from configuration import SetConfig
from instance_generator.generator import generate_instance
from preprocessor.preprocess import preprocess
from model_and_algo.column_generation import ColumnGenerationSolver

scfg = SetConfig("small_cg", 120, 5, 0, 60, 5)
inst = generate_instance(scfg, 1, seed=777)
data = preprocess(inst)
T = len(data.periods); I = len(data.cured_items)
M = len(data.machines); U = len(data.configs)

# ═══════════════════════════════════════════════════════════
# 1. 静态指标：物料量 vs 机台容量
# ═══════════════════════════════════════════════════════════
vols = [data.v_i[i] for i in range(I)]
print(f"=== 物料体积 (v_i) ===")
print(f"  min={min(vols):.1f}  max={max(vols):.1f}  mean={statistics.mean(vols):.1f}  median={statistics.median(vols):.1f}")

print(f"\n=== 机台容量 (q_m) ===")
for m in range(M):
    print(f"  m{m}: q_m={data.q_m[m]:.1f}")

# 理论装箱：每台机台每个时段最多放多少件
for m in range(M):
    q = data.q_m[m]
    count_by_capacity = sum(1 for v in vols if v <= q)
    avg_items_fit = q / statistics.mean(vols) if vols else 0
    print(f"  m{m}: 容量 {q:.0f} → 最多装 {avg_items_fit:.1f} 件(按平均体积)")
    # Items per config on this machine
    for u in range(U):
        items_u = set(data.items_by_config[u])
        items_m = set(data.items_by_machine[m])
        items_u_on_m = list(items_u & items_m)
        if items_u_on_m:
            vol_u = [data.v_i[i] for i in items_u_on_m]
            max_per_batch = max(int(q / min(vol_u)), 1) if vol_u else 0
            print(f"    配置{u}: {len(items_u_on_m)}件, 体积范围[{min(vol_u):.0f},{max(vol_u):.0f}], 单批最多{max_per_batch}件")

# ═══════════════════════════════════════════════════════════
# 2. 时间窗口
# ═══════════════════════════════════════════════════════════
lt_values = []
for i in range(I):
    for t in range(T):
        lt = data.lt_it[i][t]
        if lt > 0:
            lt_values.append(lt)

print(f"\n=== 加工时长 (lt_it) ===")
print(f"  min={min(lt_values)}  max={max(lt_values)}  mean={statistics.mean(lt_values):.1f}  median={statistics.median(lt_values):.1f}")

# Lead time histogram
lt_bins = defaultdict(int)
for lt in lt_values:
    lt_bins[lt] += 1
print(f"  各lt值频率: {dict(sorted(lt_bins.items())[:12])}")

# ═══════════════════════════════════════════════════════════
# 3. 可行开工窗口
# ═══════════════════════════════════════════════════════════
from model_and_algo.pricing_ebrp import _item_start_feasible

earliest = []; latest = []; window_widths = []
always_feasible_count = 0  # items feasible at ALL t
end_risk_count = 0
end_risk_details = []

for i in range(I):
    feasible_t = []
    for t in range(T):
        if _item_start_feasible(data, i, t, T):
            feasible_t.append(t)
            # Also check complement: τ_it = t + lt_it - 1 <= T
    if feasible_t:
        earliest.append(feasible_t[0])
        latest.append(feasible_t[-1])
        window_widths.append(feasible_t[-1] - feasible_t[0] + 1)
        if len(feasible_t) >= T * 0.9:
            always_feasible_count += 1
        # End-of-horizon risk: if latest feasible start + lead time might go past T
        last_t = feasible_t[-1]
        lt_last = data.lt_it[i][last_t] if last_t < T else 0
        if last_t + lt_last > T - 3:  # finishes within last 3 periods or past horizon
            end_risk_count += 1
            end_risk_details.append((i, last_t, lt_last, last_t + lt_last))

print(f"\n=== 可行开工窗口 ===")
print(f"  earliest start: min={min(earliest):.0f}  max={max(earliest):.0f}  mean={statistics.mean(earliest):.1f}")
print(f"  latest start:   min={min(latest):.0f}  max={max(latest):.0f}  mean={statistics.mean(latest):.1f}")
print(f"  window width:    min={min(window_widths):.0f}  max={max(window_widths):.0f}  mean={statistics.mean(window_widths):.1f}")

start_bin = defaultdict(int)
for e in earliest:
    start_bin[e // 10] += 1
print(f"  最早开工时段分布(每10期): {dict(sorted(start_bin.items()))}")

# End-of-horizon risk
print(f"\n=== 期末风险（最后3期内完工或超期） ===")
print(f"  风险件数: {end_risk_count} / {I}")
for i, lt, lt_val, finish_t in end_risk_details[:10]:
    print(f"  item {i:>3}: latest_start={lt:>3}, lt={lt_val:>2}, finish_t={finish_t:>3} (超出 {finish_t-T} 期)" if finish_t >= T else f"  item {i:>3}: latest_start={lt:>3}, lt={lt_val:>2}, finish_t={finish_t:>3}")

# ═══════════════════════════════════════════════════════════
# 4. 理论时间利用率
# ═══════════════════════════════════════════════════════════
# Total volume-time demand: sum over items of v_i * lt_i (approximate)
total_vol_time_demand = 0
for i in range(I):
    avg_lt = statistics.mean([data.lt_it[i][t] for t in range(T) if data.lt_it[i][t] > 0])
    total_vol_time_demand += data.v_i[i] * avg_lt

total_capacity_time = sum(data.q_m[m] for m in range(M)) * T
print(f"\n=== 理论时间利用率 ===")
print(f"  总需求(体积×时间): {total_vol_time_demand:.0f}")
print(f"  总产能(容量×时间): {total_capacity_time:.0f}")
print(f"  理论利用率: {total_vol_time_demand/total_capacity_time*100:.1f}%")

# ═══════════════════════════════════════════════════════════
# 5. EBRP CG 求解 + 列分析
# ═══════════════════════════════════════════════════════════
print(f"\n=== 求解 CG+EBRP（用于分析列结构）===")
cg = ColumnGenerationSolver(
    data, pricing_mode="ebrp", pricing_inprocess=True,
    time_limit=600, skip_cross_check=True, progress_interval_sec=0,
)
lrmp, rcs, cols = cg.run()
print(f"  LRMP={lrmp:.1f}  iter={len(cg.iteration_times)}  converged={cg.converged}")

# Analyze all generated columns
all_batch_vols = []
all_batch_items = []
all_batch_spans = []
time_used_by_machine = defaultdict(set)

# Build item_id → index mapping
item_id_to_idx = data.item_index

for machine in data.machines:
    m_idx = int(machine[1:])
    machine_vols = []
    for col in cols.get(machine, []):
        for (u, t), y_val in col.y.items():
            if y_val > 0.5:
                batch_vol = 0
                batch_items_count = 0
                batch_span = 0
                for (item_id, t2), x_val in col.x.items():
                    if x_val > 0.5 and t2 == t:
                        i_idx = item_id_to_idx.get(item_id)
                        if i_idx is None:
                            continue
                        batch_vol += data.v_i[i_idx]
                        batch_items_count += 1
                        batch_span = max(batch_span, data.lt_it[i_idx][t2])
                if batch_vol > 0:
                    q_m = data.q_m[m_idx]
                    all_batch_vols.append(batch_vol / q_m * 100)  # % utilization
                    all_batch_items.append(batch_items_count)
                    all_batch_spans.append(batch_span)
                    machine_vols.append(batch_vol / q_m * 100)
                    # Track time occupancy
                    for dt in range(int(batch_span)):
                        if t + dt < T:
                            time_used_by_machine[m_idx].add(t + dt)
    if machine_vols:
        print(f"  m{m_idx}: {len(machine_vols)} batches, vol util mean={statistics.mean(machine_vols):.1f}% median={statistics.median(machine_vols):.1f}%")

print(f"\n=== 各列统计 ===")
print(f"  总批次数: {len(all_batch_vols)}")
print(f"  每批体积利用率: mean={statistics.mean(all_batch_vols):.1f}%  median={statistics.median(all_batch_vols):.1f}%")
print(f"  每批物料数:     mean={statistics.mean(all_batch_items):.1f}  median={statistics.median(all_batch_items):.1f}")
print(f"  每批跨期(span): mean={statistics.mean(all_batch_spans):.1f}  median={statistics.median(all_batch_spans):.1f}")

# Time utilization (which periods are occupied by any batch across all machines)
for m_idx in range(M):
    occupied = len(time_used_by_machine.get(m_idx, set()))
    print(f"  m{m_idx}: 占用 {occupied}/{T} 时段 ({occupied/T*100:.1f}%) [LP允许多列分时，此为所有列覆盖的时段]")

print(f"\n=== 业务合理性评估 ===")
print(f"  1. 空间利用率(装箱): 平均每批{statistics.mean(all_batch_vols):.0f}%容量利用率 — {'偏低' if statistics.mean(all_batch_vols) < 50 else '合理'}（e.g. 典型制造70-85%）")
print(f"  2. 时间利用率: 理论{total_vol_time_demand/total_capacity_time*100:.1f}% — {'偏低' if total_vol_time_demand/total_capacity_time < 0.4 else '合理'}")
print(f"  3. 加工时长: mean={statistics.mean(lt_values):.1f} 相对T=120 — {'偏长' if statistics.mean(lt_values)/T > 0.3 else '合理'}")
print(f"  4. 期末在产风险: {end_risk_count}/{I}件 — {'需关注' if end_risk_count > I*0.2 else '可控'}")
print(f"  5. 物资拆分: 每件2-4台机台 → 有效机台竞争，合理")
