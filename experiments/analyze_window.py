"""Analyze time utilization bottleneck."""
import sys, statistics, random
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import configuration as cfg
cfg.LEAD_TIME_CHOICES = (6, 7, 8)

from configuration import SetConfig
from instance_generator.generator import generate_instance
from preprocessor.preprocess import preprocess
from model_and_algo.pricing_ebrp import _item_start_feasible

scfg = SetConfig("small_cg", 60, 5, 0, 42, 5)
inst = generate_instance(scfg, 1, seed=999)
data = preprocess(inst)
T = len(data.periods); I = len(data.cured_items); M = len(data.machines)

vols = [data.v_i[i] for i in range(I)]
q_ms = [data.q_m[m] for m in range(M)]

# Compute window stats
windows = []
max_batches_possible = []
for i in range(I):
    fea = [t for t in range(T) if _item_start_feasible(data, i, t, T)]
    if fea:
        e, d = fea[0], fea[-1]
        win = d - e + 1
        windows.append(win)
        # Each batch takes span ≈ max(lt) in window, ~ avg_lt
        lt_vals = [data.lt_it[i][t] for t in fea]
        avg_lt_i = statistics.mean(lt_vals)
        # If this is the ONLY item on the machine, how many batches could it fit?
        batches_fit = win / avg_lt_i
        max_batches_possible.append(batches_fit)

print("=== Window Analysis ===")
print(f"  Window width: min={min(windows):.0f}  max={max(windows):.0f}  mean={statistics.mean(windows):.1f}")
print(f"  Batch-slots per window (win/lt): min={min(max_batches_possible):.1f}  max={max(max_batches_possible):.1f}  mean={statistics.mean(max_batches_possible):.1f}")
print()

# Key calculation: items/machine vs batch capacity
items_per_m = [len(data.items_by_machine[m]) for m in range(M)]
avg_items_m = statistics.mean(items_per_m)
print(f"  Items per machine: {items_per_m}  avg={avg_items_m:.1f}")
print()

# If avg window is W, each batch takes S periods and processes B items:
# Sequential batches per machine = W / S
# Items processed per machine = (W/S) * B
# Items that CAN'T be processed = items_m - (W/S)*B
W = statistics.mean(windows)
# From EBRP results: span_mean=9.7, items/batch=3.1
S = 9.7
B = 3.1
seq_batches = W / S
seq_items = seq_batches * B
print(f"  With avg window={W:.1f}, span={S:.1f}, items/batch={B:.1f}:")
print(f"    Sequential batches possible: {seq_batches:.1f}")
print(f"    Sequential items processable: {seq_items:.1f} out of {avg_items_m:.1f}")
print(f"    Feasible items ratio: {seq_items/avg_items_m*100:.1f}%")
print()

# Time utilization from this perspective
# Items that CAN be processed = min(seq_items, avg_items_m)
processable = min(seq_items, avg_items_m)
time_usage = processable * S / T * 100  # % of T used by actual processing
print(f"  Time usage (processable items): {time_usage:.1f}% of T")
print(f"  Volume-time utilization (theoretical): {processable * statistics.mean(vols) * S / (statistics.mean(q_ms) * T) * 100:.1f}%")
print()

# ---- What would it take to reach 50%? ----
print("=== How to reach 50%+ time utilization ===")
print()
print("  Formula: util = I * v_mean * lt_mean / (M * q_mean * T)")
v_mean = statistics.mean(vols)
q_mean = statistics.mean(q_ms)
lt_mean = 8.0  # from earlier run

# With current I, what T would achieve 50%?
for target in [0.40, 0.50, 0.60]:
    T_needed = I * v_mean * lt_mean / (M * q_mean * target)
    print(f"  For {target*100:.0f}%: T = {T_needed:.1f} (items/m={I*3/M:.1f})")

# With current T, what I would achieve 50%?
print()
for target in [0.40, 0.50, 0.60]:
    I_needed = target * M * q_mean * T / (v_mean * lt_mean)
    items_per_m_needed = I_needed * 3 / M  # assuming each item goes to 3 machines
    print(f"  For {target*100:.0f}%: I = {I_needed:.0f} (items/m≈{items_per_m_needed:.0f})")

# The REAL bottleneck: window width
print()
print("  --- Root cause ---")
print(f"  Window params (small_cg): length_frac=(0.20,0.38), so window ≈ {0.20*T:.0f}-{0.38*T:.0f} of T={T}")
print(f"  With LT={lt_mean:.0f}, each machine can process at most {W/lt_mean:.1f} batches")
print(f"  ≈ {W/lt_mean*B:.0f} items sequentially in {W:.0f} periods")
print(f"  But each machine has {avg_items_m:.0f} items! Window is the bottleneck, not capacity.")
print()
print(f"  SOLUTION: widen the window. E.g., length_frac=(0.40,0.65) gives")
print(f"  window ≈ {0.40*T:.0f}-{0.65*T:.0f} periods → {0.65*T/9:.1f} batches → {0.65*T/9*B:.0f} items")
print(f"  Items/m should then be ~{0.65*T/9*B:.0f} for 100% feasible")
