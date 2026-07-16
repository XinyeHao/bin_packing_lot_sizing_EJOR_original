"""Diagnose packed key collision."""
import sys, os
ROOT = r'C:\Users\haoxi\PycharmProjects\bin_packing_and_scheduling'
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ['GRB_CS_QUIET'] = '1'

import configuration as cfg
cfg.LEAD_TIME_CHOICES = (8, 9, 10)
cfg.WINDOW_GENERATION_BY_SET['small_cg'] = {
    'length_frac': (0.45, 0.65), 'rel_frac': (0.03, 0.25), 'min_length': 6
}
from configuration import SetConfig
from instance_generator.generator import generate_instance
from preprocessor.preprocess import preprocess
from model_and_algo.pricing_ebrp import EBRPContext

scfg = SetConfig('small_cg', 36, 5, 0, 50, 5)
inst = generate_instance(scfg, 1, seed=101)
data = preprocess(inst)

for m_idx in range(len(data.machines)):
    ctx = EBRPContext.build(data, m_idx)
    n_local = len(ctx.local_items)
    start_mask = 0
    # Current buggy shift computation:
    shift = (start_mask | 0x3FF).bit_length().bit_length() or 1
    shift = max(8, shift * 8)
    print(f'M{m_idx}: n_local={n_local}, shift={shift}, bits for mask={n_local}, '
          f'sufficient={shift >= n_local}')
    if shift < n_local:
        t1, m1 = 1, 0
        t2, m2 = 0, 1 << shift
        if m2 < (1 << n_local):
            p1 = (t1 << shift) | m1
            p2 = (t2 << shift) | m2
            print(f'  COLLISION: pack({t1},{m1})={p1} == pack({t2},{m2})={p2}')
