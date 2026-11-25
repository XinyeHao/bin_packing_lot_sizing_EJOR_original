import os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'experiments'))
import run_rf_small as r
import inspect,textwrap
from model_and_algo import pricing_ebrp as p, column_generation as c
source=textwrap.dedent(inspect.getsource(p.solve_subproblem_ebrp))
source=source.replace('pair = (1 << i, 1 << j)','pair = (1 << ctx.global_to_local[i] if i in ctx.global_to_local else 0, 1 << ctx.global_to_local[j] if j in ctx.global_to_local else 0)')
ns=dict(vars(p));exec(source,ns)
p.solve_subproblem_ebrp=ns['solve_subproblem_ebrp']
c.solve_subproblem_ebrp=p.solve_subproblem_ebrp
original_bind=c.ColumnGenerationSolver._bind_pricing_worker_state
def bind_with_rf(self):
    original_bind(self)
    c._PRICING_WORKER_STATE['fixed_rf']=dict(self.fixed_rf)
c.ColumnGenerationSolver._bind_pricing_worker_state=bind_with_rf
for name in ('_price_machine_in_worker','_execute_pricing_job'):
    source=textwrap.dedent(inspect.getsource(getattr(c,name)))
    marker='sri_terms=(state.get("sri_terms_by_machine") or {}).get(m_idx, ()),'
    assert source.count(marker)==1
    source=source.replace(marker,marker+'\n        fixed_rf=state.get("fixed_rf"),')
    ns=dict(vars(c))
    exec(source,ns)
    setattr(c,name,ns[name])
# The function globals must use the module's live worker state after rebinding.
for name in ('_price_machine_in_worker','_execute_pricing_job'):
    fn=getattr(c,name)
    def make_forward(fn):
        def forward(*args,**kwargs):
            fn.__globals__['_PRICING_WORKER_STATE']=c._PRICING_WORKER_STATE
            fn.__globals__['_solve_pricing_subproblem']=c._solve_pricing_subproblem
            return fn(*args,**kwargs)
        return forward
    setattr(c,name,make_forward(fn))



import run_case1_rf as case
if __name__ == '__main__':
    case.main()
