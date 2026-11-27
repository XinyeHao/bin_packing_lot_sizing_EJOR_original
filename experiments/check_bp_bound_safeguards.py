"""Focused tree-bookkeeping checks with independently specified node optima."""
from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model_and_algo import branch_and_price as bp


class SyntheticTree(bp.BranchAndPriceSolver):
    def __init__(self, children=2, unresolved_last=False):
        super().__init__(SimpleNamespace(machines=["m0"]), time_limit=None,
                         max_nodes=children + 2, branch_strategy="best_bound", final_im=False)
        self.children = children
        self.unresolved_last = unresolved_last
        self.policy = SimpleNamespace(note_lb=lambda _: None,
                                      pick=lambda depth, *args: SimpleNamespace() if depth == 0 else None)

    def _start_pricing_pool(self):
        pass

    def _stop_pricing_pool(self):
        pass

    def _run_node_cg(self, node, **kwargs):
        fractional = node.node_id == 0 or (self.unresolved_last and node.node_id == self.children)
        cost = 0.0 if node.node_id == 0 else (100.0 if node.node_id == self.children else 120.0)
        if self.children > 300 and node.node_id == self.children:
            cost = 5.0
        value = 0.5 if fractional else 1.0
        return SimpleNamespace(
            last_valid_lb=cost, lrmp_obj=cost, converged=True, stop_reason=None,
            iteration_times=[], iteration_time_detail=[], _model=None,
            _artifacts={"Q": {"m0": [SimpleNamespace(X=value)]}, "R": {0: SimpleNamespace(X=0.0)}},
            data=self.data, columns={"m0": []}, sri_cuts=[],
            get_implied_y=lambda: {(0, 0, 0): value},
        )

    def _accept_integer_master(self, cg, incumbent):
        # This fixture specifies independent node optima, not physical schedules.
        return min(incumbent, cg.lrmp_obj) if incumbent is not None else cg.lrmp_obj

    def _decision_label(self, _decision):
        return "synthetic"

    def _create_children(self, node, dec, inherited, sri, next_id, dfs, lazy, heap, stats):
        for index in range(1, self.children + 1):
            child = bp.BranchNode(node_id=index, depth=1)
            self._push_heap(heap, child, 0.0, defer=False)
        return self.children + 1, self.children


def main():
    solver = SyntheticTree()
    solver._unresolved_node_bounds = {}
    child = bp.BranchNode(node_id=1, parent_lb=10.0, lb_estimate=999.0)
    heap = [(999.0, 1, child)]
    assert solver._tree_dual_bound(heap, [], []) == 10.0
    unknown = bp.BranchNode(node_id=2, lb_estimate=1000.0)
    assert solver._tree_dual_bound(heap, [], [unknown]) == float("-inf")
    solver._park_unresolved_node(child)
    assert solver._tree_dual_bound([], [], []) == 10.0
    assert solver._gap_reached(10.01, [], [], [])[0] is False
    with patch.object(bp, "_node_feasible", return_value=True), contextlib.redirect_stdout(io.StringIO()):
        result = SyntheticTree().solve()
        assert result.objective == 100.0 and result.nodes_explored == 3
        assert result.status == "optimal" and result.nodes_unresolved == 0
        unresolved = SyntheticTree(unresolved_last=True).solve()
        assert unresolved.objective == 120.0 and unresolved.lower_bound == 100.0
        assert unresolved.status == "unproven_nodes" and unresolved.nodes_unresolved == 1
        large = SyntheticTree(children=400).solve()
        assert large.objective == 5.0 and large.nodes_explored == 401
        assert large.status == "optimal"
    print(json.dumps({"checks": 6, "passed": True,
                      "covered": ["priority not bound", "unknown node blocks proof", "unresolved gap guard",
                                  "integer child retains siblings", "missing branch retains bound", "400 live nodes retained"]}))


if __name__ == "__main__":
    main()
