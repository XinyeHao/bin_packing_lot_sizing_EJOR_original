"""项目统一配置：路径、算例生成、求解器、实验与检验容差。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
INSTANCES_DIR = PROJECT_ROOT / "instances"
RESULT_DIR = PROJECT_ROOT / "result"

# ---------------------------------------------------------------------------
# 算例规模（论文 Table 1）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SetConfig:
    set_type: str
    num_periods: int
    num_configs: int
    num_end_items: int
    num_cured_items: int
    num_machines: int


SET_A = SetConfig("A", 15, 5, 5, 25, 5)
SET_B = SetConfig("B", 25, 10, 7, 35, 7)
DEMO_CONFIG = SetConfig("demo", 6, 3, 2, 4, 2)

SET_CONFIGS: dict[str, SetConfig] = {"A": SET_A, "B": SET_B}

LEAD_TIME_CHOICES = (1, 2, 3)
BOM_CHOICES = (0, 2, 4)
DEMAND_CHOICES = (1, 2)
BOM_FALLBACK_CHOICES = (2, 4)

INSTANCE_COUNT_DEFAULT = 20
INSTANCE_BASE_SEED = 42
DEMO_INSTANCE_SEED = 0
DEMO_INSTANCE_NAME = "demo_small"

# 算例随机生成参数（论文 Section 5.1）
TRAY_LENGTH_LONG_PROB = 0.8
TRAY_LENGTH_LONG_RANGE = (1.0, 2.0)
TRAY_LENGTH_SHORT_RANGE = (0.5, 1.0)
MACHINE_CAPACITY_RANGE = (5.0, 10.0)
C_STAR_RANGE = (100.0, 150.0)
HOLDING_COST_DIVISOR = 10.0
END_ITEM_HOLDING_MULTIPLIER = 1.2
BACKORDER_COST_MULTIPLIER = 10.0
SCRAP_COST_MULTIPLIER = 2.0  # sc_i = SCRAP_COST_MULTIPLIER * bc_i，须 > bc_i
DEMAND_PERIODS_PER_ITEM = 2
# 最晚进罐 D_i：在 [ceil(DEADLINE_MIN_FRACTION * T), T] 上均匀随机（T=10 时即从 3 起）
DEADLINE_MIN_FRACTION = 0.3

# ---------------------------------------------------------------------------
# 求解器默认参数
# ---------------------------------------------------------------------------

DEFAULT_TIME_LIMIT = 900.0
DEFAULT_MIP_GAP = 0.05
DEFAULT_LIM_TIME_LIMIT = 300.0

DEFAULT_CG_MAX_ITERATIONS = 1000
# 列生成 LRMP 最优性：每台机器 reduced cost >= -DEFAULT_CG_RC_TOLERANCE 时停止
DEFAULT_CG_RC_TOLERANCE = 1e-6
DEFAULT_CG_IMPROVE_TOLERANCE = 1e-6
DEFAULT_CG_INIT_TIME_LIMIT = 90.0
DEFAULT_CG_INIT_TIME_CAP = 180.0
# 0 表示按机器数并行；>0 为进程池大小
DEFAULT_CG_PRICING_WORKERS = 0
DEFAULT_CG_PRICING_TIME_LIMIT = 60.0

# Branch-and-price
DEFAULT_BP_MAX_NODES = 200
DEFAULT_BP_NODE_IM_TIME_LIMIT = 300.0
DEFAULT_BP_DEMO_MAX_NODES = 80

CGFO_THRESHOLD_I = 0.0
CGFO_THRESHOLD_II = 0.10

DEFAULT_METHOD = "im_gurobi"
SOLVER_METHODS = ("im_gurobi", "cg", "cgfo_i", "cgfo_ii", "branch_and_price", "bp")
EXPERIMENT_METHODS = ("lim", "cg", "im_gurobi", "cgfo_i", "cgfo_ii")
SET_B_TEST_METHODS = ("lim", "cg", "im_gurobi", "cgfo_ii")

# 按算例集区分的时间限制（论文 Section 5）
SET_EXPERIMENT_LIMITS: dict[str, dict[str, float]] = {
    "A": {"im_time": 900.0, "cgfo_time": 900.0, "cg_init": 90.0},
    "B": {"im_time": 1800.0, "cgfo_time": 1800.0, "cg_init": 180.0},
}

# Set-B 快速测试（5 算例）
SET_B_TEST_COUNT = 5
SET_B_TEST_BASE_SEED = 200
SET_B_TEST_LOG = "set_b_test_5.csv"
SET_B_TEST_SUMMARY = "set_b_test_5_summary.json"

# ---------------------------------------------------------------------------
# 批量实验
# ---------------------------------------------------------------------------

EXPERIMENT_DEFAULT_SETS = ("A",)
EXPERIMENT_DEFAULT_COUNT = 20
EXPERIMENT_LOG_FILE = "experiment_log.csv"
EXPERIMENT_SUMMARY_FILE = "experiment_summary.json"

PAPER_REFERENCE = {
    "table2": {
        "A": {"lblim": 5611, "lblrmp": 15960, "cpu": 93, "gap": 64.85},
        "B": {"lblim": 9474, "lblrmp": 18932, "cpu": 211, "gap": 49.96},
    },
    "table_ub_a": {
        "im": {"ub": 18028, "cpu": 900},
        "cgfo_i": {"ub": 17632, "ub_gap": -2.25, "cpu": 570},
        "cgfo_ii": {"ub": 17663, "ub_gap": -2.07, "cpu": 279},
    },
    "table_ub_b": {
        "im": {"ub": 21736, "cpu": 1802},
        "cgfo_i": {"ub": 21112, "ub_gap": -2.96, "cpu": 1540},
        "cgfo_ii": {"ub": 21034, "ub_gap": -3.34, "cpu": 1015},
    },
    "table3": {
        "A": {
            "cgfo_i_improve": 2.25,
            "cgfo_ii_improve": 2.07,
            "cgfo_i_time_red": 36.74,
            "cgfo_ii_time_red": 69.04,
        },
        "B": {
            "cgfo_i_improve": 2.96,
            "cgfo_ii_improve": 3.34,
            "cgfo_i_time_red": 14.55,
            "cgfo_ii_time_red": 43.96,
        },
    },
    "set_b_quick_test": {
        "avg_lblim": 9474,
        "avg_lblrmp": 18932,
        "avg_ub_im": 21736,
        "avg_ub_cgfo_ii": 21034,
        "gap_lrmp_lim_pct": 49.96,
    },
}

# ---------------------------------------------------------------------------
# Checker 容差与审计
# ---------------------------------------------------------------------------

CHECKER_TOL = {
    "feasibility": 1e-4,
    "bounds": 1e-4,
    "objective": 1e-2,
    "integer": 1e-5,
    "almost_equal": 1e-4,
    "column": 1e-6,
    "column_cost": 1e-4,
    "subproblem": 1e-3,
    "instance_backorder": 1e-6,
    "bound_order": 1e-3,
    "cg_convergence": 1e-6,
    "cg_column_select": 1e-6,
    "mip_gap_report": 0.01,
    "bldp_dp": 1e-9,
    "objective_divisor": 1e-9,
    "setup_derive": 0.5,
}

# 仅含完整决策变量的方法才做 IM 可行解检查
IM_FEASIBILITY_METHODS = frozenset({"im_gurobi", "cgfo_i", "cgfo_ii", "sim", "im"})

# run_checks --all-results 跳过的非解 JSON
RESULT_JSON_SKIP = frozenset(
    {
        "experiment_summary.json",
        "system_audit.json",
        SET_B_TEST_SUMMARY,
    }
)

AUDIT_INSTANCE_NAMES = ("demo_small", "set_a_01", "set_b_01", "set_b_03")
AUDIT_CG_INSTANCE_NAMES = ("demo_small", "set_a_01")
AUDIT_SUBPROBLEM_INSTANCES = ("demo_small", "set_b_01")
AUDIT_RESULT_PATTERN = "*im_gurobi.json"
AUDIT_OUTPUT_FILE = "system_audit.json"

CHECK_FRESH_SOLVE_TIME_LIMIT = 120.0
CHECK_FRESH_CG_ITERATIONS = 50
CHECK_COLUMN_CG_ITERATIONS = 30
CHECK_COLUMN_INIT_TIME = 30.0
