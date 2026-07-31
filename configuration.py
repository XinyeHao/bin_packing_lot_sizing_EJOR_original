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

DEFAULT_PERIOD_HOURS = 1.0

# 小/中/大实验算例统一：
# - LT∈[5,10]（占罐时长）
# - 窗长为绝对时段、与 T 无关（符合「工艺窗不随计划期变长」）
# - 窗须明显大于 LT，否则几乎无排程松弛、算例会过易（旧 (5,6) 即此问题）
# - 难度主要由 T、I（及负载）拉开，不靠拉长窗
BENCH_LEAD_TIME_CHOICES = (5, 6, 7, 8, 9, 10)
BENCH_WINDOW: dict = {
    "length_range": (12, 16),  # 绝对窗长；均值~14，相对 LT~7 约有 +7 松弛
    "rel_frac": (0.10, 0.50),  # 就绪落在 [0.1T, 0.5T]
}
# 三档规模（U=M=5）；仅 T、I 不同
# 原「中」(30,40) → 现「小」；原「大」(35,50) → 现「中」
# 现「大」= 原大各 +5 → (40, 55)
BENCH_SMALL = (30, 40)   # T, I
BENCH_MEDIUM = (35, 50)
BENCH_LARGE = (40, 55)

# 时间窗与等待阶梯联动生成（按物料）
# length_range=(lo,hi)：绝对窗长；length_frac+min_length：相对 T 的旧规则（A/B/C/demo）
WINDOW_GENERATION_BY_SET: dict[str, dict] = {
    "A": {"length_frac": (0.38, 0.58), "rel_frac": (0.06, 0.22), "min_length": 5},
    "B": {"length_frac": (0.40, 0.62), "rel_frac": (0.04, 0.18), "min_length": 6},
    "demo": {"length_frac": (0.65, 0.90), "rel_frac": (0.05, 0.20), "min_length": 4},
    "large": {"length_frac": (0.42, 0.60), "rel_frac": (0.05, 0.20), "min_length": 6},
    # Set-C：就绪后 1~2 天内需进罐（工艺有效期）；配合锚点错峰与 jitter 压低 EBRP 活跃件数
    "C": {"length_frac": (0.22, 0.38), "rel_frac": (0.02, 0.55), "min_length": 24},
    # 小/中/大实验（SetConfig small_cg）：统一 BENCH_WINDOW
    "small_cg": dict(BENCH_WINDOW),
    "med_cg": dict(BENCH_WINDOW),
}


def apply_bench_generation(
    lead_times: tuple[int, ...] | None = None,
    window: dict | None = None,
) -> None:
    """将小/中/大统一 LT 与时间窗写入 configuration 与 generator 模块。"""
    import instance_generator.generator as genmod

    global BENCH_LEAD_TIME_CHOICES
    lt = tuple(lead_times) if lead_times is not None else BENCH_LEAD_TIME_CHOICES
    w = dict(window) if window is not None else dict(BENCH_WINDOW)
    BENCH_LEAD_TIME_CHOICES = lt
    WINDOW_GENERATION_BY_SET["small_cg"] = w
    WINDOW_GENERATION_BY_SET["med_cg"] = w
    genmod.BENCH_LEAD_TIME_CHOICES = lt
    genmod.LEAD_TIME_CHOICES = lt
    genmod.WINDOW_GENERATION_BY_SET["small_cg"] = w
    genmod.WINDOW_GENERATION_BY_SET["med_cg"] = w
ITEM_STEP_COUNT_RANGE = (1, 2)       # 每件时间窗内 lt 阶跃次数
ITEM_STEP_DELTA_RANGE = (1, 3)       # 每档额外占罐小时（折算为时段）

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
# 小规模 CG 压测：多机竞争 + 窄窗，|I_m|≤18 便于 Gurobi 子问题
SET_SMALL_CG = SetConfig("small_cg", 18, 4, 0, 24, 3)
# 中规模 CG：Gurobi 定价子问题（|I_m| 约 30~40）
SET_MED_CG = SetConfig("med_cg", 28, 4, 0, 40, 4)
DEMO_CONFIG = SetConfig("demo", 6, 3, 2, 4, 2)
LARGE_CONFIG = SetConfig("large", 20, 6, 0, 28, 4)
# Set-C：周计划 IM 测试算例（T=120h≈5天×24h，M=5 热压罐，U=5，|I|≈225）
SET_C = SetConfig("C", 120, 5, 0, 0, 5)

SET_CONFIGS: dict[str, SetConfig] = {
    "A": SET_A,
    "B": SET_B,
    "C": SET_C,
    "large": LARGE_CONFIG,
    "small_cg": SET_SMALL_CG,
    "med_cg": SET_MED_CG,
}

SMALL_CG_ITEM_MACHINES_RANGE = (2, 4)   # 每件 2~4 台可加工 → 机台竞争
SMALL_CG_INSTANCE_COUNT = 6
SMALL_CG_BASE_SEED = 601
# 报废成本 ≥ factor×最廉占罐费，避免“全报废=最优”导致 CG 一轮收敛
SMALL_CG_SCRAP_SETUP_FACTOR = 1.2

MED_CG_ITEM_MACHINES_RANGE = (2, 4)
MED_CG_INSTANCE_COUNT = 3
MED_CG_BASE_SEED = 701
MED_CG_SCRAP_SETUP_FACTOR = 1.2
MED_CG_IM_TIME_LIMIT = 600.0
MED_CG_CG_TIME_LIMIT = 3600.0
MED_CG_PRICING_TIME_LIMIT = 60.0

LEAD_TIME_CHOICES = (1, 2, 3)
BOM_CHOICES = (0, 2, 4)
DEMAND_CHOICES = (1, 2)
BOM_FALLBACK_CHOICES = (2, 4)

INSTANCE_COUNT_DEFAULT = 20
INSTANCE_BASE_SEED = 42
DEMO_INSTANCE_SEED = 0
DEMO_INSTANCE_NAME = "demo_small"
LARGE_INSTANCE_NAME = "large_01"
LARGE_INSTANCE_SEED = 100
# Set-C 算例（名称 set_c_01 …；模拟 5 台热压罐、每周约 225 件、每批约 2~4 件）
SET_C_INSTANCE_COUNT = 2
SET_C_INSTANCE_BASE_SEED = 200
# 件数：225 ≈ 5 罐 × 120h × 密度 0.375；每批约 3 件 → 约 75 批/周
SET_C_SLOT_DENSITY_RANGE = (0.4, 0.4)
SET_C_MIN_ITEMS = 222
SET_C_MAX_ITEMS = 228
SET_C_TARGET_ITEMS_PER_BATCH = 3.0
SET_C_BATCH_ITEMS_RANGE = (2, 4)
# 占罐（小时为时段）：单批约 4~6h，窗内 1~2 档阶跃
SET_C_LEAD_TIME_CHOICES = (4, 5)
SET_C_STEP_DELTA_RANGE = (1, 3)
# Set-C 时间窗（业务×算法双赢，见下表）
# | 参数 | 旧值 | 新值 | 业务含义 | 算法效果 |
# | length_frac | 0.50–0.72 | 0.22–0.38 | 就绪后约 1–1.5 天须进罐 | 窗长中位 74h→39h |
# | min_length | 48h | 24h | 最短 1 天（仍容 4–6h 占罐+阶跃） | 避免强制 2 天宽窗 |
# | jitter | ±3h | ±12h | 同配置 1 个班次带内就绪 | 45 件不再同时进池 |
# | anchor_stagger | 无 | 五配置错峰 | 不同产品线周内分批就绪 | active@t 中位 217→~80 |
# 实测（set_c_01_new）：零对偶 ~0.17s/机；随机对偶 ~3.3s/机精确；CG 真实对偶常触 120s 限时
SET_C_DEMAND_START_FRACTION = 0.35
# 同配置就绪微扰（小时）：±12h ≈ 1 个班次带，保留合批又避免 45 件同时进池
SET_C_CONFIG_WINDOW_JITTER = 12
# 五类配置锚点沿周计划错峰（各占约 1/5 可用时段），避免周中 200+ 件同时可排
SET_C_CONFIG_ANCHOR_STAGGER = True
SET_C_MACHINE_CAPACITY_RANGE = (6.8, 7.2)
SET_C_TRAY_LENGTH_LONG_RANGE = (1.9, 2.5)
SET_C_TRAY_LENGTH_SHORT_RANGE = (1.5, 2.0)
SET_C_TRAY_LONG_PROB = 0.7
SET_C_MACHINE_TIME_TARGET = 0.45
SET_C_SCRAP_SETUP_FACTOR = 2.5
# 每件物料可加工机台数 |M_i| 的生成范围（1=仅单机；|M|=全机）
SET_C_ITEM_MACHINES_RANGE = (1, 2)
SET_C_IM_TIME_LIMIT = 1200.0   # Set-C IM：20 分钟
SET_C_CG_TIME_LIMIT = 600.0    # Set-C CG：10 分钟（LRMP 精确收敛目标）

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
SMALL_CG_GENERATION = {
    "machine_capacity_range": (4.0, 6.5),
    "demand_periods_per_item": DEMAND_PERIODS_PER_ITEM,
}
MED_CG_GENERATION = {
    "machine_capacity_range": (4.0, 6.5),
    "demand_periods_per_item": DEMAND_PERIODS_PER_ITEM,
}
# WIP 可开始加工时间窗 [E_i, D_i]（1-based，均指“开始进罐/加工”的时段）
# E_i ∈ [ceil(EARLIEST_START_MIN_FRACTION·T), ceil(EARLIEST_START_MAX_FRACTION·T)]
# D_i ∈ [max(E_i, ceil(DEADLINE_MIN_FRACTION·T)), T]
EARLIEST_START_MIN_FRACTION = 0.05
EARLIEST_START_MAX_FRACTION = 0.45
DEADLINE_MIN_FRACTION = 0.3

# Set-B 专用生成参数（加宽加工窗、略收紧产能/需求密度，使 IM 更接近论文量级 ~600–1800s）
# 不改变 IM 决策变量与约束结构，仅影响算例数据难度。
SET_B_GENERATION = {
    "earliest_start_min_period": 1,       # E_i ∈ [1, 10]
    "earliest_start_max_period": 10,
    "deadline_min_period": 15,            # D_i ∈ [15, T]（T=25 时为 [15,25]）
    "machine_capacity_range": (4.0, 8.0),
    "num_cured_items": 45,                # 论文 Set-B 为 35，此处略增以抬高 IM 难度
    "demand_periods_per_item": 8,
}

# ---------------------------------------------------------------------------
# 求解器默认参数
# ---------------------------------------------------------------------------

DEFAULT_TIME_LIMIT = 900.0
DEFAULT_MIP_GAP = 0.05
# BPC 树搜索相对 gap：(UB − 全局开节点最小 LB) / UB ≤ 该值则停（0=证到最优）
DEFAULT_BP_MIP_GAP = 0.01
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
DEFAULT_PROGRESS_INTERVAL_SEC = 60.0
EBRP_EXACT_MAX_ITEMS = 40   # 按单机专属物料数 |I_m| 判定，非全局 |I|
EBRP_DEFAULT_TIME_LIMIT = 120.0
EBRP_HEURISTIC_TIME_LIMIT = 1.0
IM_PRICING_DEFAULT_TIME_LIMIT = 90.0

# Branch-and-price
DEFAULT_BP_MAX_NODES = 5000
DEFAULT_BP_OPT_MAX_NODES = 20000
DEFAULT_BP_NODE_IM_TIME_LIMIT = 60.0
# Legacy flags: BPC UB path now uses CGFO + integer LRMP only; these are ignored.
DEFAULT_BP_USE_NODE_IM = False
DEFAULT_BP_ROOT_IM = False
DEFAULT_BP_FINAL_IM = False
DEFAULT_BP_DEMO_MAX_NODES = 80
DEFAULT_INSTANCE_TIME_LIMIT = 600.0  # 每个算例总时限（CG/BP/IM 合计）
DEFAULT_BP_TIME_LIMIT = 600.0  # B&P 单算例求解时限
DEFAULT_BP_OPT_TIME_LIMIT = 3600.0  # 追求最优时的 B&P 时限
# B&P 单算例时限不超过同算例 IM 耗时的该比例（须 < 0.5）
DEFAULT_BP_IM_TIME_RATIO = 0.45
# Prefer setup_y0 only at very shallow depths; allow WIP/scrap earlier for LB lift.
DEFAULT_BP_Y0_DEPTH_BUDGET = 2
DEFAULT_BP_LAZY_Y1_CG_CAP = 60.0
DEFAULT_BP_LB_STALL_EPS = 0.5
BP_STRATEGIES = (
    "new_policy",
    "best_bound",
    "ml",
    "frac_sb",  # frac Top-K pool → short strong-branch refine (no ML)
    "baseline_y0_dfs",
    "old_hybrid",
)
DEFAULT_BP_STRATEGY = "best_bound"
# frac_sb uses the same refine K / child CG caps as ML online refine
FRAC_SB_REFINE_K = 5  # user may set 3; default Top-5
ML_BRANCH_TRAIN_DIR = PROJECT_ROOT / "ml_train_data"
ML_BRANCH_MODEL_PATH = PROJECT_ROOT / "ml_branch_model.txt"
ML_BRANCH_PRETRAIN_PATH = PROJECT_ROOT / "ml_branch_model_pretrain.txt"
# Online ML branch: rank pool then short strong-branch refine on Top-K
ML_SB_REFINE_K = 5
ML_SB_CHILD_TLIM = 10.0  # hard cap per child CG side
ML_SB_MIN_REMAINING = 40.0  # skip refine if wall time left below this (Top-5 needs more budget)

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
    "C": {"im_time": SET_C_IM_TIME_LIMIT, "cg_time": SET_C_CG_TIME_LIMIT, "cgfo_time": SET_C_CG_TIME_LIMIT, "cg_init": 180.0},
}

# Set-C CG：EBRP-H 启发式定价（1s/机，并行墙钟≈1s）；精确 EBRP 见 ebrp_time_limit=None
SET_C_CG = {
    "use_gurobi_init": False,
    "rc_tolerance": DEFAULT_CG_RC_TOLERANCE,
    "pricing_inprocess": False,
    "pricing_batch": False,
    "pricing_workers": 0,
    "time_limit": SET_C_CG_TIME_LIMIT,
    "max_iterations": 20000,
    "verbose": True,
    "pricing_mode": "ebrp_heuristic",
    "ebrp_force": True,
    "ebrp_time_limit": EBRP_HEURISTIC_TIME_LIMIT,
}

# Set-C CG：显式 EBRP / 规范 Gurobi 定价变体
SET_C_CG_EBRP = {
    **SET_C_CG,
    "pricing_mode": "ebrp",
    "use_gurobi_init": False,
    "pricing_inprocess": False,
    "pricing_batch": False,
    "pricing_workers": 0,
    "ebrp_time_limit": None,
    "im_pricing_time_limit": IM_PRICING_DEFAULT_TIME_LIMIT,
}

# Set-C：强制 EBRP（限时，非精确）；绕过 |I|>40 的 Gurobi 回退
SET_C_CG_EBRP_TIMED = {
    **SET_C_CG_EBRP,
    "pricing_mode": "ebrp_force",
    "ebrp_force": True,
    "ebrp_time_limit": 60.0,
    "max_iterations": 3,
}

SET_C_CG_IM = {
    **SET_C_CG,
    "pricing_mode": "im_gurobi",
    "use_gurobi_init": False,
    "pricing_inprocess": True,
    "pricing_workers": 0,
    "im_pricing_time_limit": IM_PRICING_DEFAULT_TIME_LIMIT,
}

# Set-B 快速 CG：跳过 Gurobi 初始列 + 7 路并行 map + 轻量 RC 回传（负 RC 才 pickle 完整列）
SET_B_CG_FAST = {
    "use_gurobi_init": False,
    "rc_tolerance": DEFAULT_CG_RC_TOLERANCE,
    "pricing_inprocess": False,
    "pricing_batch": False,
    "time_limit": None,
}

SET_B_CG_CUTS = {
    **SET_B_CG_FAST,
    "lrmp_cuts": "standard",
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
    "pricing_dp": 1e-9,
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
