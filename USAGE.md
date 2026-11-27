# 使用说明

热压罐固化调度：集成模型（IM）与分支定价（BPC）。默认相对 gap 目标 `(UB − tree_LB) / UB ≤ 1%`（`DEFAULT_BP_MIP_GAP = 0.01`）。

## 1. 环境

- Python 3.10+（开发环境为 3.14）
- [Gurobi](https://www.gurobi.com/) 有效许可证（`gurobipy`）
- 依赖：

```bash
pip install -r requirements.txt
```

| 包 | 用途 |
|----|------|
| `gurobipy` | IM / 定价 MIP / LRMP |
| `numpy` | 数值 |
| `lightgbm` | BPC 的 `ml` 分支策略（可选；仅 `frac_sb` 可不装） |

在项目根目录运行命令（保证 `import configuration` 等可用）。

## 2. 快速开始

```bash
# 生成演示算例 → instances/demo_small.db
python main.py --generate demo

# IM（Gurobi）
python main.py --instance demo_small --method im_gurobi --time-limit 900

# 列生成下界
python main.py --instance demo_small --method cg --verbose

# 分支定价（默认策略见 configuration.DEFAULT_BP_STRATEGY）
python main.py --instance demo_small --method bp --time-limit 600 --mip-gap 0.01

# 结果校验
python checkers/run_checks.py --instance demo_small --cg
```

结果写入 `result/`。算例为 SQLite：`instances/{name}.db`。

## 3. 方法与入口

| `method` | 含义 |
|----------|------|
| `im_gurobi` | 完整集成 MIP |
| `cg` | 列生成（LRMP + 定价） |
| `cgfo_i` / `cgfo_ii` | CGFO 启发式上界 |
| `bp` / `branch_and_price` | 分支定价（BPC） |

统一入口：`model_and_algo/solver.py` → `solve(...)`。  
BPC 更细参数请直接构造 `BranchAndPriceSolver`（见实验脚本）。

当前上界策略默认为 `upper_bound_strategy="adaptive"`：
根节点先运行至多 5 秒的 IRMP，再运行至多 30 秒的 CGFO。
IRMP 的完整日程用于 CGFO 的 MIP 初始解，其开工位置并入 LP 正值列的允许集合。
`upper_bound_strategy="cgfo"` 可用于直接 CGFO 的对照实验。
树中上界连续 120 个节点未改善时，尝试至多 10 秒的两台机器局部修复。
树中不调用 IRMP；节点 LP 自然得到的整数解仍可直接更新上界。
连续无改善会增加调用间隔。所有新上界均保存完整整数日程并经过原模型检查，
可从 `res.best_solution`、`res.primal_history` 和 `res.primal_stats` 读取。
修复模型的固定条件只用于生成可行解，不参与 BP 下界或定价证书。
CGFO 的列支持开工集合参考 [Hao 等（2022）](https://doi.org/10.1016/j.ejor.2022.03.012)；
按改善效果控制启发式调用频率的思路可参考
[Maher 和 Rönnberg（2023）](https://doi.org/10.1007/s12532-023-00240-w)。

完整中、大规模测试（各 15 例，时限分别为 1800、3600 秒，目标 gap 1%）：

```bash
python experiments/run_adaptive_ub_suite.py --workers 1 --threads 16 --output result/diagnostics/root_irmp_cgfo_serial_v5
python experiments/analyze_adaptive_ub_suite.py --output result/diagnostics/root_irmp_cgfo_serial_v5 --audit
```

算例按中规模、大规模的种子顺序逐个求解；默认使用本机全部求解器线程。
上述命令将结果保存在 `result/diagnostics/root_irmp_cgfo_serial_v5/`，包括逐例日志、日程、代码快照、
上界改善历史和汇总对照表。使用 `--resume` 可续跑相同代码及配置下的测试；
修改算法后应以 `--output` 指定新的结果目录。

### 3.1 BPC 关键配置

```python
from configuration import apply_bench_generation, BENCH_MEDIUM, SetConfig
from instance_generator.generator import generate_instance
from preprocessor.preprocess import preprocess
from model_and_algo.branch_and_price import BranchAndPriceSolver

apply_bench_generation()  # LT∈{5..10}, length_range=[12,16], rel_frac=[0.1,0.5]
T, I = BENCH_MEDIUM       # (35, 50)；大档 BENCH_LARGE=(40,55)
raw = generate_instance(SetConfig("small_cg", T, 5, 0, I, 5), seed % 1000, seed)
data = preprocess(raw)

bp = BranchAndPriceSolver(
    data,
    max_nodes=8000,
    time_limit=1800.0,          # 中档建议 1800；大档 3600
    mip_gap=0.01,               # 1% 相对 gap
    branch_strategy="frac_sb",  # 或 "ml"
    use_node_im=False,
    root_im=False,
    final_im=False,
    cg_kwargs=dict(
        pricing_mode="ebrp",
        ebrp_force=True,
        use_gurobi_init=True,
    ),
)
res = bp.solve()
```

### 3.2 分支策略

| `branch_strategy` | 说明 |
|-------------------|------|
| `frac_sb` | 分数度 Top-20 候选池 → Top-5 短强分支精炼 |
| `ml` | 同候选池 → LightGBM 粗排序 → Top-5 短强分支；需仓库根目录 `ml_branch_model.txt` |
| `best_bound` 等 | 历史/对照策略 |

ML **不替代**强分支，只替换 frac 粗排序。模型路径：`configuration.ML_BRANCH_MODEL_PATH`。

## 4. 实验算例规模（基准）

`configuration.apply_bench_generation()` 后统一：

| 规模 | `(T, I)` | `U=M` | 建议时限 |
|------|----------|-------|----------|
| 小 | `(30, 40)` | 5 | 900s |
| 中 | `(35, 50)` | 5 | 1800s |
| 大 | `(40, 55)` | 5 | 3600s |

时间窗：`length_range=[12,16]`（绝对时段）、`rel_frac=[0.1,0.5]`、`LT∈{5..10}`。

## 5. 常用实验脚本

均在项目根目录执行。单例 worker 在子进程结束时 `disposeDefaultEnv`，避免 Gurobi 内存残留。

### 5.1 三方法对照（IM / frac_sb / ml）

```bash
# 中规模 10 个新种子（31101–32001），每例依次 IM → frac → ML
python experiments/run_three_method_10med_new.py

# 大规模 10 个新种子（41101–42001）
python experiments/run_three_method_10large_new.py
```

### 5.2 单方法批跑 / 单例 worker

```bash
# 中/大各 5 种子：IM 或 BPC
python experiments/run_im_gap1pct_5med_5large.py
python experiments/run_bpc_frac_gap1pct_5med_5large.py
python experiments/run_bpc_ml_gap1pct_5med_5large.py

# 单例（参数：scale seed limit mip_gap out.json）
python experiments/_im_gap1pct_one.py medium 30101 1800 0.01 out.json
python experiments/_bpc_frac_gap1pct_one.py medium 30101 1800 0.01 out.json
python experiments/_bpc_ml_gap1pct_one.py medium 30101 1800 0.01 out.json
```

`scale` 为 `medium` 或 `large`。

### 5.3 ML 数据采集与重训

```bash
# 采集分支标签（默认写入 ml_train_data_v7_w1216/）
python experiments/ml_full_labelA_8h.py

# 重训粗排序模型 → ml_branch_model.txt
python experiments/ml_retrain_top5_coarse.py --data-dir ml_train_data_v7_w1216 --label y_min
```

训练数据目录默认被 `.gitignore` 忽略；仓库自带已训好的 `ml_branch_model.txt` 可直接跑 `ml` 策略。

## 6. 结果与文档

| 路径 | 内容 |
|------|------|
| [`supplement_materials/test_instance.md`](supplement_materials/test_instance.md) | 方法说明与正式基准结果 |
| [`supplement_materials/all_results_im_frac_ml.md`](supplement_materials/all_results_im_frac_ml.md) | IM / frac / ML 全表 |
| [`supplement_materials/problem_description_CN.md`](supplement_materials/problem_description_CN.md) | 问题与 IM 模型 |
| [`README.md`](README.md) | 项目概览 |

统计脚本示例：`python experiments/_stats_ml_vs_frac.py`。

## 7. 目录一览

| 目录 / 文件 | 用途 |
|-------------|------|
| `main.py` | CLI：生成算例 / 求解 |
| `configuration.py` | 路径、规模、BPC/CG/ML 默认参数 |
| `instance_generator/` | 算例生成 |
| `preprocessor/` | 加载与预处理 |
| `model_and_algo/` | IM、CG、EBRP、BPC、ML 分支 |
| `experiments/` | 基准与 ML 流水线 |
| `checkers/` | 可行性与目标检验 |
| `ml_branch_model.txt` | 线上 ML 分支模型 |

## 8. 常见问题

1. **`ImportError: lightgbm`**：仅 `branch_strategy="ml"` 需要；安装 `pip install lightgbm`，或改用 `frac_sb`。
2. **找不到模型**：确认根目录存在 `ml_branch_model.txt`。
3. **Gurobi 许可证 / 内存**：长批跑请用 `_*-one.py` 子进程模式；批跑脚本已按此封装。
4. **gap 未到 1%**：达到 `time_limit` 会返回当前 UB/LB；看 `status` 与相对 gap。
