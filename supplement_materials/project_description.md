# 项目描述

配置相关装箱与固化调度集成问题：在 EJOR (2022) 模型基础上去除装配工序，引入单件 WIP、进罐时间窗（$rel_i$–$end_i$）、显式报废与阶跃占罐时长 $lt_{it}$，采用 Gurobi IM 与 Dantzig-Wolfe 列生成（及分支定价）求解。

## 研究问题

制造企业将单件 WIP 装入热压罐固化。每件物料：

- 有工装长度 $v_i$，同批总长度不超过罐容 $q_m$；
- 对应固化配置 $u(i)$，同罐同时仅一种配置；
- 须在 $rel_i \le t \le end_i$ 内开始进罐，否则报废（$R_i=1$）；
- 占罐时段数 $lt_{it}$ 由预处理给定：在窗内距 $rel_i$ 越远越大（`config_wait_steps` 分段阶跃）；
- 同 $(m,t)$、同配置可合批，占罐 $L_{u,m,t}=\max\{lt_{it}\}$，批末同时完工。

数学模型与符号见 [`problem_description_CN.md`](problem_description_CN.md)；技术背景见 [`technical_report_CN.md`](technical_report_CN.md)。

**算例字段**：`earliest_starts` = $rel_i$，`deadlines` = $end_i$；`item_config` = $u(i)$；`config_lead_times` + `config_wait_steps` → 预处理表 $lt_{it}$。

## 项目结构

| 目录 | 用途 |
|------|------|
| `instance_generator/` | 生成原始算例（Set-A / Set-B / demo） |
| `instances/` | 存放算例 JSON |
| `preprocessor/` | 算例加载、索引化（`earliest_starts` / `deadlines` 等） |
| `model_and_algo/im_model.py` | IM 集成模型（Gurobi） |
| `model_and_algo/column_generation.py` | Dantzig-Wolfe 列生成 |
| `model_and_algo/pricing_ebrp.py` | 子问题 EBRP 精确定价（§3.3） |
| `model_and_algo/pricing_im_gurobi.py` | 子问题 Gurobi MIP（§3.3） |
| `model_and_algo/branch_and_price.py` | 分支定价 |
| `model_and_algo/cgfo.py` | CGFO 启发式 |
| `model_and_algo/solver.py` | 统一求解入口 |
| `checkers/` | 算例、可行解、目标、上下界检验 |
| `experiments/` | 对比实验脚本 |
| `supplement_materials/` | 问题描述与技术报告 |

## 实现状态

| 模块 | 状态 |
|------|------|
| 算例生成、预处理、检验器 | 已实现（时间窗字段 `earliest_starts` / `deadlines`） |
| IM / CG / EBRP | 已实现；定价对齐规范 §3.3（EBRP 或 Gurobi MIP） |
| $lt_{it}$ 阶跃预处理、`config_wait_steps` | 文档已定稿，代码待实现 |
| $L$、$C$ 新 IM  formulation | 文档已定稿，代码待实现 |

## 运行示例

```bash
# 生成演示算例
python main.py --generate demo

# Gurobi 求解 IM
python main.py --instance demo_small --method im_gurobi --time-limit 900

# 列生成下界
python main.py --instance demo_small --method cg

# CGFO-II 启发式
python main.py --instance demo_small --method cgfo_ii --time-limit 900

# 检验求解结果
python checkers/run_checks.py --all-results
python checkers/run_checks.py --instance demo_small --cg

# 生成 Set-A / Set-B 算例（各 20 个）
python main.py --generate A
python main.py --generate B
```
