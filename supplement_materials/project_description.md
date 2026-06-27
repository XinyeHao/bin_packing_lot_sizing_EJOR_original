# 项目描述

> 在此填写项目背景、研究问题、方法概述与使用说明。

## 研究问题

（待补充）

## 项目结构

| 目录 | 用途 |
|------|------|
| `instance_generator/` | 生成原始算例 |
| `instances/` | 存放算例文件 |
| `preprocessor/` | 算例预处理，转为算法友好型结构 |
| `model_and_algo/im_model.py` | SIM 集成模型（Gurobi） |
| `model_and_algo/column_generation.py` | Dantzig-Wolfe 列生成 |
| `model_and_algo/bldp.py` | 子问题双层动态规划（BLDP） |
| `model_and_algo/cgfo.py` | CGFO 启发式 |
| `model_and_algo/solver.py` | 统一求解入口 |
| `postprocessor/` | 计算结果后处理 |
| `checkers/` | 算例/可行解/目标函数/上下界/列 检验 |
| `supplement_materials/` | 项目描述与补充材料 |

## 运行示例

```bash
# 生成演示算例
python main.py --generate demo

# 直接 Gurobi 求解 SIM（对应论文 IM-CPLEX）
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
