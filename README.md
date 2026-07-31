# bin_packing_and_scheduling

配置相关装箱与批量计划集成问题（基于 Hao et al. 2022, *EJOR*）的扩展实现：单件 WIP、进罐时间窗与显式报废、阶跃占罐时长（$lt_{it}$）、Dantzig-Wolfe 列生成与分支定价。

## 文档

| 文件 | 说明 |
|------|------|
| [`USAGE.md`](USAGE.md) | **使用说明**（环境、IM/BPC、ML 分支、实验脚本） |
| [`supplement_materials/test_instance.md`](supplement_materials/test_instance.md) | 基准设定与 IM / frac_sb / ML 结果 |
| [`supplement_materials/problem_description_CN.md`](supplement_materials/problem_description_CN.md) | 问题定义、符号、IM 模型、算例字段 |
| [`supplement_materials/technical_report_CN.md`](supplement_materials/technical_report_CN.md) | 技术报告（背景、建模） |
| [`supplement_materials/project_description.md`](supplement_materials/project_description.md) | 项目结构与运行说明 |

**模型要点（当前文档版本）**

- 单件物料 $i\in I$，$X_{imt}\in\{0,1\}$；须在 $rel_i \le t \le end_i$ 内开工，否则报废 $R_i=1$。
- 配置 $u(i)$ 由 `item_config` 给定；同 $(m,t)$ 同配置可合批。
- 占罐时长 $lt_{it}$ 由预处理根据 `config_wait_steps` 阶跃生成；合批跨期数 $L_{u,m,t}=\max lt_{it}$。
- 算例 JSON：`earliest_starts` ↔ $rel_i$，`deadlines` ↔ $end_i$。

> **算例存储**：每个算例一个 SQLite 文件 `instances/{name}.db`。生成：`python main.py --generate demo`。

> **注意**：列生成定价已对齐规范 §3.3（EBRP / Gurobi MIP）；主问题 LRMP 与 IM 的其余差异见 `problem_description_CN.md` §5。

## 项目结构

| 目录 | 用途 |
|------|------|
| `instance_generator/` | 生成算例 |
| `instances/` | 算例文件 |
| `preprocessor/` | 加载与索引化预处理 |
| `model_and_algo/` | IM（Gurobi）、列生成、EBRP 定价、分支定价 |
| `checkers/` | 算例 / 解 / 目标检验 |
| `experiments/` | 实验脚本 |
| `supplement_materials/` | 中文文档 |

## 运行示例

更完整的说明见 [`USAGE.md`](USAGE.md)。

```bash
# 生成演示算例
python main.py --generate demo

# Gurobi 求解 IM
python main.py --instance demo_small --method im_gurobi --time-limit 900

# 列生成下界
python main.py --instance demo_small --method cg

# 分支定价（BPC）
python main.py --instance demo_small --method bp --time-limit 600 --mip-gap 0.01

# CGFO-II 启发式
python main.py --instance demo_small --method cgfo_ii --time-limit 900

# 检验求解结果
python checkers/run_checks.py --all-results
python checkers/run_checks.py --instance demo_small --cg

# 生成 Set-A / Set-B 算例
python main.py --generate A
python main.py --generate B
```

## 参考文献

Hao, Z., et al. (2022). Integrated lot-sizing and bin packing problems with configuration decisions. *European Journal of Operational Research*.
