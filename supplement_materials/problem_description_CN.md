# 考虑时效限制与阶跃加工时长的固化装箱调度问题（无装配工序）

> 基于 Hao et al. (2022) EJOR 配置相关装箱—批量计划集成模型，本项目做如下扩展：
>
> 1. **去除装配工序**：固化产出即为最终产品；
> 2. **单件物料**：$i\in I$ 为物料编码（如 MR001），**每件恰 1 单位** WIP；
> 3. **物料—配置**：每件 $i$ **直接**对应固化配置 $u(i)\in U$（与 EJOR 的 `item_config` 一致）；同配置多件可**合批**进罐；
> 4. **加工时间窗与报废**：须在 $rel_i \le t \le end_i$ 内开始进罐，否则 $R_i=1$；
> 5. **阶跃加工时长**：预处理参数 $lt_{it}$ 在窗内随 $t$ 分段阶跃、单调不减；合批占罐 $L_{u,m,t}=\max\{lt_{it}\}$，经 $C_{i,m,t,t'}$ 接入流平衡。

> **算例 JSON 字段对应**：$rel_i$ ↔ `earliest_starts[i]`，$end_i$ ↔ `deadlines[i]`。

---

## 1. 问题描述

### 1.1 索引语义（相对 EJOR）

| 概念 | EJOR 原文 | 本模型 |
|------|-----------|--------|
| $i\in I$ | 产品类型，$X_{imt}$ 为投产量 | **单件编码**（MR001），每件 1 单位 WIP |
| $u(i)$ | 经 `item_config` 隐含 | 物料 $i$ **直接**对应固化配置 $u(i)\in U$ |
| $X_{imt}$ | 进罐数量 | 是否在 $(m,t)$ **开始进罐**（0/1） |
| 合批 | 同 setup 混装多种 $i$ | 同 $(m,t)$、**同配置** $u$ 的多件 $i\in I_u$ 可合批 |
| 加工时长 | 固定 $l_u$ | 预处理 $lt_{it}$；合批 $L_{u,m,t}=\max lt_{it}$ |

**示例**：MR001、MR003 均为 $u(i)=\texttt{cfg\_1}$，MR009 为 $u(i)=\texttt{cfg\_2}$。MR001 与 MR003 可在 $(m,t)$ 合批；MR001 与 MR009 不可。该次 setup 占罐 $L_{u,m,t}=\max\{lt_{i,t}: i\in\text{批}\}$，批内于 $t+L_{u,m,t}-1$ **同进同出**。

### 1.2 生产流程

- $X_{imt}=1$：单件 $i$ 于 $t$ 在热压罐 $m$ 开始进罐；
- $\sum_{m\in M}\sum_{t:\,rel_i\le t\le end_i} X_{imt} + R_i = 1$；
- 同 $(m,t)$ 合批须 $u(i)$ 相同（由 $I_u$ 与开始生产约束保证）；
- $Y_{umt}$ 表示 $(m,t)$ 新开配置 $u$ 的 setup；$L_{u,m,t}$ 为本次占罐跨期数；$C_{i,m,t,t'}$ 将完工量接入库存平衡。

### 1.3 阶跃加工时长（预处理）

等待小时（单件 $i$ 若在 $t$ 开工）：

$$
\omega_{it} = (t - rel_i)\cdot h.
$$

配置 $u=u(i)$ 给定基准占罐 $l_u$（`config_lead_times`）及等待阶梯 $\mathcal{S}_u$（`config_wait_steps`，可空）。额外占罐小时 $\Delta_u(\omega)$ **取档、不累加**。预处理得到：

$$
lt_{it} = l_{u(i)} + \left\lceil \Delta_{u(i)}(\omega_{it})/h \right\rceil,
\quad rel_i \le t \le end_i.
$$

在窗外可置 $lt_{it}$ 为任意值（相应 $X_{imt}$ 已被时间窗约束强制为 0）。因各件 $rel_i$ 不同，同配置两件在同一 $t$ 开工时 $lt_{it}$ 可不同。

**合批占罐**（决策变量，最优整数解中取等）：

$$
L_{u,m,t} = \max\bigl\{lt_{it} : i\in I_u,\, X_{imt}=1\bigr\}.
$$

### 1.4 基本假设

1. 每件 $i$ 恰 1 单位 WIP；$u(i)$ 已知且唯一。
2. 合批仅要求同配置：$\{i:X_{imt}=1\}\subseteq I_u$。
3. $\sum_i v_i X_{imt}\le q_m$（开工时刻装箱）；$rc_i>bc_i$。
4. 批内物料同时完工于 $t+L_{u,m,t}-1$。

---

## 2. 符号说明

### 2.1 集合与参数

| 符号 | 含义 | 算例字段 |
|------|------|----------|
| $T,\,I,\,U,\,M$ | 时段、物料、配置、热压罐 | `num_periods`, `cured_item_ids`, `config_ids`, `machine_ids` |
| $u(i)$ | 物料配置 | `item_config[i]` |
| $I_u$ | 配置 $u$ 可加工单件集 | 由 `item_config` 导出 |
| $rel_i,\,end_i$ | 最早 / 最晚开始进罐时段 | `earliest_starts[i]`, `deadlines[i]` |
| $lt_{it}$ | $t$ 开工占罐时段数（阶跃） | 预处理生成 |
| $l_u,\,\mathcal{S}_u$ | 基准占罐、等待阶梯 | `config_lead_times`, `config_wait_steps` |
| $v_i,\,d_{it}$ | 工装长度、需求 | `tray_lengths`, `demand` |
| $hc_i,\,bc_i,\,pc_{mu},\,rc_i$ | 持货、缺货、setup、报废成本 | `holding_costs`, `backorder_costs`, `production_costs`, `scrap_costs` |
| $q_m$ | 热压罐容量 | `machine_capacities` |
| $h$ | 每时段小时数 | `period_hours`（默认 1.0） |
| $\bar M$ | 大 $M$ 常数 | 可取 $|T|+\max_{i,t} lt_{it}$ |

算例 JSON 片段：

```json
{
  "item_ids": ["MR001", "MR003", "MR009"],
  "item_config": {"MR001": "cfg_1", "MR003": "cfg_1", "MR009": "cfg_2"},
  "earliest_starts": {"MR001": 1, "MR003": 2, "MR009": 1},
  "deadlines": {"MR001": 10, "MR003": 12, "MR009": 8},
  "config_lead_times": {"cfg_1": 2, "cfg_2": 3},
  "config_wait_steps": {"cfg_1": [[4, 2], [8, 3]], "cfg_2": []}
}
```

### 2.2 决策变量

| 符号 | 含义 | 类型 |
|------|------|------|
| $X_{imt}$ | 单件 $i$ 是否在 $(m,t)$ 开始进罐 | 0/1 |
| $R_i$ | 单件 $i$ 是否报废 | 0/1 |
| $Y_{umt}$ | 热压罐 $m$ 在 $t$ 开始按配置 $u$ 生产 | 0/1 |
| $C_{i,m,t,t'}$ | 在 $(m,t)$ 开工的 $i$ 于 $t'$ 完工 | 0/1 |
| $L_{u,m,t}$ | $(m,t)$ 上配置 $u$ 本次 setup 跨期数 | $\mathbb{Z}_+$ |
| $S_{it}^+,\,S_{it}^-$ | 库存 / 缺货 | 0/1 |

初值：$S_{i0}^+=S_{i0}^-=0$。

---

## 3. 数学模型（IM）

### 3.1 目标函数

$$
\min \sum_{i,t} hc_i S_{it}^+ + \sum_{i,t} bc_i S_{it}^-
+ \sum_{m,u,t} pc_{mu} Y_{umt} + \sum_i rc_i R_i
$$

### 3.2 约束体系

**成品流平衡**（报废则不交付需求）：

$$
S_{i,t-1}^+ - S_{i,t-1}^- + \sum_{m\in M}\sum_{t'\in T}C_{i,m,t',t}
= d_{it}\cdot (1-R_{i}) + S_{it}^+ - S_{it}^-
\quad \forall i \in I,\; t \in T
$$

**每期至多一种新开配置**：

$$
\sum_{u \in U} Y_{umt} \leq 1
\quad \forall m \in M,\; t \in T
$$

**开始生产约束**（有生产必有 setup，有 setup 必有生产）：

$$
\sum_{i\in I_{u}}X_{imt}\le |I_{u}|\cdot Y_{umt},
\quad \forall t\in T,\; m\in M,\; u\in U
$$

$$
Y_{umt}\le \sum_{i\in I_u} X_{imt},
\quad \forall t\in T,\; m\in M,\; u\in U
$$

**跨期数**（合批取 $\max lt_{it}$）：

$$
L_{umt}\ge lt_{it} \cdot X_{imt},
\quad \forall u\in U,\; i\in I_{u},\; m\in M,\; t\in T
$$

$$
L_{umt}\le \bar M\cdot Y_{umt},
\quad \forall u\in U,\; m\in M,\; t\in T
$$

**完工时刻**（$C_{i,m,t,t'}=1$ 当且仅当 $t'=t+L_{u(i),m,t}-1$）：

$$
t+L_{u(i),m,t}-1-t'\le \bar M\cdot (1-C_{i,m,t,t'}),
\quad \forall i\in I,\; m\in M,\; t,t'\in T
$$

$$
t+L_{u(i),m,t}-1-t'\ge -\bar M\cdot (1-C_{i,m,t,t'}),
\quad \forall i\in I,\; m\in M,\; t,t'\in T
$$

$$
\sum_{t'\in T}C_{i,m,t,t'}=X_{imt},
\quad \forall i\in I,\; m\in M,\; t\in T
$$

**机台不重叠**：

$$
L_{umt}-\bar M\cdot (2-Y_{umt}-Y_{u',m,t'})\le t'-t,
\quad \forall m\in M,\; u,u'\in U,\; t,t'\in T,\; t'> t
$$

**热压罐长度容量**（开工时刻装箱）：

$$
\sum_{i \in I} v_i \cdot X_{imt} \leq q_m
\quad \forall m \in M,\; t\in T
$$

**单件去向**：

$$
\sum_{m\in M} \sum_{t:\,rel_i\le t\le end_i} X_{imt} + R_i = 1
\quad \forall i \in I
$$

**时间窗与计划期末可行**：

$$
X_{imt} = 0,
\quad \forall i\in I,\; m\in M,\;
t < rel_i \;\text{ 或 }\; t > end_i \;\text{ 或 }\; t+lt_{it}-1 > |T|
$$

### 3.3 逻辑链

```
单件 i  (MR001, 1 单位)
    │ u(i) ∈ U
    ▼
预处理 lt_{it}  (rel_i ≤ t ≤ end_i，阶跃)
    │
    ├─ X_{imt}∈{0,1}  →  须在 [rel_i, end_i] 内开工或 Ri=1
    │
    ├─ 同 (m,t) 合批 {i: X_{imt}=1} ⊂ I_u
    │       └─ L_{u,m,t} = max_{i∈批} lt_{it}
    │
    ├─ Y_{umt}  （setup 计费；与 X 双向绑定）
    ├─ C_{i,m,t,t'}  （完工于 t+L_{u,m,t}-1）
    └─ 机台不重叠  （相邻 setup 间距 ≥ 前次 L）
```

### 3.4 EJOR 退化

当 $\mathcal{S}_u=\varnothing$ 时 $lt_{it}=l_{u(i)}$（常数）。若每批仅一件且 $L_{umt}=lt_{it}X_{imt}$，并去掉 $C$ 的批末同时完工语义、改回固定提前期流平衡，则退化为 EJOR 单配置固定占罐特例。

---

## 4. 列生成（概要，待实现）

目标分解与 EJOR 相同思路：主问题在单件层分配列（计划模式），子问题在单机—配置上生成带 $x_{imt}^k$、$y_{umt}^k$、占罐长度 $L_{u,m,t}^k$（或等价 $\theta$）的列。

- 主问题：$\sum_{m,k} x_{imt}^k Q_{mk} + R_i = 1$（单件去向）；
- 子问题定价：在 $rel_i \le \tau \le end_i$ 上按 $\max_i lt_{i\tau}$ 扩展 setup 窗口；
- 预处理阶段离线计算 $lt_{it}$ 表。

---

## 5. 文档与代码状态

| 项目 | 状态 |
|------|------|
| 单件 $I$、$u(i)$、$rel_i/end_i$、$lt_{it}$ 阶跃 | 文档已定稿 |
| $L_{u,m,t}$、$C_{i,m,t,t'}$ 替代 $Z/\Theta$ | 文档已定稿 |
| `config_wait_steps` → $lt_{it}$ 预处理 | 文档已定稿；代码待实现 |
| `im_model.py` / 列生成 / EBRP | 定价对齐规范 §3.3；大规模 $|I|>40$ 自动回退 Gurobi MIP |

详细技术报告见 [`technical_report_CN.md`](technical_report_CN.md)。
