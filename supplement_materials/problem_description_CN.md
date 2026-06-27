# 集成装箱与多级批量计划问题（配置相关装箱过程）

> 来源：Hao, X., Zheng, L., Li, N., & Zhang, C. (2022). Integrated bin packing and lot-sizing problem considering the configuration-dependent bin packing process. *European Journal of Operational Research*, 303, 581–592.

---

## 1. 问题描述

### 1.1 背景与生产流程

本文研究航空制造工厂中复合材料航空产品的生产调度问题。生产系统采用多品种、小批量模式，包含两个主要阶段（见图 1）：

**第一阶段——固化（装箱过程）**  
各物料放置于专用托盘上，送入不同长度的工业热压罐（autoclave，即“箱子”）进行固化。同一热压罐内只能处理具有**相同固化配置**（如相同温度、压力）的物料，不同配置的物料不可混装。热压罐选定某一配置并完成 setup 后，须持续运行与该配置对应的固化时长；固化结束后托盘取出，已固化物料卸载。

**第二阶段——装配**  
固化后的物料按物料清单（BOM）装配为复合成品，并向客户交付。

由于固化阶段本质上是装箱过程，同时各物料的批量计划（lot-sizing）决策需一并确定，该问题可归结为**集成装箱与批量计划问题（BP-LSP）**，其核心特征为**配置相关的装箱过程（configuration-dependent bin packing process）**：

- 热压罐**可重配置**：只有为某配置完成 setup 后，该配置对应的物料才能在其中加工；
- 加工时间**依赖配置**：选定配置后，固化状态须持续对应时长；
- 装箱决策与批量计划决策**耦合**：第一阶段装箱过多可降低生产成本，但会增加第二阶段等待装配的 holding 成本；且需装箱的物料集合由装配需求内生决定，而非传统 BPP 中给定的固定物品集。

### 1.2 基本假设

1. 所有输入参数已知，为**确定性**问题。
2. 瓶颈在固化阶段，**装配阶段视为无产能约束**。
3. 成品（end item）外部需求允许**延期交货（backorder）**；非成品（固化物料）无外部需求。
4. 固化过程持续若干时段，须考虑**提前期（lead time）**，为计划时段单位长度（UTP）的整数倍：$l_{ti} = \left\lceil \dfrac{\text{curingTime}}{\text{UTP}} \right\rceil$。例如 UTP = 8 h 时，固化时间 12 h 的配置对应提前期 $\lceil 12/8 \rceil = 2$。
5. 托盘宽度相同、长度不同；托盘在热压罐内**单行水平排列**，可容纳托盘数由热压罐长度与已分配托盘累计长度决定。
6. 各尺寸托盘数量充足。

---

## 2. 符号说明

### 2.1 集合与下标

| 符号 | 含义 |
|------|------|
| $T$ | 计划期内时段集合，下标 $t,\, t'$ |
| $I$ | 全部物料集合，下标 $i,\, i'$ |
| $End^p$ | 成品（end item）集合，$End^p \subset I$，下标 $j$ |
| $U$ | 固化配置集合，下标 $u$ |
| $M$ | 热压罐（autoclave）集合，下标 $m$ |

### 2.2 参数

| 符号 | 含义 |
|------|------|
| $l_{ti}$ | 固化物料 $i$ 的生产提前期 |
| $b_{iu}$ | 若物料 $i$ 在配置 $u$ 下固化则为 1，否则为 0 |
| $l_u$ | 固化配置 $u$ 的生产提前期；若 $b_{iu}=b_{i'u}=1$，则 $l_{ti}=l_{ti'}=l_u$ |
| $r_{ij}$ | 按 BOM，装配 1 单位成品 $j$ 所需固化物料 $i$ 的数量 |
| $d_{jt}$ | 时段 $t$ 成品 $j$ 的外部需求 |
| $v_i$ | 与固化物料 $i$ 匹配的托盘长度 |
| $q_m$ | 热压罐 $m$ 可容纳的最大累计托盘长度 |
| $p_{cum}$ | 热压罐 $m$ 在配置 $u$ 下的生产成本 |
| $h_{ci}$ | 物料 $i$ 单位库存、单位时段 holding 成本 |
| $bc_j$ | 成品 $j$ 单位延期、单位时段 backorder 成本 |

### 2.3 决策变量

| 符号 | 含义 |
|------|------|
| $S_{it}^+$ | 时段 $t$ 末物料 $i$ 的库存量 |
| $S_{jt}^-$ | 时段 $t$ 末成品 $j$ 的延期量 |
| $X_{imt}$ | 时段 $t$ 投入热压罐 $m$ 的物料 $i$ 数量 |
| $Y_{umt}$ | 若时段 $t$ 热压罐 $m$ 为配置 $u$ 做 setup 则为 1，否则为 0（setup 后生产状态持续 $l_u$ 个时段） |
| $Z_{umt}$ | 若时段 $t$ 热压罐 $m$ 在配置 $u$ 下运行则为 1，否则为 0 |
| $A_{jt}$ | 时段 $t$ 装配的成品 $j$ 数量 |

---

## 3. 数学模型

本文先给出**原始集成模型（OIM）**，再通过约束替换将其简化为**简化集成模型（SIM）**；后文所称 **IM** 均指 SIM。

### 3.1 原始集成模型（OIM）

**目标函数**——最小化计划期内总成本（holding、backorder 与热压罐生产成本）：

$$
\min \sum_{t \in T} \sum_{i \in I} h_{ci} \cdot S_{it}^+ + \sum_{t \in T} \sum_{j \in End^p} bc_j \cdot S_{jt}^- + \sum_{t \in T} \sum_{m \in M} \sum_{u \in U} p_{cum} \cdot Y_{umt} \tag{1}
$$

**约束条件：**

$$
S_{j,t-1}^+ - S_{j,t-1}^- + A_{jt} = d_{jt} + S_{jt}^+ - S_{jt}^- \quad \forall j \in End^p;\; t \in T \tag{2}
$$

$$
S_{i,t-1}^+ + \sum_{m \in M} X_{i,m,t-l_{ti}} = \sum_{\substack{j \in End^p \\ r_{ij}>0}} r_{ij} \cdot A_{jt} + S_{it}^+ \quad \forall i \in I \setminus End^p;\; t \in T \tag{3}
$$

$$
\sum_{u \in U} Y_{umt} \leq 1 \quad \forall m \in M;\; t \in T \tag{4}
$$

$$
\sum_{u \in U} Z_{umt} \leq 1 \quad \forall m \in M;\; t \in T \tag{5}
$$

$$
\sum_{t' \in \{t' : t \leq t' \leq t + l_u - 1\}} Y_{u,m,t'} \leq 1 \quad \forall u \in U;\; m \in M;\; t \in T \tag{6}
$$

$$
Y_{umt} \leq Z_{u,m,t'} \quad \forall u \in U;\; m \in M;\; t \in T,\; t' \in \{t' : t \leq t' \leq t + l_u - 1\} \tag{7}
$$

$$
X_{imt} \leq \left\lfloor \frac{q_m}{v_i} \right\rfloor \cdot \left( \sum_{u \in U} b_{iu} \cdot Y_{umt} \right) \quad \forall i \in I \setminus End^p;\; m \in M;\; t \in T \tag{8}
$$

$$
\sum_{i \in I \setminus End^p} v_i \cdot X_{imt} \leq q_m \quad \forall m \in M;\; t \in T \tag{9}
$$

$$
S_{it}^+,\; S_{jt}^-,\; A_{jt} \in \mathbb{Z}_+ \quad \forall i \in I;\; j \in End^p;\; t \in T \tag{10}
$$

$$
X_{imt} \in \mathbb{Z}_+ \quad \forall i \in I \setminus End^p;\; m \in M;\; t \in T \tag{11}
$$

$$
Y_{umt},\; Z_{umt} \in \{0, 1\} \quad \forall u \in U;\; m \in M;\; t \in T \tag{12}
$$

**约束说明：**

- **(2)**：成品流量守恒（含外部需求与 backorder）。
- **(3)**：固化物料库存平衡（需求由装配过程内生，无外部需求）。
- **(4)**：每个时段初，每台热压罐至多选择一种配置做 setup。
- **(5)**：每个时段，每台热压罐至多处于一种配置的运行状态。
- **(6)**：长度为 $l_u$ 的任意时间窗口内，配置 $u$ 的 setup 至多一次。
- **(7)**：若时段 $t$ 为配置 $u$ 做 setup，则随后 $l_u$ 个时段内热压罐均处于配置 $u$ 的运行状态。
- **(8)**：装箱与配置的关联——仅与当前配置匹配的物料可投入该热压罐。
- **(9)**：空间容量约束——投入托盘的累计长度不超过热压罐容量 $q_m$。
- **(10)–(12)**：变量非负性与整数性。

### 3.2 简化集成模型（SIM / IM）

**命题 1**：下列约束 **(13)** 可替代 OIM 中的约束 **(4)、(6)、(7)**：

$$
\sum_{t' \in \{t' : \max\{t - l_u + 1,\, 1\} \leq t' \leq t\}} Y_{u,m,t'} = Z_{umt} \quad \forall u \in U;\; m \in M;\; t \in T \tag{13}
$$

约束 (13) 表示：时段 $t$ 热压罐 $m$ 是否在配置 $u$ 下运行，取决于包括当前时段在内的前 $l_u$ 个时段内是否发生过配置 $u$ 的 setup。

**SIM 形式：**

$$
\text{[SIM]} \quad \min \ \text{目标函数 (1)}
$$

$$
\text{s.t.} \quad \text{约束 (2), (3), (5), (8)–(13)}
$$

---

## 4. 模型要点小结

| 维度 | 内容 |
|------|------|
| 问题类型 | 集成装箱（BPP）+ 多级 capacitated 批量计划（LSP） |
| 核心特征 | 可重配置热压罐、配置相关加工时间、装配导向的内生装箱需求 |
| 成本构成 | holding 成本 + backorder 成本 + 热压罐 setup/生产成本 |
| 产能约束 | 热压罐长度容量（空间装箱）+ 配置持续时长（时间维度） |
| 本文采用模型 | SIM（即 IM），基于 Kantorovich 型装箱建模 |
