# IM、CG 主/子问题 与 EBRP 子问题求解算法

---

## 1. 符号与定义

### 1.1 集合与索引

| 符号 | 定义 |
|------|------|
| $T=\{1,2,\dots,|T|\}$ | 离散计划时段集合，索引 $t,t'\in T$ |
| $I$ | 单件物料集合，索引 $i\in I$ |
| $U$ | 固化配置集合，索引 $u\in U$ |
| $M$ | 热压罐集合，索引 $m\in M$ |
| $I_u=\{i\in I\mid u(i)=u\}$ | 配置 $u$ 可加工的物料集合 |
| $M_i\subseteq M$ | 物料 $i$ 允许加工的机台集合 |
| $I_m=\{i\in I\mid b_{im}=1\}$ | 机台 $m$ 可加工的物料集合 |
| $\mathcal{F}_{im}=\{(i,t)\mid b_{im}=1,\;rel_i\le t\le end_i,\;\tau_{it}\le|T|\}$ | 物料 $i$ 在机台 $m$ 上的全部可行开工时间点 |
| $\mathcal{K}_m$ | 机台 $m$ 的调度列（column）集合，索引 $k$ |

### 1.2 已知参数

| 符号 | 定义 |
|------|------|
| $b_{im}\in\{0,1\}$ | $=1$ 当且仅当物料 $i$ 可在机台 $m$ 上加工 |
| $u(i)\in U$ | 物料 $i$ 的固化配置 |
| $rel_i\in T$ | 物料 $i$ 的最早开工时段 |
| $end_i\in T$ | 物料 $i$ 的最晚开工时段 |
| $lt_{it}\in\mathbb{Z}_+$ | 物料 $i$ 在时段 $t$ 开工的占罐时段数 |
| $\tau_{it}:=t+lt_{it}-1$ | 物料 $i$ 在时段 $t$ 开工的完工时段 |
| $v_i\in\mathbb{R}_+$ | 物料 $i$ 的工装长度 |
| $q_m\in\mathbb{R}_+$ | 热压罐 $m$ 的容量上限 |
| $d_{it}\in\{0,1\}$ | 时段 $t$ 对物料 $i$ 的需求量 |
| $w_i\in\{0,1\}$ | 物料 $i$ 的初始 WIP 量（单件模型中恒为 $1$） |
| $hc_i,\,bc_i$ | 物料 $i$ 的单位持货 / 缺货成本 |
| $pc_{mu}$ | 机台 $m$ 上配置 $u$ 的 setup 单位成本 |
| $rc_i$ | 物料 $i$ 的单位报废成本 |
| $\bar M$ | 充分大的常数（大 M） |

### 1.3 决策变量

#### IM 决策变量

| 符号 | 类型 | 含义 |
|------|------|------|
| $X_{imt}$ | $\{0,1\}$ | $=1$ 当且仅当物料 $i$ 在机台 $m$ 的时段 $t$ 开工加工 |
| $Y_{umt}$ | $\{0,1\}$ | $=1$ 当且仅当机台 $m$ 在时段 $t$ 对配置 $u$ 进行 setup（即该时段至少加工一个配置 $u$ 的物料） |
| $L_{umt}$ | $\mathbb{Z}_+$ | 机台 $m$ 上配置 $u$ 在时段 $t$ 的占罐跨期（从 $t$ 起占用罐体的时段数） |
| $R_i$ | $\{0,1\}$ | $=1$ 当且仅当物料 $i$ 被报废（没有在任何机台的可行窗内开工） |
| $S_{it}^+$ | $\{0,1\}$ | $=1$ 当且仅当时段 $t$ 结束时物料 $i$ 出现正库存（持货） |
| $S_{it}^-$ | $\{0,1\}$ | $=1$ 当且仅当时段 $t$ 结束时物料 $i$ 出现负库存（缺货），需求被延期满足 |

#### 列参数（主问题中列 $k$ 的预计算属性）

| 符号 | 定义 |
|------|------|
| $x_{imt}^k\in\{0,1\}$ | 列 $k$ 是否在机台 $m$ 的时段 $t$ 开工加工物料 $i$ |
| $y_{umt}^k\in\{0,1\}$ | 列 $k$ 是否在机台 $m$ 的时段 $t$ 对配置 $u$ 进行 setup |
| $c_m^k=\sum_{u,t}pc_{mu}\,y_{umt}^k$ | 列 $k$ 的总 setup 成本 |
| $w_i^k=\sum_{t:(i,t)\in\mathcal{F}_{im}}x_{imt}^k$ | 列 $k$ 对物料 $i$ 的窗内开工量 |
| $a_{i,t'}^{k}=\sum_{t:\tau_{it}=t'}x_{imt}^k$ | 列 $k$ 在时段 $t'$ 对物料 $i$ 的完工入库量（$t$ 开工、$t'$ 完工的所有物料量之和） |

#### 主问题决策变量

| 符号 | 类型 | 含义 |
|------|------|------|
| $\theta_{mk}$ | $\mathbb{R}_+$ | 机台 $m$ 上列 $k$ 的使用强度（凸组合系数）；IP 最优解中取 $\{0,1\}$ |
| $S_{it}^+$ | $\mathbb{R}_+$ | 时段 $t$ 结束时物料 $i$ 的正库存量（持货，单件模型中取 $\{0,1\}$） |
| $S_{it}^-$ | $\mathbb{R}_+$ | 时段 $t$ 结束时物料 $i$ 的负库存量（缺货延期，单件模型中取 $\{0,1\}$） |
| $R_i$ | $\mathbb{R}_+$ | 物料 $i$ 的报废量（单件模型中取 $\{0,1\}$） |

#### 子问题决策变量（固定机台 $m$）

| 符号 | 类型 | 含义 |
|------|------|------|
| $X_{imt}$ | $\{0,1\}$ | $=1$ 当且仅当物料 $i$ 在机台 $m$ 的时段 $t$ 开工（此处 $m$ 已固定，故无 $\scriptstyle m$ 上标） |
| $Y_{umt}$ | $\{0,1\}$ | $=1$ 当且仅当机台 $m$ 在时段 $t$ 对配置 $u$ 进行 setup |
| $L_{umt}$ | $\mathbb{Z}_+$ | 机台 $m$ 上配置 $u$ 在时段 $t$ 的占罐跨期 |

---

## 2. 集成模型（IM）

### 2.1 目标函数

总成本 = 持货成本 + 缺货成本 + setup 成本 + 报废成本：

$$\min\ \sum_{i\in I}\sum_{t\in T} hc_i\,S_{it}^+ + \sum_{i\in I}\sum_{t\in T} bc_i\,S_{it}^- + \sum_{m\in M}\sum_{u\in U}\sum_{t\in T} pc_{mu}\,Y_{umt} + \sum_{i\in I} rc_i\,R_i$$

### 2.2 约束

**(1) 流平衡**（每个时段结束时，库存、缺货、入库、需求之间的关系）：

$$S_{i,t'-1}^+ - S_{i,t'-1}^- + \sum_{m\in M}\sum_{t:\tau_{it}=t'} X_{imt} = d_{i,t'}\,(1-R_i) + S_{i,t'}^+ - S_{i,t'}^- \quad \forall i\in I,\; t'\in T$$

含义：上一时段末的净库存 + 该时段完工入库量 = 当期需求（若未报废） + 当期未净库存。

> 注：若 $R_i=1$（报废），则 $d_{i,t'}\,(1-R_i)\equiv0$，需求不计入流平衡。

**(2) 机台时段唯一配置**（每台机台每时段最多开一种配置）：

$$\sum_{u\in U} Y_{umt} \le 1 \quad \forall m\in M,\; t\in T$$

**(3) Setup-开工双向耦合**（开 setup $\iff$ 至少加工一个该配置的物料）：

$$\sum_{i\in I_u} X_{imt} \le |I_u|\,Y_{umt} \quad \forall m\in M,\; u\in U,\; t\in T$$

$$Y_{umt} \le \sum_{i\in I_u} X_{imt} \quad \forall m\in M,\; u\in U,\; t\in T$$

**(4) 单物料 Setup 依赖**（物料 $i$ 开工 $\implies$ 其配置 $u(i)$ 必须开 setup）：

$$X_{imt} \le Y_{u(i),mt} \quad \forall i\in I,\; m\in M,\; t\in T$$

**(5) 跨期定义**（占罐跨期至少为批内所有物料的加工时间）：

$$L_{umt} \ge lt_{it}\,X_{imt} \quad \forall m\in M,\; u\in U,\; i\in I_u,\; t\in T$$

$$L_{umt} \le \bar M\,Y_{umt} \quad \forall m\in M,\; u\in U,\; t\in T$$

**(6) 非重叠约束**（相邻两个 setup 批不允许在时间上重叠）：

$$L_{umt} - \bar M\,(2 - Y_{umt} - Y_{u'mt'}) \le t' - t \quad \forall m\in M,\; u,u'\in U,\; t,t'\in T,\; t'>t$$

**(7) 容量约束**（每时段开批的物料总工装长度 $\le$ 罐体容量）：

$$\sum_{i\in I} v_i\,X_{imt} \le q_m \quad \forall m\in M,\; t\in T$$

**(8) 物料去向**（每件物料要么在可行窗内开工，要么报废）：

$$\sum_{m\in M_i}\sum_{t:\,rel_i\le t\le end_i} X_{imt} + R_i = w_i \quad \forall i\in I$$

**(9) 不可行点固定为零**：

$$X_{imt}=0 \quad \forall (i,m,t) \text{ 满足 } b_{im}=0 \lor t<rel_i \lor t>end_i \lor \tau_{it}>|T|$$

---

## 3. 基于 DW 分解的主问题与子问题

IM 是含多台机台的大规模 MIP。对其应用 Dantzig-Wolfe 分解：将每台机台的"调度方案"（setup + 开工决策 + 跨期决策）视为一个**列变量**，主问题在原物料流平衡和去向约束下对各机的列进行凸组合。

### 3.1 限制主问题（LRMP）

**决策变量：**

$$ \theta_{mk} \ge 0 \quad (k\in\mathcal{K}_m,\; m\in M),\qquad S_{it}^+,S_{it}^-,R_i \ge 0 $$

**目标函数：**

$$\min\ \sum_{i,t} hc_i\,S_{it}^+ + \sum_{i,t} bc_i\,S_{it}^- + \sum_i rc_i\,R_i + \sum_{m\in M}\sum_{k\in\mathcal{K}_m} c_m^k\,\theta_{mk}$$

**约束：**

**(a) 流平衡（列形式）**——将各机各列的完工入库量代入：

$$\sum_{m\in M}\sum_{k\in\mathcal{K}_m} a_{i,t'}^{k}\,\theta_{mk} + S_{i,t'-1}^+ - S_{i,t'-1}^- - S_{i,t'}^+ + S_{i,t'}^- = d_{i,t'}\,(1-R_i) \quad \forall i\in I,\; t'\in T$$

**(b) 物料去向（列形式）**——各机各列的开工量之和 + 报废 = 初始量：

$$\sum_{m\in M}\sum_{k\in\mathcal{K}_m} w_i^k\,\theta_{mk} + R_i = w_i \quad \forall i\in I$$

**(c) 机台凸组合**——每台机台至多选一个调度方案：

$$\sum_{k\in\mathcal{K}_m} \theta_{mk} \le 1 \quad \forall m\in M$$

**对偶变量**（用于列生成定价）：

| 对偶变量 | 符号 | 符号约束 | 对应约束 |
|----------|------|----------|----------|
| 流平衡 | $\alpha_{it}$ | 自由 | 约束 (a) |
| 物料去向 | $\rho_i$ | 自由 | 约束 (b) |
| 机台凸组合 | $\beta_m$ | $\le 0$ | 约束 (c) |

### 3.2 定价子问题（固定机台 $m\in M$）

子问题解决："给定当前 LRMP 对偶解 $( \alpha, \rho, \beta )$，为机台 $m$ 寻找一个约化成本为负的可行调度列 $k$。"

**决策变量（仅单机 $m$）：**

$$X_{imt},\,Y_{umt}\in\{0,1\},\qquad L_{umt}\in\mathbb{Z}_+$$

**约化成本：**$RC = \text{新列成本} - \sum(\text{对偶}\times\text{列系数})$

展开后：

$$RC = \sum_{u\in U}\sum_{t\in T} pc_{mu}\,Y_{umt} - \beta_m - \sum_{i\in I_m}\sum_{t:(i,t)\in\mathcal{F}_{im}} \bigl(\rho_i + \alpha_{i,\tau_{it}}\bigr)\,X_{imt}$$

其中 $\alpha_{i,\tau_{it}}=0$ 当 $\tau_{it}>|T|$（完工超出规划期时流平衡对偶无贡献）。

**合并同类项**：将 $X_{imt}$ 的系数合并为 $\pi_{it}=-(\rho_i+\alpha_{i,\tau_{it}})$，则目标为：

$$RC = \sum_{u\in U}\sum_{t\in T} pc_{mu}\,Y_{umt} - \beta_m + \sum_{i\in I_m}\sum_{t:(i,t)\in\mathcal{F}_{im}} \pi_{it}\,X_{imt}$$

等价于极小化

$$\min\ \sum_{u\in U}\sum_{t\in T} pc_{mu}\,Y_{umt} + \sum_{i\in I_m}\sum_{t:(i,t)\in\mathcal{F}_{im}} \pi_{it}\,X_{imt}$$

（常数项 $-\beta_m$ 在极小化目标中可暂略，最后加回即可。）

**约束：**

$$\sum_{u\in U} Y_{umt} \le 1 \quad \forall t\in T \tag{S1：每时段至多一种配置}$$

$$\sum_{i\in I_u} X_{imt} \le |I_u|\,Y_{umt},\quad Y_{umt} \le \sum_{i\in I_u} X_{imt} \quad \forall u\in U,\; t\in T \tag{S2：Setup-开工耦合}$$

$$X_{imt} \le Y_{u(i),mt} \quad \forall i\in I,\; t\in T \tag{S3：单物料 Setup 依赖}$$

$$L_{umt} \ge lt_{it}\,X_{imt} \quad \forall u\in U,\; i\in I_u,\; t\in T \tag{S4：跨期定义}$$

$$L_{umt} \le \bar M\,Y_{umt} \quad \forall u\in U,\; t\in T \tag{S5：跨期上限}$$

$$L_{umt} - \bar M\,(2 - Y_{umt} - Y_{u'mt'}) \le t' - t \quad \forall u,u'\in U,\; t,t'\in T,\; t'>t \tag{S6：非重叠}$$

$$\sum_{i\in I_m} v_i\,X_{imt} \le q_m \quad \forall t\in T \tag{S7：容量}$$

$$\sum_{t\in T} X_{imt} \le 1 \quad \forall i\in I_m \tag{S8：物料不重复}$$

$$X_{imt}=0 \quad \forall (i,t)\notin\mathcal{F}_{im} \tag{S9：不可行开工点}$$

### 3.3 列生成流程

每轮 CG 迭代：

1. 求解当前 LRMP，得到对偶解 $(\alpha,\rho,\beta)$。
2. 对每台机台 $m\in M$，求解上述子问题 MIP。若某机的 $RC < -\varepsilon$（$\varepsilon$ 为容差），则将其最优解转换为新列 $k$ 加入 $\mathcal{K}_m$。
3. 若所有机台的 $RC \ge -\varepsilon$，列生成收敛，LRMP 已达最优。

---

## 4. 子问题求解方法：EBRP

以上子问题仍是一个 MIP（含二元变量和整数变量），可以用商用求解器（Gurobi）求解，也可以用 EBRP（Exact Batch Routing Problem）精确算法求解。

### 4.1 EBRP 术语

| 术语 | 定义 |
|------|------|
| **批** (batch) | 同一配置 $u$、同一开工时段 $t$、同一占罐跨期 $s$ 下的一组容量可行物料子集，记为 $B$ |
| $\delta(B)$ | 批 $B$ 的约化成本贡献：$\delta(B)=pc_{mu}-\sum_{i\in B}(\rho_i+\alpha_{i,\tau_{it}})$ |
| $\text{span}(B)$ | 批 $B$ 的占罐跨期：$\max_{i\in B}lt_{it}$ |
| $\text{items}(B)$ | 批 $B$ 所含物料子集 |
| **批目录** | 机台 $m$ 上全部 $(u,t)$ 对应的可行批的集合，记为 $\mathcal{B}_{ut}$ |
| $\text{next}(t)$ | $\ge t$ 的下一个存有有效批次的时段（若不存在则为 $|T|$） |

### 4.2 基本思路

子问题 MIP 的最优解可以等价地表述为：在时段轴上选取一批 **时段不重叠、物料不重复** 的批次序列 $\{B_1,B_2,\dots,B_R\}$，极小化总 $\sum_r\delta(B_r)$。

原因：
- 子问题目标 $\sum_u\sum_t pc_{mu}Y_{umt} + \sum_i\sum_t \pi_{it}X_{imt}$ 中，给定一个批 $B$，其总目标贡献恰为 $pc_{mu} + \sum_{i\in B}\pi_{it} = \delta(B)$。
- 跨期约束 (S6) 保证相邻批次不重叠；物料不重复约束 (S8) 保证各物料至多被选入一个批。

因此，EBRP 的核心策略是：

1. **预枚举全部容量可行的"批"**（第一阶段，与对偶无关，每机仅执行一次）。
2. **每轮 CG 迭代将对偶代入，计算每批的 $\delta$，经 Pareto 剪枝形成批目录**（第二阶段）。
3. **DP 选取最优批序列**（第三阶段），极小化 $\sum\delta(B)$。

这种"预枚举 + 动态规划"将通用的 MIP 分枝定界转化为专用组合算法。

### 4.3 第一阶段：静态子集枚举（每机仅首轮执行，与对偶无关）

**输入**：ProcessedInstance（$b_{im}, lt_{it}, v_i, q_m$ 等），机台索引 $m$。

**输出**：缓存 `_subset_cache[(u_idx, t_idx, s_idx)]`，存全部容量可行的物料子集。

**步骤**：

1. 对每 $(u,t)$，收集可行物料 $P_{ut}=\{i\in I_u\cap I_m: (i,t)\in\mathcal{F}_{im}\}$。
2. 提取 $P_{ut}$ 中所有不同跨期 $S_{ut}=\{lt_{it}:i\in P_{ut}\}$，升序排列。
3. 对每个跨期 $s\in S_{ut}$（满足 $t+s-1<|T|$）：
   - 候选池 $\text{pool}=\{i\in P_{ut}:lt_{it}\le s\}$；
   - 执行 **0/1 背包 DP**（物料体积缩放为整数，容量缩放为整数）：穷举所有容量 $\le q_m$ 的子集；
   - 仅保留满足"至少有一个物料的 $lt_{it}=s$"的子集——这保证每个子集的真实跨期恰为 $s$，且每个子集恰好被生成一次。

### 4.4 第二阶段：批目录生成（每轮 CG 迭代执行）

**输入**：当前对偶 $(\alpha,\rho)$，第一阶段缓存的子集列表。

**输出**：按 $t$ 索引的批列表 `batches_by_t[t]`。

**步骤**：

1. 遍历缓存的所有子集，计算每批的 $\delta(B)=pc_{mu}-\sum_{i\in B}(\rho_i+\alpha_{i,\tau_{it}})$。$\alpha$ 超出 $T$ 时取 $0$。
2. 滤除 $\delta(B)\ge 0$ 的批——空闲成本为 $0$，正 $\delta$ 不可能优于空闲。
3. **Pareto 支配剪枝**：按 $(\text{span}, |\text{items}|, \delta)$ 升序排列。若已保留批 $A$ 满足 $A.\text{items}\subseteq B.\text{items}$ 且 $A.\delta\le B.\delta$ 且 $A.\text{span}\le B.\text{span}$，则 $B$ 被 $A$ 支配并丢弃。支配方向为"物料少的支配物料多的"——$A$ 替代 $B$ 后剩余更多物料给后续批使用，永远不劣势。

### 4.5 第三阶段：DP 路由（每轮 CG 迭代执行）

**输入**：`batches_by_t`，时段数 $|T|$。

**输出**：最优总 $\delta$ 值 $V(0,\varnothing)$，子问题约化成本 $RC=V(0,\varnothing)-\beta_m$。

**DP 状态**：$(t,\mathcal{S})$，其中 $t\in[0,|T|]$ 为当前时刻，$\mathcal{S}\subseteq I_m$ 为已用物料集合（用位掩码实现）。

**动作**：
1. **空闲**：$t\to\text{next}(t)$，成本 $0$；
2. **选批**：对 $B\in\bigcup_u\mathcal{B}_{ut}$，若 $\text{items}(B)\cap\mathcal{S}=\varnothing$ 且 $t+\text{span}(B)\le|T|$，则 $t\to t+\text{span}(B)$，$\mathcal{S}\to\mathcal{S}\cup\text{items}(B)$，成本 $\delta(B)$。

**递归**：

$$V(t,\mathcal{S})=
\begin{cases}
0, & t\ge|T|,\\[6pt]
\min\Bigl\{
V(\text{next}(t),\mathcal{S}),\;
\displaystyle\min_{\substack{B\in\bigcup_u\mathcal{B}_{ut}\\ \text{items}(B)\cap\mathcal{S}=\varnothing\\ t+\text{span}(B)\le|T|}}
\bigl(\delta(B)+V(t+\text{span}(B),\mathcal{S}\cup\text{items}(B))\bigr)
\Bigr\}, & t<|T|.
\end{cases}$$

子问题约化成本 $RC=V(0,\varnothing)-\beta_m$。若 $RC < -\varepsilon$（$\varepsilon$ 为容差），则回溯 $\text{memo}[(0,\varnothing)]$ 还原最优批次序列 $\{B_1,\dots,B_R\}$，生成新列。

**不做任何前瞻剪枝**：纯记忆化 DP，仅靠哈希表缓存避免重复展开同一 $(t,\mathcal{S})$。空闲动作通过 $\text{next}(t)$ 跳过无批可用时段。

**加速技巧汇总：**

| 技巧 | 原理 | 安全性 |
|------|------|:--:|
| 滤除 $\delta(B)\ge 0$ 的批 | 空闲成本 $0$，正 $\delta$ 不可能优于空闲 | ✓ |
| Pareto 支配剪枝 | 被支配批可被支配批无损替代 | ✓ |
| 记忆化 $(t,\mathcal{S})$ | 状态无环（$t$ 严格递增，$\mathcal{S}$ 单调扩展），每状态至多展开一次 | ✓ |
| $\text{next}(t)$ 预计算 | 无批时段直接跳过，$O(1)$ 查表 | ✓ |

### 4.6 最优性证明（概要）

1. **解与批次序列的对应**：子问题 MIP 的任一可行解对应一个时段不重叠、物料不重复的批次序列 $\{B_1,\dots,B_R\}$，每批 $B_r$ 由其配置 $u_r$、开工时段 $t_r$、跨期 $s_r$ 和物料子集 $\text{items}(B_r)$ 唯一确定。MIP 中 $\sum_u\sum_t pc_{mu}Y_{umt}+\sum_i\sum_t\pi_{it}X_{imt}$ 的值恰等于 $\sum_r\delta(B_r)$。

2. **目录完备性**：任一可行批次 $B$ 的物料子集必存在于第一阶段静态缓存。证明：设 $s^*=\max_{i\in B}lt_{it}$，则 $B\subseteq\text{pool}_{s^*}$ 且至少包含一个 $lt_{it}=s^*$ 的物料；背包 DP 穷举了 $\text{pool}_{s^*}$ 中全部容量可行的包含该条件的子集。

3. **状态空间完备性**：批次序列 $(B_1,\dots,B_R)$ 对应 DP 中的合法状态转移：
   $$(0,\varnothing)\xrightarrow{B_1}(t_1+s_1,\mathcal{S}_1)\xrightarrow{B_2}\cdots\xrightarrow{B_R}(t_R+s_R,\mathcal{S}_R)$$
   若 $t_r+s_r<t_{r+1}$，DP 通过连续空闲动作（$\text{next}$ 跳转）无成本跨越间隙。

4. **DP 最优性**：递归在每个 $(t,\mathcal{S})$ 穷举空闲和全部互斥可用批并取 $\min$。$t$ 严格递增（$s_r\ge 1$）、$\mathcal{S}$ 单调扩展，状态无环且有限（至多 $|T|\cdot2^{|I_m|}$个）。记忆化保证终止于全局最优。

5. **剪枝安全性**：Pareto 支配中被移除的批可被支配批无损替代（更少物料消耗、更小 $\delta$、不更长的跨期）；$\delta(B)\ge0$ 过滤因空闲动作 $\delta=0$ 而安全；无 TD 下界剪枝，因此不存在因不精确下界导致的漏解。

综上，EBRP 精确求解子问题。

---

## 5. 实验：CG+EBRP vs CG+Gurobi（子问题）

### 5.1 实验设置

- **框架**：列生成。主问题是 LRMP，用 Gurobi 求解。
- **对比对象**：CG 框架下，子问题分别用 **EBRP**（纯版精确算法）和 **Gurobi**（商用 MIP 求解器）求解。
- **算例生成**：使用 `SetConfig("small_cg", T, U, 0, I, M)` 配置，`ITEM_MACHINES_RANGE=(2,4)`（每件物料随机分配到 2~4 台机台）。
- **小/中/大各 3 个实例**，CG 时间限制 600s/实例。
- **内存管理**：每个求解器运行完成后执行 `gc.collect()` 和 `del` 释放内存。

### 5.2 算例规模

| 规模 | $T$ | $I$ | $M$ | $U$ | $I_m$(平均) |
|------|-----|------|------|------|---------------|
| 小 | 18 | 24 | 3 | 4 | ~20 |
| 中 | 24 | 28 | 4 | 4 | ~21-22 |
| 大 | 28 | 30 | 5 | 5 | ~18-19 |

### 5.3 实验结果

| 规模 | # | EBRP | Gurobi（子问题） | 加速比 | LRMP 偏差 |
|------|---|------|-----------------|--------|----------|
| 小 | 1 | 2.6s | 15.3s | **5.8x** | 0.00% |
| 小 | 2 | 2.5s | 16.9s | **6.9x** | 0.00% |
| 小 | 3 | 2.1s | 11.2s | **5.5x** | 0.00% |
| 中 | 1 | 31.0s | 42.1s | **1.4x** | 0.00% |
| 中 | 2 | 66.2s | 46.4s | 0.7x | 0.00% |
| 中 | 3 | 12.5s | 45.1s | **3.6x** | 0.00% |
| 大 | 1 | 39.5s | 73.8s | **1.9x** | 0.00% |
| 大 | 2 | 14.8s | 45.6s | **3.1x** | 0.00% |
| 大 | 3 | 12.1s | 49.2s | **4.1x** | 0.00% |

> 注："LRMP 偏差" = $(|\text{LRMP}_{\text{EBRP}}-\text{LRMP}_{\text{Gurobi}}|/\text{LRMP}_{\text{Gurobi}})\times100\%$。

### 5.4 分析

1. **正确性：100%**。全部 9 个实例的 LRMP 目标值完全一致（偏差 0.00%），所有实例均收敛。证实纯版 EBRP（无 TD 下界剪枝）精确求解子问题。

2. **效率：多数情况下 EBRP 更快**。8/9 实例 EBRP 达到 1.4x–6.9x 加速。Gurobi 子问题求解更稳定（10–74s），EBRP 波动更大（2–66s）。

3. **EBRP 的弱点**：中等规模 #2（$|I_m|\approx22.2$）EBRP 慢了 0.7x。原因是当每机物料数接近 ~22 时，DP 状态 $(t,\mathcal{S})$ 中的 $\mathcal{S}$（物料位掩码）组合数快速增长，记忆化 DP 的展开节点数随之膨胀。EBRP 的 DP 路由是主要瓶颈——当 $|I_m|$ 增大时，状态数呈指数级增长。

4. **Gurobi 的稳定性**：Gurobi 子问题求解时间随规模线性增长，在中大规模上保持 40–75s。其分枝定界对打包+排程问题有一定优势。

5. **EBRP 的优势**：在 $|I_m|\le 20$ 时 EBRP 明显更快（5–7x）。预枚举批目录耗时 $O(2^{|P_{ut}|})$，当候选池不大时完全可控；DP 状态数也相对有限。

6. **结论**：在列生成框架下，EBRP 是一种**精确且通常更快**的子问题求解替代方案，特别适合 $|I_m|$ 在 ~20 以内的场景。对于机台物料数更大的实例，EBRP 的 DP 状态膨胀需要进一步优化（如状态空间的对称性削减、更激进的 Pareto 支配剪枝等）才能保持优势。
