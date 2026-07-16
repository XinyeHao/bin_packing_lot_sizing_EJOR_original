# 数学模型（IM / 主问题 / 子问题）

## 符号

| 符号 | 定义 |
|------|------|
| $T$ | 离散计划时段集合，$t\in T$，$|T|$ 为规划期长度 |
| $I$ | 单件物料集合 |
| $U$ | 固化配置集合 |
| $u(i)\in U$ | 物料 $i$ 的配置 |
| $I_u=\{i\in I:u(i)=u\}$ | 配置 $u$ 可加工物料集 |
| $M$ | 热压罐集合 |
| $M_i\subseteq M$ | 物料 $i$ 允许加工的机台集 |
| $b_{im}\in\{0,1\}$ | $m\in M_i$ 时 $b_{im}=1$ |
| $I_m=\{i\in I:b_{im}=1\}$ | 机台 $m$ 可加工物料集 |
| $rel_i,\,end_i$ | 物料 $i$ 最早 / 最晚开工时段 |
| $lt_{it}$ | 物料 $i$ 在 $t$ 期开工的占罐时段数 |
| $\tau_{it}:=t+lt_{it}-1$ | 物料 $i$ 在 $t$ 期开工的完工时段 |
| $v_i$ | 工装长度 |
| $q_m$ | 热压罐 $m$ 容量 |
| $d_{it}$ | 时段 $t$ 对物料 $i$ 的需求 |
| $w_i$ | 物料 $i$ 初始 WIP 量（单件模型中恒为 $1$） |
| $hc_i,\,bc_i,\,pc_{mu},\,rc_i$ | 持货、缺货、setup、报废单位成本 |
| $\bar M$ | 足够大的常数 |
| $\mathcal{F}_{im}:=\{(i,t): b_{im}=1,\; rel_i\le t\le end_i,\; \tau_{it}\le|T|\}$ | 物料 $i$ 在机台 $m$ 上的全部可行开工时间点 |

**列参数**（列 $k\in\mathcal{K}_m$，固定机台 $m$）：

| 符号 | 定义 |
|------|------|
| $x_{imt}^k\in\{0,1\}$ | 列 $k$ 是否在 $(m,t)$ 开工加工 $i$ |
| $y_{umt}^k\in\{0,1\}$ | 列 $k$ 是否在 $(m,t)$ 对配置 $u$ 开 setup |
| $L_{umt}^k\in\mathbb{Z}_+$ | 列 $k$ 在 $(m,t)$ 上配置 $u$ 的占罐跨期 |
| $c_m^k=\sum_{u,t}pc_{mu}\,y_{umt}^k$ | 列 $k$ 的 setup 成本 |
| $w_i^k=\sum_{t:\,rel_i\le t\le end_i}x_{imt}^k$ | 列 $k$ 对物料 $i$ 的窗内开工量 |
| $a_{i,t'}^{k}=\sum_{t:\,\tau_{it}=t'}x_{imt}^k$ | 列 $k$ 在 $t'$ 对物料 $i$ 的完工入库量 |

**对偶变量**（LRMP）：$\alpha_{it}$（流平衡），$\rho_i$（去向），$\beta_m$（机台凸组合）。

---

## 集成模型（IM）

**变量：**

$$
X_{imt},\,Y_{umt},\,R_i,\,S_{it}^+,\,S_{it}^-\in\{0,1\},\qquad
L_{umt}\in\mathbb{Z}_+
$$

**目标：**

$$
\min\ \sum_{i\in I}\sum_{t\in T} hc_i\,S_{it}^+
+ \sum_{i\in I}\sum_{t\in T} bc_i\,S_{it}^-
+ \sum_{m\in M}\sum_{u\in U}\sum_{t\in T} pc_{mu}\,Y_{umt}
+ \sum_{i\in I} rc_i\,R_i
$$

**约束：**

$$
S_{i,t'-1}^+ - S_{i,t'-1}^-
+ \sum_{m\in M}\sum_{t\in T:\,\tau_{it}=t'} X_{imt}
= d_{i,t'}\,(1-R_i) + S_{i,t'}^+ - S_{i,t'}^-
\quad \forall i\in I,\; t'\in T
$$

$$
\sum_{u\in U} Y_{umt} \le 1
\quad \forall m\in M,\; t\in T
$$

$$
\sum_{i\in I_u} X_{imt} \le |I_u|\,Y_{umt},
\qquad
Y_{umt} \le \sum_{i\in I_u} X_{imt}
\quad \forall m\in M,\; u\in U,\; t\in T
$$

$$
X_{imt} \le Y_{u(i),mt}
\quad \forall i\in I,\; m\in M,\; t\in T
$$

$$
L_{umt} \ge lt_{it}\,X_{imt}
\quad \forall m\in M,\; u\in U,\; i\in I_u,\; t\in T
$$

$$
L_{umt} \le \bar M\,Y_{umt}
\quad \forall m\in M,\; u\in U,\; t\in T
$$

$$
L_{umt} - \bar M\,(2 - Y_{umt} - Y_{u'mt'}) \le t' - t
\quad \forall m\in M,\; u,u'\in U,\; t,t'\in T,\; t'>t
$$

$$
\sum_{i\in I} v_i\,X_{imt} \le q_m
\quad \forall m\in M,\; t\in T
$$

$$
\sum_{m\in M_i}\sum_{t:\,rel_i\le t\le end_i} X_{imt} + R_i = w_i
\quad \forall i\in I
$$

$$
X_{imt} = 0
\quad \forall i\in I,\; m\in M,\; t\in T
\text{ s.t. } b_{im}=0 \;\lor\; t<rel_i \;\lor\; t>end_i \;\lor\; \tau_{it}>|T|
$$

---

## 限制主问题（LRMP）

**变量：**

$$
\theta_{mk}\ge 0 \quad (k\in\mathcal{K}_m,\;m\in M),\qquad
S_{it}^+,\,S_{it}^-,\,R_i \ge 0
$$

**目标：**

$$
\min\ \sum_{i,t} hc_i\,S_{it}^+ + \sum_{i,t} bc_i\,S_{it}^- + \sum_i rc_i\,R_i
+ \sum_{m\in M}\sum_{k\in\mathcal{K}_m} c_m^k\,\theta_{mk}
$$

**约束：**

$$
\sum_{m\in M}\sum_{k\in\mathcal{K}_m} a_{i,t'}^{k}\,\theta_{mk}
+ S_{i,t'-1}^+ - S_{i,t'-1}^- - S_{i,t'}^+ + S_{i,t'}^-
= d_{i,t'}\,(1-R_i)
\quad \forall i\in I,\; t'\in T
$$

$$
\sum_{m\in M}\sum_{k\in\mathcal{K}_m} w_i^k\,\theta_{mk} + R_i = w_i
\quad \forall i\in I
$$

$$
\sum_{k\in\mathcal{K}_m} \theta_{mk} \le 1
\quad \forall m\in M
$$

---

## 定价子问题（固定 $m\in M$）

**变量：**

$$
X_{imt},\,Y_{umt}\in\{0,1\},\qquad L_{umt}\in\mathbb{Z}_+
$$

**约化成本目标（展开式）：**

$$
\min\ \sum_{u\in U}\sum_{t\in T} pc_{mu}\,Y_{umt}
- \beta_m
- \sum_{i\in I}\rho_i \sum_{t:\,rel_i\le t\le end_i} X_{imt}
- \sum_{i\in I}\sum_{t'\in T}\alpha_{i,t'}
\sum_{t\in T:\,\tau_{it}=t'} X_{imt}
$$

**约化成本目标（合并 $X_{imt}$ 同类项）：**

对可行开工点 $(i,t)\in\mathcal{F}_{im}$，定义

$$
\pi_{it} := -\bigl(\rho_i + \alpha_{i,\tau_{it}}\bigr)
$$

则

$$
\min\ \sum_{u\in U}\sum_{t\in T} pc_{mu}\,Y_{umt}
- \beta_m
+ \sum_{i\in I_m}\sum_{t\in T:\,(i,t)\in\mathcal{F}_{im}} \pi_{it}\,X_{imt}
$$

等价地，

$$
\min\ \sum_{u\in U}\sum_{t\in T} pc_{mu}\,Y_{umt}
- \beta_m
- \sum_{i\in I_m}\sum_{t\in T:\,(i,t)\in\mathcal{F}_{im}}
\bigl(\rho_i + \alpha_{i,t+lt_{it}-1}\bigr)\,X_{imt}
$$

其中 $\alpha_{i,t+lt_{it}-1}=0$ 当 $t+lt_{it}-1\notin T$。

**约束：**

$$
\sum_{u\in U} Y_{umt} \le 1
\quad \forall t\in T
$$

$$
\sum_{i\in I_u} X_{imt} \le |I_u|\,Y_{umt},
\qquad
Y_{umt} \le \sum_{i\in I_u} X_{imt}
\quad \forall u\in U,\; t\in T
$$

$$
X_{imt} \le Y_{u(i),mt}
\quad \forall i\in I,\; t\in T
$$

$$
L_{umt} \ge lt_{it}\,X_{imt}
\quad \forall u\in U,\; i\in I_u,\; t\in T
$$

$$
L_{umt} \le \bar M\,Y_{umt}
\quad \forall u\in U,\; t\in T
$$

$$
L_{umt} - \bar M\,(2 - Y_{umt} - Y_{u'mt'}) \le t' - t
\quad \forall u,u'\in U,\; t,t'\in T,\; t'>t
$$

$$
\sum_{i\in I_m} v_i\,X_{imt} \le q_m
\quad \forall t\in T
$$

$$
\sum_{t\in T} X_{imt} \le 1
\quad \forall i\in I_m
$$

$$
X_{imt} = 0
\quad \forall i\in I,\; t\in T
\text{ s.t. } b_{im}=0 \;\lor\; t<rel_i \;\lor\; t>end_i \;\lor\; \tau_{it}>|T|
$$

---

## EBRP 精确子问题求解算法

### 术语与符号

| 术语 / 符号 | 定义 |
|------|------|
| **批** (batch) | 同一配置、同一开工时段、同一占罐跨期下的一组容量可行物料子集 |
| **批目录** (batch catalog) | 机台 $m$ 上所有 $(u,t)$ 对应的全部可行批的集合，记为三维结构 $\mathcal{B}[u][t]=\mathcal{B}_{ut}$。其中每批 $B\in\mathcal{B}_{ut}$ 由其物料子集 $\text{items}(B)$、跨期 $\text{span}(B)$ 及下文定义的约化成本贡献 $\delta(B)$ 刻画 |
| $\mathcal{B}_{ut}$ | 配置 $u$ 在 $t$ 开工的全部可行批次 |
| $\delta(B)$ | 批 $B$ 的约化成本贡献：$\delta(B)=pc_{mu}-\sum_{i\in B}(\rho_i+\alpha_{i,\tau_{it}})$ |
| $\text{span}(B)$ | 批 $B$ 的占罐跨期：$\max_{i\in B}lt_{it}$ |
| $\text{items}(B)$ | 批 $B$ 所含物料子集（$I_m$ 的子集） |
| **TD 下界** (Time-Dependent lower bound) | 从时段 $t$ 出发，忽略物料不重复约束和批次组合效应后，未来剩余时段总 $\delta$ 的一个下界，记作 $\text{TD}[t]$ |
| **DP 路由** (Dynamic Programming routing) | 在时段轴上用记忆化动态规划选择不重叠、物料不重复的批次序列以最小化总 $\sum\delta(B)$ 的过程。DP 状态为 $(t,\mathcal{S})$，$t$ 为当前时刻，$\mathcal{S}$ 为已用物料集 |
| $\mathcal{S}$ | DP 状态中已使用的物料集合（位掩码 $\in\{0,1\}^{|I_m|}$） |
| $\text{next}(t)$ | $\ge t$ 的下一个存有有效批次的时段（T 若不存在） |

### 概述

**基本思路**：定价子问题的目标函数可改写为各批次的约化成本贡献之和（参见上文子问题合并同类项后的目标式）。核心观察是：子问题中 Y_{umt}（setup 决策）仅通过配置 u 的 setup 成本 pc_{mu} 贡献目标，而 X_{imt}（开工决策）通过 \rho_i+\alpha_{i,\tau_{it}} 贡献；两者通过容量约束 q_m 和非重叠约束耦合。EBRP 的基本策略是——

1. **预枚举全部"批"**：将每一组"同时开工、同配置、同跨期、容量可行"的物料子集称为一个"批"。提前穷举所有可能的批，形成一个"批目录"（batch catalog）。
2. **对偶代入计算 δ**：将当前 CG 迭代的对偶值 (\alpha,\rho) 代入，计算每批的净贡献 \delta(B)=pc_{mu}-\sum_{i\in B}(\rho_i+\alpha_{i,\tau_{it}})。
3. **DP 选取最优批次序列**：在时段轴上，用动态规划选择一批不重叠、物料不重复的批次，使总 \sum\delta(B) 最小。该 DP 等价于子问题的精确求解。

这种"预枚举 + 动态规划"策略将子问题从通用 MIP 求解器（如 Gurobi 的分枝定界）转化为专用组合算法，在中小规模下效率更高。由于预枚举的批目录仅依赖容量和跨期等静态参数，第一阶段只需每机台执行一次（首轮 CG 时），后续迭代仅重新计算 δ 值即可。

EBRP 将单机定价子问题分解为四个阶段：

1. **第一阶段：静态结构构建**（每机仅首轮执行，与对偶无关）——穷举全部容量可行批子集，存入静态缓存。
2. **第二阶段：批目录生成**（每轮 CG 迭代执行）——将当前对偶 (\alpha,\rho) 代入静态缓存，计算每批的 \delta(B) 并做支配剪枝，产出批目录。
3. **第三阶段：TD 下界**（每轮 CG 迭代执行）——在批目录上计算松散下界 \text{TD}[t]，用于第四阶段的 DP 剪枝。
4. **第四阶段：DP 路由**（每轮 CG 迭代执行）——记忆化 DP 选取最优批次序列，得出子问题约化成本。

### 第一阶段：静态结构构建（`EBRPContext.build`）

**作用**：为机台 m 穷举全部容量可行的批子集（仅含物料组合，不含 \delta），存入静态缓存。该阶段与对偶值无关，每机仅执行一次。

**输入**：ProcessedInstance（含所有静态参数：b_{im}, lt_{it}, v_i, q_m, \mathcal{F}_{im} 等），机台索引 m。

**输出**：EBRPContext 对象，内含——pool_specs（全部 (u,t,s) 三元组的物料池与锚点）、_subset_cache（按三元组索引的全部容量可行物料子集）、event_times（有批次可用的时段列表）。

**步骤**：

1. 收集可行物料 $P_{ut}=\{i\in I_u\cap I_m: (i,t)\in\mathcal{F}_{im}\}$。
2. 取不同跨期 $S_{ut}=\{lt_{it}:i\in P_{ut}\}$，升序 $s_1<\dots<s_K$。
3. 对每个 $s_k$（满足 $t+s_k-1<|T|$）：
   - **池** $\text{pool}_k=\{i\in P_{ut}:lt_{it}\le s_k\}$；
   - **锚点** $\text{anchors}_k=\{i\in\text{pool}_k:lt_{it}=s_k\}$（若空，跳过）；
   - 整数缩放 $w_i=\text{round}(v_i\cdot Q)$，$C=\lfloor q_m\cdot Q\rfloor$；
   - 对每个锚点 $i^*$，**背包 DP** 枚举所有包含 $i^*$ 且浮点 $\sum_{i\in B}v_i\le q_m$ 的 $B\subseteq\text{pool}_k$：
     $$dp[c]=\{\text{容量恰好为 }c\text{ 且包含 }i^*\text{ 的全部子集}\},\quad c=w_{i^*},\dots,C.$$
     仅保留浮点校验通过的子集，存入静态缓存 `_subset_cache`（与对偶无关）。

**性质 1（目录完备性）** 对任意容量可行批次 $B$（配置 $u$、开工 $t$），设 $s^*=\max_{i\in B}lt_{it}$，若 $B$ 包含至少一个 $lt_{it}=s^*$ 的物料，则静态缓存中存有 $B$ 的物料子集。

*证明*：$B\subseteq\text{pool}_{s^*}$ 且包含锚点物料；锚点启动的背包 DP 穷举了 $\text{pool}_{s^*}$ 的全部容量可行包含该锚点的子集。

### 第二阶段：批目录生成（`EBRPContext.build_catalog`）

**作用**：将当前 CG 迭代的对偶值 (\alpha,\rho) 代入第一阶段缓存的物料子集，计算每批的 \delta(B)，经过去重和 Pareto 支配剪枝后产出完整的批目录 \mathcal{B}。

**输入**：ProcessedInstance，当前对偶 (\alpha,\rho)，第一阶段构建的 EBRPContext（含静态子集缓存）。

**输出**：三维数组 \mathcal{B}[u][t] = \mathcal{B}_{ut}，每项为一个 Batch 对象（包含 items, span, delta, item_mask）。

**步骤**：

$$
\delta(B)=pc_{mu}-\sum_{i\in B}\bigl(\rho_i+\alpha_{i,\tau_{it}}\bigr),
$$

其中 $\tau_{it}>|T|$ 时取 $\alpha_{i,\tau_{it}}=0$。同 $(u,t)$ 内按跨期分组，去重（同物品子集仅留最优 $\delta$），再做 **Pareto 支配剪枝**：若 $\text{items}(A)\subsetneq\text{items}(B)$ 且 $\delta(A)\le\delta(B)$，则 $B$ 被 $A$ 支配并移除——用 $A$ 替代 $B$ 代价不增且释放更多物料。

### 第三阶段：TD 下界（`_precompute_td_bounds`）

**作用**：在批目录上计算从每个时段 t 出发的松弛下界 \text{TD}[t]，用于第四阶段 DP 的安全剪枝。

**输入**：批目录 \mathcal{B}，时段数 |T|。

**输出**：数组 \text{TD}[0,\dots,|T|]，其中 \text{TD}[|T|]=0。

**步骤**：

$$
\begin{aligned}
\text{TD}[t]&=\min\Bigl\{
\text{TD}[t+1],\;
\min_{u:\,\eta_{ut}<\infty}\bigl(\text{TD}[t+s_{ut}^*]+\eta_{ut}\bigr)
\Bigr\},\\
\text{TD}[|T|]&=0.
\end{aligned}
$$

**性质 2（下界可容许性）** $\text{TD}[t]\le V(t,\varnothing)$（$\forall t$）。

*证明*：TD 忽略物料不重复约束（$\eta_{ut}$ 的最优批可任意复用）、忽略批次组合效应（跨期仅用单批近似），松弛使目标不高于真实最优。

### 第四阶段：DP 路由（`_solve_routing`）

**作用**：以记忆化动态规划在时段轴上选取一批不重叠、物料不重复的批次序列，极小化总 \sum\delta(B)，从而得到子问题的约化成本 RC = V(0,\varnothing)-\beta_m。这是 EBRP 的核心求解步骤。

**输入**：批目录 \mathcal{B}，TD 下界数组 \text{TD}[\cdot]，时段数 |T|，禁止的 (u,t) 集合（来自分枝定界固定变量），可选的时限 / 节点数上限。

**输出**：最优批次序列的总 \delta 值 V(0,\varnothing)，以及解码后的 X_{imt}, Y_{umt} 赋值（即新列的物料开工和 setup 决策）。

**状态与动作**：
1. **空闲**：$t\to\text{next}(t)$，成本 $0$；
2. **选批**：对 $B\in\bigcup_u\mathcal{B}_{ut}$，若 $\text{items}(B)\cap\mathcal{S}=\varnothing$ 且 $t+\text{span}(B)\le|T|$，则 $t\to t+\text{span}(B)$，$\mathcal{S}\to\mathcal{S}\cup\text{items}(B)$，成本 $\delta(B)$。

**递归**：

$$
V(t,\mathcal{S})=
\begin{cases}
0, & t\ge|T|,\\[6pt]
\min\Bigl\{
V(\text{next}(t),\mathcal{S}),\;
\displaystyle\min_{\substack{B\in\bigcup_u\mathcal{B}_{ut}\\ \text{items}(B)\cap\mathcal{S}=\varnothing\\ t+\text{span}(B)\le|T|}}
\bigl(\delta(B)+V(t+\text{span}(B),\mathcal{S}\cup\text{items}(B))\bigr)
\Bigr\}, & t<|T|.
\end{cases}
$$

子问题约化成本 $RC=V(0,\varnothing)-\beta_m$。

**加速技巧**：

| 技巧 | 原理 | 安全性 |
|------|------|:--:|
| 滤除 $\delta(B)\ge 0$ 的批 | 空闲成本 $0$，正 $\delta$ 不可能优于空闲 | ✓ |
| TD 剪枝：若 $\delta(B)+\text{TD}[t+\text{span}(B)]\ge\text{best}$，跳过 | 性质 2：$\text{TD}\le$ 剩余最优 | ✓ |
| 记忆化：$(t,\mathcal{S})$ 存表 | $t$ 递增、$\mathcal{S}$ 单调扩展，无环 | ✓ |
| $\text{next}(t)$ 预计算 | 无批时段直接跳过，$O(1)$ 查表 | ✓ |

### Gurobi 交叉验证

当 EBRP 得出 $RC>-10^{-3}$ 且未超时、无固定 setup 时，调用 Gurobi 子问题求解（时限 5s）。若 $RC_{\text{Gurobi}}<RC_{\text{EBRP}}$，以 Gurobi 结果覆盖。这是对 TD 剪枝可能因实现瑕疵导致漏解的最终兜底。

### 最优性证明

**定理** EBRP 输出的 $V(0,\varnothing)-\beta_m$ 等于子问题 IP 的全局最优值。

**证明**：

**1. 解与批次序列的对应**。子问题 IP 的任一可行解对应一个时段不重叠、物料不重复的批次序列 $\{B_1,\dots,B_r\}$，每批 $B_k$ 由配置 $u_k$、开工 $t_k$、跨期 $s_k=\max_{i\in B_k}lt_{i,t_k}$ 的容量可行物料子集构成。序列目标值为 $\sum_k\delta(B_k)$。

**2. 目录完备性**。由性质 1，$B_k$ 的物料子集存在于静态缓存。由 $\delta$ 定义，第二阶段构造的 $\mathcal{B}_{u_kt_k}$ 中存在 $B_k'$ 满足 $\text{items}(B_k')=\text{items}(B_k)$、$\text{span}(B_k')=s_k$、$\delta(B_k')=\delta(B_k)$。DP 动作空间覆盖该可行批。

**3. 状态空间完备性**。批次序列（$t_k+s_k\le t_{k+1}$）对应状态转移：
$$(0,\varnothing)\xrightarrow{B_1}(t_1+s_1,\mathcal{S}_1)\xrightarrow{B_2}\cdots\xrightarrow{B_r}(t_r+s_r,\mathcal{S}_r),$$
累积成本为 $\sum_k\delta(B_k)$。若 $t_k+s_k<t_{k+1}$，DP 通过连续空闲动作（$\text{next}$ 跳转）无成本跨越间隙。

**4. DP 最优性**。递归在每个 $(t,\mathcal{S})$ 穷举空闲和全部互斥可用批并取 min。$t$ 严格递增（$s_k\ge 1$）、$\mathcal{S}$ 单调扩展，状态无环且有限（至多 $|T|\cdot 2^{|I_m|}$）。记忆化保证每状态至多展开一次，DP 在有限步内终止于全局最优。

**5. 剪枝安全性**。Pareto 支配剪枝中，被移除的批可被支配批无损替代；TD 剪枝由性质 2 保证不排除最优分支；$\delta\ge 0$ 过滤因空闲 $\delta=0$ 而安全。

**6. 交叉验证**。Gurobi 兜底保证即使实现瑕疵导致漏解，最终结果不劣于 Gurobi 子问题精确解。

综上，EBRP 精确求解子问题。$\square$
