# IDEAS — 让 ABM+LLM hybrid 真正立住的研究方向

> 维护原则：每条想法保留**why** + **预期表现** + **能立刻跑的最小实验** + **论文里能写成什么 claim**。
> 不要稀释。如果一条想法被实施或被否决，单独标注，不删原文。

---

## 背景：为什么 hybrid 在第一版实验里没赢

1. **只跑了一对故事**。LLM perception 是按 (agent, story) 给分的，但 200 个 agent 都对同一个 fake 故事打分，结果就是一组数往结构化 logit 上加——本质是噪声，不是新信息。
2. **Vosoughi 的基准是 population-level 统计量**（reach、depth、breadth），不奖励 LLM 的强项——LLM 强在个体级语义理解，曲线对齐 / 干预排序这些指标本来就不给它发挥空间。
3. **pure ABM 的特征已经覆盖了 LLM perception 想表达的东西**。我们已经有 ideology、confirmation_bias、emotionality 三个变量，LLM 给出的 importance / emotional_intensity / relevance 跟它们高度共线——hybrid 只是把已知信号重新打包了一次。
4. **"成本/速度"这条卖点在用户场景下站不住**——延迟不是约束。所以 hybrid 必须靠学术维度赢，不能靠 pragmatic 维度赢。

---

## 4 个 sexy points（按论文影响力排序）

### 1. 反事实稳定性 — 这一条最强，建议主推

**问题**：社会科学最关心的是反事实——"如果这个人群媒介素养再高 20%，假新闻还会传得这么远吗？"

**三种 variant 的表现**：
- **纯 ABM**：改一个参数，结果完全可重复（同 seed 下 0 噪声）。
- **ABM+LLM**：改结构化参数，perception 不变 → 结果可重复；改故事，重跑 perception → 1 次 LLM 调用，可控。
- **纯 LLM persona**：改任何一个 trait（比如 media_literacy），LLM 重新生成 persona → 答案波动巨大且不可解释。

**实验设计**：每个 variant 跑 30 次反事实——把所有 agent 的 media_literacy 提高 0.2，记录 fake_reach 变化量。算这 30 次估计的 **方差**。

**预期图表**：箱线图，pure_abm 和 abm_llm 的箱子又窄又对齐，pure_llm 的箱子又宽又偏移。

**论文 claim**：
> Hybrid models are the only LLM-augmented agents that produce identifiable counterfactuals.

这是 LLM agent 文献里**最常被批的一点**（Concordia、Park 2023、Generative Agents 都被指控反事实不稳定），直接打到痛点上。

---

### 2. 参数可识别性 — 跟统计推断挂钩，最适合发 social science 期刊

**问题**：纯 ABM 的参数（confirmation_bias 系数等）能被真实 diffusion 数据反推吗？纯 LLM 的"参数"是 100B 个权重，根本没法推。Hybrid 是唯一在中间的——结构参数可识别，LLM 部分提供语义。

**实验设计**：
1. 用纯 ABM 在已知参数下生成"地面真值" diffusion 数据。
2. 三种 variant 都假装不知道真值，用 Bayesian / least-squares 反推参数。
3. 比较恢复误差。

**预期结果**：pure_abm 完美恢复；abm_llm 恢复相当好（结构参数仍然可识别，LLM 提供的扰动不影响）；pure_llm 根本没法回答这个问题。

**论文 claim**：
> ABM+LLM is the only LLM-augmented social simulator that supports inverse problems—you can fit it to observed Twitter data and read off interpretable parameters.

---

### 3. Heavy-tail 还原 — 视觉冲击力最强

**问题**：真实 misinformation cascade 的 size 分布是幂律（power law，α ≈ 2.5，Vosoughi 直接报告过）。LLM persona 因为 RLHF 后倾向于"模范市民"答案，会塌缩到 mode——这是 LLM agent 文献里另一个未解决的问题（Park et al. 2024 explicitly notes this）。

**预期表现**：
- 纯 ABM：log-normal，尾巴太轻。
- ABM+LLM：LLM perception 给 agent 间引入额外异质性 → 尾巴变重，最接近幂律。
- 纯 LLM persona：尾巴最轻（所有"clone"型 agent 决策相似 → cascade 大小集中）。

**实验设计**：用大网络（500 agents）跑足够多 repeats（50+），收集所有 cascade size，做 log-log Pareto 拟合。比较各 variant 拟合出的幂律指数 α 跟 Vosoughi 的差距。

**图表**：经典的 log-log 累积分布图，三条仿真曲线 + Vosoughi 真实数据。**ABM+LLM 的曲线最贴近真实**——这是论文里一眼能记住的图。

---

### 4. 跨故事泛化 — 最贴近"实用性"，但需要多故事数据

**问题**：训练干预策略时，只能在少数几个真实故事上有数据。能不能泛化到没见过的故事？

**实验设计**：
1. 准备 10 对 narratively 不同的 fake/true 故事（不同议题：选举/疫苗/移民/经济…）。
2. 在故事 1–8 上"标定"每个 variant 的最优干预参数。
3. 在故事 9–10 上 zero-shot 评估。
4. 比较泛化误差。

**预期**：
- 纯 ABM：参数跟故事无关，泛化"不变"但起点低。
- 纯 LLM persona：每个故事重新生成 persona，故事间不连贯。
- **ABM+LLM：LLM perception 提供故事间的语义桥梁，泛化误差最低**。

**论文 claim**：
> ABM+LLM is the only model where calibrated interventions transfer across narratives, because the LLM provides a continuous semantic embedding of "story type" that the structural rule can leverage.

---

## 推荐执行顺序

**Phase 1（1–2 天）— 反事实稳定性（#1）**
- 在 `src/comparative_evaluation.py` 加 `counterfactual_stability_test(...)`。
- 4 个反事实：media_literacy ±0.2、emotionality ±0.2。
- 三个 variant 各跑 20 次，记录每次 CF 估计。
- 出一张箱线图 + 一张数值表。
- 这是 hybrid 唯一可能"碾压式"赢的维度——pure_abm 与 abm_llm 并列第一、pure_llm 远远落后。

**Phase 2（1 天）— Heavy-tail 还原（#3）**
- 网络扩大到 500 agents、repeats 调到 50+。
- 收集所有 cascade size，log-log 散点 + Pareto 拟合。
- 出对比图 + 各 variant 的 α 估计。
- #1 给方法论硬 claim，#3 给现象学硬证据，两条加起来 hybrid 卖点立得稳。

**Phase 3（可选）— 参数可识别性（#2）**
需要写 Bayesian 反推代码，工作量更大，但可以作为 appendix 或下一篇论文的主轴。

**Phase 4（可选，依赖多故事数据）— 跨故事泛化（#4）**
需要先准备 10 对故事的 prompt，是个 prompt engineering + 数据准备活。

---

## 被讨论但暂未实施的产品方向

**交互式仿真 demo**：用户输入"A 公司发生了 [丑闻]"，500 个 agent 自动演化。
- 技术上 ABM+LLM 是唯一合理选择（pure ABM 没法从自然语言起步；pure LLM per-step 太慢）。
- 但用户认为延迟不是约束，所以"快"这条卖点不算 sexy point；该方向暂归为产品 / demo backlog，不作为论文主线。
- 如果未来要做：参考前次会话末尾的 MVP / v0.2 / v0.3 路径。

---

## 引用 anchors

- Vosoughi, Roy & Aral (2018) *Science* — 主基准，cascade depth/breadth/virality/size + 速度比 + 幂律尾。
- Pennycook & Rand (2021) *Trends in Cognitive Sciences* — belief vs share 解耦；nudge 干预理论。
- Allcott & Gentzkow (2017) *JEP* — exposure ≠ belief。
- Friggeri et al. (2014) *ICWSM* — rumor cascade 增长曲线。
- Park et al. (2023) *UIST* — generative agents；以及 Park et al. (2024) 提到 LLM agent 反事实不稳。
- Goel, Anderson, Hofman, Watts (2016) *Management Science* — structural virality 定义。
- Roozenbeek et al. (2022) — inoculation 干预。
- Guess et al. (2020) *PNAS* — 媒介素养干预效果。
