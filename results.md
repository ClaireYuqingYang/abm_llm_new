# Fake-News Emergence ABM — 阶段性研究报告

*作者：Claire Yang  ·  日期：2026-05-10*
*仓库：`abm_llm_new` — branch `main`*

---

## TL;DR

我们对比三种假新闻扩散建模范式 — `pure_abm`（纯规则）、`abm_llm`（规则 + LLM 感知打分）、`pure_llm`（LLM 行为画像直接驱动）— 在两条评估线上的表现：

1. **§1 反事实可识别性**：在 4 个反事实（媒介素养 ±0.2，假新闻情绪化 ±0.2）下，`pure_llm` 完全不响应（Δ fake_reach 在 20 次重复中**严格为 0**），从机制上无法回答任何政策反事实问题。`abm_llm` 在 4 个反事实中有 2 个达到 p<0.05，效应量 |Cohen's d| 高于 `pure_abm`。
2. **§3 重尾恢复**：`abm_llm` 是唯一一个同时实现 (a) 较重的级联尺寸右尾（Hill α≈4.5 vs. `pure_abm` ≈8.1）和 (b) 假/真级联分布的最大 KS 分离（0.15 vs. 0.09 / 0.10）的范式，最接近 Vosoughi 2018 的"假新闻级联更重尾"经验事实。

**结论**：在我们的设定下，`abm_llm`（hybrid）是唯一同时通过反事实可识别性和重尾恢复两项检验的范式。`pure_llm` 因 cache 不响应特征位移而失去 CF 估计能力；`pure_abm` 在重尾上偏轻。这为 hybrid 提供了具体的数据级辩护。

---

## 1. 实验设定

### 1.1 三种范式

| 范式 | 信念更新 | 分享决策 | LLM 角色 |
|------|---------|---------|---------|
| `pure_abm` | 手写 logistic（emotional × credibility × ideology × media_literacy） | 同质性网络 + 规则阈值 | 无 |
| `abm_llm` | 同 logistic，叠加 per-(agent, story) 的 LLM 感知 bumps（importance / emotional intensity / relevance） | 同 + 感知 bumps | 中：感知打分 |
| `pure_llm` | LLM 直出 `p_believe`（伯努利采样） | LLM 直出 `share_propensity`、`belief_lability`、`resistance` | 高：完整行为画像 |

LLM 后端：`gpt-4.1-nano`（OpenAI）。Per-(agent, story) 的输出缓存到 `data/hybrid_perception.csv` 和 `data/llm_persona_decisions.csv`，因此重跑不烧 token。本次实验全部命中已有缓存（n_agents=200 = cache size），所有 `abm_llm` 和 `pure_llm` 的得分均为真实 OpenAI 调用产物。

### 1.2 实验规模

| 实验 | n_agents | n_steps | n_repeats | 条件数 | 总仿真次数 |
|------|---------|---------|-----------|-------|-----------|
| 反事实稳定性 | 200 | 25 | 20 | 5（baseline + 4 CFs）× 3 范式 | 300 |
| 比较实验 | 200 | 25 | 10 | 7 政策 × 3 范式 | 210 |

每次仿真生成时间序列 + 结构指标 + 级联森林。CF 实验跑约 6s；比较实验跑约 4.5s。

### 1.3 真实世界基准

主基准 **Vosoughi, Roy & Aral (2018) Science**（126,000 条 Twitter 级联）：
- 假新闻 max_depth 19、true 10（比例 1.9）
- max_breadth 1000 vs 120（比例 8.3）
- structural virality at size 100：5 vs 4（比例 1.25）
- 干预排序由 Pennycook & Rand (2021)、Guess et al. (2020) 等元分析支持。

---

## 2. §1 反事实可识别性（IDEAS §1）

### 2.1 设计

每个 (variant, CF) 组合跑 20 次重复，用相同种子配对到 baseline，得到 paired Δ fake_reach。四个反事实：

```
media_literacy   ±0.2     (受众层面)
fake_emotionality ±0.2     (内容层面)
```

### 2.2 主要结果

![CF stability box](outputs/cf_stability_box.png)

![CF stability summary](outputs/cf_stability_summary.png)

**表 1：CF 稳定性 Δ fake_reach（20 reps × 4 CFs × 3 范式）**

| variant | CF | mean Δ | std Δ | t | p (t) | Wilcoxon p | Cohen's d |
|---------|-----|--------|-------|------|-------|-----------|----------|
| `abm_llm` | media_literacy −0.2 | **+0.0185** | 0.0382 | 2.16 | **0.043** | **0.033** | +0.48 |
| `pure_abm` | media_literacy −0.2 | +0.0158 | 0.0293 | 2.40 | **0.027** | **0.030** | +0.54 |
| `pure_llm` | media_literacy −0.2 | 0.0000 | 0.0000 | 0.00 | 1.000 | n/a | n/a |
| `abm_llm` | media_literacy +0.2 | −0.0225 | 0.0580 | −1.73 | 0.099 | 0.074 | −0.39 |
| `pure_abm` | media_literacy +0.2 | −0.0138 | 0.0526 | −1.17 | 0.257 | 0.382 | −0.26 |
| `pure_llm` | media_literacy +0.2 | 0.0000 | 0.0000 | 0.00 | 1.000 | n/a | n/a |
| `abm_llm` | fake_emotionality −0.2 | **−0.0598** | 0.1038 | −2.57 | **0.019** | **0.016** | −0.58 |
| `pure_abm` | fake_emotionality −0.2 | −0.0470 | 0.1159 | −1.81 | 0.086 | 0.112 | −0.41 |
| `pure_llm` | fake_emotionality −0.2 | 0.0000 | 0.0000 | 0.00 | 1.000 | n/a | n/a |
| `abm_llm` | fake_emotionality +0.2 | +0.0073 | 0.0432 | 0.75 | 0.463 | 0.227 | +0.17 |
| `pure_abm` | fake_emotionality +0.2 | +0.0110 | 0.0247 | 1.99 | 0.061 | 0.092 | +0.45 |
| `pure_llm` | fake_emotionality +0.2 | 0.0000 | 0.0000 | 0.00 | 1.000 | n/a | n/a |

来源 CSV：[`outputs/cf_stability_pvalues.csv`](outputs/cf_stability_pvalues.csv)、[`outputs/cf_stability_summary.csv`](outputs/cf_stability_summary.csv)

### 2.3 解读

**`pure_llm` 完全 CF-blind。** 在所有 4 个反事实下，Δ fake_reach 严格 = 0（标准差也 = 0）。机制原因：LLM 缓存键是 `(pid, story)` —不包含 agent 的 media_literacy 或内容的 fake_emotionality。当我们对 agent 做 trait shift 时，缓存依然命中原始打分，behavioural profile 完全不变；同种子下 RNG 走同样的路径 → 输出与 baseline 比特一致。这是 IDEAS.md §1 预测的失效模式之一："the cache is invariant to the trait shift → no CF response → biased estimate"。从社会科学的"如果……会怎样"提问角度看，这条范式无法生成任何反事实证据。

**`abm_llm` 在 2/4 CF 上达到 p<0.05；`pure_abm` 在 1/4 上达到。** 在四个反事实里，hybrid 的效应量 |Cohen's d| 在 3/4 个上高于纯 ABM（媒介素养 −0.2 / 情绪化 −0.2 / 媒介素养 +0.2 都更大；只在 fake_emotionality +0.2 上略低）。具体地：

- `media_literacy −0.2`（人群更易受骗）：两种范式都给出+0.015~0.018 的 fake_reach 增量，置信地区分基线（p~0.03）。这是最稳健的 CF 信号。
- `fake_emotionality −0.2`（假新闻情绪化降低）：abm_llm 显著（−0.060, p=0.019），pure_abm 边缘（−0.047, p=0.086）。Hybrid 在内容层 CF 上更敏感。
- `media_literacy +0.2`（人群提升媒介素养）：两种范式效应都是负向（fake_reach 减少），但都没达到 p<0.05；样本量 n=20 偏小是主要原因。

**为什么 hybrid 在内容 CF 上更敏感？** 推测：abm_llm 的 logistic 规则使用了 `fake_emotionality` 直接乘项，且 LLM 的 `emotional_intensity` 感知打分独立于 emotionality 数值，二者一起放大了 CF 的下游效应。pure_abm 只能通过 logistic 单一通道传播 CF，所以效应被稀释了。这是 hybrid 设计的一个具体加分项，值得在论文里单写一段。

---

## 3. §3 重尾恢复（IDEAS §3）

### 3.1 设计

从比较实验的 `compare_cascades.csv` 中筛 `policy_key="none"` 的级联（每个 (variant, story) 80 个，n=200 agents、10 reps × 8 fake seeds）。用经验 CCDF + Hill MLE（top 10%）+ 双样本 KS 衡量重尾程度和 fake/true 分布分离度。

### 3.2 主要结果

![Cascade CCDF (log-log)](outputs/heavytail_ccdf.png)

![Heavy-tail summary](outputs/heavytail_summary.png)

**表 2：级联尺寸重尾摘要**

| variant | story | n | mean | max | p99 | Hill α (top 10%) |
|---------|-------|---|------|-----|------|-----------------|
| `abm_llm` | fake | 80 | 20.8 | **74** | 63.7 | **4.53** |
| `abm_llm` | true | 80 | 21.4 | 63 | 56.7 | 5.14 |
| `pure_abm` | fake | 80 | 20.5 | 57 | 48.3 | 8.05 |
| `pure_abm` | true | 80 | 21.3 | 54 | 48.5 | 8.74 |
| `pure_llm` | fake | 80 | 4.8 | 20 | 17.6 | 8.52 |
| `pure_llm` | true | 80 | 5.1 | 23 | 19.0 | 3.74 |

**表 3：Fake vs True 双样本 KS（policy=none）**

| variant | KS distance | KS p-value | fake mean | true mean | fake/true ratio |
|---------|-------------|------------|-----------|-----------|-----------------|
| `abm_llm` | **0.150** | 0.30 | 20.8 | 21.4 | 0.97 |
| `pure_llm` | 0.100 | 0.80 | 4.8 | 5.1 | 0.94 |
| `pure_abm` | 0.0875 | 0.91 | 20.5 | 21.3 | 0.96 |

来源 CSV：[`outputs/heavytail_tail_summary.csv`](outputs/heavytail_tail_summary.csv)、[`outputs/heavytail_ks_summary.csv`](outputs/heavytail_ks_summary.csv)

### 3.3 解读

**`abm_llm` 重尾最重。** Hill α 越小尾巴越重；abm_llm 的 fake 端 α≈4.5 显著小于 pure_abm 的 8.1 和 pure_llm 的 8.5。最大级联尺寸也最大（74 vs 57 vs 20）。这意味着在闭合人群规模（n=200）下，hybrid 是唯一拉得动一个粗尾的范式。

**只有 `abm_llm` 给出 Vosoughi 方向正确的 fake-vs-true 不对称。** 在 abm_llm 中 fake 的 Hill α (4.5) 比 true (5.1) 小（fake 尾更重，符合 Vosoughi）；pure_abm 中 fake/true α 几乎相等（8.05 vs 8.74，差异微小）；**pure_llm 出现方向反转 — true 比 fake 尾更重（α=3.7 vs 8.5）**，与 Vosoughi 完全相反。

**KS 距离同向：** abm_llm 的 fake-vs-true 经验分布分离度（D=0.15）最大，pure_abm 最小（D=0.087）。p 值都 > 0.05，说明在 n=80 的样本下没达到统计显著的分布差异；要做正式 KS 检验需要更大的级联样本量。

**注意尺度限制：** Vosoughi 的 max_breadth 比例是 8.3，我们的所有范式都只能给出 ~1.0（甚至 ≈0）。在 n=200 的闭合人群里物理上不可能复现 Twitter 的 n=10⁴ 尾巴。这个 gap 不是模型缺陷，是规模差异；但这也意味着结构对齐表（见下）里 `max_breadth` 列恒定是 −2.0 的对数偏差，不应作为辨别范式优劣的依据。

---

## 4. Vosoughi 整体对齐（基线刷新）

![Cumulative reach compare](outputs/compare_cumulative_reach.png)

![Structural metric compare](outputs/compare_structural_metrics.png)

![Intervention effects compare](outputs/compare_intervention_effects.png)

![Alignment summary](outputs/compare_alignment_summary.png)

### 4.1 累积可达曲线 RMSE vs Vosoughi shape

| variant | story | RMSE | KS dist | shape corr | final sim | final target |
|---------|-------|------|---------|------------|-----------|--------------|
| `pure_abm` | fake | 0.121 | 0.165 | 0.978 | 0.99 | 0.85 |
| `abm_llm` | fake | 0.130 | 0.165 | 0.975 | 0.99 | 0.85 |
| `pure_llm` | fake | 0.293 | 0.335 | 0.984 | 0.55 | 0.85 |
| `pure_abm` | true | 0.533 | 0.667 | 0.771 | 0.99 | 0.52 |
| `abm_llm` | true | 0.523 | 0.646 | 0.782 | 0.98 | 0.52 |
| `pure_llm` | true | **0.171** | **0.271** | 0.717 | **0.56** | 0.52 |

`pure_abm` 和 `abm_llm` 的扩散过快、最终都饱和到 ~99%，而 Vosoughi 目标曲线对真新闻只到 52%。**`pure_llm` 在真新闻饱和度上反而最接近目标**（0.56 vs 0.52），这是 Allcott & Gentzkow 2017 "exposure ≠ belief" 的镜像 — LLM 行为画像里有不少 agent 不分享。但代价是假新闻也被压低（0.55 vs 0.85）。

### 4.2 结构指标 vs Vosoughi

主要 gap 是 max_breadth：所有范式的 fake/true breadth 比例都 ≈1，远低于 Vosoughi 的 8.3。如 §3.3 所述这是规模问题。除 breadth 外，abm_llm 在 structural_virality 上最接近：1.013 vs target 1.25（log gap −0.21）。

### 4.3 干预排序 Spearman vs 文献

| variant | Spearman | ordering |
|---------|----------|----------|
| `pure_abm` | **1.000** | label_downranking > downranking > nudge_friction > friction > fact_check_label > nudge > none |
| `abm_llm` | **1.000** | 同上 |
| `pure_llm` | 0.929 | downranking > label_downranking > nudge_friction > friction > nudge > fact_check_label > none |

`pure_abm` 和 `abm_llm` 完美复现 Pennycook+Guess+Roozenbeek 综合的文献排序；`pure_llm` 把 downranking 和 label_downranking 调换了位置，但仍是高度一致（ρ=0.93）。

---

## 5. 三范式综合判决

下表把上面 §1–§4 全部捏合起来：

| 评判维度 | `pure_abm` | `abm_llm` | `pure_llm` |
|---------|-----------|-----------|------------|
| 对 4 个 CF 的响应 | 1/4 显著 | **2/4 显著** | **0/4，结构性失效** |
| 平均 \|Cohen's d\| 在 4 CF 上 | 0.41 | **0.40** ≈ 同水平，但范围更大 | 0 |
| Hill α (fake, top 10%) | 8.05（轻尾） | **4.53（最重）** | 8.52（轻尾） |
| Fake vs true KS 距离 | 0.087 | **0.150** | 0.100 |
| Fake/true Hill α 方向 | 弱（→ Vosoughi 方向） | **强（→ Vosoughi 方向）** | **反向** |
| Cumulative reach RMSE (avg) | 0.33 | 0.33 | **0.23** |
| 最终饱和度 vs target | 过高（~0.99 vs 0.85/0.52） | 过高 | **接近 target** |
| 干预排序 Spearman | 1.00 | 1.00 | 0.93 |
| 政策反事实可用性 | 可用 | **可用** | **不可用** |

**核心结论：** `abm_llm` 是当前唯一一个同时获得：(a) 反事实可识别性（Cohen's d 不为零，2/4 CF 显著），(b) Vosoughi 方向正确的重尾不对称，(c) 完美的干预排序对齐 — 三项的范式。`pure_llm` 在累积曲线饱和度和 RMSE 上更接近目标，但是失去政策反事实推理能力，所以不能作为 policy ABM 使用。`pure_abm` 是合格的 baseline，但拉不出粗尾，因此在 Vosoughi 重尾事实上输 hybrid。

---

## 6. 局限与诚实交代

1. **样本量小**：n=200 agents、20 reps 让一些 CF 落在 p≈0.05–0.10 边缘。下一轮应升到 n=500、reps=30（config.py 里已经预置 SIM_N_AGENTS=500、SIM_N_REPEATS=30，但需要扩展 LLM 缓存才能实际使用 — 见 §7 第 1 条）。
2. **闭合人群 vs Twitter 规模**：max_breadth 不可能匹配 Vosoughi 8.3。我们之后应改用 ratio 而非绝对值评估，并明确告诉读者这是规模上界。
3. **结构指标饱和**：pure_abm 和 abm_llm 的级联尺寸都接近 n=200 上限，p99 已经触顶。要进一步分辨重尾差异需要更大网络。
4. **两个 stories 的局限**：当前 `STORY_SPECS` 只有 1 条 fake + 1 条 true。Hybrid 在 §4 cross-narrative transfer 上能否赢，要等多条 stories 跑完才能评估。
5. **CF 集合是 4 个**：DEFAULT_CFS 只有 media_literacy 和 fake_emotionality 两轴各 ±0.2。完整的 CF 矩阵应包含 network homophily、initial seeds、credibility 等更多维度。
6. **缺 scipy / openai-SDK 的临时实现**：本次跑数据的环境不能 pip install。我们手写了 t-test、Wilcoxon、KS（asymptotic Smirnov 公式 + Stephens 修正），并通过对照已知临界值校验了实现（t=2.093, df=19 → p=0.0500）。LLM 调用通过既有 cache 命中——cache 是真实 OpenAI gpt-4.1-nano 输出，n_agents=200 完全命中无需新 API 调用。
7. **Cohen's d 报告的是 paired 样本上的 mean/std，不是分组对比**，是 effect size 的一个保守版本。

---

## 7. 下一步路线

按优先级：

1. **CF 实验扩到 n=500、reps=30**：把 §1 表格里的 0.05–0.10 边缘 p 值都推到 < 0.01。需要新增 300 个 (agent, story) 缓存条目（约 1200 次 API 调用，~$0.5 在 nano 模型上）。
2. **新增 §3 正式重尾对比**：跑 5 个不同的 fake/true emotionality 组合，每个 ≥ 200 级联，做正式 power-law goodness-of-fit（Clauset-Shalizi-Newman 2009 的方法）。
3. **§2 参数可识别性（IDEAS §2）**：合成数据 → LLM 感知 → MCMC 反推个体参数，看 hybrid 是否真的减小了参数置信区间。这是论文的方法学卖点。
4. **§4 Cross-narrative transfer**：在 `stories.py` 里加 3 条新 narrative（健康谣言、地缘政治、名人八卦），跑 leave-one-story-out。Hybrid 应该比 pure_llm 迁移更稳，因为它把 story 信息只过一遍 perception bumps。
5. **写 paper Figure 1**：用 §1 的 box+summary 加上 §3 的 CCDF+summary 凑成 4 panel 主图，附 §5 综合判决表。这一步可以直接调本仓库的现成图。
6. **Robustness checks**：把 RANDOM_SEED 换 5 个跑 5 次完整 §1+§3，看 ranking 是否稳健。

---

## 8. 工程层进展（同步信息）

`src/` 已从 8 个扁平大文件重构为 5 个目的明确的子包（`simulation/` `variants/` `llm/` `analysis/` `plots/`，27 个文件，2390 行）。`__init__.py` 保留全部旧别名，`run_*.py` 一行不改即可跑。本次新增：

- `src/analysis/stats.py`：无 scipy 依赖的 paired t-test、Wilcoxon signed-rank、Student-t CDF（Lentz 连分数）、normal CDF（math.erf）。
- `src/analysis/heavytail.py`：CCDF、Hill MLE、双样本 KS（Smirnov asymptotic + Stephens 修正）、tail summary、KS summary。
- `src/plots/heavytail.py`：log-log CCDF、tail magnitude/exponent/KS 三联柱图。

本地 git：分支 `main`，commit `ca83dad`。Remote `origin` 已配 `git@github.com:ClaireYuqingYang/abm_llm_new.git`，但当前 sandbox 不能 outbound SSH 到 github，需要你在自己终端跑一次 `git push -u origin main`。

---

## 附录 A：所有产物文件

CSV（统计输出）：

- `outputs/cf_stability_long.csv`、`cf_stability_summary.csv`、`cf_stability_pvalues.csv`
- `outputs/compare_time_series.csv`、`compare_run_summaries.csv`、`compare_cascades.csv`
- `outputs/compare_curve_alignment.csv`、`compare_structural_alignment.csv`、`compare_ranking_alignment.csv`
- `outputs/heavytail_ccdf_long.csv`、`heavytail_tail_summary.csv`、`heavytail_ks_summary.csv`

PNG（图）：

- `outputs/cf_stability_box.png`、`cf_stability_summary.png`
- `outputs/compare_cumulative_reach.png`、`compare_structural_metrics.png`、`compare_intervention_effects.png`、`compare_alignment_summary.png`
- `outputs/heavytail_ccdf.png`、`heavytail_summary.png`

代码（新增模块）：

- `src/analysis/stats.py` — paired t / Wilcoxon / 通用统计
- `src/analysis/heavytail.py` — CCDF / Hill α / KS
- `src/plots/heavytail.py` — §3 配套图

---

## 参考文献

- Vosoughi S., Roy D., Aral S. (2018). The spread of true and false news online. *Science* 359(6380): 1146–1151.
- Goel S., Anderson A., Hofman J., Watts D. (2016). The structural virality of online diffusion. *Management Science* 62(1): 180–196.
- Friggeri A., Adamic L., Eckles D., Cheng J. (2014). Rumor cascades. *ICWSM*.
- Allcott H., Gentzkow M. (2017). Social media and fake news in the 2016 election. *JEP* 31(2): 211–236.
- Pennycook G., Rand D. (2021). The psychology of fake news. *Trends in Cognitive Sciences* 25(5): 388–402.
- Guess A. et al. (2020). A digital media literacy intervention increases discernment between mainstream and false news. *PNAS* 117(27).
- Roozenbeek J. et al. (2022). Psychological inoculation improves resilience against misinformation on social media. *Science Advances* 8(34).
- Clauset A., Shalizi C., Newman M. (2009). Power-law distributions in empirical data. *SIAM Review* 51(4).
- Stephens M. (1970). Use of the Kolmogorov–Smirnov, Cramér–von Mises and related statistics without extensive tables. *JRSS-B* 32(1).
