# 主要思路与 Major Prompt 记录

这个文件用于持续记录本项目的重要思路、核心 prompt、关键指导和后续改进方向。
之后遇到会影响项目方向、模型设计、实验设计、prompt 策略或论文表达的内容，都优先记录在这里。

## 使用方式

- 当用户给出关键指导、重新定义项目方向、提出模型/实验/prompt 改进时，在这里追加记录。
- 每条记录保持简洁，但要足够具体，方便之后的会话恢复当时的意图。
- 优先记录长期有效的决定、理由、设计原则和可复用 prompt；临时 debug 细节一般不放这里。
- 如果某个 prompt 很重要，尽量保存原文；如果太长，可以保存忠实的浓缩版本。

## 当前项目方向

- 项目主题：围绕虚假新闻感知、传播与政策干预的 agent-based modeling / simulation。
- 当前仓库包含仿真、数据读取、评估、特征选择和可视化相关代码。
- 重要长期指令：随着项目推进，把用户的主要想法、major prompt、关键指导和重要改进持续保存在本文件中。

## 用户关键指导日志

### 2026-04-25

- 创建本文件，作为专门保存用户主要思路、major prompt、关键指导和未来改进的地方。
- 长期指令：当用户给出重要指导或有意义的改进时，记录到这里，方便后续工作继承。
- 当前协作语言偏好：先用中文与用户协作。

### 2026-04-27

- 文件本身也先使用中文结构和中文说明，后续记录优先用中文。

### 2026-05-05

- 用户希望把现有 ABM 结果与"已发表论文中的真实结果"对比，重点研究 **纯 ABM / ABM+LLM / 纯 LLM persona** 三种范式的差异。
- 选定基准：Vosoughi, Roy & Aral (2018) *Science*（cascade depth/breadth/structural virality/size + time-to-reach 曲线 + intervention 排序）。补充参考：Friggeri 2014、Allcott·Gentzkow 2017。
- 决定：模型本身需要重写为三种 variant；LLM variant 跑真 API 但缩小规模 (200 agents × 25 steps × 5 repeats)。
- 新增模块：
  - `src/real_world_benchmarks.py` — Vosoughi 风格统计量与目标曲线
  - `src/llm_personas.py` — 双层 LLM 缓存（perception 与 persona profile）
  - `src/diffusion_variants.py` — 三种 variant + 父指针 cascade 追踪
  - `src/comparative_evaluation.py` — RMSE/KS/Spearman 对齐
  - `src/comparison_plots.py` 与 `run_comparative_experiment.py`
- 关键发现（mock 跑出的 5×7×3 实验结果，真 LLM 待用户在本机重跑）：
  - **干预排序对齐**最强：abm_llm Spearman 1.0；pure_llm 0.96；pure_abm 0.89。混合范式对干预效果排序最忠实。
  - **结构性 ratio 对齐最弱**：三种 variant 的 fake/true 比都在 1 附近，与 Vosoughi 1.25–8.3 差距大；提示当前闭合人群+对称种子下 ABM 难以再现 Twitter 上 false vs true 的不对称。
  - **传播曲线**：三种 variant 都对 true 故事过冲（saturate too fast）；fake 与目标形状相关性 0.93–0.98，pure_llm 略慢、最接近 Vosoughi 曲线。
- Backlog：
  - 让 fake 故事的初始种子更分散、true 故事更集中，看是否能拉开结构性 ratio。
  - 加入 Pennycook & Rand (2021) 的 belief-vs-share 偏差作为另一条对齐维度。
  - 跑真 OpenAI API 后比较 mock vs real LLM 对结构性指标的影响。

### 2026-05-06

- 文件夹清理：把旧 single-variant pipeline 移到 `archive/`；`src/config.py` 砍掉
  stale 字段（USE_REAL_DATA / HF_* / FEATURE_* / FEATURE_SELECTION_* / LR_* /
  PERCEPTION_NOISE）。
- IDEAS.md 落地：4 个 sexy points 写入 `IDEAS.md`，论文主推 #1（反事实稳定性），
  #3（heavy-tail）跟进。
- **Phase 1 实现 — 反事实稳定性测试**：
  - 新增 `src/counterfactual_stability.py` + driver `run_counterfactual_stability.py`。
  - `diffusion_variants.run_variant_experiment` 加 `cf_overrides` 参数，支持
    `media_literacy_shift / fake_emotionality_shift / true_emotionality_shift /
    skepticism_shift / confirmation_bias_shift / platform_trust_shift /
    impulsivity_shift`。
  - Baseline + 4 CF（media_literacy ±0.2, fake_emotionality ±0.2）×
    3 variant × N reps，paired Δ fake_reach。
  - 输出：`outputs/cf_stability_long.csv` /
    `outputs/cf_stability_summary.csv` /
    `outputs/cf_stability_box.png` / `outputs/cf_stability_summary.png`。
- **关键发现 (mock 模式, 150 agents × 18 steps × 8 reps)**：
  - **pure_llm 在所有 4 个 CF 下都返回 EXACT ZERO delta**。原因：LLM persona
    profile 在 baseline 被 cache，cache key 是 (pid, story)，完全不包含 trait 值。
    CF 改变 `agents['media_literacy']` 但 cache 不动 → pure_llm 决策与 baseline
    完全一致 → fake_reach 一致 → delta = 0。
    **这本身就是 IDEAS.md §1 论点的最强版本：pure LLM 给出 biased zero-effect
    CF 估计**（policy 视角下完全无用）。
  - pure_abm 与 abm_llm 都有非零响应，方向基本符合理论
    （media_literacy_-0.2 → fake spread ↑；fake_emotionality_+0.2 → fake spread ↑），
    小 N reps 下噪声偏大但模式清晰。
  - 视觉冲击：boxplot 上 pure_llm 三条全部塌缩到 y=0 一条直线，pure_abm 与
    abm_llm 的箱子有合理宽度并基本对齐。这是论文里 "identifiable CF" 最直接
    的视觉证据。
- 待办：
  - 用 `PERCEPTION_MODE=openai` + `--n_reps 20` 在用户机器上重跑（确认真 LLM
    数据下 pure_abm 与 abm_llm 的 CF 分布形状）。
  - 实现"无 cache 重新调用模式"：pure_llm 每次 CF 重新调 LLM，看 delta variance
    是否爆炸。这能补上 IDEAS.md 中提到的 "LLM 重新生成 persona → 答案波动巨大"
    路径，跟 "biased zero" 一起构成 pure_llm 失败的两端。
  - 增大 N reps（→ 20）以稳定 pure_abm/abm_llm 的均值方向。

## Major Prompt 记录区

当某个 prompt 成为项目核心时，记录在这里。

```text
Prompt 标题：
Prompt 原文：
用途：
备注 / 结果：
```

## 关键设计想法

记录长期有效的建模、实验、评估、可视化、论文写作和项目叙事想法。

## 改进 Backlog

记录重要但暂时还没有实现的改进方向。
