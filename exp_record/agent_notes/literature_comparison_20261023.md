# 文献对比：近一年 VLA 运行时验证 / 修复 / 交接方向 vs 我们的方法

撰写时间：2026-10-23  
检索范围：2025-10 至 2026-10 的 arXiv / CVPR 2026 相关文献  
核心关键词：frozen VLA、verifier-guided action selection、handoff、recovery、action candidate ranking、runtime verification

---

## 一、我们当前方法的定位（2026-10-23 版）

### 核心设计
围绕**冻结 OpenVLA（H1，单步 7 维动作）** 的运行时验证 + 有界修复框架，新增两个轻量神经评分模块，不 finetune VLA，不上完整 world model，不做 action chunk：

| 模块 | 输入 | 输出 | 作用 |
|------|------|------|------|
| **模块 A：RecoveryScorer** (`src/repair/recovery_scorer.py`) | DINOv2 视觉特征(384) + 控制器物理状态(24) + 候选动作(7) | 该动作能带来可靠恢复的概率 [0,1] | 在 VLA 生成的 4 个候选动作（1 greedy + 3 sampled）中选概率最高的执行 |
| **模块 B：HandoffScorer** (`src/repair/handoff_scorer.py`) | DINOv2 视觉特征(384) + 控制器物理状态(24) | 交接后任务能否保持完成的概率 [0,1] | 局部恢复达成后，再判断是否允许把控制权交回主策略 |

### 关键边界
- **不生成动作**：四个候选全部由冻结 VLA 生成，scorer 只打分选哪个，不改动作数值。
- **单步决策**：每次只选下一步执行哪个动作，不是一次性规划多步修复路径。
- **不训练 world model**：没有 JEPA/流式模型，没有 MCTS 仿真搜索，没有 conformal calibration。
- **数据隔离**：训练只给**已执行**的候选打标签（成功=1，失败=0），未执行的候选不打标签，避免反事实标签泄露。

代码已 commit 并 push：`5abb70d`（`main` 分支）。

---

## 二、近一年相关文献清单

| 论文 | 发表/时间 | 核心机制 | 与我们的主要区别 |
|------|-----------|----------|------------------|
| **VEGAS** (CVPR 2026) | CVPR 2026 | 采样一组候选动作，用生成式验证器（MLLM）选最可靠的；通过数据合成训练 verifier | 用 MLLM 作为 verifier，需合成失败数据训练；我们用轻量 MLP，输入视觉+物理+动作，不合成数据 |
| **Action-Draft-and-Verify (ADV)** | arXiv 2603.18091 | 扩散动作专家 draft 多个候选 action chunks，VLM 用 perplexity 评分 rerank | 用 action chunks + VLM 概率（NLL）评分；我们用单步动作 + 轻量 MLP |
| **VeriVLA** | arXiv 2610.21985 (2026-10-15) | Flow-based VLA，两轴触发器，JEPA-based latent world model rank 候选 action chunks | 用 latent world model 预测未来 latent 状态；我们无 world model，无 JEPA |
| **PACT-VLA** | arXiv 2610.28019 (2026-10-20) | 异步推理场景，pre-commit observation 选 chunk，FlowRes 残差评分 | 针对异步推理延迟；我们同步执行，单步动作 |
| **SVA / Look Before You Leap** | arXiv 2607.03751 (2026-10-05) | MCTS 在仿真里探索 VLA 输出分布，蒸馏成 Q-value 模型，选最高 Q-value | 用 MCTS + 仿真蒸馏 Q 模型；我们无 MCTS，无 Q 模型，直接 MLP 打分 |
| **CheckVLA** | arXiv 2607.26789 (2026-07-29) | action-conditioned world model 验证执行，conformal calibration 决定干预时机 | 用 world model + conformal threshold + action chunks；我们无 world model |
| **VLA-Corrector** | arXiv 2609.06508 (2026-09) | stage-aware observable state，Learned Verifier + Prompt Recovery（改 prompt 让 VLA 生成纠正动作） | 改 prompt 诱导 VLA 生成纠正动作；我们选现有候选，不改 prompt |
| **CoRe / Counterfactual Realignment** | arXiv 2608.14822 | training-free，imagined continuation，合成观察回滚，最小物理重排 | 用合成观察 + 物理回滚；我们无回滚，直接选动作 |
| **FLARE** | CVPR 2026 | Retry + Reset，MLLM 监控，训练 Reset skills | 训练额外 Reset 技能；我们不训练额外技能 |
| **HELM** | arXiv 2604.18791 (2026-04) | Episodic Memory + State Verifier + Harness Controller，rollback/replan | 有记忆和回滚；我们无记忆，无回滚 |
| **CommitFlow** | arXiv 2609.21908 (2026-09) | Semantic Commitment Monitor + BoundaryFlow local correction（修改 base action） | 修改动作数值；我们不修改动作，只选候选 |
| **ActFovea** | arXiv 2607.29169 (2026-07) | spatiotemporal visual-action consistency，foveated regions | 视觉动作一致性监控；我们用状态+动作打分 |
| **Diagnosing Semantic Handoff Failures** | arXiv 2607.06256 (2026-07) | 技能组合中的 handoff 失败诊断，VLM verifier | 技能级 handoff；我们原子动作级交接 |

---

## 三、详细对比：我们与主流方向的异同

### 1. 动作候选选择（最贴近我们的模块 A）

**VEGAS (CVPR 2026)**  
- 机制：test-time sample ensemble of candidate actions + generative verifier 选最可靠的。  
- 关键发现：用 MLLM off-the-shelf 作为 verifier 无改进，需 LLM-driven data synthesis 构造 failure curriculum 训练 verifier。  
- 与我们不同：VEGAS 的 verifier 是 MLLM 级别，需合成大量 failure 数据；我们的 RecoveryScorer 是轻量 MLP（384+24+7 → 1），训练数据来自已有 audited 修复轨迹（十几个 episode），不合成数据。  
- 启发：verifier 训练需要 failure-aware 数据，不能只靠成功轨迹；我们目前正样本极少（只有 root4 slot3 一条成功），这与 VEGAS 的教训一致，需警惕过拟合。

**Action-Draft-and-Verify (ADV)**  
- 机制：扩散动作专家 draft 多个 action chunks，VLM 用 perplexity（length-normalized NLL）在一次 forward pass 里 rerank。  
- 与我们不同：ADV 作用于 **action chunks**（多步动作序列），评分器是 VLM 本身（perplexity）；我们作用于 **单步 7 维动作**，评分器是独立轻量 MLP。  
- 启发：如果未来升级到 action chunk，可以参考 ADV 的“单次 forward 评分所有候选”以降低延迟。

**VeriVLA**  
- 机制：Flow-based VLA，两轴触发器检测关键状态，JEPA-based latent world model 预测 action-conditioned 未来 latent 状态和 task progress delta，rank 候选 chunks。  
- 与我们不同：用 latent world model 预测未来，我们不用 world model，只用当前状态+动作打分。  
- 启发：action-conditioned 的未来预测比纯状态分类更能区分候选动作，是模块 A 未来可扩展方向（但目前资源不足）。

**SVA / Look Before You Leap**  
- 机制：MCTS 在仿真里探索 VLA 输出分布，蒸馏成 lightweight Q-value 模型，部署时选最高 uncertainty-regularized Q-value。  
- 与我们不同：SVA 需要仿真 MCTS 收集轨迹并训练 Q 模型；我们不做 MCTS，不训练 Q 模型，直接用 MLP 做二分类概率。  
- 启发：pass@k 研究（pass@1 33% → pass@32 92%）说明 VLA 内部确实包含有能力的行为，选对候选很有价值；这支持了我们“从 VLA 候选里选”的基本假设。

### 2. 验证与交接（最贴近我们的模块 B）

**CheckVLA**  
- 机制：action-conditioned world model 验证执行，conformal calibration 界定不必要干预概率，决定何时介入及重写多少 suffix。  
- 与我们不同：CheckVLA 用 world model + conformal threshold + action chunks；我们用轻量 MLP + 固定阈值 + 单步动作。  
- 启发：conformal calibration 能给出“干预概率上界”的统计保证，我们当前 scorer 阈值 0.5 是经验值，缺乏统计保证，是后续可改进点。

**VLA-Corrector**  
- 机制：Learned Verifier 估计进度和风险，异常时用 stage-conditioned recovery prompt 让同一冻结 VLA 生成纠正动作。  
- 与我们不同：VLA-Corrector **改 prompt** 诱导 VLA 生成新动作；我们**不改 prompt**，只在 VLA 已生成的 4 个候选里选一个。  
- 启发：stage-aware 的失败模式识别（approach/alignment/grasp/transport/placement）可参考，用于改进我们的 trigger 逻辑。

### 3. 恢复与回滚（我们目前不做，但有参考价值）

**CoRe / Counterfactual Realignment**  
- 机制：training-free，imagined continuation（合成观察回滚到可行状态），最小物理重排后返回控制。  
- 与我们不同：CoRe 做物理回滚和重排；我们不回滚，直接选动作继续。  
- 启发：如果当前状态已无法通过选动作恢复，回滚是必要的兜底，但超出当前资源范围。

**HELM**  
- 机制：Episodic Memory + State Verifier + Harness Controller，失败时 rollback 到最近 checkpoint 并 replan。  
- 与我们不同：HELM 有记忆和回滚；我们无记忆，无回滚。  
- 启发：memory-augmented verification 能缓解“状态分类孤立看”的问题，是模块 B 未来可扩展方向。

**FLARE (CVPR 2026)**  
- 机制：Retry（注入扰动示范）+ Reset（MLLM 离线分析 OOD 状态，训练 Reset skills），在线 MLLM 监控。  
- 与我们不同：FLARE 训练额外 Reset 技能；我们不训练额外技能。  
- 启发：训练专门的 recovery policy 效果好，但需要额外数据和训练，与“冻结 VLA、轻量评分”的主线有偏差。

### 4. 动作修正（我们明确不做）

**CommitFlow**  
- 机制：Semantic Commitment Monitor 检测语义承诺违反，BoundaryFlow **修改 base action**（local correction）。  
- 与我们不同：CommitFlow 改动作数值；我们不改动作，只选候选。  
- 启发：局部修正动作能更精准纠正偏差，但打破“冻结 VLA 动作不变”的契约，需谨慎。

**ActFovea**  
- 机制：spatiotemporal visual-action consistency，foveated regions，检测 runtime 风险并构造 candidate observations 验证。  
- 与我们不同：ActFovea 做视觉动作一致性监控；我们做状态+动作打分。  
- 启发：视觉-动作一致性可作为额外的验证信号，并入模块 B 的特征。

---

## 四、方法论定位总结

| 维度 | 我们的方法 (2026-10-23) | 主流近一年文献 |
|------|------------------------|---------------|
| **VLA 是否冻结** | 是，完全冻结 | 绝大多数也是冻结（VEGAS/ADV/VeriVLA/SVA/CheckVLA/VLA-Corrector/CoRe/FLARE/HELM/CommitFlow/ActFovea 均冻结） |
| **是否修改动作数值** | 否，原样执行 | CommitFlow 修改；ADV/VeriVLA/PACT 选 chunk；VLA-Corrector 改 prompt |
| **是否训练 world model** | 否 | VeriVLA (JEPA)、CheckVLA (world model) 训练 |
| **是否用 MCTS/仿真搜索** | 否 | SVA 用 MCTS |
| **是否有 conformal 统计保证** | 否，阈值经验值 | CheckVLA 有 |
| **候选粒度** | 单步 7 维动作 | ADV/VeriVLA/PACT 用 action chunks |
| **训练数据来源** | 已 audited 修复轨迹（已执行候选 only） | VEGAS 用合成 failure 数据；SVA 用 MCTS 仿真轨迹 |
| **交接验证** | 模块 B scorer + 原 fusion gate | 大多文献不专门研究“交接后能否保持完成”，更多关注“选哪个动作”或“是否干预” |

### 一句话定位
> 我们的方法属于 **“frozen VLA + 轻量动作条件评分器 + 交接就绪评分器”** 的 test-time 方案。与 VEGAS/SVA/CheckVLA 等方向同属“在 VLA 生成的候选里做选择”，但更轻量（无 MCTS、无 world model、无 conformal 校准），更适合 UG4 毕设资源约束；与 VLA-Corrector/CommitFlow 等“修改 prompt 或动作”的方向互补，不冲突。

---

## 五、对我们下一步的建议（基于文献对比）

1. **数据瓶颈是最大风险**：VEGAS 明确指出 off-the-shelf verifier 无效，需 failure-aware 数据。我们目前只有十几个 episode，其中成功样本极少（root4 slot3 一条）。在声称“scorer 有效”之前，必须先在独立 root（如 10-19）上验证泛化，不要过拟合。

2. **模块 B 的统计保证缺失**：CheckVLA 用 conformal calibration 界定干预概率，我们当前 handoff_scorer 阈值 0.5 纯经验。如果要 claim 安全性，需加 calibration 或至少在验证集上报告 false accept / false reject。

3. **单步 vs chunk**：ADV/VeriVLA/PACT 都作用于 action chunks，未来如果 VLA 升级到输出多步动作，模块 A 可扩展为 chunk-level scorer（参考 ADV 的单次 forward 批量评分）。

4. **回滚兜底缺失**：CoRe/HELM/FLARE 都有回滚或 Reset 机制。当前框架在预算耗尽后“未经验证交接”，缺乏兜底。下一步可考虑轻量回滚策略，但需先确认 scorer 选动作本身有效。

代码已保存：`git commit 5abb70d`，已 push 到 `origin/main`。
