# Core Research Plan (Oct 2026 – Sep 2027)

**Dissertation:** Integrating Neural Physics Engines into Embodied Foundation Models for Enhanced Physical Reasoning and Robust Generalization
**Replaces:** `Experiment_plan_Phase_1.md` (3–4 week validation plan). That plan remains the blueprint for Months 1–3 infrastructure.
**Prior work used:** OrbiSim (Li et al., arXiv 2605.16395), CheckVLA (Liu et al., arXiv 2607.26789), Genesis (`genesis-world`)
**Last updated:** 2026-10-12

## 当前主线：日常生活中的精细接触操作，architecture-first（2026-10-12）

**研究定位：** 在日常 manipulation 的接触阶段，用可验证的物理反馈与时序预测帮助 VLA 安全、精确地完成任务，而不只是避免一次碰撞。主贡献仍为 **differentiable physics verifier / constrained local repair architecture**；单纯扩大 MLP、训练范围或候选数量不作为主贡献。系统优化支持可运行性，解释层支持诊断；真实机械臂仍后置。

**核心问题：** 给定相同 VLA proposals、观测权限和执行机会，什么架构能可靠识别接触阶段的风险，并选择仍能推进任务的局部动作？效果不足可能来自 temporal representation、feedback/state estimation、risk/progress ranking、repair expressiveness 或 control timing，不能预先归因为模型太小，也不能默认物理信息必然带来收益。

### 生活化任务 shortlist（候选，不是已确认 benchmark task IDs）

| 优先级 | 日常任务候选 | 物理反馈应解决的问题 | 可行性与验收需核实 |
|---|---|---|---|
| 第一批候选 | 将抽屉/柜门轻柔关闭并到位 | 末端接触、阻力变化、slam/卡住；完成与低冲击兼顾 | articulation/contact labels、动作接口、可用 OpenVLA baseline；以位置到位与可验证接触风险评估，不凭画面判成功 |
| 第一批候选 | 将杯/瓶准确放入窄间隙杯架或收纳槽 | 对齐、侧向接触、卡住、碰撞后重新调整 | 真实几何 clearance、抓取/放置分阶段成功条件、pose/contact labels；不是普通无约束 pick-and-place |
| 后续候选 | 对齐并压合收纳盒盖 | 多点接触、错位、压入是否完成、过大接触力 | 先核实 lid/contact/compliance 是否真实支持；不宣称气密性或 snap-fit，除非模拟了对应机制 |
| 后续候选 | 按压家电按钮至有效行程 | 接触后行程、过推与反复施力 | button joint、有效 travel 和风险信号；电器功能、热/液体语义不默认存在 |

第一批只选 **一个** 可运行、物理标注可信且 baseline 能进入接触阶段的任务；上表顺序不是任务已落实的承诺。Peg/connector insertion 可作为机制 stress test，但日常任务应承担最终展示。Prepare Coffee / Pan Transfer 等厨房 composite tasks 继续作为后续候选，不用视觉复杂度代替接触机制证据。

### 架构假设与验证路径

1. **Temporal risk representation。** 当前 risk 分支对每步 action 重用相同当前 context，full-H max score 对动作顺序不敏感。Audit 检验它是否漏掉真实未来接触；若证据支持，测试 order-aware/direct chunk-risk prediction，而不是直接增宽 MLP 或重开旧 autoregressive-risk 分支。用同数据重训旧 formulation 作对照，分离 architecture 与 data coverage。
2. **Observed motion/contact history。** 当前仅有前一帧 observation/action，没有 force/torque history 输入。更长 history 或接触反馈为待验证设计，不能写成已有功能。仿真 force labels 不等于真实传感器反馈；机器人 EE pose 用 forward kinematics，object pose/velocity 和接触反馈分别定义 sensing 路径。隐藏 mass/friction 不作为 runtime 输入。
3. **Repair expressiveness。** 当前缩放只能“刹车”，不能主动对齐、侧移或退让。如果离线证实当前有限候选均不能救回、但方向性动作有解，再定义受限 directional/retraction candidates；如果已有好候选但 ranking 错误，先修预测/选择，不盲目增加候选数量。
4. **Task progress 与接触阶段。** Goal distance 只是 proxy，不能自动代表抽屉到位、插入完成或盒盖正确压合。每个任务需要独立验证 pose/stage/completion signals；risk-safe 但无进展、或减速改变后续接触，是不同 failure modes。
5. **Feedback/control timing。** 可研究 `VLA proposal → feedback-conditioned physics verifier → constrained local repair → execute → feedback` 的分工。更快内环不是已实现能力；若改变反馈或控制自由度，所有 comparison arms 必须匹配可用信息和控制机会，并加入 classical feedback/impedance baseline。当前 audit 与已有对比继续保持 H10 / `chunk_k=5`，不通过偷偷增加 verifier replanning 频率制造收益。

**执行顺序：** 完成 frozen validation audit → 输出“failure mechanism / evidence / unresolved alternatives / minimal ablation” → 选择单一架构改进 → matched data/budget/information 验证 → 冻结并独立测试 → 为一个日常精细接触任务训练 simulator-specific predictor、校准、评估。任务/benchmark feasibility investigation 可并行，不能用新场景掩盖当前机制问题。Kernel optimisation 与新 scale sweep 暂不抢占主线。

**评估要求：** SR、CVR、safe success、paired rescue/harm 与置信区间必须一起报告；按任务增补对齐误差/完成行程、接触风险、接触阶段 completion 和 stall。先验证 simulator contact/solver/control-step reliability，再固定 task-native 成功与风险定义，不随意指定未验证的 force/tilt 阈值。Baseline 比较需控制 privileged state/feedback access、candidate budget、training data、执行 cadence 与 inner-controller 差异。缩放 surrogate acceptance 不是 certified safety。

**Benchmark 与边界：** LIBERO / RoboCasa365 / robosuite-MuJoCo 是待核实载体，不预设具备上述几何、接触、液体或热学语义。公开任务与自定义 precision variants 明确分开；实际 IDs、checkpoint/action conventions 和依赖确认后再决定。公开论文 SR 与我们的 safe success 不混表；当前实现不是官方 CheckVLA reproduction，也不是 SEAL semantic verifier。Box push 保留为 audit/ablation testbed，不作为最终生活化展示。

**文档权威性：** 本节及 Week 4 measured results 覆盖下文历史时间表与早期推断。后续章节保留设计沿革；旧 cadence-confounded 数字、未复核 literature claims、计划状态或 hypothetical benchmark 不能作为当前有效性/完成状态证据。实现与实验 artifact 状态以版本化代码和 `exp_record/week4.md` 为准。

## 完整 audit 后的 cause-first gate（2026-10-12）

**状态更新：** `verifier_audit_20261012_a` 已完成全部 20 cells / 80 arm replays，`summary.json` 为 `complete=true`。全部 outcomes、cadence 和执行 prefix risk 检查通过；此前“audit 正运行”段落为当天历史状态。CPU attribution artifact：`src/results/push_torch/audit_attribution_20261012_a/{analysis.json,report.md}`。这不是新的 efficacy 实验，validation headline 不变。

**主要证据（14 OOD cells，448 episodes/arm；以下以 bisection 为主）：**

- **Detection 与可控性并存。** 145 个 unsafe five-step proposals 中触发 59、漏检 86；86 次漏检仅 14 次存在安全 scale，72 次八个候选均 unsafe。129 个实际首次违规 episode 中，107 个在该 decision 没有安全 scale，22 个有；后者分为 14 次漏检和 8 次触发后仍 unsafe。不能把“无安全 scale”写成任意动作均无解，更不能将这类局部计数当作新算法的全局收益上界。
- **H10 不能替代执行标签。** H10 unsafe proposals 为 265，触发 78、漏检 187；95 次漏检存在安全 scale。Step 6–10 是不 replanning 的 counterfactual，不能把 H10 miss 数直接当实际 episode recall。396 次触发中有 337 个原 proposal prefix-safe，其中 318 个 H10 也 safe；prefix-safe lookahead trigger 不自动等于误干预或 causally harmful。
- **Harm 是后续安全损失，而非单纯 stall。** Bisection paired rescue/harm = 8/12；12 个 harms 全部 violation，只有 3 个同时 success loss。12 个首次 execution divergence 都与 baseline state/proposal 一致、原 H10 safe、修复后 prefix safe，但在 5–8 control steps 后首次违规。Discrete/progress 的 9 个 harms 同样全部 safety loss，仅各 2 个同时 success loss。局部风险下降不能证明整条闭环轨迹更安全；当前不能宣称减速是唯一原因。
- **时序表达有明确限制，但尚无净收益因果证据。** Bisection 的 11,186 个顺序探针 score 全部不变，1,838 个真实 H10 safety labels 改变。探针相关且可能 off-policy；用于约束 architecture hypothesis，不作为新增 runtime candidates。
- **不能把主要瓶颈直接归为 progress ranking。** Bisection 的 learned-accepted、GT-prefix-safe 候选中，2,208 个 decision 有 ≥2 个可比选项，只有 9 个的 predicted-progress choice GT regret >1 mm；triggered 子集为 9/297。这是经过 filter/GT safety 条件化后的局部 goal-distance proxy，不证明总体 progress 准确或 task success 可预测。

**当前最小诊断实验：** 已 detached 启动 `intervention_branch_20261012_a`。只用原 validation 的 bisection arm，覆盖其全部 12 harms + 8 rescues（8 cells），在每个 episode 首次 execution-changing intervention 时跳过一次修复，之后仍运行同一个 frozen verifier。所有 32 rows 保留；先做整 batch 未修改 replay control，再验证 branch 前及未选中 rows 一致、H10/k5/microbatch 8 不变。Selection 使用 hindsight outcome 是离线诊断，不是可部署的 skip rule，结果不能写作新方法 SR。日志 `~/scratch/intervention_branch_20261012_a.log`；尚不把 launch 当 completed 证据。

**分支结果后的选择条件：** 如果跳过首个干预能反转 harms，需要先设计能降低不必要干预的 temporal/rollout-conditioned risk ablation，并同时检查 rescues 是否损失；若 harms 不反转，检查后续 interventions/接触轨迹再归因。若某些 case 到当前 decision 已无安全 scale，先用 zero-action / bounded retreat 等离线 witness 或更早 decision branches 区分 finite-action-set 不足与 intervention timing，不能直接上线更大 candidate set。Contact history、directional repair 与 inner-loop 改动暂不合并实施。

**防止数据泄漏：** Audit 的 validation labels 只用于诊断与设计选择，不进入拟合。若 temporal risk ablation 获批，另行生成 disjoint training-seed trajectories/candidates；旧 formulation 与新架构用相同 training-only data、coverage 和预算训练，在独立 calibration 上重新定阈值，再在未用于训练的 validation 上比较。Confirmation 16000+ 与 final-test 46000+ 不用于调参。History length、risk head、repair freedom 与 cadence 逐项隔离。

## Week 4 当前结论与优先级（2026-10-12；覆盖下面启动状态）

**Audit 已启动（本日后续更新）：** `scripts/audit_verifier_validation.sh` 复放原 candidate validation 的全部 20 cells × 32 episodes × 4 arms；不改权重、τ 或 controller，不使用 confirmation/final-test 做选择。记录每个 decision 的 5-step/H10 真实与预测 risk、candidate ranking/progress、same-state repair counterfactual，以及动作顺序探针。逐 episode outcomes、cadence 和实际执行 prefix risk 必须与原结果一致，否则停止。保留 seated/zero-action stages；原因未能唯一确定时继续做局部/整条轨迹 counterfactual，不先认定需要新 head。日志 `~/scratch/verifier_audit_20261012_a.log`，执行记录见 `exp_record/week4.md`。下文“无进程”指此前结果检查时的状态。

- **两组 9 Oct 实验均已完成，两组原训练/eval 进程已退出；现另运行 audit。** Metadata/source/artifact hashes 和所有 arm 的 H10 / `chunk_k=5` / policy cadence 已核实；详见 `exp_record/week4.md`。
- **扩大 confirmation（n=64/cell）：** pooled OOD baseline SR/CVR/safe = 84.71%/31.92%/62.28%；physics decision τ = 82.92%/28.35%/63.50%，safe +1.23 pp，配对 fixed-grid CI [+0.11,+2.46] pp；legacy τ safe = 63.84%，+1.56 pp；GT shadow safe = 77.01%。收益比初期 n=8 小，尚不支持明显且稳定提升的主张。
- **Algorithm validation（n=32/cell）：** baseline safe = 65.63%；bisection/discrete = 64.73%；progress = 65.18%。三者均未建立相对 baseline 的净收益，progress 相对 bisection 仅 +0.45 pp 且 CI 跨零。不能宣称 SEAL-inspired selection 已有效。
- **本周第一步不是扩大 sweep 或立即迁移 verifier。** 优先 replay validation 上首次 violation 前的 decision、false trigger、harm/rescue，使用 simulator 离线标注比较相同候选的 5-step prefix / H10 risk、candidate ranking 和 predicted/true progress。Confirmation 中 254 个违规 episode 有 157 个在 first trigger 前违规或未触发，这不是 decision-level recall，也不足以唯一归因为 risk prediction。
- **若 audit 证实 temporal-risk mismatch，第二步测试 direct chunk-risk head。** 当前 risk 对每步 action 重用相同当前 context，不是真正可靠的未来接触轨迹预测。新 head 使用 context + 完整 candidate chunk 直接预测逐步风险，保留 differentiability；与同候选数据重训的原 risk formulation 做对照，分离数据 coverage 与算法效果。之后单独 ablate longer history conditioning；不能读取真实 mass/friction，也不直接打开旧 autoregressive-risk 分支。
- **Repair 保持可归因。** 暂不把 progress 提升为默认方案；可靠 risk/ranking 建立后再测试 recovery。所有新 score 独立 recalibrate，仅 validation 选参数；final-test seed base 46000 继续保留未使用。LIBERO verifier transfer 仍需稳定且有实际意义的 safe-success 净收益 gate；可另行准备 baseline 可行性。当前只启动诊断 replay，没有启动新训练或算法 efficacy sweep。

## 当前执行更新（2026-10-09）

- **小样本 matched-protocol validation 已完成。** Physics safe success 为 63.4–64.3%，baseline 59.8%，oracle 77.7%；n=8/cell 尚不能证明稳定收益。Decision-only calibration 和 mass2 coverage 暂未带来额外整体提升。当前实现不是完整 CheckVLA reproduction，也未用 SEAL 的 VLM semantic verification。
- **扩大确认已启动：** `scripts/expand_verifier_confirmation.sh`，完整 OpenVLA `llm` H10 DAgger×2、所有 arm `chunk_k=5`、20 cells × 64 episodes/arm，独立 seeds 16000–16019；保留原 predictor，比较 frozen old/new taus，不用 confirmation outcomes 调参。
- **第一轮算法 validation 已启动：** `scripts/validate_candidate_repair.sh`，20 cells × 32 episodes/arm，seeds 26000–26019；固定 predictor/decision taus，对比 bisection、离散候选最小修改、risk-filtered progress selection。候选固定为 8 个 scale；progress 使用 learned rollout 在实际执行第 5 步的预测目标距离，不使用 GT future 或隐藏 mass/friction。没有合格候选时返回原 proposal 并记录 abstention，不称其为 safe fallback。
- **归因边界：** discrete→progress 隔离 selection objective；bisection→discrete 同时改变 search 和 acceptance-failure handling，不能把所有差异归因为 search。Progress state prediction 可能 drift，这轮检验是否有用；不是完整 SEAL。History-conditioned predictor / stall recovery 仍未启用。
- **评估 gate：** 保存逐 episode SR/CVR/safe outcomes、rescue/harm、stall、task cost；报告配对 fixed-grid bootstrap CI。每 cell 原子保存 partial JSON，最终结果显式 `complete=true`，记录 source/artifact SHA-256。Validation 选择完成后冻结参数，再用独立 final-test seeds 46000+。先确认 verifier 稳定净收益，再迁移 LIBERO；kernel optimisation 和真实机械臂后置。
- **运行记录与日志：** 详见 `exp_record/week3.md`；`~/scratch/verifier_confirmation_20261009_a.log`、`~/scratch/candidate_validation_20261009_a.log`。两组使用独立健康 GPU、`setsid nohup` 脱离 shell 运行；启动不等于完成。

## 当前执行优先级（2026-10-08；覆盖下文旧时间表与完成状态）

**当前不做真实机械臂；主线是 verifier 性能 → LIBERO / RoboCasa365 仿真。**

1. **先修 evaluation protocol，而不是先扩大 MLP。** Controller 原实现只为 `none` 和触发后的轨迹缓存 chunk；verifier/oracle 未触发时每步重新查询 policy。旧对比不满足所有 arm `chunk_k=5` 的要求，不能作为 matched-protocol contribution 证据。新协议为 `matched_open_loop_v2`，所有 arm 查询/执行频率一致，日志记录 policy-call counts。
2. **Actor-specific calibration。** 仅采样实际 policy decision 时刻的 safe chunks，完整 `llm` 与 `vis` 分开校准。使用独立 `--calibration_path`，保留原 predictor/taus；结果保存实际路径、SHA-256、H、chunk_k、参数，禁止从 tag 推断 predictor。相关 chunk 样本不自动满足 exchangeability，不能宣称 OOD conformal safety guarantee。
3. **Validation gate。** `scripts/validate_verifier_protocol.sh` 使用 calibration episode seeds 42000+、validation seeds 6000+、完整 OpenVLA `llm` H10 DAgger×2，对比原 predictor/旧 τ、原 predictor/decision τ、mass2 predictor/decision τ。初期 n=8/cell 仅用于诊断，不作为 headline。最终使用未参与选择的初始种子，报告 SR/CVR/safe success 和不确定性。
4. **再做算法改进。** 根据修正协议下的误触发、repair controllability、stall 和任务进度损失分析，分别 ablate risk+progress candidate selection、stall-aware recovery、history-conditioned prediction/calibration；一次只改一个机制。参数仅由 validation 决定，不对 test cell 的真实质量/摩擦分别调 τ；kernel 优化暂后置。
5. **跑通复杂仿真 benchmark。** 优先调查 LIBERO 的真实 task IDs、可用 OpenVLA checkpoint 与控制约定，先完成单任务 baseline rollout，再扩展任务集与独立物理约束评测。RoboCasa365 用于后续厨房/复合任务，需要另外验证 embodiment、checkpoint、资源及任务语义；不能将其论文 SR 搬为 LIBERO safe-success 数字。
6. **迁移必需项。** 新 state/action adapter、MuJoCo 接触风险标注、robot/object history、重新训练 predictor、重新校准；复用的是接口与验证/修复逻辑，不是 box-push MLP 权重。特权状态使用必须明确，并提供 state-access-matched 对照。
7. **最终展示。** 复杂仿真任务的同 seed `OpenVLA alone` / `OpenVLA + learned verifier` 并排 web demo，风险/动作/阶段进度时间线，以及完整测试集的配对收益与损害统计。Box push 仅作机制诊断。Prepare Coffee / Pan Transfer 等为候选目标，不冒充已验证的 LIBERO task。

**验收标准：** pipeline 跑通不等于贡献成立；在一致协议和冻结测试下获得 learned-verifier safe-success 净收益，并在公开机械臂仿真任务上复现，才支持有效性主张。真实机械臂暂不作为当前 gate。

---

> **Revision 2026-09-29 (pending supervisor confirmation):**
> 1. The **headline result** is now closed-loop: *does adding the verifier to a real VLA improve safe task completion?* RQ1 explains that result and RQ3 shows it fits the compute budget.
> 2. New primary metric: **safe success** (task completed **and** no constraint violation during the episode).
> 3. A real VLA moves from a late "transfer check" to the **main RQ2 policy**. Integration and model selection happen in M2 (Nov). BC stays as the cheap policy for large sweeps.
> 4. **Task A uses a box peg** in both simulators (Genesis 1.4 differentiable mode does not detect cylinder–sphere contacts).
> 5. **Compute budget: one RTX A5000 (24 GB)** for VLA + verifier together, to mimic local/edge compute.
> 6. **Task A = box peg pushed by a two-fingertip pusher** (single-point pushing of a box is unstable; see `exp_record/week1.md`, 29 Sep).
> 7. **Public VLA benchmark added (P2b / P5):** a subset of LIBERO tasks with physical perturbations (friction/mass) and force limits, current VLAs with vs. without the verifier. This follows how CheckVLA (RoboCasa365), SAFE (LIBERO, SimplerEnv) and OrbiSim (robosuite Push etc.) are evaluated.
> 8. **Deformable scope (T3) made explicit:** a simple deformable task with a clear strain/force constraint (e.g. lift-and-fold one cloth corner without over-stretching, or press a soft object below a strain limit). Full garment folding with a VLA is out of scope: current folding VLAs rely on large real-robot datasets, and the physics predictor would need full cloth dynamics.

---

## 0. What changes with one year instead of one month

A one-month plan can only run **comparative validation experiments**. A one-year plan should **build something with a clear contribution boundary**. That means:

1. **Research loop, not engineering pipeline.** "Baseline, then VLA, then demo" is an engineering order. The research order is: fix the one question to answer in 12 months → build only the infrastructure that question needs → iterate, with room for failure and change of direction.
2. **Pick one main contribution.** Of the three candidate directions (physical reasoning chain / differentiable verifier / system optimisation), one is the main contribution and the others support it. The three are not treated equally.
3. **Real-world evidence becomes necessary.** A year of simulation-only work invites the objection "self-validation inside a simulator". This plan keeps a small, low-risk real-data validation (offline, no full closed loop).

---

## 1. Choosing the main contribution (decision needed — Month 1, Week 2)

| Direction | What the thesis becomes | Fit with the proposal | 1-year risk | Recommended role |
|---|---|---|---|---|
| **A. Differentiable verifier** (CheckVLA-style, physics predictor inside) | Replacing the visual world model with a differentiable physics predictor as the runtime verifier: *when is its signal reliable, and for which constraints* | High — directly "neural physics engine inside an embodied foundation model" | Medium | **Main contribution (recommended)** |
| **B. System optimisation** (checkpointing, gradient truncation, fast/slow scheduling) | A gradient-transport system for differentiable physics in long-horizon VLA loops | Medium — close to a systems paper, further from "physical understanding" | Medium-low | **Support**: makes A feasible at long horizons; its own chapter |
| **C. Physical reasoning chain** (explicit causal physical-constraint chain in VLA reasoning) | VLA reasoning that explicitly includes physical-constraint causality | Highest ambition | High — three open sub-problems (extracting constraints / organising them into a chain / validating the chain) | **Light support only**: the verifier's trigger trace (observation → constraint → intervention) is treated as a constrained, checkable instance of a physical reasoning chain; no free-form CoT generation |

**Recommendation: A main, B support, C as interpretation layer.** This keeps the proposal's narrative, keeps B's systems depth as a second contribution chapter, and avoids C's open-endedness.

> If you choose **B** as main: the core experiments in §4 become the "workload" and §5 becomes the centre. If you choose **C**: this plan needs a rewrite. Do not start Month 2 until this is decided.

The rest of this document assumes **A main / B support / C light**.

---

## 1a. Revised 30 Sep (start of Week 2) — what the first week of measurements changed

> **历史分析，12 Oct 校正：** 下述旧数字含 legacy cadence confound，不作为当前因果/效率结论。“CheckVLA gradient branch” 指本地 reference ablation，并非官方方法；代码公开性、未来 transfer 与 surrogate safety 均不能由这里的早期判断确认。以 §1b 和 Week 4 audit 为准。

Four things are now settled by data rather than by argument. Three of them contradict earlier parts of this document.

**1. Genesis is demoted from GT engine to optional cross-check.** Two blocking measurements: with the same expert and initial states, SR is 1.00 in torch and 0.06 in Genesis (the peg drifts sideways and goes around the end of the wall, for both box and cylinder, cause still unknown), and it costs 34–52 ms/step against torch's 6–7. Neither is fixable inside the budget we have. **The GT remains the hand-written differentiable torch sim** (`sims/torch_push.py`: box peg, two fingertips, 10-d state, analytic box–point contact). Genesis is retained only as a *transfer* target for P2b: if the predictor transfers to it, that is evidence the learned signal is physics rather than torch-sim artefact. This is a downgrade in ambition and an upgrade in honesty — the earlier plan assumed Genesis would be the differentiable engine inside the loop, and the measurements say it cannot be trusted to be.

**2. The verifier is a distilled surrogate, not the physics engine run in the loop.** This is a real divergence from how A was originally framed ("use Genesis as an analytical world model"). In practice we *train* the risk model: `OrbiSimDynamics` is an MLP ensemble (hidden 256 × E=5, 15000 iterations) and `VisionWM` is a GRU. The engine is only the GT the surrogate is scored against. Consequences to accept rather than hide:
- the contribution becomes *"a cheap differentiable physics surrogate that can be run in the VLA loop, and a characterisation of when its signal is trustworthy"* — which is closer to OrbiSim's own distillation claim than to "run the engine inline";
- the verifier reads true simulator state (friction and mass stay hidden), so it is **privileged**. This must be stated wherever the VLA framing is used, and P2b's LIBERO transfer is what tests whether it survives without privilege;
- the surrogate is 24–30 ms/step against the vision model's 451 ms, so the cheap predictor is also the one that works — a systems result, reported as such.

**3. CheckVLA's gradient branch and hard prefix are both inert on this task, and that is now measured rather than assumed.** `suffix_repair` holds a hard prefix, and the violating contact is the *first* step of the chunk, so it cannot change the one thing that matters: GT risk 1.193 → 1.193, against 0.125 for the bisected down-scale on the same chunks. It nonetheless scores *lower* on the predictor (0.300 vs 0.256, winning on 29% of envs), so the "keep whichever the predictor rates safer" selection picked it and discarded the bisection (1.193 → 0.419). This is what produces the 49% no-op rate in the repair audit. It is **not** the mechanism behind the RQ3 `stab` table, which is a different code path (direct BPTT through the simulator, no prefix, no verifier) and fails for the opposite reason: unregularised BPTT is the *best* of the three there (CVR 0.88 vs 1.00), so over-regularisation is what hurts. Two independent mechanisms, one conclusion — gradients through this contact model are not worth optimising against. Both are now off by default (`--use_grad`, `--hard_prefix`) and kept only as ablations. Measured on friction 0.2/mass 1.5 with 128 envs: bisection 1.073 → 0.105, with the gradient branch 1.073 → 0.333.

**Consequence for the numbers already collected:** the v5 headline table was produced with the gradient branch *enabled* (it was the old unconditional default), so every RQ2/RQ2b/RQ3-sched number for the `checkvla_*` arms describes the contaminated selection. Those arms must be re-run before the table is quoted anywhere. The `none` and `gt_shadow` arms are unaffected.

**4. What we may claim about the headline, and what we may not.** The current claim is *not* "the privileged predictor repairs better" — the repair audit falsifies that: per intervention the vision model lowers the true risk of the chunk it executes by 41.4% against orbisim's 28.9%, with half the no-op rate. The claim is that the two differ in **how conservative they are** (interventions 3721 vs 1394; recall 0.934 vs 0.377; lead 8.8 vs 5.2 steps), and a CVR comparison taken at each predictor's own calibrated tau is **confounded by that**. `rq2_matched_tau.py` exists to remove the confound; until it is in, the orbisim-over-vision gap is suggestive, not established.

---

## 1b. 与相关方法的实现边界（12 Oct 修订）

当前方法不是“官方 CheckVLA 换一个 predictor”。论文机制与本地 stand-in、可证实结果与研究假设必须分开；官方代码公开性未在此完成重新核实，不作为已确认结论。

| 机制 | CheckVLA 论文描述 | 当前本地实现 |
|---|---|---|
| 验证时机 | chunk 执行过程中，以新观测核对预测 | chunk 边界、执行前验证 |
| 风险证据 | prediction/observation deviation 与时序 risk head | simulator-labelled learned physical risk；当前 risk 逐步重用同一 context |
| 校准单位 | nominal-success episode 的首次误干预控制 | policy-decision safe chunk 分数分位数；不构成 OOD safety guarantee |
| 修复 | risk-guided VLA suffix regeneration，保留不可撤回动作 | bisection/scaled-candidate repair；梯度与 hard-prefix 默认关闭 |
| Progress memory | episodic keyframe memory | 未实现 |

SEAL 的 reasoning-conditioned candidates 和 VLM semantic alignment verification 当前也未实现。我们的 scale/progress ablation 仅受 candidate validation 思路启发；内部 VisionWM 不是官方 CheckVLA world model，state-based predictor 享有 privileged information。

**撤回早期强结论：** legacy cadence-confounded 对比不能证明 calibration 是唯一机制、physics 优于 vision、梯度无普遍价值或“忠实复刻”官方修复。Surrogate acceptance 不等于 certified physical safety。当前贡献是假设驱动的 architecture 设计及可靠性诊断；必须由 matched-protocol 独立实验与日常精细接触任务验证，而不是靠旧数字或方法名称支持。

---

## 2. Core research question

> **在日常精细接触操作中，如何设计 feedback-conditioned、order-aware 的物理验证与局部修复架构，使同一 VLA 在接触阶段更安全且仍能完成任务？哪些改善来自时序表示、反馈信息或修复能力，而不是更大的模型、数据或控制预算？**

Three levels, each an upgrade of the Phase-1 RQ:

| RQ | Upgrade of | Question | Falsifiable hypothesis |
|---|---|---|---|
| **RQ1 — Signal quality** | Phase-1 RQ1 | Not "is the prediction accurate" but "under which conditions are the predictor's risk and gradient signals reliable" (horizon, contact regime, OOD distance, Genesis gradient-path validity) | H1: trigger AUROC and gradient agreement fall off sharply beyond a critical horizon *h\** that depends on contact regime. Beyond *h\** the physics predictor is no better than the vision WM. |
| **RQ2 — Closed-loop benefit on a real VLA, per constraint type** | Phase-1 RQ2 | **Headline:** does adding the verifier to a real VLA raise **safe success** (success with no violation) under physical OOD shift, and is the gain due to the physics predictor rather than the CheckVLA scaffolding (vs. the vision-WM arm)? Then: how force / deformation / contact-mode constraints differ in what they need from the trigger (lead time, calibration, repairability) | H2a: VLA + physics verifier > VLA alone and > VLA + vision verifier on safe success in OOD cells. H2b: the physics predictor's advantage over the vision WM is largest for constraints with hidden physical state (force, internal strain), and smallest for visually salient ones (large visible deformation). |
| **RQ3 — System feasibility** | Phase-1 RQ3 | Not "does checkpointing reduce memory" but "within which resource envelope (GPU memory, control latency) the verifier architecture remains usable, and what it degrades to outside it" | H3: with checkpointing + adaptive truncation + async scheduling, verifier quality at horizon *h\** is kept within budget (e.g. <50 ms/step, one consumer GPU), and naive BPTT cannot reach *h\**. |

**目标 headline（尚未建立）：** 同一 OpenVLA 在一个经过验证的日常精细接触仿真任务中，加入 physics verifier/local repair 后的 safe-success 净收益；与 information/control-matched 对照和 classical feedback baseline 比较。Task A OOD grid 是诊断与 ablation 支持证据，不是最终生活化 demo。

**Supporting deliverables:** a *reliability map* (RQ1): signal quality as a function of (horizon × constraint type × contact regime × OOD shift), which explains where and why the headline gain holds; plus the system (RQ3) that makes the usable region reachable within the single-GPU budget.

A verifier can only fix failures with a *physical* cause. Before the main VLA runs, check that the base VLA's failures in the OOD cells are mainly physical (overshoot, excess force) and not perception or grasping errors.

---

## 3. Timeline overview

Core experiments and system optimisation run **in parallel, not in series**. System limits decide which experiment designs are feasible, and experiment needs decide what to optimise.

| Phase | Months | Calendar | Main task | Milestone / gate |
|---|---|---|---|---|
| **P1 Infrastructure** | 1–3 | Oct–Dec 2026 | Real components in place of stand-ins (especially a real VLA policy), gradient path audit, end-to-end predict → trigger → repair. **GT stays the hand-written differentiable torch sim; Genesis is demoted to an optional cross-check** (see §1a) | **G1 (end Dec):** end-to-end loop runs with a real VLA policy as the actor |
| **P2a Releasable physics benchmark — PLANNED** | 2–4 | Nov 2026 – Jan 2027 | Freeze Task A into a runnable protocol with the actor as an argument; ≥2 actors on the same verifier | **G1b (end Feb):** third party reproduces in ≤2 h on one A5000, or demoted to internal protocol |
| **P2 Core experiments — PLANNED; initial push tests only** | 4–7 | Jan–Apr 2027 | Vary horizon, task complexity, constraint type; measure signal quality & trigger effectiveness | **G2 (end Mar):** first reliability map for Task A; *h\** located or H1 rejected |
| **P3 System optimisation — DESIGNED** | 6–9 (parallel) | Mar–Jun 2027 | Checkpointing / truncation / scheduling ablations at the horizons P2 needs | **G3 (end Jun):** usable-envelope result (H3) |
| **P2b Public VLA benchmark (LIBERO) — NEXT** | 7–10 (parallel) | Apr–Jul 2027 | LIBERO subset + physical perturbations and force limits; predictor retrained on robosuite state; current VLAs with vs. without verifier | **G2b (end Jul):** with/without-verifier results on the LIBERO subset, or a documented reason it could not transfer |
| **P4 Real-data validation — PENDING HARDWARE** | 8–10 | May–Jul 2027 | Offline trigger tests on real robot trajectories | **G4 (end Jul):** signal meaningful on real data, or a documented sim-to-real gap |
| **P5 Converge & Write — PLANNED** | 10–12 | Jul–Sep 2027 | Complementary experiments, figures, writing | Full thesis draft end Aug; final Sep |

Buffer: about 1 month total, spread across P2/P3 overlap. If a gate fails, see §8.

---

## 4. Phase details

### P1 — Infrastructure (Months 1–3)

Starting point: the `src/` framework (plug-in interfaces, stand-ins, Genesis backend written but never run). See `src/README.md` for component status.

| Month | Work | Output |
|---|---|---|
| **M1** | ~~Run Genesis on the server~~, Franka demo recording. **Decide main contribution (§1).** Literature check: is OrbiSim / CheckVLA code public? **Task A → box peg in the torch GT; Genesis ↔ torch parity.** | Decision recorded; box Task A in the torch GT. **Parity failed (30 Sep) → Genesis demoted, see §1a.1** |
| **M2** | Replace stand-ins: real OrbiSim-Dynamics via `adapters/orbisim_official.py` (or, if unavailable, a documented re-implementation following the paper — named "OrbiSim-style", not "OrbiSim"). Real/official CheckVLA logic if public; otherwise the reference verifier with its design written up (add event-driven keyframe banks; align hard prefixing with the paper). Risk head for the predictor (route a or b in the adapter). **VLA selection and integration:** short-list 2–3 chunk-output VLAs, measure memory and ms/step on one A5000, pick one; wire Genesis camera (`render_rgb`) and the Franka action mapping; collect Genesis Task A demos and fine-tune; baseline VLA (no verifier) on the OOD grid, with its failure causes classified. | Real components behind the interfaces; chosen VLA running in the loop with baseline OOD numbers |
| **M3** | **Gradient path audit on Genesis** for Task A (rigid) and a Genesis-native deformable task (MPM/FEM, since PBD cloth is not differentiable). End-to-end loop on Genesis. Minimal checkpointing prototype started (the P3 system track starts early, as in the Phase-1 review). | Audit map (Fig B2); **G1** |

Base policy decision (revised 2026-09-29): the **real VLA is the policy for the headline RQ2 runs** (with vs. without verifier, safe success). The BC proxy (behaviour cloning: a small network trained by supervised learning to imitate the scripted expert's action chunks; a VLA is the same idea at large scale with images and language) is kept for the large sweeps (RQ1 horizon × constraint × OOD, RQ3 ablations), where running thousands of VLA episodes is too slow on one A5000. Every BC-based conclusion used to explain the headline should be spot-checked on the VLA.

### P2 — Core experiments (Months 4–7)

A factorial design over the axes the RQs name. Every cell compares **physics predictor vs. vision WM vs. hybrid** inside the same verifier.

| Axis | Levels | Notes |
|---|---|---|
| Prediction horizon *h* | 5, 10, 20, 40, 80 steps | locate *h\** (H1) |
| Constraint type | contact force · deformation/strain · contact-mode change (slip/stick, making/breaking contact) | H2 |
| Task complexity | T1 rigid push-insertion (box peg, two-fingertip pusher) → T2 multi-contact rigid (e.g. peg-in-hole with tight clearance) → T3 simple deformable task with a strain/force limit (Genesis MPM/FEM; e.g. lift-and-fold one cloth corner, or press a soft object) | increasing contact richness; T3 is deliberately not full garment folding |
| OOD shift | friction, mass, stiffness grids (from Phase 1), plus distance-to-training-range | calibration under shift |
| Predictor | physics (OrbiSim-Dynamics) · vision WM · hybrid (physics state + visual residual) | the hybrid is optional, added if the first two are close |
| Policy | VLA (headline arms: none / vision verifier / physics verifier) · BC (full sweep) | VLA on the informative cells only if one A5000 cannot cover the full grid |

Measurements for every cell:
- **Signal quality:** risk calibration (reliability curve, ECE), AUROC, gradient agreement with Genesis (gradient-valid regions only), fidelity decay vs horizon
- **Trigger effectiveness:** precision/recall, timely recall, lead time, repair success (does the repaired suffix avoid the violation in Genesis), **safe success** (primary), SR and CVR under closed loop. Reporting SR and CVR next to safe success shows whether the verifier helps or only makes the policy more cautious.
- **Cost:** predictor ms/step, memory

Order: M4 T1 all axes → M5 T2 → M6–7 T3. After **G2 (end Mar)**, T2/T3 cover only the informative axis levels, pruned by what T1 showed.

**Long-horizon behaviour of the repair (added 1 Oct).** Not yet measured; the current task is single-stage, 80 steps, and every result is SR/CVR at that fixed length. Two things to keep apart:
- *Task length is not prediction length.* The verifier re-anchors on the true state every chunk and predicts only `chunk_k` steps ahead, so prediction error does not compound over the task. Fidelity decay (H1, *h\**) is a property of the **chunk length**, which is what the horizon axis above sweeps.
- *What does compound is the repair.* Each intervention only shrinks the chunk; over many chunks the delays add up. At a fixed time budget that turns into lost SR (over-damping, already seen at mis-calibrated τ: vision at rate 0.19 drives CVR to 0.01 at SR 0.05).

Planned measurements:
1. **Multi-stage task** (T1-long: push through 2–3 seats in sequence, T = 200–400) with **time-to-success** and cumulative lost progress Σ(1 − scale)·|a| reported alongside SR/CVR, with and without the verifier.
2. **Stall detector:** after k consecutive shrinking repairs, escalate (best-of-N candidate or re-query the policy) instead of shrinking again.
3. **Risk + progress selection** over best-of-N candidates (see `exp_record/week2.md`, SEAL note), so a direction change is available and standing still is never the optimum.

### P3 — System optimisation (Months 6–9, parallel with P2)

The three components already exist in `src/systems/` as prototypes. P3 turns them into a studied system:

| Component | Research question | Ablation |
|---|---|---|
| Truncated checkpointing + CPU offload | memory vs horizon vs recompute cost; best segment size per task | naive / GPU ckpt / offload / offload+truncation, horizons up to 400 steps |
| Adaptive gradient truncation | does relaxation/clipping in chaotic contact windows improve repair quality, or only gradient variance? | none / clip / smooth-relax, per contact regime from the audit |
| Fast/slow async scheduling | how stale can slow-engine (Genesis) verdicts be before they stop helping? | fast-only / async (varying slow latency) / sync |

Output: the **usable-resource envelope** (Fig: verifier quality vs memory budget × latency budget), answering H3.

### P2b — Public VLA benchmark (Months 7–10, parallel with P3/P4)

**Why:** reviewers expect results on a public benchmark with existing VLAs, as in related work:

| Paper | Benchmark | Policies | Reported |
|---|---|---|---|
| CheckVLA (2607.26789) | RoboCasa365 (sim, 365 household mobile-manipulation tasks) | several VLAs under one training recipe | success rate vs. periodic replanning; timely recall at a matched 5% false-alarm rate; sim only |
| SAFE (NeurIPS 2025) | LIBERO, SimplerEnv, plus real robot | OpenVLA, π0-FAST, … | failure-detection ROC-AUC, detection time |
| OrbiSim (2605.16395) | robosuite Push, Isaac Lab Stack, AdaManip, Physion Drape | own policies | prediction fidelity, RL success |

**Problem:** public benchmarks do not test physical constraints. Most failures are grasping or placement errors, which a physics verifier cannot fix.

**Plan:** "LIBERO + physics":
1. Choose a LIBERO subset where contact matters (pushing, placing, inserting). LIBERO runs on robosuite/MuJoCo, which exposes contact forces and lets us vary friction and mass.
2. Add physical OOD perturbations (friction, mass) and per-task force limits; define safe success as in §4.
3. Retrain the physics predictor on robosuite object state (same architecture as for Genesis; the vision WM likewise), and recalibrate tau per policy.
4. Run 1–2 current VLAs that fit one A5000 (from the M2 short-list; e.g. OpenVLA-OFT or π0-FAST): none / vision verifier / physics verifier.
5. Report safe success, SR, CVR, and the detection metrics used by CheckVLA/SAFE (timely recall at matched false-alarm rate, AUROC), so the numbers can be compared.

**Why LIBERO rather than RoboCasa365:** LIBERO is the most common VLA benchmark, uses a fixed-base Franka (as our demo), and is lighter to run on one GPU. RoboCasa365 is mobile manipulation and heavier.

**Prerequisites:** the chosen VLA runs in the loop (G1) and the Task A results have located where the physics signal is reliable (G2). If robosuite transfer fails, report the Task A results as the main evidence and the benchmark attempt as a limitation.

### P2a — Releasable physics-constraint benchmark (Months 2–4, **before** P2b, not after)

Introduced 30 Sep. The problem with going straight to P2b is that it couples three unknowns — a VLA we have never run, a new simulator, and a new benchmark — so a failure in any of them costs a semester and the failure is diagnosed slowly. P2a removes the two unknowns we control.

**What it is:** the Task A setting, frozen into a benchmark protocol that others can run, with the actor treated as a variable rather than part of the task. Released as `physicsbench-push` (or similar): seeded initial states, the published friction × mass perturbation grid, fixed episode length and success/violation predicates, and an eval harness taking a policy as an argument.

**Why it is a contribution and not just our testbed:** CheckVLA evaluates on RoboCasa365, which contains **no physical constraints** — its failures are grasping and placement errors that no physics verifier can address. There is therefore no public benchmark on which a physics verifier can be shown to help. Producing one, together with the honest split of which failures a physics verifier *can* and *cannot* fix, is a standalone contribution and it is the one most likely to be cited.

**Actor ladder** (each row is an arm, same protocol, same verifier — the verifier is never retrained per actor):
1. scripted expert (sanity: verifier must not break a correct policy)
2. `bc` (current stand-in — the row every later arm is compared against)
3. small chunk-output VLA, run open-loop over `chunk_k` like every other arm
4. the VLA used in P2b, if time permits

**Numbers to produce:** safe success / SR / CVR per actor × per perturbation cell, plus the detection metrics (AUROC, timely recall at matched false-alarm rate) and the resource cost. The interesting result is the *interaction*: the same verifier helps a weak actor more than a strong one, because a strong actor's failures are less often physical. That is a claim about verifier scope, and it is only visible with ≥2 actors — which is why a single-actor testbed could never have shown it.

**Decision gate G1b (end Feb 2027):** if the harness is not runnable by others (documented, one-command, ≤2 h on one A5000), it stays an internal protocol and P2b carries the external-validity burden alone.

**Prerequisites:** none beyond what already runs. This is the cheapest credible external artifact in the plan.

**Camera status (30 Sep, done).** The blocker for actors 3–4 was that `Env.render_rgb` raised `NotImplementedError` for the torch backend, so the VLA had no input and Genesis — the *optional* cross-check — could not be made to carry it. `sims/camera.py` is now a batched analytic ray-cast renderer on the torch GT: a table plane plus five yaw-oriented boxes (peg, two fingertips, gripper body, end-stop wall), Lambert + Blinn-Phong, per-env randomised appearance. It is a **pure renderer** — reads the state tensor, returns pixels, never writes `self.state` and never calls `sim.step` — so no result already computed with `--backend torch` is affected by its presence. The wall and the printed seat are in the scene on purpose: the thesis is about the contact-force constraint at the end-stop, so a camera that hid the wall could not support the claim.

Whether the camera is *worth having* is measured, not asserted (`src/tests/test_camera_observability.py`): a ridge from pixels to each task variable, fitted on 2048 states and scored on 1024 unseen. Held-out R² **0.974** for `bx`, **0.979** for `by` and `ty`, and **0.974** for the gap `x_w − bx` — the clearance the constraint actually depends on. Yaw is 0.40, as it should be, since a square is symmetric under 90°. The same test asserts the complementary fact, which is the more useful one for the argument: **friction and mass are invisible** (R² −0.087, −0.070). They are not in the state tensor, so they cannot appear in the image, and a visual policy therefore cannot substitute for the privileged physics signal. That is a claim about the *design*, and it is now falsifiable.

Cost, 64 envs at 224×224 on one A5000: **42 ms** at `ss=1`, 146 ms at `ss=2`, so `ss` defaults to 1 — supersampling is 3.5× for smoother edges, and 146 ms is 29 ms/step even amortised over a 5-step chunk, i.e. more than half the 50 ms budget spent on antialiasing. The frame is cached per chunk (the same reason the policy is called once per chunk), which `src/tests/test_rgb_cache.py` verifies is action-identical to rendering every step, at 5.0× fewer rendered rows.

### P4 — Real-data validation (Months 8–10)

Minimal and offline — **no full VLA closed loop on hardware**:
1. Get real trajectories with physical measurements (contact force from a wrist F/T sensor; deformation from vision/markers where possible). Options, in order of preference: (a) a lab robot at Edinburgh, if access can be arranged — **ask the supervisor in M1**; (b) public manipulation datasets that include force/torque — candidates to be identified and checked in M1.
2. Run the verifiers offline on these trajectories: does the physics predictor's trigger fire before measured force/deformation limit crossings?
3. Report the same RQ1/RQ2 metrics on real data, and the sim-to-real gap relative to Genesis.

If no real data with force labels can be obtained by M6, P4 shrinks to a **sim-to-sim transfer** study (train in the torch GT, test in Genesis, or across Genesis solver settings), and this is stated as a limitation.

### P5 — Convergence & writing (Months 10–12)

- Complementary experiments requested by the gates and supervisor feedback
- Final VLA runs: headline with/without-verifier comparison on the chosen VLA over the Task A OOD grid (building on the P2 results), plus the final LIBERO + physics numbers (P2b)
- Demo: Franka arm video in Genesis (`src/demo/`), normal vs OOD vs OOD + verifier
- Writing: chapter plan in §7

---

## 5. Metrics & figures (thesis-level)

| Figure | Content | RQ |
|---|---|---|
| F1 | System diagram: policy → predictor → verifier → repair; fast/slow engines | — |
| F2 | Gradient path audit map, per task × contact regime (torch GT; Genesis columns only where the probe ran) | RQ1 |
| **F6b** | **P2a: safe success by actor × perturbation cell** — the same verifier on 2–4 actors, showing the gain depends on how physical the actor's failures are | RQ2 (P2a) |
| F3 | **Reliability map:** AUROC / calibration vs horizon × constraint type, physics vs vision | RQ1 (headline) |
| F4 | Fidelity and gradient-agreement decay vs horizon, with *h\** marked | RQ1 |
| F5 | Trigger lead time & repair success per constraint type | RQ2 |
| F6 | SR/CVR over OOD grids, 4–5 arms | RQ2 |
| **F0** | **Headline: safe success of the VLA over the OOD grid — none vs. vision verifier vs. physics verifier (with SR and CVR alongside)** | RQ2 (headline) |
| F11 | Demo video: same seed and OOD cell, VLA without vs. with verifier (force, risk score and trigger overlaid); episode chosen to be representative of F0, not the best case | RQ2 |
| F12 | VLA + verifier memory and ms/step on one A5000 vs. the 50 ms budget | RQ3 |
| F13 | Public benchmark: LIBERO + physics subset — safe success, SR, CVR for none / vision / physics verifier on 1–2 VLAs, plus timely recall at matched false-alarm rate | RQ2 (P2b) |
| F7 | Memory / latency vs horizon, with and without each optimisation | RQ3 |
| F8 | Usable envelope: verifier quality vs resource budget | RQ3 |
| F9 | Real-data offline trigger traces vs measured force | P4 |
| F10 | Verifier trace as a physical reasoning chain (observation → constraint → intervention) | C (interpretation) |

---

## 6. What the existing `src/` code covers

| Needed | Exists in `src/` | Gap |
|---|---|---|
| Plug-in interfaces for policy / predictor / verifier / GT | `interfaces.py`, `registry.py` | — |
| **GT engine** | `sims/torch_push.py` — hand-written differentiable box–point contact (the working GT); `sims/genesis_push.py` (parity fails, 30 Sep) | T2 multi-contact; Genesis demoted to optional transfer target, not a deliverable |
| Releasable benchmark (P2a) | `experiments/rq2_eval.py` arms + the OOD grid already do most of it | freeze protocol, seed the initial states, take the actor as an argument, one-command harness, docs |
| Real OrbiSim / CheckVLA | adapter templates only | M2 |
| Gradient path audit | `experiments/audit_gradients.py` (torch GT + Genesis probe) | per-regime Genesis audit |
| RQ1/RQ2 metrics | `rq1_calibration.py`, `rq2_eval.py` | horizon & constraint-type sweeps, calibration curves/ECE, repair success |
| System track | `systems/bptt.py`, `systems/scheduler.py` | Genesis-native checkpointing; latency sweeps |
| Demo | `demo/genesis_franka_demo.py` (never run) | run; add target marker, OOD comparison video |
| Real VLA policy | `adapters/openvla_policy.py` (skeleton, single-action OpenVLA) | M2: chunk-output VLA selection, fine-tuning; **camera now done** (`sims/camera.py`, gap 0.974 held out, friction/mass provably invisible) — remaining: 2-D velocity action → EE-delta mapping, demos |
| Public benchmark | — | P2b: LIBERO install, physics perturbation + force-limit wrapper, robosuite state adapter for the predictor, eval script |
| Real data | — | P4 |

Known issue carried over: the stand-in predictor under-predicts rare risk spikes (fix written, unverified). Real predictors will need the same check: rare-event risk calibration is part of RQ1.

---

## 7. Thesis structure (target)

1. Introduction — physical reliability of VLAs; why runtime verification
2. Background — VLAs, world models, differentiable physics (Genesis, OrbiSim), runtime verification (CheckVLA)
3. System — architecture, predictors, verifier, Genesis integration, gradient path audit
4. **Signal quality & trigger effectiveness (RQ1, RQ2)** — main contribution
5. **Making it usable: system design (RQ3)** — supporting contribution
6. Real-data validation (P4)
7. Discussion — the verifier as a physical reasoning chain; limitations (policy proxy, Genesis gradient coverage, sim-to-real)
8. Conclusion

---


## Algorithm Improvements (Priority Queue)

| Priority | Issue | Solution | Status |
|---|---|---|---|
| P0 | Heavy cells (mass 2.0×) verifier increases violations | 1) Extend mass range to 2.5× 2) Per-cell τ calibration 3) Add VLA states to training | IN PROGRESS |
| P1 | Only brakes, never steers | Best-of-N candidates + risk+progress joint selection (from SEAL) | PLANNED |
| P2 | Stall under repeated triggers | Stall detector: after k consecutive shrinks, escalate to re-plan | PLANNED |
| — | Kernel optimization | NOT NEEDED (verifier 30ms, bottleneck is 7B VLA at 59ms) | WONT FIX |


## 8. Risks & decision gates

| Risk | Probability | Mitigation / fallback |
|---|---|---|
| OrbiSim / CheckVLA code not public | Medium-High | Documented re-implementations named "OrbiSim-style" / "CheckVLA-style"; the contribution statement is about *differentiable physics predictors* as a class, with OrbiSim as the reference design |
| ~~Genesis gradients unavailable for key contact modes~~ | **Realised (30 Sep)** | Genesis parity fails (SR 0.06 vs 1.00) and it is 5× slower. Not mitigated — *removed from scope*. torch GT is the engine; Genesis is an optional transfer target for P2b |
| A releasable benchmark cannot be made reproducible by others | Medium | G1b forces a one-command ≤2 h harness on one A5000, or P2a is demoted to an internal protocol rather than a claimed artefact |
| Differentiable deformable task in Genesis too unstable | Medium | T3 reduced to a smaller qualitative study; T1/T2 carry RQ1/RQ2 |
| H1 rejected (no clear *h\**, physics ≈ vision everywhere) | Medium | Still a publishable negative result; pivot the main contribution to B (system envelope) at G2 |
| No real data with force labels | Medium | Sim-to-sim transfer study (see P4) |
| Scope creep into direction C | Medium | C is only the interpretation of verifier traces; no free-form reasoning generation |
| Compute | Low-Medium | Budget fixed to one A5000 (24 GB); P3 results give the budget for P2 grid sizes |
| VLA too large or slow for one A5000 alongside the verifier | Medium-High | Measure in M2 before choosing; prefer smaller chunk-output VLAs; run VLA arms only on informative cells; the gap to the 50 ms budget becomes an RQ3 result |
| Base VLA fails in OOD cells for non-physical reasons (perception, grasping) | Medium | Classify failure causes in M2; fine-tune on Genesis Task A demos; report which failures the verifier can and cannot address |
| Verifier lowers SR by making the policy over-cautious | Medium | Safe success as the primary metric, with SR and CVR reported next to it; tune the repair margin and conformal alpha on held-out cells |
| Public-benchmark failures are not physical, so the verifier has nothing to fix | Medium-High | Choose contact-heavy LIBERO tasks and add physical perturbations; classify failure causes; report the split |
| Predictor does not transfer to robosuite state | Medium | Same architecture retrained on robosuite; if it still fails, keep Task A as the main evidence and document the gap |

**Gates:**
- **G1 (end Dec 2026):** end-to-end loop on the torch GT with real or justified components, **and the chosen VLA running in the loop with baseline (no-verifier) OOD numbers**. The Genesis half of the original gate is removed (30 Sep, §1a.1).
- **G1b (end Feb 2027):** P2a harness runnable by a third party in ≤2 h on one A5000, else demoted to an internal protocol.
- **G2 (end Mar 2027):** *h\** located or H1 clearly rejected. If rejected → re-weight toward B.
- **G3 (end Jun 2027):** usable envelope measured.
- **G2b (end Jul 2027):** LIBERO + physics with/without-verifier results, or a documented reason the transfer failed.
- **G4 (end Jul 2027):** real or sim-to-sim validation done.

---

## 9. Immediate next steps (next 2–3 weeks)

Done in Week 1: main contribution decided (A); Genesis installed and gradient probe working; OrbiSim / CheckVLA code confirmed not public; compute fixed to one A5000.

**Done in Week 2** (see `exp_record/week2.md` for the measurements): full torch pipeline at 64 envs with the safe-success metric; orbiim / vision / vision_noact trained and conformally calibrated; RQ1 (detection + controllability), RQ2 (pooled OOD, 5 arms), RQ2b (shift ladder), RQ3 (memory + scheduling), the repair audit, and the two empty-step defects in the normaliser and the rollout risk. Genesis parity measured and failed → demoted (§1a.1). The gradient repair branch and the hard prefix both measured inert → off by default (§1a.3). Peak latency is 183 ms/step mean, 441 ms p95, so the 50 ms budget is **not** met at `chunk_k=1` but amortises to ~37 ms/step at `chunk_k=5`.

1. **Run the rate-matched τ sweep** (`experiments/rq2_matched_tau.py`). The headline orbisim-over-vision gap is confounded by the two predictors' different conservatism until this lands; it is the single highest-value thing outstanding. Smoke test in progress.
2. **Freeze the P2a protocol** (P2a, new): seed the initial states, make the actor a harness argument, write the one-command entry point, add the scripted-expert arm. This is the cheapest credible external artefact and it needs no new dependency.
3. **Start the VLA short-list properly** — this is now the only thing between us and "the verifier is on a VLA", and everything downstream (P2b, F0, F13) depends on it. Measure one candidate's memory and step latency *with the verifier resident* on the A5000 before committing, since 3.6× over budget is the current state.
4. Read out ablation (`scratch/readout_ablation.py`, written but never run at full scale): current-state vs extrapolated-state vs delta read-out. Note the trap recorded in `exp_record/week1.md` — the extrapolated read-out loses on uniformly sampled validation windows and wins under the real RQ1 protocol, so this must be judged on the RQ1 protocol only.
5. Any change to `models/nets.py` requires retraining both predictors (~300 s each) or the results silently use stale weights.
6. **Supervisor:** confirm the 30 Sep revision (Genesis out, surrogate framing, P2a ahead of P2b, contribution delta in §1b); report the GPU-0 fault (`ERR!`/`N/A`) to the admin; ask about lab robot / F/T sensor access for P4.
