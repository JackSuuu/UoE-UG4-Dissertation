# Core Research Plan (Oct 2026 – Sep 2027)

**Dissertation:** Integrating Neural Physics Engines into Embodied Foundation Models for Enhanced Physical Reasoning and Robust Generalization
**Replaces:** `Experiment_plan_Phase_1.md` (3–4 week validation plan). That plan remains the blueprint for Months 1–3 infrastructure.
**Prior work used:** OrbiSim (Li et al., arXiv 2605.16395), CheckVLA (Liu et al., arXiv 2607.26789), Genesis (`genesis-world`)
**Last updated:** 2026-10-16

## 唯一当前主线：本地VLA在有噪声精细任务中的物理预测与安全完成（2026-10-13，优先于下文）

### 16 Oct最新用户指令：A2融合预测门控与几何反馈，以完整效果为目标

用户授权设计并执行融合实验，不要求为了突出物理预测而牺牲效果。五层架构不变：共享预测/事件gate→阶段几何默认修复→预测可信且显著更优时选择其他合格候选；按真实执行owner记录贡献。新增[A2冻结协议](exp_record/assembly_a2_protocol_20261016.md)，先精确重放全部30条A1真实轨迹诊断预测残差/抓持/阶段（不筛掉成功或失败），再跑4组：VLA、原反馈预算80、阶段几何预算80、阶段几何+可信预测预算80。原10案例仅作回归；3100/3101新seed×5入口作为前瞻开发，共20案例×4组=80条；160步/5步完整终点不变，各组共同计算11候选×3步，microbatch8+3。保留A1预测器，本轮不混入再训练收益。

阶段管理加入防提前松爪、失去抓持停止空移并交还actor、入座后松爪/撤离；预测需通过上一真实动作残差、候选不确定度/风险/进展约束才覆盖几何选择。预检7项机制测试与四组32步真实仿真fixture均通过独立完整预测/决策/图像/物理回放；fixture不代表新在线VLA成绩。混合组在此fixture没有预测覆盖，必须如实报告而非调整门限制造预测参与。执行状态和结果见[执行记录](exp_record/assembly_a2_execution_20261016.md)与只读`status_assembly_hybrid.py`。

### 15 Oct最新用户指令：启动A1接触装配，真实接入自研预测—验证—修复（覆盖旧“暂不训练”安排）

**A1已完成、当前优先于下方启动状态：** 40条采集/6400transition、训练与校准已结束；30/30在线轨迹通过精确回放。VLA3/10、完整框架6/10、共享预测gate但几何选动作8/10；框架相对VLA救回3条、未误伤原成功case，但预测排序没有超过几何排序。30个原视频和4个三方法并排视频位于`assembly_a1_20261015_a/review/index.html`索引；原结果完整归档为`backups/assembly_a1_20261015_a`（310文件，本地校验通过）。

用户要求逐项排查4条框架失败，新增[诊断记录](exp_record/assembly_a1_diagnosis_20261015.md)：逐动作预测误差/评分分解/预算统计；4种固定事后事件×8候选×3步精确prefix仿真分支；另以同模型/控制/终点、仅预算40→80重跑四条失败并核对共同prefix。这是failure-selected开发诊断，不计入原30条、不改阈值、不当确认集提升。初步分型为3000/c1最终未脱离、3000/c3早期脱手且后续预测崩溃、3001/c2约0.096mm几何边界拒绝、3001/c3持续纠偏反而横向漂移。后续完整反事实结果以诊断记录为准。

**诊断已完成：** 128条3步候选分支均精确prefix；预算80的4条新live都通过原40次修正前共同prefix和全轨迹独立回放。3000/c1以80次修正/148步成功，3001/c2以47次/101步成功；3000/c3仅46次仍失败、3001/c3以70次仍失败且末态横向误差约102mm。失去抓持后的3个检查点所有候选对零件真实作用完全相同，预测仍给虚假差异；3001/c3预测risk排序偏好+y而真值局部排序偏好−y。四条原失败153/160次选中的修正仍高于trigger阈值，因为当前acceptance只要求相对objective改善，没有候选过阈或预测可信度要求。下一版本应分别验证预测支持/适用性、风险与完成目标排序、有效进展预算；不能只加预算、不能把选出的2条救回改写成原A1更高成绩。原310文件+4并排视频已独立从GitHub/LFS恢复核验。

用户要求直接检验我们框架的问题，不能以简单控制替代框架。新[A1开发协议](exp_record/assembly_a1_protocol_20261015.md)：方形螺母套柱，物理抓持后从柱口上方开始；冻结OpenVLA逐帧H1动作→自研OrbiSim-role学习动力学→RefCheckVLA风险验证→有界候选修复→新观测交还。五层架构不变。官方OrbiSim/CheckVLA适配器仍未实现，OrbiSim作者主页仍“Code coming soon”；不会把MuJoCo分支或自研MLP写成官方模型。

实际装配入口已通过三个位置真实夹持预检；12mm偏移的持续下压出现柱口停滞/接触负载，而对准下压可入座。这些是装配fixture，不是VLA效能。新预测器使用40D特权状态、3成员MLP、真正自回归3步state/risk；独立采集训练/校准/验证seed，microbatch8，训练后冻结。2个新装配开发seed×5位置×3arm共30条；主角framework，对照VLA以及共享预测gate但用实测几何选择候选的feedback（这是ranking消融，不称完全无预测baseline）。全部保留，不按结果选样。目标为插入并松爪稳定5步/最多160步；接触负载与延迟单列，不宣称安全或20Hz实时。

M1仍停止，原9条完成结果、2条中断和replay不一致问题原样保留；新A1不改M1协议，不消费LIBERO最终root。**A1已实际启动**：`24b219d`先推送后启动，目录`/home/ami/scratch/assembly_a1_20261015_a`，supervisor1659007，采集worker1659011/GPU2；随后自动训练、GPU1/2/3配对在线评估。35个预检文件和12个源码/协议文件已独立远端恢复核对。新增[执行记录](exp_record/assembly_a1_execution_20261015.md)与只读`status_assembly_development.py`；当前进度以launch/status为准，下方旧“没有训练运行”仅表示当时状态。

### 15 Oct后续纠偏：行为验证与修复贡献分线（优先于下方启动时状态）

用户明确要求先定义不正确行为，再做参数化失败发现；不以挑出框架获胜场景替代方法验证。已新增[验证/修复双线契约](exp_record/libero_validation_repair_contract_20261015.md)，逐项写明现有600步内20interval完整完成判据、相位条件适用范围、精度单位、安全未知项和墙钟实时要求。A线输出baseline失败条件/约束违反；B线输出同case对VLA、hold、timed及后续强规划对照的完整任务救回/误伤。30%–70%失败率仅为开发描述性信号，不是最终case筛选门槛；3个root不够确定稳定区间。不能为制造失败事后把drawer中心误差收紧到1mm。

**当前实际状态：M1已自动停止，非运行中。** root2/sham/phase在同worker内独立回放时pre-action fingerprint断言失败，supervisor终止其他workers。新独立进程用未改动verifier完整重放该条成功；原不一致原因仍未定位。现有9条完成且通过回放：VLA3/3、hold0/3、timed1/2、phase0/1，verified尚未执行；其余中断/未执行不能计作任务失败。保留所有不利结果，不因sham误伤更换阈值或提高扰动。当前没有新rollout/training启动。

**具体失败边界：** 五条已完成干预失败末步都native成功、碗稳定保留、抽屉关闭，但夹爪仍接触柜体，`released`是唯一未满足终验条件；不是碗放置误差或危险碰撞的已证结论。root2 phase曾stable-clearance交还，仍未最终完成，说明局部恢复不等于剩余任务成功。各episode实测actor推理中位0.706–0.726秒，所有live query超过50ms；当前同步实现已不满足20Hz实时要求。安全/CVR仍未知。只读双线审计脚本`audit_libero_mining_contract.py`核对artifact hashes、重算全部completion dwell并保留未核验case；报告`backups/libero_mining_contract_audit_20261015.json`。

**下一比较要求：** 明确native H1本就每帧闭环，不能把同帧重复查询命名“重规划”。现有hold/timed是机制对照；计划的强对照为同信息权限/预算的receding-horizon shooting planner，并同时区分共享trigger的机制归因和共同总预算下完整planner能力。尚未实现/运行MPC，不把现有两事件三候选重命名MPC，不现在启动额外RL训练。先解决当前可重现性并保留原注册比较，再登记新控制/场景/实时实验。

### 15 Oct用户授权批量场景发现与文献逻辑集成（当前最高优先级）

**已实际启动M1：** 代码/冻结协议/39case清单/全部预检以`a37e354`先推GitHub，随后启动`/home/ami/scratch/libero_failure_mining_20261015_a`。supervisor1499667，GPU1/2/3worker1499671/1499672/1499673，脱离SSH运行。每个root逐case运行VLA/hold/timed/phase/verified全部5arm，不会baseline成功就取消。总计195条真实actor延续，每条自动精确回放，逐case生成5方法视频。当前为运行中，尚无完成的效能结论。[启动记录/状态命令](exp_record/libero_failure_mining_launch_20261015.md)。

用户明确要求从单一易成功场景的反复修补，转为**批量定制开发场景→寻找VLA失败/框架成功机制→冻结后独立确认**。新[M1协议](exp_record/libero_failure_mining_protocol_20261015.md)覆盖此前只做9个D2case的下一步安排；D1/D2历史结果不改，五层架构不变。首批39配置×5方法=195真实actor延续：drawer三个已消费开发root，各sham+4/8/12/16步释放延迟+world-x/y正负10/20mm动作偏置。是同任务的39开发配置，不是39任务或正式独立测试。所有方法都跑所有配置，不因baseline成功取消干预。

**已接入实现：** `libero_mining.py`和新runner实现早期release-opportunity事务、release-conditioned强hold对照、10+30定时撤离、实测phase修复、额外候选物理验证repair。三候选×3步pulse-and-settle，在entry/首次released各一次，所有arm同预算shadow预测，只有verified arm可据此改动作；forecast重放真实effective prefix、核对state/controller/image，不能读未来故障。每条按原600步/20interval完整任务验收，交回后继续真实VLA关抽屉。框架selected和外部faulted effective分开，保存事件记忆、动作—结果核对、全部候选预测、实际效果、失败handoff与逐步视频。

**文献核对与边界：** [集成审计](exp_record/libero_literature_integration_20261015.md)。SEAL官方`NVlabs/actalign`固定到`ed65d42e1466d9eb85ce8097f5034817a7e0ddc1`，已读sample evaluator与verifier prompt，checkpoint仍TBD；当前移植propose/predict/verify运行逻辑，非π0+VLM官方复现。CheckVLA核对原论文/作者页，未定位官方发布入口；新增command/effect与event bank目前为诊断证据，已有release/clear历史驱动实际控制；没有已集成其learnedWM/conformal/flow-suffix/keyframe-policy reader的说法。H1不能伪装chunk。文献有效不能替代本地配对结果。

**预检状态：** 11项新/相关单测通过；真实sim五arm与精确forecast/action/trace回放预检进行中，第一次fixture在numpy bool JSON序列化处停止，已修正并保留失败记录，未消费在线actor结果。M1预检通过后启动独立SSH会话的GPU1/2/3 workers，逐case自动验证/视频，完成195条后输出全量成功/救回/误伤/双方失败地图并归档。当前尚无M1效能结论，final根未打开。

**五arm预检已通过：** root0延迟8步的记录动作fixture中，hold/timed/phase实际修改各12步，verified修改11步；原VLA修改0步。所有实际effective动作、预测分支、因果trace、状态/controller/camera精确回放通过，sham精确复现；已生成5面板视频/trace页。报告`backups/libero_mining_preflight_20261015.json`。这些是fixture非在线效能；另行进行x/y动作偏置真实物理接口预检，随后启动全量真实actor批次。

**全部启动预检通过：** x+20mm/y−20mm物理偏置fixture也通过，每个仅4步外部执行端动作改变、框架改变0步，状态实际改变且全量action/forecast/trace回放一致；报告`backups/libero_mining_pose_preflight_20261015.json`。运行路径和协议准备冻结，真实actor启动另记PID与目录。

### 14 Oct用户明确授权受控扰动：最终benchmark与D2（最高执行优先级）

**纠正D1后的推进判断：** 干净场景VLA3/3应成为no-harm/sham锚点，不是放弃drawer或停止效能研究的理由。用户明确要求通过可恢复、受控、同条件的故障注入检验框架干预和完整任务收益。[最终benchmark设计](exp_record/libero_final_benchmark_design_20261014.md)优先于下方“先返回book”的建议；五层架构不变，继续drawer的release-delay机制，再冻结方法迁移book和干净任务。

**研究证明链：** 受控故障→可观测物理偏差→因果事件触发→有界实际修复→交还控制→完成剩余关抽屉；必须在相同actor/观测权限/故障/控制预算下优于原VLA及强简单几何shield，并报告sham误伤。只比检测AUC、只恢复子目标或把注入器改变的动作算框架干预都不够。

**D2已准备的可执行部分：** 新增外部`ReleaseDelay`执行端注入器、无结果筛选的案例manifest生成脚本和4项通过的单测。已有D1轨迹在root0/1/2的首次释放机会为169/147/163步，按共同prefix精确重放，冻结0/4/8个20Hzinterval延迟，共9个开发case身份。注入在gate/repair选出动作之后作用于夹爪；方法不得读开关/时长/anchor，只读实测接触/几何。`nominal→selected→effective`分开记录。manifest状态`prepared_not_executed`；这不是已完成的在线恢复实验，真实sim故障/零扰动回放预检正在后台执行。新release-pending监测/控制及其逐步解释需在新D2runner接入，D1冻结实现不改。

**D2注入器真实sim预检已通过：** root0的169步共同前缀分别完整重放3次，integration/controller/camera完全一致；sham后续8步与原轨迹完全一致；4/8步延迟正确改变物理夹爪状态。8步延迟分支在该窗口持续碗/夹爪接触，sham已脱离。只是使用已记录动作的接口回归，不是在线actor失败/框架恢复结论；报告已保存`backups/libero_stress_smoke_20261014.json`，当前无后台实验运行。

**最终评测框架固定：** 3个Spatial+3个Object干净任务检验no-harm；book/caddy与drawer两个long任务分别做clean+两档释放延迟+两档横向动作偏置。每任务20–49为final根，0–5开发、6–9校准、10–19仅必要的参考/训练；最终数据当前未打开。选定8任务不是全LIBERO套件分数。核心四方法480case×4=1920轨迹；共享repair的动作跳变/特征距离检测对照仅long的300case×2=600，最终计划上限2520，分阶段执行，先不启动大规模表。Spatial/Object须确认对应冻结官方checkpoint能力，不能把10checkpoint迁移失败当标准baseline。

**基线与指标按实际实现命名：** 当前原VLA是每帧H1闭环，非open-loop chunk；ACC需重叠action chunks，H1只能叫temporal action-jump。Hard-stop非CBF；embedding距离非自动等于V-JEPA/LLMD复现。已核对LIBERO-Plus、VLA-FAIL、RoboRecover与VLA-FixBench来源并写入设计。用户提及74ms/AUC0.727vs0.996/0.909/19步/Task0的80%在当前记录未建立对应LIBERO来源，不能直接写成果。主指标为全任务SR、paired rescue/harm和root聚类CI；故障case无条件完成率与trigger条件恢复率分开；提前量在共同无修复shadow轨迹测；开销实测median/p95，20Hz仿真不等于实时。接触分类/数值验证前Collision/CVR/safe仍null。

### 14 Oct最新实施：D1定向阶段交接实验与可观测决策链

**D1已核验结束（覆盖下方启动记录）：** [结果](exp_record/libero_drawer_target_results_20261014.md)。原VLA在drawer开发root0/1/2按完整物理验收均成功3/3，分别368/564/344步；按预登记`no_observed_post_event_headroom`停止，未启动其余9条干预轨迹。1276个真实actor动作及其证据→判据→路由→动作→结果链全部原root精确回放通过，生成3个单VLA面板2倍速视频和可点击trace页。**没有本轮框架提升结论。** 关抽屉后的延迟主要是柜体/夹爪未脱离，非碗再次被碰；root2稳定支撑触发晚于释放和关抽屉，说明事件出现不等于及时覆盖目标阶段。下一效能目标应回到已证明存在失败的book/caddy阶段交接，再单独登记；不为正结果扩大扰动/root搜索。当前无实验运行。40源文件+3视频已本地归档核验，远端恢复随后记录。

**D1远端恢复已通过：** `81eb54e`推送4个LFS对象/26.3MB后，独立从GitHub下载并完整恢复校验40源文件及3个视频；[验证记录](backups/libero_drawer_target_remote_verified_20261014.json)。

**已实际启动：** `d3889e2`实现/冻结协议/预检已推GitHub，随后启动`libero_drawer_target_20261014_a`。脱离SSH的supervisor779352；健康GPU1/2/3对应root0/1/2、baseline worker779356/779357/779358，当前阶段`baseline_target_coverage`。根据预登记覆盖/事件后失败规则自动决定是否接着跑全四arm；完成后精确回放、生成可观测决策链视频/HTML并归档。尚无该任务效能结论。

**研究问题唯一化：** 已放入并获得支撑之后，按实测“释放→净空→交还控制”执行局部修复，能否改善后续真实VLA关抽屉及完整任务完成？[冻结D1协议](exp_record/libero_drawer_target_protocol_20261014.md)、[实现与运行记录](exp_record/libero_drawer_target_launch_20261014.md)。五层接口已接通：共享子目标tracker→冻结OpenVLA→接触/几何/历史monitor→正常/接触转换gate→有界hold-open/向抽屉外撤离。此次只检验release-to-close机制；早期对齐/停滞仅记录，不堆叠新干预算法。

**有针对性的对照与止损：** drawer任务开发root0/1/2；先跑三条原VLA确认目标事件和事件后失败。若事件不可达，或所有到达事件的baseline均已完整成功，按预登记停止，不跑无效重复对照、不事后改阈值。存在事件后失败才在全部三root比较VLA、gate+hold/open、固定10hold+30外撤、实测阶段条件repair，最多12轨迹，每条600步、修复最多40步、每步都查同一H1actor。hold/open明确含夹爪释放，不伪称无动作的纯gate或H1重规划。主要比较phase-repair对两个简单控制的完整关抽屉/释放保持收益和误伤。

**用户要求的显式可观测链已实现：** 每步落盘`decision_trace.jsonl`，按“t−1前实测证据→子目标→事件判据/阈值/连续次数→门控模式/原因/剩余预算→原动作vs实际动作→t执行后结果→handoff条件”完整记录；视频同步overlay，HTML可点击事件跳到对应帧。离线从原root逐动作精确replay重建并核对链，解释来自实际执行条件，不事后编造理由。

**判据与预检：** bowl使用40个collision boxes完整角点，夹爪含mesh和pad的完整碰撞顶点；native In仅参考点、Close为q>0。真实仿真正例可满足完整关抽屉+支撑+脱离1s，部分越界反例虽native In却被完整几何拒绝；3环境instrumented/uninstrumented integration/controller/camera完全一致。16项单测通过；短合成动作端到端导出/精确回放通过，最终trace修订的四arm复检也通过，source/protocol hash与待启动版本一致，3份预检报告已入backups。它们不是actor成功率数据。当前仍是privileged规则护栏、H1/k1，Collision/CVR/safe仍null，不训练新模型。

### 14 Oct用户锁定：五层架构与长序列验证（最高优先级）

**架构固定为：高层子目标 → 冻结OpenVLA骨干 → 风险与事件边界监测 → 双模门控决策 → 局部修复模块。** 权威定义已写入[framework design](exp_record/libero_framework_design_20261014.md)，[逐层改动与下一任务实施顺序](exp_record/libero_five_layer_implementation_20261014.md)。后续只在这五层内部根据证据修正、做匹配消融，不再更换主架构或铺开新研究方向。高层先用共享的任务milestone状态机；双模明确为正常跟踪／接触与阶段转换管理；修复限有界Retract、Re-align、Slow Approach，优先几何启发式，必要时小残差，不训练复杂大规模RL修复器。

**论文定位：** Runtime Physical Safety & Affordance Monitoring + Contact/Phase Transition Repair，作为围绕冻结VLA的轻量Physical Guardrail。几何、接触转换和实测进展支持面向执行的有限“物理理解”；不要求PINN/V-JEPA/完整神经3D引擎，也不把规则监测写成已学习动力学或通用物理推理。算力效率、安全与效果必须实测，不预先承诺最高效率、成绩或正结果。

**当前数据正确定位与已完成结果：** 旧native3/3不等于真实完整任务100%；释放稳定放置仅1/3，当前book/caddy本来就在`libero_10`。preservation是五层中的局部机制探针，不是完整五层实现。`_b`24条rollout三个worker均exit0；已修复汇总阶段MP4 reader长度提示引起的预分配错误，在同一批数据上独立复核全部通过并生成6个四方法2倍速视频。[完整结果](exp_record/libero_preservation_results_20261014.md)：VLA/hold/retreat/physics放置完成分别2/6、3/6、2/6、3/6；完成且40步actor续跑末20步保持分别1/6、2/6、1/6、1/6。native全部6/6，CVR/safe仍null。physics相对hold没有完成率优势且保持较差；只支持在固定第3–5层补早期事件覆盖和阶段感知handoff，不宣称框架胜出。全部root0–5仅开发，当前无rollout/训练在运行。

**唯一下一long-horizon任务：** `KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it`。已核对BDDL：抽屉初始Open，终态要求Close与In同时满足；依次验证抓取→对齐运输→放入释放→夹爪撤出→关抽屉→保持，不虚构“必须先开抽屉”。先完成当前诊断归档，再接五层phase/event/gate/repair接口、核查此任务判据/actor能力，最后冻结同观测/预算/停止规则的原策略、Gate-only/Replan-only、Event-Gated+Local Repair对照。

**动作与指标实事求是：** 当前checkpoint是nativeH1，不能称原生open-loop chunk；真实task-compatible chunk提议及actor-only归因尚需解决，历史H10/k5不被悄悄替换。每帧调用H1后又在同帧replan是退化消融，不拿来填三组正式chunk表。碰撞率须先固定允许/禁止contact-pair×phase及事件分母，不能将正常抓取/支撑接触算违规；force/CVR/safe在阈值和数值稳定性核验前仍null。扰动强度和split事先登记，不人为削弱baseline制造差距。

### 以下为按时间累积的实施记录；活动状态以上述最新条目为准

**14 Oct运行中止与修复（更新下方启动状态）：** `_a`在root2/physics第二次前瞻decision190因相机hash不一致自动停止全部worker；integration/controller一致，尚无完整框架对比结果，root3–5尚未执行。原因是robosuite1.4.1多环境渲染未重新绑定EGL context，首个单次smoke没覆盖分支销毁后的下一次主环境图像。已添加本地逐render绑定，连续3次/18分支检查通过；补测实际修正后prefix重放后，按原冻结协议在新目录`libero_preservation_20261014_b`重跑全部24轨迹。[故障、修复及中止数据说明](exp_record/libero_preservation_renderer_fix_20261014.md)。中止的100文件归档进入Git LFS；没有利用部分结果改候选/阈值/打分。

**修复验收通过：** 连续8次/48分支检查154.5s全部通过，候选选择`[0,0,0,0,2,2,2,2]`覆盖修正后历史重放；完整integration/controller/相机、live不变性、预测首步=执行首步均精确一致，未放宽像素校验。7项单测仍通过。此为接口回归探针，不是新的效能结果。

**最新活动运行：** 修复`ec9ff87`推送后，`libero_preservation_20261014_b`已启动，supervisor749068、GPU1/2/3worker749073/749074/749075，全部24条按原协议重跑，结果待完成。中止`_a`的100文件LFS归档已从GitHub独立下载并完整恢复校验通过。

**14 Oct当前执行决策（优先于此前“未启动”状态）：** 用户授权在当前场景接入真实预测→选择→执行闭环。已冻结并推送[placement-preservation机制协议](exp_record/libero_preservation_protocol_20261014.md)（`9336782`），[实现与启动记录](exp_record/libero_preservation_launch_20261014.md)。准确物理selector与VLA、hold-open、retreat-open匹配对比，root0–2及新开发root3–5，共24轨迹；当前仍是privileged H1/k1诊断，非learned efficacy。首个有效放置后的20步内每步最多6个5步pulse-and-settle候选，仅执行第一步；达到原1s放置验收后关闭修正，再执行40步原VLA观察恢复后的保持。未扩大为多任务或训练新模型。7项单测通过；root2完整190状态replay、6分支完整integration/controller/图像一致性、live不变性、预测首步=实际首步已通过。早期alignment失败不在本次trigger覆盖范围，不能靠事后放宽规则制造收益。

**备份已远端恢复验证：** `f33915a`包含14个LFS对象/2.45GB，重新从GitHub克隆下载并完整恢复1,674文件均通过SHA256；LIBERO固定submodule及任务初始状态hash一致。见[恢复报告](backups/remote_recovery_verified_20261014.json)。预训练7B权重仍依赖固定HF revision重新下载。

**当前运行状态：** `libero_preservation_20261014_a`已在`b9d7a31`推送后启动；supervisor743321，健康GPU1/2/3worker743325/743326/743327，root分配0/3、1/4、2/5。24条开发对照待完成；自动汇总通过后导出方法对比2倍速MP4。此前“无实验运行”的状态仅对应历史continuation结束时。

**14 Oct归因澄清与泛化顺序：** continuation的worker0/1/2全部是冻结OpenVLA baseline；物理模块仅作诊断/验收，尚未嵌入动作选择或修复闭环。worker2视频中的调整不能归因于我们的框架，且其短暂放好后再接触，未通过1s稳定验收。先在单一book/caddy场景完成匹配的VLA／简单控制／准确物理前瞻机制对照，再按证据逐项调整算法。泛化证据先来自未见root及预注册的观测/执行噪声、隐藏动力学条件，最后冻结方法到第二任务；同任务3个开发worker不是泛化，第二任务追加训练属于适配而非zero-shot。研究恢复入口见[Git LFS备份、视频和固定LIBERO submodule](backups/README.md)。

**14 Oct审查与下一项方法决策（优先于下文历史建议）：** [场景检测、公平对照与架构设计](exp_record/libero_framework_design_20261014.md)。全部rollout已完成，当前released placement仅1/3，尚无机器人任务的learned框架效果证据。已生成全部3条轨迹的2倍速独立MP4及13.025s并排总览，并接入汇总脚本的自动导出；视频按仿真时间加速，包含geometry/脱离/dwell/re-contact诊断，不是方法对比。当前history规则在root2于step188观察到placement opportunity、194观察到re-contact；不是提前6步预测成功。简单slow-contact stall规则漏掉root1，不通过事后调阈值宣称解决。

**下一步比较的必要结构：** 固定同一full7B actor/观测权限/状态估计/候选动作/反馈与预算，比较A原VLA、B共享检测器＋简单phase/geometry/撤离控制、O准确物理前瞻选择（privileged机制诊断），只有O优于B且完整续跑有净收益才做L允许观测输入的learned预测替代。L必须同时优于A与B，不能把多观测、简单终止逻辑或更大候选预算算物理预测贡献。先冻结一个小预算机制协议，不自动扩collector或训练；force/impulse阈值与数值敏感性仍待核实，CVR/safe不能补造。

**有证据依据的架构方向：** 从固定当前context的scalar risk转向短段实测history→接触/任务状态及不确定性→因果动作条件的相对pose/velocity/contact/progress预测；先小型residual模型与有限候选，不默认GRU/大模型。repair要能表达有界横向/姿态/撤离及明确gripper分支，不能仅缩小原动作。保留gate/event与连续repair目标的单位分离，并检查actor恢复执行后的完成。当前H1只作机制诊断；正式H10/k5研究仍须真实任务兼容H10提议和基线可行性，不能重复H1凑数。真实sim几何/contact仅诊断或训练标签，不能偷偷变成learned runtime输入。具体实现和效果须由后续匹配实验验证，本次未训练/启动新干预实验。

**最新续跑完成（覆盖下文运行中状态）：** [完整结果](exp_record/libero_continuation_results_20261013.md)。native-ever3/3，几何/脱离/稳定1s完成1/3：root0在313步通过，root1/2均到520步未通过；所有原baseline重合prefix的tokens/actions相同、state误差0，子步/视频/判据独立核验通过。root1始终未完整进入目标隔间且抓取后保持夹爪接触；root2曾全部条件通过0.3s，随后夹爪重新接触并破坏稳定。root0证明存在能完成的新验收例子，但末动作仍close，不能称为主动open-gripper release或安全撤离。下一项机制诊断应围绕对齐和完成后继续执行，加入匹配的简单停止/保持/撤离控制，防止把终止逻辑收益冒充learned physics收益。force数值尚未数值收敛/阈值认证，CVR/safe仍null。当前无实验运行，未启动新训练或干预。

**最新续跑实施（覆盖下文无实验运行状态）：** [冻结协议](exp_record/libero_continuation_protocol_20261013.md)与代码`9f39e2b`在launch前提交推送，3条同初始状态full7B H1/k1续跑已在健康GPU1/2/3启动，总预算仍每条520policy steps。几何入后隔间＋底部到位＋全部子步脱离夹爪＋慢速/支撑连续1s才计released placement；native成功仅记录事件。[实施/检查](exp_record/libero_continuation_launch_20261013.md)。5项测试通过；185步原轨迹加入监测后state完全相同；独立无VLA的合成正确放置fixture满足60个连续interval，确认判据在真实模型中可满足。该fixture的native predicate反而false，说明几何判据是替代性任务验收，不是native-success的逻辑子集，两者须分别报告。当前续跑结果待核验，不宣称安全提升。

**最新pilot已完成（覆盖下文启动状态）：** [结果与判据纠正](exp_record/libero_baseline_results_20261013.md)。3条native成功，185/229/166步；所有artifact长度/映射/hash核验通过。但三条终态夹爪都仍闭合并接触书本，未证明释放后稳定放置。源码追踪确认SiteObjectState.check_contact恒True，因此该In实际上只检查reference-point进区域，先前协议对“必须接触caddy”的解读不成立（以结果erratum纠正，冻结协议不改写）。native3/3不等于完整精细安全完成，CVR/safe/严格precision仍null。下一步先冻结继续执行＋松手/几何/稳定判据的有限baseline续跑，不能直接训练verifier。当前无实验运行。

**13 Oct实施进展：** 已接通LIBERO书本入收纳盒任务：Panda/OSC、官方initial states、完整任务full7B权重、native7D动作、官方图像预处理及视频/状态/接触记录。环境smoke和真实模型单动作smoke通过，预处理与官方函数逐像素一致；2项新sim ABI测试＋10项已有闭环测试通过，独立TF5/action-map检查通过。[实现记录](exp_record/libero_integration_20261013.md)。[3条完整任务pilot协议](exp_record/libero_baseline_protocol_20261013.md)与代码已在launch前提交`bf7c6c6`，开发initial indices0/1/2分别在健康GPU1/2/3持久化启动；当前本条只记录启动，不宣称完成率。H1/k1仅验证task-capable baseline，不能替代H10/k5匹配研究。现有In判据不要求松手/完整几何容纳/稳定；CVR、safe与严格precision暂记null，后续依据实际轨迹与mechanics冻结。未启动learned训练或物理干预实验。

**最新范围确认：先证明本地VLA执行闭环有效，LLM与端云协同只列future work。** 核心贡献目标是“物理预测/验证与必要的局部修正，能否在观测/执行/动力学存在不确定性时提高完整、精确、安全的任务完成”。具体loss、微调、网络或repair算法可以按证据调整，不能把某个训练技巧当成必须实现的主线。成功的VLA实例支持执行层逻辑，不自动证明LLM通用性或真实世界安全保证。[范围、任务与噪声设计](exp_record/local_vla_scope_20261013.md)。

**精细任务选择（逐个推进）：** 首选LIBERO-10“将书放入收纳盒后隔间”（确切ID和In目标已核实；实际间隙/碰撞与精度判据仍需检查）；第二候选“碗放入底层抽屉并关闭”（In＋Close目标已核实，轻柔/稳定到位需独立物理指标）；robosuite `NutAssemblySquare`作为后续受控对齐接触机制任务，现有成功条件并非严格精度认证，且task-matched VLA未核实。官方LIBERO-10 full7B checkpoint公开可用，未有本地rollout结果。先前“打开中层抽屉”降为可选接口smoke，不代表最终精细任务。只接通一个primary，不并行扩多套环境。

**数据与noise原则：** 用目标场景的独立training roots采集真实策略成功/失效及有机制依据的干预数据，拟合/校准物理预测与修正；现有audit/validation/final outcomes不变成训练标签。先固定task-capable VLA，避免同时微调actor和verifier后无法归因。不是“消除noise”，而是区分观测误差、执行误差、隐藏动力学与数值误差，用允许的观测history估计状态及不确定性，预测→执行→反馈更新；真实mass/friction不进learned runtime。先无噪声baseline，再预登记单因素及组合扰动，配对初始状态和外生噪声，相同观测/控制机会，验收完整task SR/CVR/safe与精度、稳定完成、rescue/harm及CI。force阈值/precision容差须由可核验mechanics决定，不能用事后挑noise强度制造收益。

### 已确认的task-first执行路径与接入检查

**最新用户确认：最终操作任务做不好，继续优化微调实验没有意义。** 当前下一步改为：①先跑通一个机械臂仿真任务及完整OpenVLA正常任务baseline；②在相同任务验证有限预算的准确物理干预能否提高完整安全完成率；③有机制可行性后，再训练/验证learned物理预测与修复贡献。此决定覆盖下文“简单任务learned收益证明后才做任何目标任务可行性”的执行顺序，以及原模型初始化微调建议；不抹去R5负结果，不把换benchmark算改进。[英文决策与兼容性核查](exp_record/simulation_task_first_20261013.md)。

**已核查的首个接入目标：** LIBERO-Goal `open_the_middle_drawer_of_the_cabinet`，官方BDDL与公开full7B任务checkpoint均存在；任务是打开中层抽屉，不是先前举例的轻柔关闭。先验证端到端完成，再核实joint到位/稳定及接触力标准；原Open predicate不自动等于精细/轻柔完成。当前仅MuJoCo已装，LIBERO/robosuite未装，任务checkpoint未缓存。现有二维H10头不能直接控制Panda；官方native H1 baseline可先作可行性复现，但不能与H10/k5 verifier混比，也不能重复单步动作假装真实H10。共享base只允许增量依赖，须先解决TF5 native decode与旧sim依赖兼容性。本次完成核查与优先级同步，未启动新训练/rollout。

### 保留的研究问题与实验沿革（当前执行顺序以上述用户确认为准）

**唯一问题：在相同完整OpenVLA、观测权限和执行机会下，learned physics verifier能否通过正确识别需要干预的动作，提高安全任务完成率，同时不明显牺牲任务成功？** 先在固定box-push任务证明，再转移到一个生活化接触任务。架构、SEAL和更复杂任务是手段，不是并行研究目标。[英文决策与证据链](exp_record/research_mainline_20261013.md)。本节覆盖下文所有“下一步”“当前主线”和SEAL启动建议；历史协议/结果不改写。

**13 Oct完成结论（覆盖下文R5计划/运行状态）：** [完整R5结果](exp_record/immediate_gate_results_20261013.md)。24采集batches、3配对训练seeds、180闭环batches/5760trajectories全部完成并独立核验。event相对同数据regression safe分别+1.34[−2.01,+4.91]/+3.35[−0.45,+7.14]/+4.24[+0.67,+7.81]pp，仅点估计方向一致，未通过全seed正CI标准。相对policy则safe−10.71/−11.83/−7.81pp（全部CI上界<0），CVR+10.27/+11.61/+7.37pp，safe rescue/harm3/51、1/54、2/37；ID也明显退步。**本配置不提升、不进入独立确认，停止该loss-only分支；目前没有新的实验启动。** 原direct/ramp仅保留既有reference，不将其称为已满足整体目标。

**失败解释的边界：** R5数值support通过，但128个训练即时事件决策中120来自手工brake-resumption，natural仅3、frozen仅5；dev62个正例全来自brake-resumption，原H10/R5 gate本就62/62检出。event达到100%dev recall未证明补上原gate的真实漏检。大部分safe harm包含新增违规（47/51、52/54、35/37），并非只是保守导致任务没完成；触发频率也并未普遍大幅增加。证据支持“该数据/架构/阈值规则下单换event objective不够”，不能单独断言全因data、threshold或网络容量。support对自然危险/漏检机制的代表性不足需明确记录，不能据此再自动放宽标准或广泛扫collector。下一项研究必须另有单一失效证据与预定停止规则，架构建议不是自动排队任务。

**13 Oct执行更新（用户已授权运行）：** [R5数值协议](exp_record/immediate_gate_20261013.md)与实现已形成：fresh train/dev/calibration roots76000/77000/78000；固定24batches/768trajectories，none/frozen/单次brake_ramp三个collector；即时违规与安全接触支持逐split/physics-bin先验检查，失败自动停止。通过才训练regression/event各3seeds，严格共享初始化、采样与3000updates，冻结原repair；加原H10和frozen-R5控制，共9arms×20cells×32episodes。57测试通过，CPU端到端smoke已完成（合成训练fixture仅验证管线，不是效果证据）。本条覆盖下文“本次不启动”的先前决策状态；实际launch另记于week4。独立测试不动。

**面向精细操作的架构检查：** [代码与证据评审](exp_record/architecture_review_20261013.md)。优先①将event gate与continuous physical repair评分解耦（本轮正在检验）；②只有残余错误支持representation问题时，才设计小型可观测接触状态＋causal action-prefix预测，不能把失败的GRU重跑说成新突破；③若检测改善但完成仍损失，再设计到位/对齐/续接感知的repair，不能只最小化一小段风险。当前obs没有force-history、原GRU也不是measured-history estimator。当前success只要求位置误差<2cm及平动速度<.05m/s，未直接检查yaw/depth/dwell，6mm几何不等于6mm精度已被成功指标验证。生活化迁移前需冻结真正精度/稳定完成条件；本轮指标和任务不改。这些建议不构成并行启动授权。

**为什么早期看起来好：** 早期BC/stand-in、vis与full OpenVLA混在叙述中；历史v5 learned safe约68.5→70.4%，大幅68.5→80.1%属于GT。旧controller还存在none每5步replan、非trigger verifier每步replan的不公平，10月8日才完整修复；不能断言这个bug解释全部历史收益。修正后n8/cell的+3.6～4.5pp信号，在独立896 OOD episodes上变成+1.23[+0.11,+2.46]pp（decision tau），safe rescue/harm21/10、SR−1.79pp；另一448验证cohort为−0.89pp。小正收益仍有效，但“大幅、稳定、完整VLA的learned收益已经证明”从未成立。最近ramp相对原verifier+2.01pp，仍非相对policy的稳定净收益。

**已选瓶颈：干预判定gate。** GT gate＋原learned repair safe77.46%，原gate＋原repair64.73%，policy65.63%；这是最强的已测组件替换证据，但GT同时改变估计与阈值语义并使用隐藏物理，未证明可学习性，且SR仍损失。连续修复伤害已获得有限改进，GRU失败、anticipation采集失败也已记录，不再每个现象各开一条线。

**下一项只改gate的学习目标，冻结repair：** 先形成一份有数值预算/停止条件的R5即时事件gate对照协议：frozen原gate、相同新合规训练数据上的regression control、同数据同结构/输入预算的event-loss gate；full actor/H10/k5、原repair、隐藏参数隔离保持固定。原direct为归因比较，ramp保留完成结果，首轮不同时组合新的repair。全部训练seeds与闭环SR/CVR/safe、paired rescue/harm及CI报告。R5有效不等于H10预警有效。

**数据阻塞必须直说：** 现有两轮collector没有合格H10-only anticipation support，完整H10 gate训练仍blocked。R5新研究也必须先按新协议预定数值检查fresh train/dev/calibration支持，不能把旧pilot偷偷改判合格。该固定预算检查若失败就停止并明确调整研究范围，不继续广泛collector扫参、不放宽标准、不扩physics ranges、不转而启动SEAL/新GRU来绕过。若支持通过才做一次匹配loss对照；检测指标改善但闭环无收益仍不算主线成功。

**统一验收：** 相对policy OOD safe≥+3pp、独立paired95% lower>0，SR损失≤2pp，stress review及既定多seed/第二cohort要求保留。未达到则不能把component增量、oracle收益或低CVR包装成framework成功。已用confirmation不能重复充当新独立测试。

**当前暂停：** SEAL候选扩展、继续调平滑、更大/更长history架构、广泛coverage sweep、完整生活化verifier迁移。SEAL只在明确需要更有表达力的repair时重新讨论。本次不启动新GPU任务；即时交付是上述唯一gate对照的可检验协议，而不是又一组支线实验。以后每个实验必须先写清“哪个已测问题→只改什么→什么闭环结果算成功→失败在哪里停止”。

## 长期任务与历史架构假设（2026-10-12，非当前并行主线）

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

**文档权威性：** 文件顶部“唯一当前主线”决定执行优先级；本节保留长期任务与历史假设。后续章节保留设计沿革；旧 cadence-confounded 数字、未复核 literature claims、计划状态或 hypothetical benchmark 不能作为当前有效性/完成状态证据。实现与实验 artifact 状态以版本化代码和 `exp_record/week4.md` 为准。

## 已完成结果与已暂停的SEAL下一步草案（2026-10-13）

**正式完成：180/180 batches、5,760 trajectories，三worker均已退出，当前没有该轮GPU任务。** 180 raw hashes、全部episode vectors、live-risk违规重建、prefix-label parity和三份shard hashes独立复核通过。[完整结果与SEAL接入评估](exp_record/repair_transition_results_20261013.md)。以下进度/launch描述保留作历史记录，以本段为准。

**结论：** learned ramp相对原verifier safe **+2.01[+0.89,+3.35]pp**、safe rescue/harm9/0，恢复原12 harms中7个并保留8/8 rescues；但相对policy仅+1.12[−0.67,+2.68]pp、SR−2.23pp，未过整体收益门槛。固定混合与ramp差异不显著；pause safe64.51%，虽减少局部跳变/违规，却未恢复任何原harm。GT ramp相对GT direct safe仅+0.45[−1.12,+2.01]pp、rescue/harm8/6。保留ramp为有希望的component/control，不宣称独立确认或普适衔接机制，不继续广泛调ramp。

**SEAL可以借鉴，但接入重点应是新的可恢复动作候选及验证。** 官方`NVlabs/actalign`和`NVlabs/Libero-10-r`代码已公开，当前官方README的reasoning-policy checkpoints仍标TBD；本机尚无LIBERO/robosuite/RoboCasa。原SEAL是reasoning π0＋随机候选＋真实simulator预测＋VLM计划一致性判断、variable horizon，不能直接当作当前deterministic OpenVLA H10/k5的一个开关。50ms latency目标当前后置，不作为否决理由。完整官方复现还需checkpoint/训练资源及接口兼容性；不能直接覆盖shared base依赖。

**建议下一项有限实验（设计草案，未launch）：** 固定K=8，scale-only与等预算非共线候选（侧向纠正/退让再接触等）对照；先在新training roots验证“是否存在安全且恢复policy后仍完成的候选”，GT branch仅诊断，不进入训练/runtime；有witness再比较同候选集的learned与GT selector，隔离candidate coverage与ranking/gating。过去scale/progress没有净收益，并未测试真正转向/再接触；反过来也不能据此声称新候选一定有效。完整数值/roots/control-query预算须在新标签前登记。R5/H10 event gate与anticipation coverage问题仍保留，不能用候选扩展掩盖漏检。长期生活化架构可组合语义进度验证与物理约束验证，正式transfer仍需learned efficacy。

## 历史启动与进度：直接验证 repair-aware transition 算法（2026-10-13）

**进度复核（启动约46分钟）：** 检查时171/180 batches完成，shard2正常结束，其余两张GPU仍运行，完整aggregate尚未生成。none/direct/ramp/flat在全部20cells已完成，80 raw hashes及outcomes/live-risk flags通过独立复核。OOD448：SR/CVR/safe(%)，none87.05/29.24/65.63，direct84.60/28.79/64.73，ramp84.82/26.79/66.74，flat85.04/26.56/66.52。Ramp vs direct safe **+2.01[+0.89,+3.35]pp**、safe rescue/harm9/0，恢复原12 harms中的7个且保留原8 rescues全部；SR+0.22[−0.67,+1.12]pp。相对none safe仅+1.12[−0.67,+2.68]pp、rescue/harm10/5，SR−2.23pp；相对flat safe+0.22[−0.45,+0.89]pp。因此已有修复衔接的增量验证证据，但未达到整体learned收益标准，也未证明渐变优于固定混合。等待pause/GT controls及最终完整校验；这是已用于设计的validation，非独立确认。

**研究问题仍然是：learned physics verifier 能否稳定提高安全任务完成率。** 最近两轮 boundary/coverage 实验解释了采集失败，并没有证明 framework 改善；不再沿着 collector 参数无限扩展。当前已有证据：(1) learned gate 是主要已测瓶颈，GT-gate 替换收益大，但 learnability 未证明；(2) 静态 max-score 有 action-order 盲点，但 GRU order-sensitive 并没有带来有效检测；(3) 原 verifier 的真实干预既救回也伤害任务，修复如何改变下一决策值得直接验证。因此算法/架构改动必须对应具体证据，而非“多做实验”本身。

**实际 verifier premise audit：** 14 OOD cells/448 episodes 中129个违规、12个相对 none 的 safe harms；340个 pre-violation post-repair decisions。16个首次违规发生于实际修改动作后的下一段，**包含全部12 harms**；其中直接恢复原动作的段只有7个违规、包含4 harms，且没有一个是该段第一步违规。另8 harms 在继续修复的段里。不能把 collector 的“恢复第一步就危险”当成真实 verifier 的已证机制；本轮改测连续 repaired decision chunks 的衔接。

**算法已经实现并启动：** repair-aware prefix shaping → unchanged gate → unchanged repair。用上一条实际执行 velocity 与“上一段是否被 repair 真正修改”这一 bit，下一决策将前5步按0.2/0.4/0.6/0.8/1.0逐渐衔接回新proposal，再评分；后5步不变。对照原direct、固定0.6混合、pause5steps，检验是否只是降速/拖延。Transform本身不反复触发，只有新的实际repair才re-arm；abstention/no-op不触发。不是检查之后偷偷改动作，也不增加gate查询/候选搜索。

**实验与资源：** [预登记协议](exp_record/repair_transition_20261013.md)，commit `0318a41` 在正式launch前完成。固定20-cell validation seeds26000–26019、full OpenVLA H10/k5/microbatch8。9 arms = none + learned gate×4 transitions + GT gate×4 transitions；repair均用原frozen predictor。180 arm batches/5,760 trajectories，640 paired roots（OOD448/ID192），不作为独立confirmation。三GPU按cell index mod3分7/7/6cells，全方法都在每个cell配对。Run `repair_transition_20261013_a`，launcher PIDs **671696/671697/671698**；日志 `~/scratch/repair_transition_20261013_a_shard{0,1,2}.log`。`setsid nohup`/stdin `/dev/null`持久运行，最后完成的worker自动合并，不启动训练。

**预先规定怎么判断：** primary固定learned_ramp，报告SR/CVR/safe、paired rescue/harm、CI及原12 harms恢复/8 rescues保留；与original、policy、flat/pause分别比较。Incremental证据要求vsoriginal safe paired95% lower>0，且SR下降不超过2pp；整体候选仍需vsnone ≥+3pp safe、positive CI、SR损失≤2pp及stress review，随后还须独立验证。局部更平滑/更安全不算task-level改进；只有GT有效不算learned成功；无效则停止这组衔接规则，回到gate数据/目标/观测问题，不无限调ramp。

**算法与架构顺序：** 现在隔离controller continuity算法（模型/阈值冻结）；gate仍需coverage-only regression→event-aware objective、独立R5/H10监督与frozen repair对照；若合格覆盖和event loss后仍失败，再检验order-sensitive representation/更长实际history或dynamics-context。现有static order blindness不等于旧GRU可用，更不支持直接扩大网络。52项测试、9-arm CPU smoke和自动汇总已通过。

**启动验证：** 三个worker各自首个none arm已完成（检查时 **3/180 batches**），outcomes/live risk均复现source factorial。Python PIDs shard0/1/2为 `671707/671705/671706`，均在运行；尚无pooled新方法结论。

## 已完成结果：boundary/gate diagnosis，定位 collector policy-resumption hazard（2026-10-13）

**该轮状态：** 三 GPU 全部完成 **84/84 batches、2,688 trajectories、128 shared training roots**，自动分析完成，原launcher/Python 均已退出；该次completion review没有启动job，现有新任务见上节。168 raw hashes、analysis→source hashes、四个 none batches 跨三个 GPU 的 state/context/proposal/labels/七模型 scores/order probes 均复核通过，跨 GPU bit-exact。详细结果：[event_boundary_results_20261013.md](exp_record/event_boundary_results_20261013.md)。下面启动记录为历史状态。

**Primary screen 全失败：** 18 个 perturbed onset/arm 每组 anticipation 至多 **3 roots**（要求24），跨组去重仅 **5 roots**，紧接 pulse 的下一 decision anticipation 为 **0**。弱/渐变扰动大多不在 pulse 内违规，说明“加速太猛”只能解释先前失败的一部分，继续扩大同类 sweep 没有充分依据。

**新定位：恢复 policy 时的立即危险。** Early brake_ramp 的 pulse survival=100%，但 **125 roots** 在下一次原 OpenVLA proposal 的第一个执行 step 就违规；early hard_brake 有114个 next-decision immediate events，其中69个在第一步。原 predictor 在 early hard_brake diagnostic-check roots 的 recall **90% [81.97,96.83]**，static 三 seeds **20–25%**、GRU **3.33–11.67%**；本轮测试的 safe-score recalibration 未救回新 heads。None-state action reversal 在272 decisions/126 roots改变H10 event，而原 predictor/static max-score均不变；GRU能感知顺序仍未可靠识别真实事件。因此不能直接推论“加GRU就解决”或“只调阈值即可”。

**下一步目标修正：** 先登记 training-only paired resumption-transition diagnosis，定位 pulse结束处的接触状态/动作跳变，再对照原恢复、固定预算渐变衔接及 pause/re-contact；保留实际history与k5。接触重建/时间状态错配等机制目前仍是假说。Immediate-event data/loss 对照与 anticipation-support 问题明确分开；未通过原标准，不自动训练完整 H10 gate，不拿R5结果冒充预警成功。架构改动随后在 matched data/loss/budget 下验证。

## 实验设计与启动记录：三 GPU boundary/gate diagnosis（现已完成）

**直接目标：** 分别检验 (1) collector 是否跳过可预警区域；(2) frozen risk score 是否有辨别力、只是 threshold 不适合；(3) action-order 表示是否存在可测结构盲点。先收证据，再决定采集、event objective、calibration 或 architecture 的改动。不是直接扩大网络或再训练失败的 GRU。

**预登记并启动：** [event_boundary_20261013.md](exp_record/event_boundary_20261013.md)，实现/协议 commit `e103cfd` 在生产 launch 前完成。三个健康 RTX A5000 并行 early/middle/late goal-gap onset **0.20/0.12/0.06 m**，launcher PIDs **662796/662797/662798**；run IDs `event_boundary_20261013_a_{early,middle,late}`，各自日志 `~/scratch/<run_id>.log`。通过 `setsid nohup`、stdin `/dev/null`、离线加载与 CPU thread caps 持久运行，结束后自动做每个 shard 的分析。

**Matched design：** 每个 onset 跑 none / hard_drive / hard_brake / gain1.10 / x-ramp0.03 / x-ramp0.06 / brake-ramp-to0.25 七种 collector，全部最多一次五步干预。复用 **128 training-only roots**（74000–74003），从 reset 完整重放以保留可达性、实际 history 与 RGB；共 **84 batches、2,688 trajectories**，不能当成 2,688 独立 roots。H10/k5、full actor/microbatch8、八 scale labels 与 physics ranges 保持冻结，none 必须重现 pilot source。48 tests 与七 arm CPU smoke/自动分析已通过。

**启动复核：** 三个 shard 均完成首个 none batch（检查时合计 **3/84**），通过原 pilot replay；跨三个 GPU 的首批 state/context/proposal/eight-scale labels/七模型 scores/reverse labels 与 scores 均 **bit-exact** 一致。Python PIDs early/middle/late `662806/662807/662805` 正常运行。

**Primary screen：** 每种 onset/collector 在 post-pulse、pre-first-violation scope 至少 **24 anticipation roots**，四 physics strata 每个 ≥2，且 pre-violation pulse survival ≥90%。同时报告下一决策/以后决策、SR/CVR/safe、paired rescue/harm 和 CI。固定预算结束，不自动放宽标准、扩充范围或启动训练；通过后仍需 fresh train/dev collector pilot 才能做 production gate training。

**算法/架构定位：** 原 frozen predictor + static/sequence 全部三 seeds 仅作 observers。64 root fit /64 root diagnostic-check 分组比较原 tau 与 nominal 1%/5% safe-score thresholds，报告实际 false triggers、recall/precision 与 root-bootstrap CI；positive roots<12 或 safe roots<32 标为证据不足。另对每个 original proposal 做离线 reverse-order probe，检查 GT event 改变而模型 max-score 不变；不执行、不加入 runtime candidate budget、不计作原 proposal coverage。Threshold 改善支持先测试 calibration/event-gate；score 仍差需先 matched data/loss 对照；order blind spot 支持受控 order-aware 表示实验，但不能据此直接提升旧 GRU。更长 history/隐变量识别只在覆盖和 event objective 检验后再做。

## 已完成：coverage pilot 未通过（2026-10-13）

**Pilot 状态：** `event_gate_coverage_20261012_a` 已完成 **24/24 batches、768 trajectories**，PID `598105/598108` 已退出。独立复核 48 artifact hashes、coverage、配对 initial state/proposal/params、cadence、live trace→prior violation、executed-prefix parity 与 split isolation 均通过。没有启动新 gate training；后续 boundary diagnosis 运行状态见上节。

**Coverage 实测（独立 root 去重）：** train immediate/anticipation-only/safe-contact **33/0/126**，要求 **32/24/64**；dev **15/1/64**，要求 **16/12/32**。提前预警严重不足，不能用大量 safe windows 或 post-violation events 替代。Drive 在违规前施加的 pulses 中，train **117/125**、dev **57/64** 在该五步内就首次违规；它主要直接制造违规，未留下足够下一决策可预警的状态。Brake 提供 train/dev **26/14 immediate roots**，但 anticipation 仅 **0/1**。这是 collector failure，不是新 gate efficacy 结果。

**Completion review 提出的方向（现已实现为上节实验）：** 用本轮 training-only roots 做 boundary-reachability diagnosis，比较更早 onset、更弱/渐变五步扰动及 brake/re-contact，先测 pulse survival 和下一次原 OpenVLA proposal 的 R5/H10 event。保留 full actor、实际 history/RGB、H10/k5；数值 schedules/stop rules 先登记，不能任意编辑 state 伪造可达性。找到有效区域后冻结 observation-only collector，在 fresh splits 按原 coverage 标准重试；若仍没有 anticipation，检验 in-range event support 与采集可达性，不能只扩大数据量、偷偷改 physics ranges 或拿 audit labels 训练。之后才做 frozen-repair 的 coverage-only regression / event-aware gate 对照。

**下一项实验已实现：event-gate coverage pilot。** 先用 disjoint training/design-dev seeds 验证 event coverage，之后才决定 gate training。Train seeds 74000–74003 共 128 root episodes，dev seeds 75000–75001 共 64 roots；每个 root 配对 none / frozen / 单次 bounded drive / 单次 brake，共 768 trajectories，不能按 768 独立 roots 统计。Physical ranges、full OpenVLA H10/k5/microbatch8、八个 labelled scales 保持不变。Collector pulse 只读当前 goal gap 与预采样常量，GT future labels 不进入 collector 决策；这是训练数据探索，不是扩大 deployment action family。

**预登记 coverage screen：** original proposals、pre-first-violation scope、跨 branches 按 root 去重。Train 至少 immediate/anticipation-only/safe-contact **32/24/64 roots**，dev 至少 **16/12/32 roots**；friction/mass 四个固定 strata 每类至少 train 2/dev 1 root。达不到则修正 collector，再训练不自动触发；达到也只是 pilot feasibility，不是独立测试或部署许可。45 项测试与 CPU end-to-end smoke 已通过。详细 protocol：[event_gate_coverage_20261012.md](exp_record/event_gate_coverage_20261012.md)。

`risk_controller_diagnosis_20261012_a` 已完成 **20 cells × 11 arms**，原 launcher/Python PID 591432/591435 已退出。220 个 raw hashes、episode outcomes、risk trace→violation、cadence，以及 none/三个 L/L 的 source replay 均复核通过。本次状态检查没有启动新训练。下节的 launch/pending 描述是历史状态。

**关键结果（14 OOD cells，448 episodes/arm）：** none SR/CVR/safe = **87.05/29.24/65.63%**。保持 repair 只换 GT gate：frozen **80.80/8.04/77.46%**，static_s0 **83.71/8.04/78.79%**，sequence_s0 **81.25/8.48/76.79%**；相对 none 的 safe rescue/harm 分别为 **53/0、59/0、50/0**。GRU G/L 相对其 L/L 的 safe delta **+35.49 pp [+31.25,+39.73]**、rescue/harm **162/3**；仅换 repair 评分的 L/G 为 **+6.70 pp [+3.79,+9.60]**、**39/9**。固定网格 paired CI、仅新模型 s0 diagnostic，不是三 seed efficacy 验证。G/G safe=76.12%，不是方法上界。

**解释边界：** 当前最强的 tested lever 是 event detection / gating 及其后续闭环效应；repair 也有影响。GT gate 同时替换风险估计与 learned-quantile threshold 为真 H10>1，未分离 calibration 与 representation；且可访问隐藏 dynamics，不能证明现有 observation context 足以学到同等 gate。GT-gated arms 仍牺牲 **3.35–6.25 pp SR**；GRU G/L 相对 none 的 ΔSR=**−5.80 [−8.48,−3.35] pp**、ΔCVR=**−20.76 [−23.21,−18.30] pp**、Δsafe=**+11.16 [+8.93,+13.62] pp**。不以 privileged 收益通过 learned-verifier transfer gate。

**下一步顺序：**
1. 先登记并实施 fresh **training-only contact/event coverage pilot**，物理 train ranges 保持当前范围；从可达 near-contact 与 post-repair states 采集原 proposals/八个 scales，记录 collector 干预。按 episode/branches 划分，train-only normalization。报告 safe-contact、R5-unsafe、H10-only-unsafe 的独立 episode 数，不能以相关 windows 充数。生产训练前预登记 coverage 数量与多样性验收条件；当前未声称满足。
2. Runtime 拆成 **learned event gate → frozen original repair predictor/rule**。监督 `y5=max(r[:5])>1`、`y10=max(r[:10])>1`，保留 anticipation；GT 只用于离线 labels，隐藏物理参数不作为 learned gate 输入。
3. 同一合格数据先重训 regression-gate controls 隔离 coverage 改动，再分别 ablate event-aware objective；保留 matched static/causal 架构、信息/参数/预算与固定 repair。不混改数据、loss、gate、history 与候选空间。重点看 matched false-trigger rate 下 recall/precision 和 contact-stage calibration，不只看 MAE。
4. 在独立 dev/calibration 上预登记 threshold-selection rule，再跑闭环 SR/CVR/safe + paired rescue/harm。若可靠 gating 仍伤 completion，再单独测试 acceptance/continuation；若 event discrimination 仍差，检验 observability/history。方法冻结后才做 independent confirmation。

完整记录：[closed-loop factorial completion](exp_record/sequence_risk_20261012.md#completion-update-closed-loop-factorial-diagnosis)。LIBERO baseline feasibility 可独立准备；正式 verifier transfer/hardware 继续后置。

## 已完成：temporal-risk validation 失败与离线因果诊断（2026-10-12）

`sequence_risk_20261012_a` 已完成全部 20 cells × 8 arms；**不推广 static 或 GRU 新模型**。14 OOD cells、每 arm 448 episodes：none SR/CVR/safe = **87.05/29.24/65.63%**，frozen = **84.60/28.79/64.73%**；static 三 seeds 的 safe = **53.57/51.34/53.57%**，GRU = **41.29/36.83/47.32%**。GRU−static 的 paired safe delta 为 **−12.28 [−16.52,−8.04] / −14.51 [−18.75,−10.27] / −6.25 [−10.27,−2.23] pp**；区间条件于固定网格，三个训练 seeds 使用相同 evaluation episodes，不能当作三组独立样本。完整 SR/CVR、rescue/harm、CIs 见 [temporal-risk postmortem](exp_record/sequence_risk_20261012.md)。ID cells 也明显退化，不能只归因于 OOD。

**已核实的原因线索：**

1. **表示 proxy 改善不等于干预有效。** GRU development MAE 与 same-state candidate ranking 更好，但 baseline 固定轨迹上的 H10 unsafe trigger recall 仅 9.68–11.69%，static 为 2.42–3.23%，frozen 为 26.21%。在相同 3,050 个 GT safety-boundary candidate pairs 上，GRU 排序错误 370–443 对，static 702–906 对；固定 GT denominator、prediction ties 单独报告。改善 within-state ordering 不能替代 across-state detection。
2. **当前新数据没有充分覆盖 hazard。** Train 6,144 个 originals 只有 20 个 H10 unsafe，而且全部 R5 unsafe；原 proposal 中缺失 H10-only anticipation examples。49,152 个 scale candidates 只有 152 unsafe，来自 34 episodes。Dev unsafe originals 仅 H10=3、R5=2，因此 100% dev recall 不足以支持可靠性结论。30% hot-decision sampling 并未解决 event diversity；也没有覆盖新模型干预后的所有 reachable states。
3. **离线交叉替换优先指向 gate。** 新增 `sequence_diagnosis_20261012_b` 在 baseline pre-first-violation 的 5,572 个 OOD states，保持同一 repair rule，仅把 gate 换成 GT H10>1：GRU H10 local rescues 从 17/17/19 增至 107/107/107；仅把 repair score 换为 GT 则为 18/18/20。GT gate+GT repair 为 110/248 H10、16/131 R5 local rescues；不是闭环收益，也不是全 action space 上界。不能把 GT H10 risk 写成 continuation value。
4. **算法风险仍在。** GRU 有 51/69/73 次 execution-changing repair 未达到自身 relative target；responsiveness 不等于 accepted safety。同状态 L/L 没有诱发新的 R5 violation，却存在严重闭环 harms，必须检查干预后的状态分布与后续 replanning，不能只靠当前 chunk 的局部指标。

**下一步已实现并启动：** `risk_controller_diagnosis_20261012_a`，policy-alone + frozen/static_s0/sequence_s0 各自 L/L、G/L、L/G + 共享 G/G，共 11 arms × 20 cells × 32 episodes。L/G 分别指 learned/GT gate 或 repair；GT gate 是物理阈值 1，learned gate 保留各自 τ，GT repair 只替换评分，六次 bisection、relative target、floor、abstention、H10/k5/microbatch8 不变。s0 是诊断约定，不挑最佳 seed；正式架构结论仍需其余 seeds。None 和 L/L 必须复现原结果。此项使用 privileged GT 做 component diagnosis，不作为可部署方法或公平 information-budget efficacy 对照。

**停止盲目扩架构：** 先根据闭环 G/L 与 L/G 的 paired outcomes 定位 gate/repair；若 G/G 也弱，再分开检验 candidate family、acceptance 和 continuation。再训练前先登记 train/dev 的独立 hazard episode 与 anticipation coverage 验收，保持 static control；data collection、event loss、gate 与 action freedom 分步改变。当前不进入 verifier 正式迁移，LIBERO baseline feasibility 可独立准备。

**状态与验证：** source/artifact/dataset/checkpoint hashes、episode split、train-only normalizers、baseline replay/cadence/outcomes 全复核通过；42 项测试与 11-arm BC smoke 通过。离线 `_a` CPU 尝试因 simulator 数值 parity 不符停止，`_b` 使用原 CUDA 设备并逐 cell 严格通过；未放宽容差。正式闭环 launcher PID `591432`，日志 `~/scratch/risk_controller_diagnosis_20261012_a.log`，`setsid nohup` detached；本条记录时结果尚待完成。

## 已完成：matched-data temporal-risk validation 配置（2026-10-12）

**不是笼统地把 verifier 做大。** 当前证据支持三个待分别解决的瓶颈：risk 对 action order 的表示不足、intervention 的后续闭环影响、finite scale family 在部分 decision 无解。第一轮仅验证时序表示，尚不改变 runtime trigger/repair 算法、历史长度或 sensing 权限。

**已实现并完成 `sequence_risk_20261012_a`：**

- `src/models/sequence_risk.py`：static MLP 与 causal GRU risk ensembles（E=5）。Static hidden=157，共 269,265 trainable parameters；GRU hidden=128，共 269,450，差约 0.07%。相同 obs / obs_prev / a_prev context、normalizers、两维物理 risk outputs；GRU 的第 j 步仅依赖 context 和前 j 个动作。两者共用 frozen state/progress dynamics。原 checkpoint 保留作 anchor。
- `src/experiments/validate_sequence_risk.py`：fresh collection → 同数据训练 → independent calibration → audit same-state validation → 全网格闭环 validation。12×32=384 training episodes（seeds 71000–71011），4×32=128 development（72000–72003），4×32=128 calibration（73000–73003）。参数分布保留现有 predictor training ranges：friction multipliers [0.5,1.5]、mass [0.6,1.6]；没有顺带扩大 OOD coverage。
- Train/dev batch 交替使用 uncorrected policy 与 frozen bisection，覆盖其 reachable states；calibration 仅 uncorrected OpenVLA、无 execution noise。在每个真实 decision 离线标注原 proposal 和同一组八个 scales 的 H10 risks，不把 GT labels 放入 controller。训练 normalizer 仅 fit train decisions；episode 和全部 candidates 不跨 split。
- 两架构分别跑训练 seeds 0/1/2，各 3,000 steps、batch 256、同一 per-step MSE（risk target clip=3）、30% hot-decision sampling、相同 optimizer/LR schedule/批次。固定末次 checkpoint、全部 seeds 报告，不挑最好 seed。参数/步数匹配不等于 FLOPs 或 wall-clock 匹配，因此另外记录各结构 training time。
- 每个新 head 独立用 calibration 的 safe original H10 decision scores 校准 α=0.05。完整模型仍是 frozen OpenVLA-7B + H10 DAgger×2 head、forward microbatch=8、chunk_k=5；bisection 原规则保持一致。此轮 candidate dataset 固定八个 scales；bisection runtime 仍使用原六次连续 search，插值泛化是两模型共有的额外测试，不冒充离散执行。
- Offline validation 在原 audit 的 baseline/frozen trajectories 上同状态比较 prefix/H10 detection、safe-trigger rate、candidate ranking；这部分仅 evaluation，不参与拟合。之后 20 cells×32 episodes，baseline、frozen anchor、3 static、3 GRU 共 8 arms；none/frozen 必须逐 episode 重现原 validation。报告各 training seed 的 SR/CVR/safe、paired rescue/harm、fixed-grid CI 与 sequence-vs-static 直接配对差值。16000+ confirmation 和 46000+ final-test 不用于选择。

**验证与运行状态：** 8 项新测试 + 29 项原 audit/cadence/candidate tests 全通过；CPU end-to-end BC smoke 完成；offline smoke 的 frozen prefix counts 精确复现 59 true triggers / 86 misses / 337 safe-proposal triggers。Smoke 不构成架构有效性证据。正式实验原 launcher PID `587112` 已退出，`status.json` 为 `complete=true`；日志 `~/scratch/sequence_risk_20261012_a.log`，输出 `src/results/push_torch/sequence_risk_20261012_a/`，GPU UUID `GPU-c164f9d9-f157-daeb-d7be-5c98029c2a2b`。负结果与后续诊断以上一节为准。

**结果如何决定下一步：** 若 GRU 只降低 MAE、未改善 same-state discrimination/ranking 或闭环 safe success，不能升为默认；若 same-state 明显改善但闭环仍失败，下一项固定模型测试 intervention gate/continuation，而非立刻增宽网络；若 static 与 GRU 都改善且差异不明，则主要收益可能来自 policy-matched data 或 risk/state 解耦，不能声称 temporal novelty；若均失败，检查 coverage、隐藏 dynamics/history 与候选可控性。

### 进入 LIBERO / simulated arm / physical robot 的 gate

1. **LIBERO 单任务 baseline feasibility 可以现在准备。** 它是 action/camera/robot-controller/checkpoint 兼容性与可重复 rollout 检查，不要求先证明 verifier gain。当前 pushing 的 2D velocity head 不能直接控制机械臂；需要任务匹配的 policy/checkpoint、EE/gripper action adapter、task-native success 和 MuJoCo contact labels。环境检查：MuJoCo 3.14.0 可用，robosuite/LIBERO/RoboCasa 当前未安装；兼容性未验证，不在共享 base 中贸然替换依赖。
2. **正式 verifier transfer 的 prospective go/no-go。** 本轮结果出来前登记工程目标：相对 policy-alone 的 pooled OOD safe success 至少 **+3 pp**，独立 episode 测试的 fixed-grid paired 95% CI 下界 >0；3 个 training seeds 均报告且点估计方向一致，再以第二组独立 episode seeds 复核方向。SR 下降超过 2 pp 或关键 stress cells 明显退化，需重新分析，不单靠 pooled gain 推进。这是项目资源分配目标，不是统计定理、安全认证或 benchmark 通用标准；本轮 validation 只做筛选，不能代替独立测试。
3. **LIBERO verifier transfer 仍是新的实验。** 重建 state/action/contact adapters，重新采集训练标签、训练并校准；首先一个 baseline 能进入接触阶段的任务，对照相同 sensing/control/candidate budget，再扩展任务。Pushing 通过 gate 只说明值得迁移，不保证 LIBERO 有效；信息量增加与快内环须有 matched classical feedback/impedance 对照。
4. **真实机械臂继续后置。** 先在目标 embodiment 的 simulated arm 上证明可重复任务完成和 verifier 净收益，验证实际可取得的 proprioception/object/contact 输入、动作单位/坐标系/频率及真实接口。硬件阶段另定范围与实验协议；没有某个 pushing SR 达标就自动进入 hardware 的规则，也不要求仿真达到不现实的 100% success。

## 完整 audit 后的 cause-first gate（2026-10-12）

**状态更新：** `verifier_audit_20261012_a` 已完成全部 20 cells / 80 arm replays，`summary.json` 为 `complete=true`。全部 outcomes、cadence 和执行 prefix risk 检查通过；此前“audit 正运行”段落为当天历史状态。CPU attribution artifact：`src/results/push_torch/audit_attribution_20261012_a/{analysis.json,report.md}`。这不是新的 efficacy 实验，validation headline 不变。

**主要证据（14 OOD cells，448 episodes/arm；以下以 bisection 为主）：**

- **Detection 与可控性并存。** 145 个 unsafe five-step proposals 中触发 59、漏检 86；86 次漏检仅 14 次存在安全 scale，72 次八个候选均 unsafe。129 个实际首次违规 episode 中，107 个在该 decision 没有安全 scale，22 个有；后者分为 14 次漏检和 8 次触发后仍 unsafe。不能把“无安全 scale”写成任意动作均无解，更不能将这类局部计数当作新算法的全局收益上界。
- **H10 不能替代执行标签。** H10 unsafe proposals 为 265，触发 78、漏检 187；95 次漏检存在安全 scale。Step 6–10 是不 replanning 的 counterfactual，不能把 H10 miss 数直接当实际 episode recall。396 次触发中有 337 个原 proposal prefix-safe，其中 318 个 H10 也 safe；prefix-safe lookahead trigger 不自动等于误干预或 causally harmful。
- **Harm 是后续安全损失，而非单纯 stall。** Bisection paired rescue/harm = 8/12；12 个 harms 全部 violation，只有 3 个同时 success loss。12 个首次 execution divergence 都与 baseline state/proposal 一致、原 H10 safe、修复后 prefix safe，但在 5–8 control steps 后首次违规。Discrete/progress 的 9 个 harms 同样全部 safety loss，仅各 2 个同时 success loss。局部风险下降不能证明整条闭环轨迹更安全；当前不能宣称减速是唯一原因。
- **时序表达有明确限制，但尚无净收益因果证据。** Bisection 的 11,186 个顺序探针 score 全部不变，1,838 个真实 H10 safety labels 改变。探针相关且可能 off-policy；用于约束 architecture hypothesis，不作为新增 runtime candidates。
- **不能把主要瓶颈直接归为 progress ranking。** Bisection 的 learned-accepted、GT-prefix-safe 候选中，2,208 个 decision 有 ≥2 个可比选项，只有 9 个的 predicted-progress choice GT regret >1 mm；triggered 子集为 9/297。这是经过 filter/GT safety 条件化后的局部 goal-distance proxy，不证明总体 progress 准确或 task success 可预测。

**最小诊断实验已完成：** `intervention_branch_20261012_a` 完成全部 8 cells，覆盖原 validation bisection 的全部 12 harms + 8 rescues。在每个 episode 首次 execution-changing intervention 时跳过一次修复，之后仍运行同一个 frozen verifier；所有 32 rows 保留。整 batch control replay、branch 前与未选中 rows 一致性、H10/k5/microbatch 8 检查全部通过；复核 summary/audit/raw hashes 和 NPZ outcomes 一致。12 个 harms 跳过后全部不再 violation，11 个恢复 safe success；剩余 1 个仍未完成。8 个 rescues 跳过后全部 violation，7 个仍完成、1 个失败。进程正常退出。Selection 使用 hindsight outcome 是离线诊断，不能把 11/12 或 11−8 换算成可部署新方法的收益。

**分支结果后的选择条件：** 如果跳过首个干预能反转 harms，需要先设计能降低不必要干预的 temporal/rollout-conditioned risk ablation，并同时检查 rescues 是否损失；若 harms 不反转，检查后续 interventions/接触轨迹再归因。若某些 case 到当前 decision 已无安全 scale，先用 zero-action / bounded retreat 等离线 witness 或更早 decision branches 区分 finite-action-set 不足与 intervention timing，不能直接上线更大 candidate set。Contact history、directional repair 与 inner-loop 改动暂不合并实施。

**防止数据泄漏：** Audit 的 validation labels 只用于诊断与设计选择，不进入拟合。若 temporal risk ablation 获批，另行生成 disjoint training-seed trajectories/candidates；旧 formulation 与新架构用相同 training-only data、coverage 和预算训练，在独立 calibration 上重新定阈值，再在未用于训练的 validation 上比较。Confirmation 16000+ 与 final-test 46000+ 不用于调参。History length、risk head、repair freedom 与 cadence 逐项隔离。

### 完成 counterfactual 后的 architecture / algorithm 路线（12 Oct）

**框架评价：** evaluation/replay 基础设施已能支持可信诊断；learned-verifier 的稳定净收益仍未建立。这里的 branch 支持一个具体因果结论：在选中的冻结 validation 轨迹上，单次早期修复及其后续闭环后果既能造成 safety loss，也能避免 safety loss；它没有隔离物理惯性、姿态、接触、后续 policy proposal 与后续 verifier 决策各自的贡献。主瓶颈应表述为 **intervention discrimination + temporal consequences**，不是“所有修复无效”或“已证明换 temporal head 必然有效”。

**已有 score 不能靠一个阈值完美分开这些 cases。** 首次干预的当前 score：12 harms 为 0.5634–0.9440，8 rescues 为 0.5610–0.7873，当前 τ=0.55549。区间交错，因此高分才干预的单一阈值无法同时去掉所有这些 harms 并保留所有这些 rescues；不排除普通 calibration 改善平均结果。12 harms 的原 H10 全 safe，7/8 rescues 的原 H10 unsafe，但还有一个 H10-safe rescue 在跳过后 46 control steps 才违规：H10 open-loop 风险仍不等于长期 closed-loop utility。

**第一项 architecture ablation：direct causal action-sequence risk。**

```text
当前 obs、obs_prev、a_prev → context encoder → z0
candidate a1 → GRU(z0,a1) → z1 → risk1
candidate a2 → GRU(z1,a2) → z2 → risk2
... → z10 → risk10
```

- 第一版保留相同观测信息、ensemble 规模、H10/k5 与固定八个 scale；选小型 GRU，不先增加 Transformer、force sensing 或 history length。也可把 causal action-prefix MLP/TCN 留作后续替代，但不同时 sweep 多种结构。
- `risk_j` 依赖 `a_1…a_j` 和观测 context，不只依赖 `a_j`；不依赖尚未可靠的 autoregressive predicted observation，也不直接重启旧 rollout-risk 分支。分步物理风险与 ensemble 输出保持可微，兼容现有 predictor interface。
- 为隔离表示作用，建立 static-risk control 与 sequence-risk head，在同一新 training-only candidate dataset、相近参数量、同一损失/步数/ensemble/校准协议下训练；两者共用不变的 state/progress predictor。另保留原 frozen model 作 anchor。不要把单独拆出 risk head、换数据、换 objective 的收益都归给 GRU。
- 数据从 disjoint training seeds 的真实 OpenVLA decision states 产生，覆盖原 proposal、既定 scales、修复后能到达的状态及 safe/near-contact/unsafe transitions；两结构的 coverage 相同。整个 episode 及其 branches/candidates 同属一个 split，normalizer 只 fit training episodes；不能沿用 `train.py` 中用全部 train/val observations fit normalizer 的写法作为新严格 held-out 协议。
- 第一轮用匹配的 per-step risk regression；接触事件 BCE / within-state ranking loss 后续分别 ablate。先检验 early warning、prefix/H10 recall–false-trigger tradeoff、candidate ranking、校准与闭环 safe success，而不是只优化 global MAE/AUROC。

**第二项 algorithm ablation：separate immediate risk, anticipation, and intervention value。**

- 从同一逐步 risk trace 分别得到 execution-prefix `R5` 与 open-loop lookahead `R10`，各自校准。保留 H10 anticipation：原 rescues 的多数 first-divergence prefix 本来就 safe，不能一律改成 prefix-only trigger。
- Candidate set 必须含原 proposal。触发后，比较 candidate 的绝对物理风险、相对风险改善、修改量和任务进度；不能只追求 `min(0.8,0.75*s0)` 这样的 surrogate 相对降幅。无合格 candidate 时记录 unsupported/abstain，返回原动作也不宣称 safety guarantee。
- 最小试验先保持 sequence model 和 candidate budget 固定，比较原 trigger/acceptance 与 independently calibrated prefix+lookahead gate；随后单独测试 paired risk-improvement margin/minimal-change selection。Ensemble spread 是未校准的 disagreement，不称概率保证；paired candidate improvement 需要独立校准，不能把两个边际 confidence 分数相减当有效置信下界。
- 更长期的 architecture 目标是 **closed-loop continuation / recoverability value**：执行 candidate 前 5 步后，接回固定版本 policy/controller，对 future violation 与 completion 做分开预测。用新 training-seed states 的 paired branches 监督，而非当前 20 个 outcome-selected cases。它与 `R10` 的训练目标和标签必须分开。先做短 continuation 诊断，再决定是否值得训练 value head，不立即做完整 MPC 或在线 GT 搜索。

**第三项：history 与 repair expressiveness，按诊断独立推进。** 历史输入先用已有观测/动作的较长 window 推断隐含 dynamics；force/contact sensing 是额外信息，另做 information-matched ablation。对首次违规时八个 scale 均 unsafe 的 cases，离线比较 same-state zero action、bounded retreat，以及前一 decision 的相同动作集合，区分有限候选不足与介入过晚；仅有局部可救 witness 仍不等于整条任务可完成。确有方向动作必要时，再比较固定预算的 directional family 与 scale-only，而非简单追加更多候选。Grad repair 需等 score 的 finite-perturbation/GT ranking 与 constrained optimization audit 通过，再作为单独算法比较。

**实施顺序与成功 gate：** `frozen anchor → matched-data static / sequence risk → fixed-model gate ablation → history OR bounded directional repair → independent final test → one daily precision-contact task`。先预定 practically meaningful safe-success 目标，再检查多训练 seeds、paired harm/rescue、SR/CVR/safe 与固定物理网格 CI；改善 detector 指标但闭环无净收益就不推广。日常任务仍先调查轻柔关抽屉或杯瓶入窄槽 baseline；新增快内环需匹配 classical feedback/impedance baseline，不归功于 learned verifier。

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
