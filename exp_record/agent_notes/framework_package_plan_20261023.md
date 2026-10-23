# 框架独立包装计划 (Framework Package Plan)

日期：2026-10-23  
状态：规划中，未实施  
相关记录：`src_architecture_evaluation_20261023.md`, `literature_comparison_20261023.md`

## 一、为什么要包装

当前框架核心逻辑散落在 `src/` 和 `scripts/run_framework_comparison.py` 中，且 runner 内硬编码了大量实验专用路径（baseline 引用、GPU lane、初始状态、场景文件）。要把框架交给其他人或其他场景使用，需要抽取成一个可独立安装、可直接下载冻结 VLA 后即可接入的 Python package。

## 二、目标包结构

```
vla_repair_framework/            # 独立 PyPI / git package
├── pyproject.toml               # 构建元数据与依赖
├── README.md                    # 快速上手：下载 VLA -> 初始化 Framework -> 每步调用 step()
├── vla_repair_framework/
│   ├── __init__.py              # 对外标准 API
│   ├── framework.py             # 核心编排类 Framework
│   │   - __init__(vla_model, task_adapter, config)
│   │   - step(observation, candidate_actions=None) -> executed_action
│   │   - verify(state) -> bool
│   │   - handoff(state) -> bool
│   ├── config.py                # 预算、阈值、阶段定义等配置（可从 YAML/JSON 加载）
│   ├── controller.py            # 状态机（从 src/controller.py 抽取，去除 microwave 硬编码）
│   ├── repair/                  # 修复相关
│   │   ├── __init__.py
│   │   ├── recovery_scorer.py   # 模块A：动作条件恢复评分器
│   │   ├── handoff_scorer.py    # 模块B：交接就绪评分器
│   │   ├── physics_evidence.py  # 规则式物理证据提取
│   │   └── candidate_pool.py    # 候选池选择逻辑
│   ├── verification/            # 验证相关
│   │   ├── __init__.py
│   │   ├── physical_verifier.py # 规则式物理检查（无学习参数）
│   │   └── fusion_mlp.py        # 融合 MLP（可选，可替换为规则）
│   ├── perception/              # 感知相关
│   │   ├── __init__.py
│   │   └── frozen_visual_encoder.py  # 严格 DINOv2 编码
│   ├── task_adapters/           # 任务适配器（通用化关键）
│   │   ├── __init__.py
│   │   ├── base.py              # BaseTaskAdapter（抽象类）
│   │   └── microwave_adapter.py # 微波炉任务适配器示例
│   ├── memory/                  # 记忆相关（可选）
│   │   └── keyframe_bank.py
│   └── training/                # 离线训练（不属于运行时核心）
│       ├── __init__.py
│       ├── train_recovery_scorer.py
│       └── train_handoff_scorer.py
├── tests/                       # 单元测试（模型免，CPU 可跑）
│   ├── test_controller.py
│   ├── test_repair_candidates.py
│   └── test_physics_evidence.py
├── examples/                    # 使用示例
│   └── microwave_example.py     # 下载冻结 VLA -> 构建 Framework -> 运行一步
└── data/                        # 空，或仅存示例配置
```

## 三、通用化重点（任务适配器）

当前硬编码在 `src/controller.py` 中的内容必须从框架核心中移出，交由 `TaskAdapter` 提供：

| 当前硬编码 | 通用化后由 TaskAdapter 提供 |
|-----------|---------------------------|
| `target = 'white_yellow_mug_1'` | `adapter.target_object` |
| `receptacle = 'microwave_1'` | `adapter.receptacle_object` |
| `region = 'microwave_1_heating_region'` | `adapter.region_name` |
| `prompts = dict(acquire=..., transfer=..., ...)` | `adapter.get_prompt(stage)` |
| `recovery_contracts = dict(acquire=['held_clear','contained'], ...)` | `adapter.get_recovery_contract(stage)` |
| 肯定/否定谓语 `In`, `Close` | `adapter.get_success_predicates()` |

Framework 类本身不应知道“杯子”或“微波炉”，只需知道：
- 当前阶段（stage）
- 当前恢复目标（recovery goal）
- 接受合同（acceptance contract）
- 成功谓语（success predicates）

这样同一个 Framework 实例可以通过更换 TaskAdapter 处理不同任务。

## 四、运行时依赖（保持最小）

- 冻结 OpenVLA（用户自行下载，通过 adapter 注入）
- DINOv2 预训练权重（可选，视觉特征）
- NumPy, PyTorch（仅 scorer 和 fusion MLP 需要；纯规则路径可不装 Torch）
- 没有 RL、没有可微分物理引擎、没有完整 world model

## 五、实施步骤

### Phase 1：抽取核心（当前可做）
1. 从 `src/controller.py` 抽出 `NativeFramework`，去除 microwave 硬编码，改为通过 `task_adapter` 注入目标、容器、提示语、合同。
2. 把 `src/repair/*`、`src/verification/physical_verifier.py`、`src/perception/frozen_visual_encoder.py` 移入新包，保持接口兼容。
3. 建立 `task_adapters/base.py` 与 `microwave_adapter.py`，把 `src/controller.py` 里硬编码的 microwave 参数迁移过去。
4. 保留现有 `src/` 与 `scripts/` 不动，新包作为并行实现，避免破坏现有实验。

### Phase 2：标准 API（之后做）
1. 实现 `Framework.__init__(vla_model, task_adapter, config)`。
2. 实现 `Framework.step(observation, candidate_actions=None)`：内部完成监测、候选选择、执行、验证、交接决策。
3. 实现 `Framework.verify(state)` 和 `Framework.handoff(state)` 对外暴露。

### Phase 3：训练与评估脚本（之后做）
1. 把 `scripts/train_recovery_scorer.py`、`scripts/train_handoff_scorer.py` 移入新包 `training/`，支持从任务适配器读取数据路径。
2. 添加 CPU 可跑的单元测试，不依赖 GPU 或 VLA。

### Phase 4：发布（之后做）
1. 添加 `pyproject.toml`，声明依赖（numpy, torch 为可选依赖）。
2. 添加 `README.md` 与 `examples/microwave_example.py`，演示“下载冻结 VLA -> 接入 Framework -> 运行一步”。
3. 发布到 PyPI 或作为 git submodule 供后续实验复用。

## 六、与现有实验的关系

- 新包 **不替换** 当前 `src/` 和 `scripts/`，现有实验（`/home/jack/scratch/` 下的 run、plan、episode）继续使用原有路径。
- 新包在验证通过后，可作为后续实验的导入目标，逐步替换 `src/` 中对应模块。
