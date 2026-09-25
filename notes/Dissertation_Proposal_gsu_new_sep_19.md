# Dissertation Proposal (Revised — Sep 2026)

**Title:** Integrating Neural Physics Engines into Embodied Foundation Models for Enhanced Physical Reasoning and Robust Generalization in Embodied Agents

## Principal Goal

The principal goal of this research is to resolve the generalization bottleneck in embodied agents navigating complex, unseen physical environments by embedding differentiable Neural Physics Engines (NPEs) into the decision-making pipelines of Embodied Foundation Models (EFMs) — increasingly realized today as Vision-Language-Action (VLA) models and world models. This aims to endow agents with physical intuition, enabling zero-shot generalization in contact-rich and deformable object manipulation tasks, thereby bridging the sim-to-real gap.

## Background & Problem Statement

While current probabilistic Large Language Models (LLMs) demonstrate exceptional proficiency in semantic analysis and contextual language understanding, they struggle to serve directly as the "brains" of AI robotics. Existing models primarily rely on statistical probabilities for text or action prediction and lack a fundamental understanding of physical laws. Consequently, when an agent faces complex, dynamic, and unseen physical environments, its generalization capability degrades severely.

This is particularly evident in tasks involving deformable objects (e.g., folding garments) or high-precision force control (e.g., cable insertion). In these scenarios, traditional gradient-free reinforcement learning is highly inefficient and prone to execution failure due to predictive deviations. Moreover, agents relying purely on surface-level pattern recognition — rather than tracking the intrinsic properties of objects and the causal/spatial relationships between them — tend to exhibit brittle behavior, such as misjudging grasp conditions on fragile items. Therefore, successfully injecting deterministic, causally-grounded physical laws into probabilistic neural foundation models is a critical challenge for the next decade of embodied AI.

Current EFMs are also increasingly defined by the heterogeneity of their training data. The "Embodied Data Pyramid" framework categorizes data sources into real robot data, UMI-style data, egocentric/exocentric video, simulation data, and general-purpose data — each contributing differently to an agent's perception, reasoning, action generation, and world-prediction capabilities. This project's methodology is designed to exploit this heterogeneity rather than relying on a single data modality.

## Proposed Research & Methodology

This project proposes a universal paradigm combining computational linguistics and physical simulation. Key research components include:

1. **Integration of Neural Physics Engines (via Genesis).** Diverging from traditional black-box exploration, this project leverages cutting-edge universal differentiable physics platforms — most notably **Genesis**, a generative physics engine developed collaboratively across 20+ research labs that has become a de facto standard for embodied AI and robotics simulation since 2024–2025. Unlike earlier tools largely constrained to rigid-body dynamics (e.g., MuJoCo, Brax), Genesis unifies rigid body, fluid, and soft-material simulation in a single differentiable system — directly matching this project's target of deformable-object manipulation (e.g., garment folding). Physical gradients (gravity, friction, deformation dynamics) can be backpropagated directly to the control policy, offering substantially higher sample efficiency than gradient-free reinforcement learning.

2. **Causal and Structured Physical Chain-of-Thought (CoT).** Building on the reasoning patterns of models like DeepSeek-R1, we will implement a Physical CoT mechanism that is both *structured* and *causally-grounded*, rather than free-form text. Before outputting an action token, the model will be forced to produce a machine-parseable trajectory of physical states — following the pattern demonstrated by 2026 work such as PhysX-CoT (which reframes single-image 3D generation as an explicit structured physical reasoning process spanning decomposition, 2D/3D grounding, geometric and surface cues). In parallel, informed by CausalPhys-style findings, the agent's reasoning will explicitly infer intrinsic object properties and track spatial/causal relationships between entities (e.g., Observation: fragile material → Constraint: torque limit → Causal link: excess force → deformation/breakage → Action: motion planning), rather than depending on surface-level pattern recognition alone. Scaffolding reasoning with causal graphs is intended to prevent the brittle failure modes (e.g., misjudged grasp conditions) that arise from purely correlational VLA reasoning.

3. **Spatial RAG & Episodic Memory.** Traditional text-based Retrieval-Augmented Generation (RAG) will be transformed into 3D Scene Graphs. By leveraging vector databases and geometric element clustering, the robot will perform real-time reasoning based on spatial geometry and historical physical interactions rather than just linguistic context.

4. **Target Scenario Validation.** The evaluation will focus on contact-rich and deformable object manipulation. We will validate the architecture's effectiveness in zero-shot or few-shot execution of highly complex tasks, such as garment folding or socket insertion, and will discuss how heterogeneous data sources from the Embodied Data Pyramid (simulation, egocentric video, real robot logs) are combined during training/evaluation.

## Expected Outcomes

1. A systematic framework for injecting causally-structured physical rules into the reasoning pipelines of Embodied Foundation / VLA models.
2. Development of a simulation evaluation environment (built on Genesis, with comparison against MuJoCo/Brax baselines) supporting physical gradient backpropagation and reinforcement learning.
3. Experimental validation demonstrating that this architecture achieves higher generalization success rates in handling deformable objects compared to pure data-driven models, and that causally-grounded CoT reduces brittle failure modes relative to unstructured CoT baselines.

## Completion Criteria

**Baseline completion:** A thorough workload characterisation and reproduction of existing benchmarks, including characterizing existing embodied AI baselines and reproducing the Physical CoT's improvement on simple grasping tasks within a simulated (Genesis) environment.

**Outstanding completion:** A novel systems contribution — namely, a Genesis-driven, causally-structured Physical CoT Embodied Foundation Model. This includes rigorous experimental evaluation demonstrating superior zero-shot generalization capabilities in complex deformable object manipulation tasks, alongside preliminary sim-to-real transfer validation.

## Additional Information

| Category | Details |
|---|---|
| Difficulty | 3 - Hard |
| Essential Skills | Python and/or C++ programming, understanding of computer architecture and memory hierarchy, familiarity with PyTorch and LLM architectures, basic knowledge of profiling and performance analysis. Familiarity with multimodal Vision-Language-Action (VLA) model architectures. |
| Desirable Skills | CUDA programming, familiarity with hardware. Experience with differentiable physics simulators (**Genesis**, MuJoCo, Brax), Unity environment development, deep understanding of underlying mechanisms for LLMs (e.g., LangChain, vector databases). |
| Resources Required | GPUs, potentially CPUs for simulation. Computation clusters for differentiable physics simulation, potentially physical robotic arms for real-world (sim-to-real) evaluation. |

## Citations

```
@inproceedings{10.1145/3731569.3764843,
  title = {KTransformers: Unleashing the Full Potential of CPU/GPU Hybrid Inference for MoE Models},
  author = {Chen, Hongtao and Xie, Weiyu and Zhang, Boxin and Tang, Jingqi and Wang, Jiahao and Dong, Jianwei and Chen, Shaoyuan and Yuan, Ziwei and Lin, Chen and Qiu, Chengyu and Zhu, Yuening and Ou, Qingliang and Liao, Jiaqi and Chen, Xianglin and Ai, Zhiyuan and Wu, Yongwei and Zhang, Mingxing},
  booktitle = {Proceedings of the ACM SIGOPS 31st Symposium on Operating Systems Principles},
  year = {2025}
}
```

Additional citations to be added upon comprehensive literature review, focusing on:

- **Genesis**: a universal, differentiable generative physics platform for robotics and embodied AI (multi-lab collaborative release, 2024–2025).
- **PhysX-CoT** (2026): structured physical chain-of-thought for single-image 3D generation via part-level state trajectories (decomposition, 2D/3D grounding, geometric/surface cues).
- **CausalPhys**-style work on causal reasoning over intrinsic object properties and spatial relationships in physical agents.
- **Embodied Data Pyramid**: taxonomy of data sources (real robot, UMI-style, egocentric/exocentric, simulation, general) for training embodied/VLA foundation models (2026).
- Differentiable physics foundations (Brax, DiffTaichi, MuJoCo MJX) as prior-generation baselines for comparison against Genesis.

---
*Note: This revision is a draft update layer over the original `Dissertation_Proposal_gsu.pdf`. Please verify all cited 2025/2026 works (Genesis, PhysX-CoT, CausalPhys, Embodied Data Pyramid) against primary sources/arXiv before final submission — venue names, authors, and exact publication dates should be confirmed and added to the BibTeX above.*
