"""RQ3 systems: BPTT checkpointing, gradient stabilization, async scheduling,
CPU/GPU hybrid inference, and expert deferral."""
from .bptt import Stabilizer, checkpointed_bptt, measure, naive_bptt, step_cost
from .scheduler import AsyncEngine
from .hybrid_engine import (
    ArithmeticIntensityScheduler,
    CUDAGraphBPTT,
    FusedStepOperator,
    HardwareBudget,
    HybridSchedule,
    TaskProfile,
)
from .expert_deferral import (
    DeferredSegment,
    DeferralStats,
    ExpertDeferralScheduler,
)

__all__ = [
    "Stabilizer", "checkpointed_bptt", "measure", "naive_bptt", "step_cost",
    "AsyncEngine",
    "ArithmeticIntensityScheduler", "CUDAGraphBPTT", "FusedStepOperator",
    "HardwareBudget", "HybridSchedule", "TaskProfile",
    "DeferredSegment", "DeferralStats", "ExpertDeferralScheduler",
]
