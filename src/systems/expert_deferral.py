"""
RQ3 — Expert Deferral: pipeline overlap for differentiable BPTT.

Inspired by KTransformers' "Expert Deferral" mechanism: break the rigid
dependency between consecutive BPTT segments so that slow-engine work
(Genesis re-simulation, gradient computation) overlaps with fast-engine
work (policy inference, risk prediction).

Core idea:
  * "Immediate" gradient segments: computed synchronously, feed into the
    next control step's action correction.
  * "Deferred" gradient segments: computed asynchronously in the background,
    their gradients are incorporated K steps later (exploiting the residual
    connection's robustness to delayed gradient signals).

This maximizes CPU/GPU overlap: while the GPU processes the immediate
segment's backward pass, the CPU (or a background thread) can start
recomputing the deferred segment's forward pass.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class DeferredSegment:
    """A BPTT segment whose gradient computation is deferred."""
    segment_id: int
    start_step: int
    end_step: int
    state_at_start: torch.Tensor = None
    actions: torch.Tensor = None
    gradient: torch.Tensor = None
    computed: bool = False
    submit_time: float = 0.0
    complete_time: float = 0.0


@dataclass
class DeferralStats:
    """Statistics for expert deferral scheduling."""
    n_immediate: int = 0
    n_deferred: int = 0
    n_overlap_steps: int = 0
    avg_deferral_depth: float = 0.0
    gpu_utilization: float = 0.0
    cpu_utilization: float = 0.0
    total_speedup: float = 1.0
    gradient_staleness: list = field(default_factory=list)


class ExpertDeferralScheduler:
    """
    Schedules BPTT segments as immediate or deferred to maximize
    pipeline overlap, following KTransformers' expert deferral principle.

    The key insight: in a control loop at step t, we need gradients from
    steps [t-K, t] to correct the action at step t+1. But we DON'T need
    gradients from steps [t-2K, t-K] immediately — they can be computed
    in the background and incorporated at step t+K.

    This creates a 2-stage pipeline:
      Stage 1 (immediate): gradient for step t -> action correction at t+1
      Stage 2 (deferred):  gradient for step t-K -> action correction at t+1+K
    """

    def __init__(self, deferral_depth: int = 2, max_deferred: int = 4):
        self.deferral_depth = deferral_depth
        self.max_deferred = max_deferred
        self.deferred_queue: list[DeferredSegment] = []
        self.completed_deferred: list[DeferredSegment] = []
        self.stats = DeferralStats()
        self._lock = threading.Lock()
        self._background_thread = None
        self._stop_event = threading.Event()

    def classify_segment(self, segment_id: int, current_step: int) -> str:
        """
        Classify a segment as 'immediate' or 'deferred'.

        Immediate: needed for the next control step's action correction.
        Deferred: can be computed in the background.
        """
        age = current_step - segment_id
        if age <= self.deferral_depth:
            return "immediate"
        return "deferred"

    def submit_deferred(self, segment: DeferredSegment,
                        compute_fn: callable):
        """
        Submit a segment for deferred (background) computation.

        compute_fn: callable that takes (state, actions) and returns
                    (loss, gradient). Runs in a background thread.
        """
        with self._lock:
            if len(self.deferred_queue) >= self.max_deferred:
                # Drop oldest deferred segment (latest-wins policy)
                dropped = self.deferred_queue.pop(0)
                self.stats.n_deferred -= 1
            self.deferred_queue.append(segment)
            self.stats.n_deferred += 1

        def _worker():
            segment.submit_time = time.perf_counter()
            try:
                loss, grad = compute_fn(segment.state_at_start, segment.actions)
                segment.gradient = grad
                segment.computed = True
                segment.complete_time = time.perf_counter()
                with self._lock:
                    self.completed_deferred.append(segment)
                    if segment in self.deferred_queue:
                        self.deferred_queue.remove(segment)
            except Exception as e:
                print(f"[deferral] segment {segment.segment_id} failed: {e}")

        self._background_thread = threading.Thread(target=_worker, daemon=True)
        self._background_thread.start()

    def collect_deferred(self, current_step: int) -> list[DeferredSegment]:
        """
        Collect completed deferred gradients that are now needed.

        A deferred gradient from step t is needed at step t + deferral_depth.
        """
        with self._lock:
            needed = []
            remaining = []
            for seg in self.completed_deferred:
                if current_step - seg.segment_id >= self.deferral_depth:
                    needed.append(seg)
                    self.stats.gradient_staleness.append(
                        current_step - seg.segment_id)
                else:
                    remaining.append(seg)
            self.completed_deferred = remaining
            return needed

    def get_effective_gradient(self, immediate_grad: torch.Tensor,
                               deferred_grads: list[torch.Tensor],
                               staleness_weights: np.ndarray = None) -> torch.Tensor:
        """
        Combine immediate and deferred gradients into a single effective gradient.

        Deferred gradients are weighted by staleness (older = less weight),
        following the principle that residual connections are robust to
        slightly stale gradient signals.
        """
        if not deferred_grads:
            return immediate_grad

        if staleness_weights is None:
            # Exponential decay: weight = 0.5^staleness
            staleness_weights = np.array(
                [0.5 ** s for s in range(1, len(deferred_grads) + 1)])

        weights = staleness_weights / staleness_weights.sum()
        effective = immediate_grad.clone()
        for i, grad in enumerate(deferred_grads):
            effective = effective + weights[i] * grad

        return effective

    def compute_overlap_speedup(self, immediate_time: float,
                                deferred_time: float,
                                n_deferred: int) -> float:
        """
        Estimate speedup from overlapping immediate and deferred computation.

        Without overlap: total = immediate_time + n_deferred * deferred_time
        With overlap:    total = max(immediate_time, n_deferred * deferred_time)
        """
        serial_time = immediate_time + n_deferred * deferred_time
        overlap_time = max(immediate_time, n_deferred * deferred_time)
        return serial_time / max(overlap_time, 1e-9)

    def reset(self):
        """Reset all state for a new episode."""
        with self._lock:
            self.deferred_queue.clear()
            self.completed_deferred.clear()
        self.stats = DeferralStats()
        self._stop_event.clear()
