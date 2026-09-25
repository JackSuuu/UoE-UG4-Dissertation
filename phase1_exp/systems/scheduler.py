"""
RQ3 systems optimisation 3: fast/slow dual-engine asynchronous scheduling.

The fast engine (policy + OrbiSim-Dynamics verifier) runs in the control loop
every step. The slow engine (full GT re-simulation — Genesis or the torch GT)
lives in a separate worker *process* with its own simulator instance and
verifies submitted (state, chunk) pairs in the background. The control loop
never blocks on it: ``submit`` is dropped if the worker is busy (latest-wins),
and ``poll`` returns whatever verdict has arrived, tagged with its staleness
(control steps between submission and consumption).

``slow_delay_ms`` adds an artificial per-request delay so that the torch GT can
emulate the cost of a full-fidelity Genesis re-simulation when needed.
"""
from __future__ import annotations

import queue
import time

import torch
import torch.multiprocessing as mp


def _worker(task, backend, n, mults, params, device, slow_delay_ms, q_in, q_out):
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from sims.base import make_env
    dev = torch.device(device)
    env = make_env(task, backend, n, dev, mults)
    if backend == "torch":
        env.reset({k: v.to(dev) for k, v in params.items()}, 0)
    else:
        env.reset(None, 0)
    while True:
        msg = q_in.get()
        if msg is None:
            break
        step_id, state, chunk = msg
        t0 = time.perf_counter()
        env.set_state(state.to(dev))
        risk = env.shadow_rollout(chunk.to(dev))
        viol = (risk.amax(dim=(1, 2)) > 1.0).cpu()
        if slow_delay_ms > 0:
            rem = slow_delay_ms / 1e3 - (time.perf_counter() - t0)
            if rem > 0:
                time.sleep(rem)
        q_out.put((step_id, viol, (time.perf_counter() - t0) * 1e3))


class AsyncEngine:
    def __init__(self, task, backend, n, mults, params, device, slow_delay_ms=0.0):
        ctx = mp.get_context("spawn")
        self.q_in, self.q_out = ctx.Queue(maxsize=1), ctx.Queue()
        p_cpu = {k: v.detach().cpu() for k, v in params.items()} if params else {}
        self.proc = ctx.Process(target=_worker, daemon=True, args=(
            task, backend, n, mults, p_cpu, str(device), slow_delay_ms, self.q_in, self.q_out))
        self.proc.start()
        self.step = 0
        self.busy = False
        self.staleness, self.slow_ms, self.n_dropped = [], [], 0

    def reset_stats(self):
        self.step = 0
        self.staleness, self.slow_ms, self.n_dropped = [], [], 0

    def submit(self, state, chunk):
        self.step += 1
        if self.busy:
            self.n_dropped += 1
            return
        try:
            self.q_in.put_nowait((self.step, state.detach().cpu(), chunk.detach().cpu()))
            self.busy = True
        except queue.Full:
            self.n_dropped += 1

    def poll(self, B, device):
        viol = torch.zeros(B, dtype=torch.bool)
        try:
            while True:
                sid, v, ms = self.q_out.get_nowait()
                self.busy = False
                self.staleness.append(self.step - sid)
                self.slow_ms.append(ms)
                viol |= v
        except queue.Empty:
            pass
        return viol.to(device)

    def close(self):
        try:
            self.q_in.put(None, timeout=1.0)
        except Exception:
            pass
        self.proc.join(timeout=5)
        if self.proc.is_alive():
            self.proc.terminate()
