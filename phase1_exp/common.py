"""Shared utilities: device, seeding, OOD grids, IO."""
import json
import os
import random

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "results")


def get_device(pref: str = "auto") -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def out_dir(task: str, backend: str = "torch") -> str:
    d = os.path.join(RESULTS, f"{task}_{backend}")
    os.makedirs(d, exist_ok=True)
    return d


def save_json(obj, path):
    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if torch.is_tensor(o):
            return o.detach().cpu().tolist()
        raise TypeError(type(o))

    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=conv)


def load_json(path):
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# OOD grids (multipliers on nominal physical parameters) — plan §2.2
# Axis 1 / axis 2 form the 2-D heatmap in Fig A.
# ---------------------------------------------------------------------------
OOD_GRIDS = {
    "push": {
        "axes": ("friction", "mass"),
        "friction": [0.2, 0.6, 1.0, 1.4, 1.8],   # nominal, ±40%, ±80%
        "mass": [0.5, 1.0, 1.5, 2.0],            # ±50%, x2
    },
    "cloth": {
        "axes": ("stiffness", "mass"),
        "stiffness": [0.5, 1.0, 2.0, 4.0],       # 0.5x, 2x, 4x
        "mass": [0.5, 1.0, 2.0],
    },
}

# Parameter ranges used to train the *predictors* (verifiers). Deliberately
# narrower than the OOD grid so extreme cells remain out-of-distribution.
# The base policy (VLA proxy) is trained at nominal (all multipliers = 1) only.
PREDICTOR_TRAIN_RANGES = {
    "push": {"friction": (0.5, 1.5), "mass": (0.6, 1.6)},
    "cloth": {"stiffness": (0.6, 2.0), "mass": (0.6, 1.6), "friction": (0.6, 1.4)},
}


def grid_cells(task: str):
    g = OOD_GRIDS[task]
    a1, a2 = g["axes"]
    return [{a1: v1, a2: v2} for v1 in g[a1] for v2 in g[a2]]


def is_ood(task: str, cell: dict) -> bool:
    rng = PREDICTOR_TRAIN_RANGES[task]
    for k, v in cell.items():
        if k in rng and not (rng[k][0] <= v <= rng[k][1]):
            return True
    return False
