"""Versioned, masked visual/physical handoff classifier. Class 1 means safe."""
from pathlib import Path
import numpy as np
import torch
from torch import nn

STAGES = ('acquire', 'transfer', 'contain_release', 'close')
STAGE_TO_IDX = {name: i for i, name in enumerate(STAGES)}
FEATURE_NAMES = ('in_goal', 'close_goal', 'grasp', 'held_clear', 'table',
    'mug_microwave', 'gripper_microwave', 'mug_gripper', 'x', 'y', 'z', 'rise_m')
FEATURE_SCHEMA = 'controller_state_values12_validity12_v2'
SCHEMA_VERSION = 2


def extract_physics_features(measured):
    """Consume controller evidence.state, never unparsed simulator diagnostics.

    Each absent/nonfinite value has value zero AND validity zero; false is
    value zero with validity one. Train/runtime share this exact extraction.
    """
    if not isinstance(measured, dict) or not any(k in measured for k in ('in_goal', 'position', 'held_clear')):
        raise ValueError('Expected controller evidence.state, not raw diagnostics')
    position = measured.get('position')
    if not isinstance(position, (list, tuple, np.ndarray)) or len(position) != 3:
        position = [None] * 3
    raw = [measured.get(k) for k in FEATURE_NAMES[:8]] + list(position) + [measured.get('rise_m')]
    values, masks = [], []
    for index, value in enumerate(raw):
        valid = isinstance(value, (bool, np.bool_)) if index < 8 else (
            isinstance(value, (int, float, np.number)) and np.isfinite(value))
        values.append(float(value) if valid else 0.)
        masks.append(float(valid))
    return np.asarray(values + masks, dtype=np.float32)


class VerificationFusionMLP(nn.Module):
    def __init__(self, vision_dim=384, physics_dim=24, num_stages=4, hidden1=64, hidden2=32):
        super().__init__()
        if physics_dim != 24 or num_stages != len(STAGES):
            raise ValueError('Unsupported fusion feature schema')
        self.vision_dim, self.physics_dim, self.num_stages = vision_dim, physics_dim, num_stages
        self.hidden1, self.hidden2 = hidden1, hidden2
        self.register_buffer('physics_mean', torch.zeros(12))
        self.register_buffer('physics_std', torch.ones(12))
        self.net = nn.Sequential(nn.Linear(vision_dim+physics_dim+num_stages, hidden1),
            nn.ReLU(), nn.Dropout(.1), nn.Linear(hidden1, hidden2), nn.ReLU(), nn.Linear(hidden2, 2))
        self.metadata = {}

    def forward(self, vision_feat, physics_feat, stage_onehot):
        values, mask = physics_feat[..., :12], physics_feat[..., 12:]
        normalized = (values-self.physics_mean)/self.physics_std
        physics = torch.cat([normalized*mask, mask], dim=-1)
        return self.net(torch.cat([vision_feat, physics, stage_onehot], dim=-1))

    def fit_normalizer(self, physics):
        physics = np.asarray(physics)
        for i in range(12):
            observed = physics[physics[:, i+12] == 1, i]
            self.physics_mean[i] = float(observed.mean()) if len(observed) else 0.
            self.physics_std[i] = max(float(observed.std()), .01) if len(observed) else 1.

    def predict_verify_probability(self, vision_feat, physics_feat, stage):
        if stage not in STAGE_TO_IDX:
            raise ValueError('Unknown recovery contract')
        vf, pf = np.asarray(vision_feat, dtype=np.float32), np.asarray(physics_feat, dtype=np.float32)
        if vf.shape != (self.vision_dim,) or pf.shape != (self.physics_dim,):
            raise ValueError('Fusion input dimension mismatch')
        if not np.isfinite(vf).all() or not np.isfinite(pf).all() or np.linalg.norm(vf) < 1e-6:
            raise ValueError('Missing/nonfinite visual or physical features')
        self.eval()
        device = next(self.parameters()).device
        stage_input = torch.eye(self.num_stages, device=device)[STAGE_TO_IDX[stage]][None]
        with torch.inference_mode():
            logits = self(torch.from_numpy(vf[None]).to(device), torch.from_numpy(pf[None]).to(device), stage_input)
            return float(logits.softmax(-1)[0, 1])


def save_model(model, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(schema_version=SCHEMA_VERSION, feature_schema=FEATURE_SCHEMA,
        positive_class=1, state_dict=model.state_dict(), vision_dim=model.vision_dim,
        physics_dim=model.physics_dim, num_stages=model.num_stages,
        hidden1=model.hidden1, hidden2=model.hidden2, metadata=model.metadata), path)


def load_model(path, device='cpu'):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if (checkpoint.get('schema_version') != SCHEMA_VERSION or
            checkpoint.get('feature_schema') != FEATURE_SCHEMA or checkpoint.get('positive_class') != 1):
        raise ValueError('Legacy/unversioned fusion weights rejected; retrain with real features and outcome labels')
    model = VerificationFusionMLP(**{k: checkpoint[k] for k in
        ('vision_dim', 'physics_dim', 'num_stages', 'hidden1', 'hidden2')})
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.metadata = checkpoint['metadata']
    return model.to(device).eval()


def train_on_pairs(model, pairs, epochs=40, lr=1e-3, batch_size=128, device='cpu', seed=0):
    if not pairs or len({p['label'] for p in pairs}) < 2:
        raise ValueError('Training requires both observed outcome classes')
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    physics = np.stack([p['physics_feat'] for p in pairs])
    model.fit_normalizer(physics)
    model.to(device).train()
    v = torch.from_numpy(np.stack([p['vision_feat'] for p in pairs])).to(device)
    p = torch.from_numpy(physics).to(device)
    s = torch.eye(4, device=device)[[STAGE_TO_IDX[x['stage']] for x in pairs]]
    y = torch.tensor([x['label'] for x in pairs], device=device)
    counts = torch.bincount(y, minlength=2).float()
    criterion = nn.CrossEntropyLoss(weight=len(pairs)/(2*counts))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.001)
    losses = []
    for _ in range(epochs):
        total = 0.
        for idx in torch.randperm(len(pairs), device=device).split(batch_size):
            optimizer.zero_grad()
            loss = criterion(model(v[idx], p[idx], s[idx]), y[idx])
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(idx)
        losses.append(total/len(pairs))
    model.eval()
    return losses
