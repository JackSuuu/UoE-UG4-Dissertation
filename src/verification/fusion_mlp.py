"""Small learned heads with strict role/schema/provenance-checked artifacts.

Old microwave-stage checkpoints are deliberately rejected, not guessed/padded.
Torch is imported only when constructing/loading a real head.
"""
import math

from .features import FEATURE_DIM, FEATURE_SCHEMA


class FusionMLP:
    def __init__(self, *, hidden=64, device='cpu'):
        import torch
        if type(hidden) is not int or hidden < 2:
            raise ValueError('Invalid hidden width')
        self.device, self.hidden = device, hidden
        self.net = torch.nn.Sequential(torch.nn.Linear(FEATURE_DIM, hidden), torch.nn.ReLU(),
                                       torch.nn.Linear(hidden, 1)).to(device)
        self.ready = False

    def probability(self, features):
        import torch
        if not self.ready:
            raise RuntimeError('Untrained fusion weights cannot be used for inference')
        values = tuple(float(v) for v in features)
        if len(values) != FEATURE_DIM or not all(math.isfinite(v) for v in values):
            raise ValueError('Fusion feature mismatch')
        with torch.inference_mode():
            score = float(self.net(torch.tensor([values], dtype=torch.float32, device=self.device)).sigmoid()[0, 0])
        if not math.isfinite(score):
            raise ValueError('Nonfinite learned score')
        return score

    @classmethod
    def load(cls, path, *, role, encoder_signature, device='cpu'):
        import torch
        from ..perception.frozen_visual_encoder import file_sha
        artifact = torch.load(path, map_location='cpu', weights_only=True)
        if (artifact.get('schema') != FEATURE_SCHEMA or artifact.get('feature_dim') != FEATURE_DIM
                or artifact.get('role') != role or artifact.get('trained') is not True
                or artifact.get('encoder_signature') != encoder_signature):
            raise ValueError('Incompatible/untrained artifact or encoder; old stage models are not substitutes')
        train, validation = artifact.get('training_roots', []), artifact.get('validation_roots', [])
        if (not train or not validation or set(train) & set(validation)
                or any(type(r) is not int or r not in range(10, 20) for r in [*train, *validation])):
            raise ValueError('Model needs disjoint train/validation roots within training roots 10–19')
        threshold = artifact.get('threshold')
        if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError('Missing validation-selected threshold')
        if artifact.get('label_schema') != 'observed_generated_contract_nominal24_v1':
            raise ValueError('Unknown training label semantics')
        head = cls(hidden=artifact['hidden'], device=device)
        head.net.load_state_dict(artifact['state_dict'], strict=True)
        head.net.eval().requires_grad_(False)
        head.ready, head.threshold = True, float(threshold)
        head.provenance = dict(path=str(path), sha256=file_sha(path), schema=FEATURE_SCHEMA,
            role=role, encoder_signature=encoder_signature, training_roots=train,
            validation_roots=validation, label_schema=artifact['label_schema'],
            calibration='empirical_validation_not_conformal_coverage')
        return head
