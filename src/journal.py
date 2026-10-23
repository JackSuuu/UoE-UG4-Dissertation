"""Append-only runtime trace with exact planner/policy input images."""
import json
from pathlib import Path

from .observation import Observation


class Journal:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        (self.directory/'images').mkdir()
        self.stream = (self.directory/'events.jsonl').open('x')

    def _encode(self, value):
        if isinstance(value, Observation):
            name = f'images/{value.index:06d}.png'
            path = self.directory/name
            if path.exists():
                if path.read_bytes() != value.image_png:
                    raise ValueError('Image changed at an existing endpoint index')
            else:
                path.write_bytes(value.image_png)
            return dict(value.to_dict(), image=name)
        if isinstance(value, dict):
            return {k: self._encode(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [self._encode(v) for v in value]
        return value

    def record(self, event):
        self.stream.write(json.dumps(self._encode(event), allow_nan=False)+'\n')
        self.stream.flush()

    def save(self, name, value):
        path = self.directory/name
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
        temporary.replace(path)

    def close(self):
        self.stream.close()
