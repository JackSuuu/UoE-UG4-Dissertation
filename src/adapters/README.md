# Backend bridges

- `libero_environment.py`: task-independent scene inventory and predicate measurements.
- `libero_image.py`: image preprocessing, applied once by the environment bridge.
- `openvla_policy.py`: frozen official H1 action generation; only the standard LIBERO gripper mapping.

Backends implement [`../interfaces.py`](../interfaces.py). No model/simulator is
imported until a backend is constructed or used. The caller owns cleanup;
`Session` alone owns policy queries and environment steps during an episode.
