# External runtime adapters

Only external VLA/runtime integration belongs here:

- `libero_openvla.py`: OpenVLA wrapper and policy-image conversion.
- `tf5_compat.py`: Transformers compatibility helpers.

The controller is [`../controller.py`](../controller.py). Perception,
verification, prediction, repair, memory and training have their own packages;
see the [source layout](../README.md) and [implementation status](../COMPONENT_STATUS.md).

Historical archived runners import these utilities as `adapters.*` with `src/`
on their import path. That external-adapter namespace is intentionally retained.
