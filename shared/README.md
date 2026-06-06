# shared

Framework-agnostic pieces reused by the FlowDAgger backends.

- `experts/` scripted experts that read privileged simulator state and return
  corrective actions. `base_expert.py` defines the interface;
  `metaworld_expert.py` implements it.
- `task_configs.py` the task registry consumed by the trainers.
- `intervention_handler.py` decides when the expert takes over during a rollout
  (beta-decay schedule or action-disagreement modes) and, when an inverter and
  action converter are supplied, logs the inverted noise targets.

Import these as a package from a backend, with the repo root on `sys.path`:

```python
from shared.experts.metaworld_expert import MetaworldScriptedExpert
from shared.intervention_handler import InterventionHandler
from shared.task_configs import get_task_config
```
