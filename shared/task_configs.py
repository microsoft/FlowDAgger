"""Per-task configuration registry.

Each entry provides the suite name, task description, and expert_kwargs that
override expert defaults. To add a task, add a dict entry to TASK_CONFIGS.

Ships one reference task: metaworld_assembly (MetaWorld V3 assembly on the
pi05_metaworld base).
"""

import os

TASK_CONFIGS = {
    # MetaWorld V3 assembly on pi05_metaworld. Native shape: 4D state, 4D
    # action, single 256x256 corner3 camera.
    "metaworld_assembly": {
        "name": "assembly",
        "suite": "metaworld",
        "env_id": "assembly-v3",
        "prompt": "Pick up a nut and place it onto a peg",
        "expert_class": "MetaworldScriptedExpert",
        "openpi_config": "pi05_metaworld",
        # Auto-downloaded from the Hub on first run unless openpi_checkpoint
        # points at an existing local dir.
        "hf_checkpoint": "mmurray-ms/pi05-metaworld",
        "openpi_checkpoint": os.environ.get("METAWORLD_CHECKPOINT", ""),
        "max_timesteps": 300,
        "resolution": 256,
        "camera_name": "corner3",
        "expert_kwargs": {},
    },
}


def get_task_config(task_id):
    """Return the config dict for task_id, or raise if it is not registered."""
    if task_id in TASK_CONFIGS:
        return TASK_CONFIGS[task_id]
    raise KeyError(
        f"Unknown task_id {task_id!r}. Available: {sorted(TASK_CONFIGS)}. "
        f"Add an entry to shared/task_configs.py::TASK_CONFIGS."
    )
