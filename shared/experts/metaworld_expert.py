"""Scripted experts for MetaWorld V3 tasks.

MetaWorld ships a scripted ``SawyerXYZV3Policy`` per task (in
``metaworld.policies``) that consumes the 39D raw obs and emits a 4D action
``[dx, dy, dz, gripper]``. This wraps one such policy as a ``BaseExpert``.
pi05_metaworld is trained natively on the 4D action, so it passes through
unchanged. Dispatch is keyed by env id via ``_POLICY_BY_ENV`` below.
"""

from __future__ import annotations

import numpy as np

from shared.experts.base_expert import BaseExpert


# MetaWorld scripted policy class per env id (defined in ``metaworld.policies``).
# To add a task, add an ``env_id -> policy class`` entry.
_POLICY_BY_ENV = {
    "assembly-v3": "SawyerAssemblyV3Policy",
}


def _load_policy(env_id: str):
    if env_id not in _POLICY_BY_ENV:
        raise KeyError(
            f"No MetaWorld scripted policy registered for {env_id!r}. "
            f"Add it to _POLICY_BY_ENV in shared/experts/metaworld_expert.py."
        )
    class_name = _POLICY_BY_ENV[env_id]
    from metaworld import policies as _P
    cls = getattr(_P, class_name, None)
    if cls is None:
        raise AttributeError(
            f"metaworld.policies has no class {class_name!r} (env_id={env_id!r}); "
            f"check the registry against the installed metaworld version."
        )
    return cls()


class MetaworldScriptedExpert(BaseExpert):
    """Wraps a MetaWorld scripted policy as a ``BaseExpert``.

    Reads the 39D raw obs from the adapter's cached ``_last_raw_obs`` rather
    than poking mujoco directly, which keeps it portable across MetaWorld
    versions. env_id must be present in the V3 task table.
    """

    def __init__(self, env_id: str):
        self.env_id = env_id
        self._policy = _load_policy(env_id)
        self._last_raw_obs: np.ndarray | None = None

    # -- BaseExpert API ------------------------------------------------

    def reset(self, env):
        raw = self._pull_raw_obs(env)
        self._last_raw_obs = raw

    def act(self, env) -> np.ndarray:
        """Return the scripted 4D action ``[dx, dy, dz, gripper]``, clipped to
        [-1, 1]."""
        raw = self._pull_raw_obs(env)
        a4 = np.asarray(self._policy.get_action(raw), dtype=np.float32).reshape(-1)
        if a4.shape[0] != 4:
            raise RuntimeError(
                f"MetaWorld policy for {self.env_id} returned {a4.shape}, expected (4,)"
            )
        return np.clip(a4, -1.0, 1.0).astype(np.float32)

    def compute_off_nominal_distance(self, env) -> float:
        """Hand-to-object distance in meters.

        Uses ``obs[4:7]`` (object position), not ``obs[36:39]``, because
        MetaWorld V3 zeroes out the public goal slot. Trends to zero as the
        task is solved for button-press/reach/press-y style tasks; tasks with
        an intermediate pick+place stage may want a custom distance.
        """
        raw = self._pull_raw_obs(env)
        hand = raw[0:3]
        obj = raw[4:7]
        return float(np.linalg.norm(hand - obj))

    # -- helpers ------------------------------------------------------

    def _pull_raw_obs(self, env) -> np.ndarray:
        """Fetch the latest 39D raw obs, preferring the adapter's cached
        ``_last_raw_obs`` and falling back to ``raw_env._get_obs()``."""
        cached = getattr(env, "_last_raw_obs", None)
        if cached is not None:
            return np.asarray(cached, dtype=np.float32)
        raw_env = getattr(env, "raw_env", env)
        if hasattr(raw_env, "_get_obs"):
            return np.asarray(raw_env._get_obs(), dtype=np.float32)
        raise RuntimeError(
            "MetaworldScriptedExpert.act() could not find a raw obs on the env. "
            "Expected a MetaworldAdapter with _last_raw_obs, or a metaworld "
            "SawyerXYZEnv exposing _get_obs()."
        )
