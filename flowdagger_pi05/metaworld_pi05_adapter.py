"""Wraps a MetaWorld V3 env in pi05_metaworld's native shape (4D state + 4D
action + single image), so no state/action widening is needed.

Obs dict (per step / reset):
    image            (H, W, 3) uint8 HWC, single corner3 cam
    state            (4,) float32, hand_xyz + gripper_norm
    metaworld_obs    (39,) float32, full raw env obs (scripted expert only,
                                    not seen by pi0.5)
    success          float (0 or 1)

step() takes a 4D action (xyz_delta + gripper) and returns
(obs, reward=float(success), done=False, info).

Goal-observable trick: the goal is exposed in obs[36:39] (cached
observation_space cleared) so the scripted expert can read it. pi0.5 never
sees env_state, so this leak is invisible to the model.
"""

from __future__ import annotations

import numpy as np


class MetaworldPi05Adapter:
    class _ActionSpace:
        def __init__(self):
            self.shape = (4,)
            self.low = np.array([-1.0] * 4, dtype=np.float32)
            self.high = np.array([1.0] * 4, dtype=np.float32)

        def sample(self):
            return np.random.uniform(-1.0, 1.0, size=(4,)).astype(np.float32)

    def __init__(
        self,
        env_name: str,
        seed: int = 0,
        camera_name: str = "corner3",
        resolution: int = 256,
    ):
        import metaworld

        self.env_name = env_name
        self._seed = seed
        self._camera_name = camera_name
        self._resolution = resolution

        base_cls = metaworld.ALL_V3_ENVIRONMENTS[env_name]
        self.raw_env = base_cls(
            render_mode="rgb_array",
            width=resolution,
            height=resolution,
            camera_name=camera_name,
        )
        self.raw_env._partially_observable = False
        self.raw_env._freeze_rand_vec = False
        # MetaWorld V3's _get_state_rand_vec falls back to the global np.random
        # when seeded_rand_vec is False, so even env.reset(seed=X) won't
        # reproduce an init. Enable it so init poses come from self.np_random
        # (which gymnasium reseeds from reset(seed=)).
        self.raw_env.seeded_rand_vec = True
        self.raw_env._set_task_called = True
        try:
            del self.raw_env.sawyer_observation_space
        except AttributeError:
            pass
        self.raw_env.seed(seed)
        self.action_space = self._ActionSpace()

        self._last_raw_obs: np.ndarray | None = None
        self._last_info: dict = {}

    # -- gym-style API ------------------------------------------------

    def reset(self, *_args, seed: int | None = None, **_kwargs):
        # Use a passed seed verbatim (deterministic replay); else advance the
        # internal seed for episode variety.
        if seed is None:
            self._seed = (self._seed + 1) % (2**31)
            seed = self._seed
        else:
            self._seed = int(seed)

        # MetaWorld V3's Env.reset() ignores the seed argument, so install a
        # fresh np_random ourselves. _get_state_rand_vec consumes this Generator
        # on the path enabled by seeded_rand_vec=True.
        from gymnasium.utils import seeding
        self.raw_env._np_random, _ = seeding.np_random(int(seed))

        raw_obs, info = self.raw_env.reset()
        self._last_raw_obs = np.asarray(raw_obs, dtype=np.float32)
        self._last_info = dict(info) if info else {}
        return self._build_obs_dict(self._last_raw_obs)

    def seed(self, seed: int):
        self._seed = int(seed)
        self.raw_env.seed(int(seed))

    def step(self, action_4d):
        a = np.asarray(action_4d, dtype=np.float32).reshape(-1)
        if a.shape[0] != 4:
            raise ValueError(f"Expected 4D action, got shape {a.shape}")
        raw_obs, _r, term, trunc, info = self.raw_env.step(np.clip(a, -1.0, 1.0))
        self._last_raw_obs = np.asarray(raw_obs, dtype=np.float32)
        self._last_info = dict(info) if info else {}

        success = float(info.get("success", 0.0))
        # env_max_reward=1 convention: collect_traj/perform_control_eval
        # early-stop on reward >= env_max_reward.
        reward = success
        done = False  # ignore_done-style; loop terminates on reward/success
        return self._build_obs_dict(self._last_raw_obs), reward, done, self._last_info

    def render(self, *_args, **_kwargs):
        return self.raw_env.render()

    def close(self):
        return self.raw_env.close()

    # -- helpers ------------------------------------------------------

    def _render(self) -> np.ndarray:
        frame = self.raw_env.render()
        if frame is None:
            return np.zeros((self._resolution, self._resolution, 3), dtype=np.uint8)
        frame = np.asarray(frame)
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        return frame

    def _build_obs_dict(self, raw_obs: np.ndarray) -> dict:
        return {
            "image": self._render(),
            "state": raw_obs[:4].astype(np.float32),
            "metaworld_obs": raw_obs.astype(np.float32),
            "success": float(self._last_info.get("success", 0.0)),
        }

    @property
    def sim(self):
        return getattr(self.raw_env, "sim", None)
