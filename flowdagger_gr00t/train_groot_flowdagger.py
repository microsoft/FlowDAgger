#!/usr/bin/env python
"""FlowDAgger for GR00T N1.7 (flow-matching head, libero_10 ckpt).

Goal: lift GR00T N1.7 zero-shot SR on libero_90 task 57 (cream cheese to tray)
by training a noise policy that outputs the initial noise for the (frozen) flow
sampler directly. Noise-space DAgger supervised by inverted expert actions only
(no SAC).

Algorithm:
    1. Rollout: w = noise_policy(obs_feat); flow-sample from w -> action chunk
       -> step env.
    2. At each chunk boundary, use the expert with probability
       intervention_probability (1.0 in the verified recipe).
    3. On intervention: LiberoExpert chunk -> encode to model space ->
       reverse-Euler invert to w*. Add (obs_feat, w*) to buffer.
    4. BC update: MSE(noise_policy(obs_feat), w*). Periodic eval.

GR00T flow head:
    train:  noisy = (1-t)*noise + t*action; target velocity = action - noise.
    sample: actions = noise; for k steps: actions += dt * model(actions, t).
    Inversion operates in model space (B, 40, 132); the runtime decoder slices
    to (B, 16, 7) for LIBERO_PANDA.

Backbone + state features are cached once per env step and fed to both the flow
sampler and the noise policy.

Usage:
    python train_groot_flowdagger.py --task_id 57
"""
import argparse
import os
import sys
import time
from dataclasses import dataclass

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
# LIBERO's checked-in init-state tensors predate PyTorch's weights-only default.
# Only use the pinned, trusted LIBERO checkout documented in README.md.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import numpy as np
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))  # repo root, for `shared`

# Obs/action conversion scaffold (do not reimplement).
from eval_libero_groot import (  # noqa: E402
    DEFAULT_CKPT,
    LIBERO_DUMMY_ACTION,
    SETTLE_STEPS,
    SUITE_MAX_STEPS,
    obs_to_groot,
    postprocess_gripper,
)

# Flow-matching inversion primitives. Inversion operates in model space
# (B, action_horizon=40, action_dim=132), not runtime-decoded (B, 16, 7).
from groot_inversion import (  # noqa: E402
    encode_features,
    forward_euler,
    perstep_fp_reverse,
    predict_velocity,  # noqa: F401  (kept for custom samplers)
    reverse_euler,  # noqa: F401  (sanity-check / warm-start variant)
)
# Expert action -> model-space encoder.
from groot_expert_inversion import (  # noqa: E402
    decode_chunk_from_model_space,
    encode_chunk_to_model_space,
)

# Framework-agnostic shared pieces.
from shared.experts.libero_expert import LiberoExpert  # noqa: E402
from shared.task_configs import get_task_config  # noqa: E402


@dataclass
class FlowDaggerConfig:
    """FlowDAgger + noise-policy training configuration."""
    suite: str = "libero_90"
    task_id: int = 57
    seed: int = 0
    ckpt: str = DEFAULT_CKPT
    # Runtime (env-facing) action shape: (16, 7).
    action_horizon: int = 16  # GR00T N1.7 default
    action_dim: int = 7       # LIBERO_PANDA: xyz + rpy + gripper
    # Model (inversion-space) action shape. The noise policy operates on the
    # full padded tensor; runtime decode reads only the (16, 7) prefix.
    model_action_horizon: int = 40  # max_action_horizon (padded)
    model_action_dim: int = 132     # max_action_dim (padded)
    replan_steps: int = 8
    episodes: int = 20
    seed_expert_episodes: int = 10
    bc_steps_per_episode: int = 100
    bc_batch_size: int = 64
    lr: float = 1e-4
    buffer_size: int = 100_000
    noise_policy_hidden_dims: tuple = (512, 512, 512)
    noise_policy_use_layer_norm: bool = True
    feature_pool: str = "mean"  # "mean" | "cls" | "mean+state"
    # LiberoExpert position_gain. The LIBERO-90 task 57 reference run uses 5;
    # higher gains saturate actions and reproduced worse in the verified recipe.
    expert_position_gain: float = 5.0
    # Inverter: per-step fixed-point iterations. Higher is more accurate at
    # roughly linear cost.
    inverter_fp_per_step: int = 10
    intervention_probability: float = 1.0
    eval_interval: int = 10
    eval_episodes: int = 25
    log_interval: int = 10
    device: str = "cuda"


class NoisePolicy(nn.Module):
    """MLP noise policy: pooled VLM/state features -> noise w of shape
    (action_horizon, action_dim) directly. The flow sampler is initialized from
    this w instead of fresh randn (this is FlowDAgger, not residual DAgger).

    action_horizon/action_dim here are the model-space dims
    (model_action_horizon=40, model_action_dim=132), not the runtime (16, 7).
    Inversion only works in model space.
    """
    def __init__(self, feature_dim, action_horizon, action_dim,
                 hidden_dims=(512, 512, 512), use_layer_norm=True, activation=nn.GELU):
        super().__init__()
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        layers, in_dim = [], feature_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            if use_layer_norm:
                layers.append(nn.LayerNorm(h))
            layers.append(activation())
            in_dim = h
        layers.append(nn.Linear(in_dim, action_horizon * action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, obs_features: torch.Tensor) -> torch.Tensor:
        return self.net(obs_features).view(-1, self.action_horizon, self.action_dim)


class FlowDaggerBuffer:
    """Stores (obs_features, w_star) pairs."""
    def __init__(self, feature_dim, action_horizon, action_dim, max_size=100_000):
        self.max_size = max_size
        self.obs = np.zeros((max_size, feature_dim), dtype=np.float32)
        self.w = np.zeros((max_size, action_horizon, action_dim), dtype=np.float32)
        self.size = 0
        self.ptr = 0

    def add(self, obs_feat, w_star):
        self.obs[self.ptr] = obs_feat
        self.w[self.ptr] = w_star
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return self.obs[idx], self.w[idx]

    def __len__(self):
        return self.size


class GrootFlowDaggerLoop:
    """FlowDAgger trainer for GR00T N1.7 on LIBERO."""
    def __init__(self, cfg: FlowDaggerConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.policy = None
        self._feature_dim = None  # lazy-init once VLM forward yields it
        self.noise_policy: NoisePolicy | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.buffer: FlowDaggerBuffer | None = None
        self.env = None
        # LiberoExpert: task 57 auto-detects object/target with no kwargs; for
        # other tasks pull kwargs from shared/task_configs.py. The expert must
        # be able to take over from arbitrary mid-episode states, so call
        # expert.reset(env) at every takeover boundary.
        self.expert: LiberoExpert | None = None

    # ------------------------------------------------------------------
    # Setup helpers (lazy-init so individual pieces can be poked).
    # ------------------------------------------------------------------
    def _build_expert(self):
        """Instantiate the scripted LiberoExpert with per-task kwargs.

        Task 57 needs no kwargs (auto-detects). position_gain is overridden
        from cfg.
        """
        entry = get_task_config(self.cfg.task_id)
        expert_kwargs = dict(entry.get("expert_kwargs", {}))
        expert_kwargs.setdefault("position_gain", self.cfg.expert_position_gain)
        if entry.get("expert_class") and entry["expert_class"] != "LiberoExpert":
            raise NotImplementedError(
                f"Task {self.cfg.task_id} requires expert_class="
                f"{entry['expert_class']}; this backend only wires LiberoExpert."
            )
        return LiberoExpert(**expert_kwargs)

    def _invert_action_chunk(self, model_space_action: torch.Tensor, ctx: dict) -> torch.Tensor:
        """Invert a model-space action tensor (B, 40, 132) -> noise w*.

        Uses per-step fixed-point reverse Euler. The one-pass reverse_euler is
        too coarse for BC supervision.
        """
        action_head = self.policy.model.action_head
        return perstep_fp_reverse(
            action_head, model_space_action, ctx,
            fp_per_step=self.cfg.inverter_fp_per_step,
        )

    def _sample_action_from_noise(self, w: torch.Tensor, ctx: dict) -> torch.Tensor:
        """Forward-Euler GR00T flow sampler with a custom initial noise tensor."""
        action_head = self.policy.model.action_head
        return forward_euler(action_head, w, ctx)

    def _lazy_init_noise_policy(self, feature_dim: int):
        if self.noise_policy is not None:
            return
        self._feature_dim = feature_dim
        # Noise policy operates in model space (inversion only works there).
        self.noise_policy = NoisePolicy(
            feature_dim=feature_dim,
            action_horizon=self.cfg.model_action_horizon,
            action_dim=self.cfg.model_action_dim,
            hidden_dims=self.cfg.noise_policy_hidden_dims,
            use_layer_norm=self.cfg.noise_policy_use_layer_norm,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.noise_policy.parameters(), lr=self.cfg.lr)
        self.buffer = FlowDaggerBuffer(
            feature_dim,
            self.cfg.model_action_horizon,
            self.cfg.model_action_dim,
            max_size=self.cfg.buffer_size,
        )

    def _compute_obs_features(self, groot_obs: dict):
        """Run GR00T backbone + state encoder once; return (pooled_feat, ctx).

        pooled_feat: (1, feature_dim) torch.float32, conditioning for NoisePolicy.
        ctx: dict consumed by forward_euler / reverse_euler / perstep_fp_reverse
            with keys backbone_features, state_features, embodiment_id,
            backbone_outputs.
        """
        raw = encode_features(self.policy, groot_obs)
        # backbone_features shape (1, seq_len, hidden).
        bb = raw["backbone_features"]
        if self.cfg.feature_pool == "mean":
            pooled = bb.mean(dim=1)
        elif self.cfg.feature_pool == "cls":
            pooled = bb[:, 0]
        elif self.cfg.feature_pool == "mean+state":
            sf = raw["state_features"].mean(dim=1)
            pooled = torch.cat([bb.mean(dim=1), sf], dim=-1)
        else:
            raise ValueError(f"unknown feature_pool {self.cfg.feature_pool}")
        ctx = {
            "backbone_features": raw["backbone_features"],
            "state_features": raw["state_features"],
            "embodiment_id": raw["embodiment_id"],
            "backbone_outputs": raw["backbone_output"],
        }
        return pooled.float(), ctx

    def _sample_with_noise(self, feature_cache: dict, w: torch.Tensor) -> np.ndarray:
        """Run the GR00T Euler sampler initialized from w instead of fresh randn.

        w must be shape (1, model_action_horizon=40, model_action_dim=132).
        Decode keeps the gripper in policy [0,1] convention (to_env_gripper=
        False); postprocess_gripper then does the single [0,1]->env mapping.
        """
        model_space = self._sample_action_from_noise(w, feature_cache)
        chunk = decode_chunk_from_model_space(self.policy, model_space, to_env_gripper=False)
        return postprocess_gripper(chunk)

    # ------------------------------------------------------------------
    # Expert chunk acquisition + action-frame encoding.
    # ------------------------------------------------------------------
    def _encode_expert_chunk_to_model_space(self, expert_chunk: np.ndarray) -> torch.Tensor:
        """Encode a 7D robosuite expert chunk into the model's (1, 40, 132) space.

        LIBERO_PANDA action configs are all absolute, so no state is needed.
        Gripper conversion and padding happen inside. Output cast to device.
        """
        return encode_chunk_to_model_space(self.policy, expert_chunk).to(
            device=self.device, dtype=torch.bfloat16
        )

    def setup(self):
        """One-shot load of policy + env + task + expert. Idempotent."""
        if self.policy is not None:
            return
        import pathlib
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        from gr00t.data.embodiment_tags import EmbodimentTag
        from gr00t.policy.gr00t_policy import Gr00tPolicy

        print(f"Loading GR00T from {self.cfg.ckpt}...")
        self.policy = Gr00tPolicy(
            embodiment_tag=EmbodimentTag.LIBERO_PANDA,
            model_path=self.cfg.ckpt,
            device=self.cfg.device,
            strict=False,
        )

        b = benchmark.get_benchmark_dict()[self.cfg.suite]()
        task = b.get_task(self.cfg.task_id)
        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        self.env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
        self.env.seed(self.cfg.seed)
        self._task_prompt = task.language
        self._init_states = b.get_task_init_states(self.cfg.task_id)
        self._max_steps = SUITE_MAX_STEPS[self.cfg.suite]
        task_name = task.name

        self.expert = self._build_expert()
        print(f"Setup done. task={task_name} | prompt: {self._task_prompt}")

    def _reset_episode(self, episode_idx: int):
        """Reset env to a fresh episode start; return (obs, init_idx)."""
        self.env.reset()
        init_idx = episode_idx % self._init_states.shape[0]
        obs = self.env.set_init_state(self._init_states[init_idx])
        for _ in range(SETTLE_STEPS):
            obs, _, _, _ = self.env.step(LIBERO_DUMMY_ACTION)
        return obs, init_idx

    def _is_success(self, done: bool, reward: float) -> bool:
        """LIBERO sets done on success."""
        return bool(done) and reward > 0

    def rollout_episode(self, episode_idx: int, force_expert: bool = False) -> dict:
        """One episode. On expert chunks, invert and add (feat, w*) to buffer."""
        self.setup()
        env = self.env
        cfg = self.cfg
        obs, init_idx = self._reset_episode(episode_idx)
        self.expert.reset(env)

        t = 0
        reward = 0.0
        done = False
        expert_chunks = 0
        policy_chunks = 0
        while t < self._max_steps and not done:
            groot_obs = obs_to_groot(obs, self._task_prompt)
            pooled_feat, ctx = self._compute_obs_features(groot_obs)
            self._lazy_init_noise_policy(pooled_feat.shape[-1])

            # Seed episodes are all-expert; afterward choose independently at
            # each chunk boundary according to intervention_probability.
            if force_expert:
                use_expert = True
            else:
                use_expert = np.random.rand() < cfg.intervention_probability

            if use_expert:
                # Execute expert live, capturing each action. No save/restore
                # (that would break the controller state).
                chunk = np.zeros((cfg.action_horizon, cfg.action_dim), dtype=np.float32)
                n_taken = 0
                for step in range(cfg.action_horizon):
                    if t >= self._max_steps:
                        break
                    a = np.asarray(self.expert.act(env), dtype=np.float32)
                    chunk[step] = a
                    n_taken += 1
                    obs, reward, done, _ = env.step(a.tolist())
                    t += 1
                    if done:
                        break
                # Pad remaining slots with a gripper-hold no-op so the encoder
                # gets a valid (action_horizon, 7) tensor.
                if n_taken < cfg.action_horizon:
                    chunk[n_taken:] = np.array([0.0] * 6 + [chunk[n_taken - 1, 6]],
                                               dtype=np.float32)
                model_space = self._encode_expert_chunk_to_model_space(chunk)
                w_star = self._invert_action_chunk(model_space, ctx)
                self.buffer.add(
                    pooled_feat[0].detach().cpu().numpy(),
                    w_star[0].float().detach().cpu().numpy(),
                )
                expert_chunks += 1
            else:
                # Noise from learned policy (or randn if not ready).
                if self.noise_policy is None or len(self.buffer) < cfg.bc_batch_size:
                    w = torch.randn(
                        1, cfg.model_action_horizon, cfg.model_action_dim,
                        device=self.device, dtype=torch.bfloat16,
                    )
                else:
                    with torch.no_grad():
                        w = self.noise_policy(pooled_feat.to(self.device)).to(torch.bfloat16)
                env_chunk = self._sample_with_noise(ctx, w)
                policy_chunks += 1
                for step in range(min(cfg.replan_steps, cfg.action_horizon)):
                    if t >= self._max_steps:
                        break
                    obs, reward, done, _ = env.step(env_chunk[step].tolist())
                    t += 1
                    if done:
                        break

        success = self._is_success(done, reward)
        return {"success": success, "steps": t, "expert_chunks": expert_chunks,
                "policy_chunks": policy_chunks, "init_idx": int(init_idx)}

    def train_step(self) -> dict:
        """One BC gradient step: L = MSE(noise_policy(obs_feat), w_star)."""
        obs_np, w_np = self.buffer.sample(self.cfg.bc_batch_size)
        obs = torch.from_numpy(obs_np).to(self.device)
        w_target = torch.from_numpy(w_np).to(self.device)
        pred = self.noise_policy(obs)
        loss = ((pred - w_target) ** 2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item()}

    def evaluate(self, num_episodes: int) -> float:
        """Deterministic eval; no expert. Uses learned noise policy (or randn
        if untrained)."""
        self.setup()
        env = self.env
        cfg = self.cfg
        if self.noise_policy is not None:
            self.noise_policy.eval()
        succ = 0
        for ep in range(num_episodes):
            obs, _ = self._reset_episode(ep)
            t = 0
            reward = 0.0
            done = False
            while t < self._max_steps and not done:
                groot_obs = obs_to_groot(obs, self._task_prompt)
                pooled_feat, ctx = self._compute_obs_features(groot_obs)
                self._lazy_init_noise_policy(pooled_feat.shape[-1])
                if self.noise_policy is None or len(self.buffer) < cfg.bc_batch_size:
                    w = torch.randn(
                        1, cfg.model_action_horizon, cfg.model_action_dim,
                        device=self.device, dtype=torch.bfloat16,
                    )
                else:
                    with torch.no_grad():
                        w = self.noise_policy(pooled_feat.to(self.device)).to(torch.bfloat16)
                env_chunk = self._sample_with_noise(ctx, w)
                for step in range(min(cfg.replan_steps, cfg.action_horizon)):
                    if t >= self._max_steps:
                        break
                    obs, reward, done, _ = env.step(env_chunk[step].tolist())
                    t += 1
                    if done:
                        break
            if self._is_success(done, reward):
                succ += 1
        if self.noise_policy is not None:
            self.noise_policy.train()
        return succ / num_episodes

    def train_loop(self):
        cfg = self.cfg
        self.setup()
        # Seed the buffer with expert-only rollouts before the noise policy is
        # ever queried.
        for ep in range(cfg.seed_expert_episodes):
            traj = self.rollout_episode(ep, force_expert=True)
            print(f"[seed {ep+1}/{cfg.seed_expert_episodes}] success={traj['success']} "
                  f"steps={traj['steps']} buffer={len(self.buffer or [])}")
        best_sr = 0.0
        t0 = time.time()
        for ep in range(cfg.episodes):
            traj = self.rollout_episode(ep)
            if self.buffer is not None and len(self.buffer) >= cfg.bc_batch_size:
                losses = []
                for _ in range(cfg.bc_steps_per_episode):
                    losses.append(self.train_step()["loss"])
                if (ep + 1) % cfg.log_interval == 0:
                    print(f"[ep {ep+1}] loss={np.mean(losses):.5f} "
                          f"expert_chunks={traj['expert_chunks']} "
                          f"policy_chunks={traj['policy_chunks']} "
                          f"buf={len(self.buffer)}")
            if (ep + 1) % cfg.eval_interval == 0:
                sr = self.evaluate(cfg.eval_episodes)
                if sr > best_sr:
                    best_sr = sr
                print(f"[ep {ep+1}] SR={sr:.0%} best={best_sr:.0%} "
                      f"elapsed={time.time()-t0:.0f}s")
        print(f"Done. Best SR: {best_sr:.0%}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--suite", default="libero_90", choices=list(SUITE_MAX_STEPS.keys()))
    p.add_argument("--task_id", type=int, default=57)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt", default=DEFAULT_CKPT)
    p.add_argument("--action_horizon", type=int, default=16)
    p.add_argument("--action_dim", type=int, default=7)
    p.add_argument("--replan_steps", type=int, default=8)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed_expert_episodes", type=int, default=10)
    p.add_argument("--bc_steps_per_episode", type=int, default=100)
    p.add_argument("--bc_batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--buffer_size", type=int, default=100_000)
    p.add_argument("--expert_position_gain", type=float, default=5.0,
                   help="LiberoExpert gain; task 57 reference uses 5")
    p.add_argument("--inverter_fp_per_step", type=int, default=10,
                   help="Per-step fixed-point iterations for the noise inverter")
    p.add_argument("--intervention_probability", type=float, default=1.0)
    p.add_argument("--eval_interval", type=int, default=10)
    p.add_argument("--eval_episodes", type=int, default=25)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--device", default="cuda")
    return p


def main():
    args = build_argparser().parse_args()
    cfg = FlowDaggerConfig(**{k: v for k, v in vars(args).items() if hasattr(FlowDaggerConfig, k)})

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    loop = GrootFlowDaggerLoop(cfg)
    loop.train_loop()


if __name__ == "__main__":
    main()
