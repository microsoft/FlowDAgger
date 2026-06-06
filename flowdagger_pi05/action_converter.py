"""Convert expert env-space actions to/from pi0.5's internal normalized space.

env -> internal:  NxD -> pad_to_dim(32) -> DeltaActions(mask) -> Normalize
internal -> env:  32D -> Unnormalize -> AbsoluteActions -> slice[:action_dim_env]

pi05_metaworld uses native 4D [dx, dy, dz, gripper] trained on raw deltas, so
delta_mask is None. delta_mask is only for policies whose first dims are
state-relative.
"""

import logging
import numpy as np

log = logging.getLogger(__name__)


class ActionConverter:
    """Converts between expert env-space actions and pi0 normalized internal space.

    Args:
        norm_stats_actions: "actions" norm stats from the pi0 checkpoint;
            .mean/.std of shape (32,) or (1, 32).
        delta_mask: DeltaActions mask; True dims are made relative to state.
            For LIBERO: (True,)*6 + (False,). None disables DeltaActions.
    """

    def __init__(
        self,
        norm_stats_actions,
        norm_stats_state=None,
        delta_mask=(True, True, True, True, True, True, False),
        action_dim_internal=32,
        action_dim_env=7,
        action_horizon=50,
        pre_norm_scale=1.0,
        gripper_scale=1.0,
    ):
        raw_mean = np.asarray(norm_stats_actions.mean).flatten()
        raw_std = np.asarray(norm_stats_actions.std).flatten()

        # Quantile normalization, when the checkpoint provides q01/q99 stats.
        self.use_quantiles = (hasattr(norm_stats_actions, 'q01')
                              and norm_stats_actions.q01 is not None)
        if self.use_quantiles:
            raw_q01 = np.asarray(norm_stats_actions.q01).flatten()
            raw_q99 = np.asarray(norm_stats_actions.q99).flatten()

        # Pad to internal dim if norm stats are raw action dim (e.g. 7D finetune)
        if len(raw_mean) < action_dim_internal:
            self.mean = np.zeros(action_dim_internal, dtype=np.float32)
            self.std = np.ones(action_dim_internal, dtype=np.float32)
            self.mean[:len(raw_mean)] = raw_mean
            self.std[:len(raw_std)] = raw_std
            if self.use_quantiles:
                self.q01 = np.zeros(action_dim_internal, dtype=np.float32)
                self.q99 = np.ones(action_dim_internal, dtype=np.float32)
                self.q01[:len(raw_q01)] = raw_q01
                self.q99[:len(raw_q99)] = raw_q99
        else:
            self.mean = raw_mean
            self.std = raw_std
            if self.use_quantiles:
                self.q01 = raw_q01
                self.q99 = raw_q99

        # State norm stats: pi0's DeltaActions/AbsoluteActions operate in
        # normalized space, not raw state space.
        if norm_stats_state is not None:
            self.state_mean = np.asarray(norm_stats_state.mean).flatten()
            self.state_std = np.asarray(norm_stats_state.std).flatten()
        else:
            self.state_mean = None
            self.state_std = None

        self.delta_mask = np.asarray(delta_mask) if delta_mask is not None else None
        self.action_dim_internal = action_dim_internal
        self.action_dim_env = action_dim_env
        self.action_horizon = action_horizon

        self.pre_norm_scale = float(pre_norm_scale)
        self.gripper_scale = float(gripper_scale)

        log.info(
            f"ActionConverter: env_dim={action_dim_env}, internal_dim={action_dim_internal}, "
            f"delta_mask={''.join('T' if m else 'F' for m in delta_mask) if delta_mask else 'None'}, "
            f"pre_norm_scale={self.pre_norm_scale}"
        )

    def expert_to_internal(self, expert_actions, state):
        """Convert expert env-space actions to pi0 internal normalized space.

        Args:
            expert_actions: (T, action_dim_env) or (action_dim_env,) env space.
            state: (S,) current robot state (from obs_to_qpos). For LIBERO:
                [eef_pos(3), axis_angle(3), gripper(2)].

        Returns (action_horizon, action_dim_internal).
        """
        if expert_actions.ndim == 1:
            expert_actions = expert_actions[np.newaxis, :]

        T = expert_actions.shape[0]

        # 0. Scale up before normalization (improves inversion SNR). Gripper is
        # the last channel (libero 7D=[xyzrpy,grip], metaworld 4D=[xyz,grip]).
        if self.pre_norm_scale != 1.0 or self.gripper_scale != 1.0:
            expert_actions = expert_actions.copy()
            grip_idx = self.action_dim_env - 1
            expert_actions[:, :grip_idx] *= self.pre_norm_scale
            expert_actions[:, grip_idx:grip_idx + 1] *= self.pre_norm_scale * self.gripper_scale

        # 1. Pad to internal dimension
        padded = np.zeros((T, self.action_dim_internal), dtype=np.float32)
        padded[:, :self.action_dim_env] = expert_actions

        # 2. Apply DeltaActions: subtract state from masked dimensions
        if self.delta_mask is not None:
            dims = len(self.delta_mask)
            offset = np.where(self.delta_mask, state[:dims], 0.0)
            padded[:, :dims] -= offset[np.newaxis, :]

        # 3. Normalize
        if self.use_quantiles:
            normalized = (padded - self.q01) / (self.q99 - self.q01 + 1e-6) * 2.0 - 1.0
        else:
            normalized = (padded - self.mean) / (self.std + 1e-6)

        # Zero padding dims to match training: openpi Normalizes the raw N-D
        # action *then* zero-pads to action_dim_internal, so the model sees
        # literal zeros in dims [action_dim_env:]. Leaving them as the
        # normalized-from-zero value (e.g. -1 when padded q01/q99 default to
        # (0, 1)) would force the inverter to push noise far from N(0,1).
        normalized[:, self.action_dim_env:] = 0.0

        # 4. Tile to action_horizon if needed
        if T < self.action_horizon:
            # Repeat last row to fill horizon
            padding = np.tile(normalized[-1:, :], (self.action_horizon - T, 1))
            normalized = np.concatenate([normalized, padding], axis=0)

        return normalized.astype(np.float32)

    def internal_to_env(self, internal_actions, state):
        """Convert pi0 internal normalized actions back to env space.

        Args:
            internal_actions: (T, action_dim_internal) internal space.
            state: (S,) current robot state.

        Returns (T, action_dim_env).
        """
        # 1. Unnormalize
        if self.use_quantiles:
            unnorm = (internal_actions + 1.0) / 2.0 * (self.q99 - self.q01 + 1e-6) + self.q01
        else:
            unnorm = internal_actions * (self.std + 1e-6) + self.mean

        # 2. AbsoluteActions: add state back to masked dimensions
        if self.delta_mask is not None:
            dims = len(self.delta_mask)
            offset = np.where(self.delta_mask, state[:dims], 0.0)
            unnorm[:, :dims] += offset[np.newaxis, :]

        # 3. Slice to env dimension
        result = unnorm[:, :self.action_dim_env]

        # 4. Undo pre-norm scale
        if self.pre_norm_scale != 1.0 or self.gripper_scale != 1.0:
            result = result.copy()
            grip_idx = self.action_dim_env - 1
            result[:, :grip_idx] /= self.pre_norm_scale
            result[:, grip_idx:grip_idx + 1] /= self.pre_norm_scale * self.gripper_scale

        return result

