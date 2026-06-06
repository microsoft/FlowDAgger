"""Steering policy: image encoder + MLP head (PixelMultiplexer) that predicts
the noise vector fed to the base policy sampler. Trained by behavior cloning
(MSE on inverted expert noise targets).
"""

import functools
import copy
from typing import Dict, Optional, Sequence, Union

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import checkpoints, train_state
from flax.core.frozen_dict import FrozenDict
from typing import Any

from jax_utils import (
    eval_actions_jit,
    sample_actions_jit,
    batched_random_crop,
    color_transform,
    Params,
    PRNGKey,
)
from nets import (
    Encoder,
    PixelMultiplexer,
    PrecomputedFeatureEncoder,
    LearnedStdTanhNormalPolicy,
    LearnedStdNormalPolicy,
)
from buffer import DatasetDict


class TrainState(train_state.TrainState):
    batch_stats: Any


def _get_batch_stats(actor):
    if hasattr(actor, 'batch_stats'):
        return actor.batch_stats
    return None


@functools.partial(jax.jit, static_argnames=('color_jitter', 'num_cameras'))
def _bc_step(rng, actor, batch, color_jitter, num_cameras):
    """One BC step: MSE(actor.mode(), target_noise)."""
    aug_pixels = batch['observations']['pixels']
    if batch['observations']['pixels'].squeeze().ndim != 2:
        rng, key = jax.random.split(rng)
        aug_pixels = batched_random_crop(key, batch['observations']['pixels'])
        if color_jitter:
            rng, key = jax.random.split(rng)
            if num_cameras > 1:
                for i in range(num_cameras):
                    aug_pixels = aug_pixels.at[:, :, :, i*3:(i+1)*3].set(
                        (color_transform(key, aug_pixels[:, :, :, i*3:(i+1)*3].astype(jnp.float32)/255.)*255).astype(jnp.uint8)
                    )
            else:
                aug_pixels = (color_transform(key, aug_pixels.astype(jnp.float32)/255.)*255).astype(jnp.uint8)

    observations = batch['observations'].copy(add_or_replace={'pixels': aug_pixels})

    def loss_fn(params):
        if hasattr(actor, 'batch_stats') and actor.batch_stats is not None:
            dist = actor.apply_fn(
                {'params': params, 'batch_stats': actor.batch_stats},
                observations,
            )
        else:
            dist = actor.apply_fn({'params': params}, observations)

        predicted = dist.mode()
        target = batch['actions'].reshape(predicted.shape)
        mse = jnp.mean(jnp.square(predicted - target))
        return mse, {'bc_loss': mse}

    grads, info = jax.grad(loss_fn, has_aux=True)(actor.params)
    new_actor = actor.apply_gradients(grads=grads)

    return rng, new_actor, info


class SteeringPolicy:
    """BC steering policy predicting noise vectors for a base policy.

    Duck-types the collect_traj interface: sample_actions(),
    action_chunk_shape, _rng.
    """

    def __init__(
        self,
        seed: int,
        observations: Union[jnp.ndarray, DatasetDict],
        actions: jnp.ndarray,
        lr: float = 3e-4,
        decay_steps: Optional[int] = None,
        hidden_dims: Sequence[int] = (256, 256),
        cnn_features: Sequence[int] = (32, 32, 32, 32),
        cnn_strides: Sequence[int] = (2, 1, 1, 1),
        cnn_padding: str = 'VALID',
        latent_dim: int = 50,
        dropout_rate: Optional[float] = None,
        encoder_type='siglip_pi0',
        encoder_norm='group',
        color_jitter=True,
        use_spatial_softmax=True,
        softmax_temperature=1,
        use_bottleneck=True,
        action_magnitude: float = 1.0,
        num_cameras: int = 1,
        # 'tanh': bounded to +/- action_magnitude. 'linear': raw output (no
        # squash), so the actor can reach w* values outside that range.
        output_bound: str = 'tanh',
    ):
        self.color_jitter = color_jitter
        self.num_cameras = num_cameras

        self.action_dim = np.prod(actions.shape[-2:])
        self.action_chunk_shape = actions.shape[-2:]

        rng = jax.random.PRNGKey(seed)
        rng, actor_key = jax.random.split(rng, 2)

        if encoder_type == 'small':
            encoder_def = Encoder(cnn_features, cnn_strides, cnn_padding)
        elif encoder_type in ('dinov2', 'resnet_pretrained', 'siglip_pi0', 'vlm_pi0'):
            encoder_def = PrecomputedFeatureEncoder()
        elif encoder_type in (
            'impala', 'impala_small', 'resnet_small',
            'resnet_18_v1', 'resnet_34_v1',
            'resnet_small_v2', 'resnet_18_v2', 'resnet_34_v2',
        ):
            raise ValueError(
                f"encoder type {encoder_type!r} was removed in the open-source "
                "minimal build; use 'small' for a pixel ConvNet or a precomputed "
                "feature encoder ('siglip_pi0', 'vlm_pi0')."
            )
        else:
            raise ValueError(f'encoder type not found: {encoder_type}')

        if decay_steps is not None:
            lr = optax.cosine_decay_schedule(lr, decay_steps)

        if len(hidden_dims) == 1:
            hidden_dims = (hidden_dims[0], hidden_dims[0], hidden_dims[0])

        if output_bound == 'tanh':
            policy_def = LearnedStdTanhNormalPolicy(
                hidden_dims, self.action_dim,
                dropout_rate=dropout_rate,
                low=-action_magnitude, high=action_magnitude,
            )
        elif output_bound == 'linear':
            # No squash: raw MLP head; scale set by init / BC targets.
            policy_def = LearnedStdNormalPolicy(
                hidden_dims, self.action_dim,
                dropout_rate=dropout_rate,
            )
        else:
            raise ValueError(
                f"output_bound must be 'tanh' or 'linear', got {output_bound!r}"
            )

        actor_def = PixelMultiplexer(
            encoder=encoder_def,
            network=policy_def,
            latent_dim=latent_dim,
            use_bottleneck=use_bottleneck,
        )
        actor_def_init = actor_def.init(actor_key, observations)
        actor_params = actor_def_init['params']
        actor_batch_stats = actor_def_init.get('batch_stats', None)

        self._actor = TrainState.create(
            apply_fn=actor_def.apply,
            params=actor_params,
            tx=optax.adam(learning_rate=lr),
            batch_stats=actor_batch_stats,
        )

        self._rng = rng

    def sample_actions(self, observations: np.ndarray) -> np.ndarray:
        """Deterministic noise prediction (actor.mode())."""
        actions = eval_actions_jit(
            self._actor.apply_fn, self._actor.params,
            observations, _get_batch_stats(self._actor),
        )
        return np.asarray(actions)

    def update(self, batch: FrozenDict) -> Dict[str, float]:
        """Single BC gradient step."""
        self._rng, self._actor, info = _bc_step(
            self._rng, self._actor, batch,
            self.color_jitter, self.num_cameras,
        )
        return info

    @property
    def _save_dict(self):
        return {'actor': self._actor}

    def save_checkpoint(self, dir, step, keep_every_n_steps):
        checkpoints.save_checkpoint(
            dir, self._save_dict, step,
            prefix='checkpoint', overwrite=False,
            keep_every_n_steps=keep_every_n_steps,
        )

    def restore_checkpoint(self, dir):
        import pathlib
        assert pathlib.Path(dir).exists(), f"Checkpoint {dir} does not exist."
        # prefix MUST match save_checkpoint's prefix='checkpoint'. Flax defaults
        # to 'checkpoint_' (trailing underscore), which does NOT match files
        # named 'checkpoint<step>' and silently no-ops the restore (leaving
        # random init). Confirm a matching checkpoint exists before restoring.
        found = checkpoints.latest_checkpoint(dir, prefix='checkpoint')
        if found is None:
            raise RuntimeError(
                f"No steering-policy checkpoint found under {dir} (prefix='checkpoint')")
        output_dict = checkpoints.restore_checkpoint(
            dir, self._save_dict, prefix='checkpoint',
        )
        self._actor = output_dict['actor']
        print(f'restored steering policy from {found}')
