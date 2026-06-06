"""Network modules for the steering policy: weight inits, MLP trunk, ConvNet
pixel encoder, PixelMultiplexer, passthrough encoder for precomputed features,
and the learned-std (tanh / linear) policy heads.
"""

from typing import Any, Callable, Dict, Optional, Sequence, Union

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict


# --- weight inits ---------------------------------------------------------

def default_init(scale: float = jnp.sqrt(2)):
    return nn.initializers.orthogonal(scale)

def xavier_init():
    return nn.initializers.xavier_normal()

def kaiming_init():
    return nn.initializers.kaiming_normal()


# --- MLP trunk ------------------------------------------------------------

def _flatten_dict(x: Union[FrozenDict, jnp.ndarray]):
    if hasattr(x, 'values'):
        obs = []
        for k, v in sorted(x.items()):
            if k == 'state':  # flatten action chunk to 1D
                obs.append(jnp.reshape(v, [*v.shape[:-2], np.prod(v.shape[-2:])]))
            elif k == 'prev_action' or k == 'actions':
                if v.ndim > 2:
                    # deal with action chunk
                    obs.append(jnp.reshape(v, [*v.shape[:-2], np.prod(v.shape[-2:])]))
                else:
                    obs.append(v)
            else:
                obs.append(_flatten_dict(v))
        return jnp.concatenate(obs, -1)
    else:
        return x


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: int = False
    dropout_rate: Optional[float] = None
    init_scale: Optional[float] = 1.
    use_layer_norm: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        x = _flatten_dict(x)

        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=default_init(self.init_scale))(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                if self.dropout_rate is not None:
                    x = nn.Dropout(rate=self.dropout_rate)(
                        x, deterministic=not training)
                if self.use_layer_norm:
                    x = nn.LayerNorm()(x)
                x = self.activations(x)
        return x


# --- pixel encoders -------------------------------------------------------

class Encoder(nn.Module):
    features: Sequence[int] = (32, 32, 32, 32)
    strides: Sequence[int] = (2, 1, 1, 1)
    padding: str = 'VALID'

    @nn.compact
    def __call__(self, observations: jnp.ndarray, training=False) -> jnp.ndarray:
        assert len(self.features) == len(self.strides)

        x = observations.astype(jnp.float32) / 255.0
        x = jnp.reshape(x, (*x.shape[:-2], -1))

        for features, stride in zip(self.features, self.strides):
            x = nn.Conv(features,
                        kernel_size=(3, 3),
                        strides=(stride, stride),
                        kernel_init=default_init(),
                        padding=self.padding)(x)
            x = nn.relu(x)

        return x.reshape((*x.shape[:-3], -1))


class PixelMultiplexer(nn.Module):
    encoder: Union[nn.Module, list]
    network: nn.Module
    latent_dim: int
    use_bottleneck: bool = True

    @nn.compact
    def __call__(self,
                 observations: Union[FrozenDict, Dict],
                 actions: Optional[jnp.ndarray] = None,
                 training: bool = False):
        observations = FrozenDict(observations)

        x = self.encoder(observations['pixels'], training)
        if self.use_bottleneck:
            x = nn.Dense(self.latent_dim, kernel_init=xavier_init())(x)
            x = nn.LayerNorm()(x)
            x = nn.tanh(x)

        x = observations.copy(add_or_replace={'pixels': x})

        if actions is None:
            return self.network(x, training=training)
        else:
            return self.network(x, actions, training=training)


class PrecomputedFeatureEncoder(nn.Module):
    """Passthrough encoder for precomputed features (pi0 SigLIP/VLM).

    Expects observations['pixels'] to hold (B, feature_dim, 1) float feature
    vectors instead of raw images; extraction happens at collection time.
    Output: (B, feature_dim) float32.
    """
    @nn.compact
    def __call__(self, observations: jnp.ndarray, training=False) -> jnp.ndarray:
        x = observations.astype(jnp.float32)
        if x.shape[-1] == 1:
            x = x[..., 0]
        return jax.lax.stop_gradient(x)


# --- policy heads ---------------------------------------------------------

class LearnedStdNormalPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    dropout_rate: Optional[float] = None
    log_std_min: Optional[float] = -20
    log_std_max: Optional[float] = 2

    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 training: bool = False) -> distrax.Distribution:
        outputs = MLP(self.hidden_dims,
                      activate_final=True,
                      dropout_rate=self.dropout_rate)(observations,
                                                      training=training)

        means = nn.Dense(self.action_dim, kernel_init=default_init(1e-2))(outputs)

        log_stds = nn.Dense(self.action_dim, kernel_init=default_init(1e-2))(outputs)
        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = distrax.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds))
        return distribution


class TanhMultivariateNormalDiag(distrax.Transformed):

    def __init__(self,
                 loc: jnp.ndarray,
                 scale_diag: jnp.ndarray,
                 low: Optional[jnp.ndarray] = None,
                 high: Optional[jnp.ndarray] = None):
        distribution = distrax.MultivariateNormalDiag(loc=loc,
                                                      scale_diag=scale_diag)

        layers = []

        if not (low is None or high is None):

            def rescale_from_tanh(x):
                x = (x + 1) / 2  # (-1, 1) => (0, 1)
                return x * (high - low) + low

            def forward_log_det_jacobian(x):
                high_ = jnp.broadcast_to(high, x.shape)
                low_ = jnp.broadcast_to(low, x.shape)
                return jnp.sum(jnp.log(0.5 * (high_ - low_)), -1)

            layers.append(
                distrax.Lambda(
                    rescale_from_tanh,
                    forward_log_det_jacobian=forward_log_det_jacobian,
                    event_ndims_in=1,
                    event_ndims_out=1))

        layers.append(distrax.Block(distrax.Tanh(), 1))

        bijector = distrax.Chain(layers)

        super().__init__(distribution=distribution, bijector=bijector)

    def mode(self) -> jnp.ndarray:
        return self.bijector.forward(self.distribution.mode())


class LearnedStdTanhNormalPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    dropout_rate: Optional[float] = None
    log_std_min: Optional[float] = -20
    log_std_max: Optional[float] = 2
    low: Optional[float] = None
    high: Optional[float] = None

    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 training: bool = False) -> distrax.Distribution:
        outputs = MLP(self.hidden_dims,
                      activate_final=True,
                      dropout_rate=self.dropout_rate)(observations,
                                                      training=training)

        means = nn.Dense(self.action_dim, kernel_init=default_init(1e-2))(outputs)

        log_stds = nn.Dense(self.action_dim, kernel_init=default_init(1e-2))(outputs)
        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = TanhMultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds), low=self.low, high=self.high)
        return distribution
