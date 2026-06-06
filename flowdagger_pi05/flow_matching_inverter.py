"""Noise inversion for pi0's flow matching (rectified flow).

Rectified flow ODE:
    Training:  x_t = t*noise + (1-t)*actions,  target v_t = noise - actions
    Inference: x_{t+dt} = x_t + dt*v_t   with dt = -1/N, t: 1->0

Inversion strategies:
- 'euler_reverse': run the ODE forward in time (t: 0->1) with dt = +1/N.
  Deterministic, no gradients.
- 'adam': optimize w* = argmin_w ||denoise(obs,w) - target||^2 + lambda*||w||^2
  via jax.grad through the Euler denoising loop.
- 'hybrid': Euler reverse init + Adam refinement.

Model params are shared (same device arrays) with the sampling policy, so
inversion adds no extra GPU memory.
"""

import logging
import time as _time

import jax
import jax.numpy as jnp
import numpy as np
import optax
import einops
from flax import nnx

from openpi.models.pi0 import make_attn_mask
from openpi.models import model as _model

log = logging.getLogger(__name__)


def _stack_observations(obs_list):
    """Concatenate a list of batch=1 Observations into one batched Observation."""
    return jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *obs_list)


def _build_euler_fn(graphdef, num_steps, direction):
    """Build a JIT-compiled Euler integration function (state, observation, x_init) -> x_out.

    direction: 'reverse' for t=0->1 (clean->noise) or 'forward' for t=1->0 (noise->clean).
    """
    if direction == 'reverse':
        t_start = 0.0
        dt = 1.0 / num_steps
        cond_fn = lambda carry: carry[1] <= 1.0 - dt / 2
    else:  # forward (denoising)
        t_start = 1.0
        dt = -1.0 / num_steps
        cond_fn = lambda carry: carry[1] >= -dt / 2

    def euler_fn(state, observation, x_init):
        model = nnx.merge(graphdef, state)
        observation = _model.preprocess_observation(None, observation, train=False)

        # Prefix KV cache (same as Pi0.sample_actions)
        prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = model.PaliGemma.llm(
            [prefix_tokens, None], mask=prefix_attn_mask, positions=positions
        )

        batch_size = observation.state.shape[0]

        def step(carry):
            x_t, time_val = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(
                observation, x_t, jnp.broadcast_to(time_val, (batch_size,))
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn = einops.repeat(
                prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1]
            )
            full_attn_mask = jnp.concatenate(
                [prefix_attn, suffix_attn_mask], axis=-1
            )
            pos = (jnp.sum(prefix_mask, axis=-1)[:, None]
                   + jnp.cumsum(suffix_mask, axis=-1) - 1)
            (_, suffix_out), _ = model.PaliGemma.llm(
                [None, suffix_tokens], mask=full_attn_mask,
                positions=pos, kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            v_t = model.action_out_proj(
                suffix_out[:, -model.action_horizon:]
            )
            return x_t + dt * v_t, time_val + dt

        x_out, _ = jax.lax.while_loop(cond_fn, step, (x_init, t_start))
        return x_out

    return jax.jit(euler_fn)


def _build_perstep_fp_reverse_fn(graphdef, num_steps, fp_per_step=3):
    """Build a JIT-compiled per-step fixed-point inversion function.

    Inverts the exact discrete Euler denoising map by solving the implicit
    equation at each step with fixed-point iteration. More accurate than plain
    Euler reverse because it inverts the discrete computation rather than
    approximating the continuous ODE.

    Forward Euler denoising step (dt_fwd = -1/K):  x_{t+dt_fwd} = x_t + dt_fwd * v(x_t, t)
    To invert, solve the implicit:                 x_t = x_{t+dt_fwd} - dt_fwd * v(x_t, t)
    via FP iteration:
        x_t^(0)   = x_{t+dt_fwd} - dt_fwd * v(x_{t+dt_fwd}, t+dt_fwd)  # Euler reverse guess
        x_t^(j+1) = x_{t+dt_fwd} - dt_fwd * v(x_t^(j), t)             # re-eval at correct point

    num_steps must match pi0's denoising step count. fp_per_step=3 is usually
    enough (contraction factor ~0.1-0.3 per iteration at dt=0.1).
    """
    dt_fwd = -1.0 / num_steps  # forward denoising dt (negative, t: 1->0)
    dt_rev = 1.0 / num_steps   # reverse dt (positive, t: 0->1)

    def perstep_fp_fn(state, observation, x_init):
        model = nnx.merge(graphdef, state)
        observation = _model.preprocess_observation(None, observation, train=False)

        prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = model.PaliGemma.llm(
            [prefix_tokens, None], mask=prefix_attn_mask, positions=positions
        )

        batch_size = observation.state.shape[0]

        def get_velocity(x_t, time_val):
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(
                observation, x_t, jnp.broadcast_to(time_val, (batch_size,))
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn = einops.repeat(
                prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1]
            )
            full_attn_mask = jnp.concatenate(
                [prefix_attn, suffix_attn_mask], axis=-1
            )
            pos = (jnp.sum(prefix_mask, axis=-1)[:, None]
                   + jnp.cumsum(suffix_mask, axis=-1) - 1)
            (_, suffix_out), _ = model.PaliGemma.llm(
                [None, suffix_tokens], mask=full_attn_mask,
                positions=pos, kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            return model.action_out_proj(
                suffix_out[:, -model.action_horizon:]
            )

        # Reverse from actions (t=0, x_init) to noise (t=1), step by step.
        x = x_init
        t_current = jnp.array(0.0)

        def outer_step(carry):
            x_prev, t_prev = carry
            t_next = t_prev + dt_rev  # time we're stepping TO

            # The forward step that produced x_prev was:
            #   x_prev = x_next + dt_fwd * v(x_next, t_next)
            # so x_next = x_prev + dt_rev * v(x_next, t_next). Seed the guess
            # with v at (x_prev, t_prev), then FP-refine x_next below.
            v_init = get_velocity(x_prev, t_prev)
            x_next = x_prev + dt_rev * v_init

            def fp_body(j, x_est):
                v_est = get_velocity(x_est, t_next)
                return x_prev + dt_rev * v_est

            x_next = jax.lax.fori_loop(0, fp_per_step, fp_body, x_next)

            return x_next, t_next

        cond_fn = lambda carry: carry[1] <= 1.0 - dt_rev / 2

        x_out, _ = jax.lax.while_loop(cond_fn, outer_step, (x, t_current))
        return x_out

    return jax.jit(perstep_fp_fn)


def _build_midpoint_fn(graphdef, num_steps, direction):
    """Build a JIT-compiled midpoint ODE solver (FireFlow-style).

    Reuses the previous step's midpoint velocity to estimate the current
    midpoint, then evaluates velocity there: second-order accuracy at near
    first-order cost (one extra model eval on the first step).

    Returns a function (state, observation, x_init) -> x_out. direction is
    'reverse' for t=0->1 (clean->noise) or 'forward' for t=1->0 (noise->clean).
    """
    if direction == 'reverse':
        t_start = 0.0
        dt = 1.0 / num_steps
        cond_fn = lambda carry: carry[1] <= 1.0 - dt / 2
    else:  # forward (denoising)
        t_start = 1.0
        dt = -1.0 / num_steps
        cond_fn = lambda carry: carry[1] >= -dt / 2

    def midpoint_fn(state, observation, x_init):
        model = nnx.merge(graphdef, state)
        observation = _model.preprocess_observation(None, observation, train=False)

        prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = model.PaliGemma.llm(
            [prefix_tokens, None], mask=prefix_attn_mask, positions=positions
        )

        batch_size = observation.state.shape[0]

        def get_velocity(x_t, time_val):
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(
                observation, x_t, jnp.broadcast_to(time_val, (batch_size,))
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn = einops.repeat(
                prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1]
            )
            full_attn_mask = jnp.concatenate(
                [prefix_attn, suffix_attn_mask], axis=-1
            )
            pos = (jnp.sum(prefix_mask, axis=-1)[:, None]
                   + jnp.cumsum(suffix_mask, axis=-1) - 1)
            (_, suffix_out), _ = model.PaliGemma.llm(
                [None, suffix_tokens], mask=full_attn_mask,
                positions=pos, kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            return model.action_out_proj(
                suffix_out[:, -model.action_horizon:]
            )

        # First step: Euler + midpoint velocity to seed the loop.
        v_0 = get_velocity(x_init, jnp.array(t_start))
        x_1 = x_init + dt * v_0
        x_mid_0 = x_init + (dt / 2) * v_0
        v_mid_seed = get_velocity(x_mid_0, jnp.array(t_start + dt / 2))

        def step(carry):
            x_t, time_val, v_mid_prev = carry
            x_mid_est = x_t + (dt / 2) * v_mid_prev
            v_mid = get_velocity(x_mid_est, time_val + dt / 2)
            x_next = x_t + dt * v_mid
            return x_next, time_val + dt, v_mid

        x_out, _, _ = jax.lax.while_loop(
            cond_fn, step, (x_1, t_start + dt, v_mid_seed)
        )
        return x_out

    return jax.jit(midpoint_fn)


def _build_tiled_adam_refine_fn(graphdef, adam_denoise_steps, full_denoise_steps,
                                adam_steps, adam_lr, grad_clip, reg_weight,
                                action_horizon=50):
    """Like _build_adam_refine_fn but optimizes a single (B, 1, 32) noise vector
    tiled to (B, action_horizon, 32) inside the loss.
    """
    adam_dt = -1.0 / adam_denoise_steps
    full_dt = -1.0 / full_denoise_steps

    optimizer = optax.chain(
        optax.clip_by_global_norm(grad_clip),
        optax.adam(adam_lr),
    )

    def tiled_adam_fn(state, observation, target_actions, init_noise_seed):
        """init_noise_seed: (B, 1, 32) - single noise vector per batch element."""
        model = nnx.merge(graphdef, state)
        observation = _model.preprocess_observation(None, observation, train=False)

        prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = model.PaliGemma.llm(
            [prefix_tokens, None], mask=prefix_attn_mask, positions=positions
        )

        batch_size = observation.state.shape[0]

        @jax.checkpoint
        def velocity_step(x_t, time_scalar):
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(
                observation, x_t, jnp.broadcast_to(time_scalar, (batch_size,))
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn = einops.repeat(
                prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1]
            )
            full_attn_mask = jnp.concatenate(
                [prefix_attn, suffix_attn_mask], axis=-1
            )
            pos = (jnp.sum(prefix_mask, axis=-1)[:, None]
                   + jnp.cumsum(suffix_mask, axis=-1) - 1)
            (_, suffix_out), _ = model.PaliGemma.llm(
                [None, suffix_tokens], mask=full_attn_mask,
                positions=pos, kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            return model.action_out_proj(
                suffix_out[:, -model.action_horizon:]
            )

        anchor_seed = init_noise_seed

        def loss_fn(seed_param):
            x_t = jnp.tile(seed_param, (1, action_horizon, 1))
            time_val = 1.0
            for _ in range(adam_denoise_steps):
                v_t = velocity_step(x_t, jnp.array(time_val))
                x_t = x_t + adam_dt * v_t
                time_val += adam_dt
            mse = jnp.mean(jnp.square(x_t - target_actions), axis=(-2, -1))
            reg = reg_weight * jnp.mean(
                jnp.square(seed_param - anchor_seed), axis=(-2, -1)
            )
            return jnp.mean(mse + reg), mse

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        opt_state = optimizer.init(init_noise_seed)

        def scan_step(carry, _):
            seed, opt_st = carry
            (loss, mse), grads = grad_fn(seed)
            updates, new_opt_st = optimizer.update(grads, opt_st, seed)
            new_seed = optax.apply_updates(seed, updates)
            return (new_seed, new_opt_st), None

        (refined_seed, _), _ = jax.lax.scan(
            scan_step, (init_noise_seed, opt_state), None, length=adam_steps
        )

        x_t = jnp.tile(refined_seed, (1, action_horizon, 1))
        time_val = 1.0
        for _ in range(full_denoise_steps):
            v_t = velocity_step(x_t, jnp.array(time_val))
            x_t = x_t + full_dt * v_t
            time_val += full_dt
        final_mse = jnp.mean(
            jnp.square(x_t - target_actions), axis=(-2, -1)
        )

        return jnp.tile(refined_seed, (1, action_horizon, 1)), final_mse

    return jax.jit(tiled_adam_fn)


def _build_tiled_focused_adam_refine_fn(graphdef, adam_denoise_steps, full_denoise_steps,
                                         adam_steps, adam_lr, grad_clip, reg_weight,
                                         action_horizon=50, active_dims=7, active_positions=20):
    """Like tiled_adam but loss only computed on first active_positions and active_dims.
    Focuses gradient on the dims/positions that actually matter for env actions.
    """
    adam_dt = -1.0 / adam_denoise_steps
    full_dt = -1.0 / full_denoise_steps

    optimizer = optax.chain(
        optax.clip_by_global_norm(grad_clip),
        optax.adam(adam_lr),
    )

    def tiled_focused_adam_fn(state, observation, target_actions, init_noise_seed):
        """init_noise_seed: (B, 1, 32) - single noise vector per batch element."""
        model = nnx.merge(graphdef, state)
        observation = _model.preprocess_observation(None, observation, train=False)

        prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = model.PaliGemma.llm(
            [prefix_tokens, None], mask=prefix_attn_mask, positions=positions
        )

        batch_size = observation.state.shape[0]

        @jax.checkpoint
        def velocity_step(x_t, time_scalar):
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(
                observation, x_t, jnp.broadcast_to(time_scalar, (batch_size,))
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn = einops.repeat(
                prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1]
            )
            full_attn_mask = jnp.concatenate(
                [prefix_attn, suffix_attn_mask], axis=-1
            )
            pos = (jnp.sum(prefix_mask, axis=-1)[:, None]
                   + jnp.cumsum(suffix_mask, axis=-1) - 1)
            (_, suffix_out), _ = model.PaliGemma.llm(
                [None, suffix_tokens], mask=full_attn_mask,
                positions=pos, kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            return model.action_out_proj(
                suffix_out[:, -model.action_horizon:]
            )

        anchor_seed = init_noise_seed

        def loss_fn(seed_param):
            x_t = jnp.tile(seed_param, (1, action_horizon, 1))
            time_val = 1.0
            for _ in range(adam_denoise_steps):
                v_t = velocity_step(x_t, jnp.array(time_val))
                x_t = x_t + adam_dt * v_t
                time_val += adam_dt
            # Loss only on the first active_positions and active_dims.
            x_focused = x_t[:, :active_positions, :active_dims]
            target_focused = target_actions[:, :active_positions, :active_dims]
            mse = jnp.mean(jnp.square(x_focused - target_focused), axis=(-2, -1))
            reg = reg_weight * jnp.mean(
                jnp.square(seed_param - anchor_seed), axis=(-2, -1)
            )
            return jnp.mean(mse + reg), mse

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        opt_state = optimizer.init(init_noise_seed)

        def scan_step(carry, _):
            seed, opt_st = carry
            (loss, mse), grads = grad_fn(seed)
            updates, new_opt_st = optimizer.update(grads, opt_st, seed)
            new_seed = optax.apply_updates(seed, updates)
            return (new_seed, new_opt_st), None

        (refined_seed, _), _ = jax.lax.scan(
            scan_step, (init_noise_seed, opt_state), None, length=adam_steps
        )

        # Final eval at full steps (over all dims, for comparison).
        x_t = jnp.tile(refined_seed, (1, action_horizon, 1))
        time_val = 1.0
        for _ in range(full_denoise_steps):
            v_t = velocity_step(x_t, jnp.array(time_val))
            x_t = x_t + full_dt * v_t
            time_val += full_dt
        final_mse = jnp.mean(
            jnp.square(x_t - target_actions), axis=(-2, -1)
        )

        return jnp.tile(refined_seed, (1, action_horizon, 1)), final_mse

    return jax.jit(tiled_focused_adam_fn)


def _build_adam_refine_fn(graphdef, adam_denoise_steps, full_denoise_steps,
                          adam_steps, adam_lr, grad_clip, reg_weight,
                          action_dim_env=7):
    """Build a JIT-compiled Adam refinement function.

    Compiles the whole Adam loop (forward + backward + optimizer update) into a
    single XLA program. Closures are created once during tracing, not per call,
    so repeated invocations do not leak GPU memory.

    Returns a function
    (state, observation, target_actions, init_noise) -> (refined_noise, final_mse).
    adam_denoise_steps uses fewer Euler steps during optimization for speed;
    final_mse is evaluated at full_denoise_steps. reg_weight weights the
    ||w - w_init||^2 anchor.
    """
    adam_dt = -1.0 / adam_denoise_steps
    full_dt = -1.0 / full_denoise_steps

    optimizer = optax.chain(
        optax.clip_by_global_norm(grad_clip),
        optax.adam(adam_lr),
    )

    def adam_refine_fn(state, observation, target_actions, init_noise):
        model = nnx.merge(graphdef, state)
        observation = _model.preprocess_observation(None, observation, train=False)

        # Prefix KV cache, reused across all Adam steps + final eval.
        prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = model.PaliGemma.llm(
            [prefix_tokens, None], mask=prefix_attn_mask, positions=positions
        )

        batch_size = observation.state.shape[0]

        @jax.checkpoint
        def velocity_step(x_t, time_scalar):
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(
                observation, x_t, jnp.broadcast_to(time_scalar, (batch_size,))
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn = einops.repeat(
                prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1]
            )
            full_attn_mask = jnp.concatenate(
                [prefix_attn, suffix_attn_mask], axis=-1
            )
            pos = (jnp.sum(prefix_mask, axis=-1)[:, None]
                   + jnp.cumsum(suffix_mask, axis=-1) - 1)
            (_, suffix_out), _ = model.PaliGemma.llm(
                [None, suffix_tokens], mask=full_attn_mask,
                positions=pos, kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            v_t = model.action_out_proj(
                suffix_out[:, -model.action_horizon:]
            )
            return v_t

        anchor_noise = init_noise

        def loss_fn(noise_param):
            x_t = noise_param
            time_val = 1.0
            for _ in range(adam_denoise_steps):
                v_t = velocity_step(x_t, jnp.array(time_val))
                x_t = x_t + adam_dt * v_t
                time_val += adam_dt
            mse = jnp.mean(jnp.square(x_t - target_actions), axis=(-2, -1))
            reg = reg_weight * jnp.mean(
                jnp.square(noise_param - anchor_noise), axis=(-2, -1)
            )
            return jnp.mean(mse + reg), mse

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        opt_state = optimizer.init(init_noise)

        def scan_step(carry, _):
            noise, opt_st = carry
            (loss, mse), grads = grad_fn(noise)
            updates, new_opt_st = optimizer.update(grads, opt_st, noise)
            new_noise = optax.apply_updates(noise, updates)
            return (new_noise, new_opt_st), None

        (refined_noise, _), _ = jax.lax.scan(
            scan_step, (init_noise, opt_state), None, length=adam_steps
        )

        # Final eval at full denoise steps (reuses prefix cache).
        x_t = refined_noise
        time_val = 1.0
        for _ in range(full_denoise_steps):
            v_t = velocity_step(x_t, jnp.array(time_val))
            x_t = x_t + full_dt * v_t
            time_val += full_dt
        final_mse = jnp.mean(
            jnp.square(x_t - target_actions), axis=(-2, -1)
        )

        return refined_noise, final_mse

    return jax.jit(adam_refine_fn)


class FlowMatchingInverter:
    """Inverts pi0's flow matching to find noise w* that reproduces target actions.

    model must be the raw pi0 model instance shared with the sampling Policy.
    See the module docstring for the available inversion methods.
    """

    def __init__(
        self,
        model,
        method='euler_reverse',
        num_denoise_steps=10,
        refine_steps=20,
        adam_lr=0.01,
        regularization_weight=0.01,
        grad_clip=1.0,
        adam_denoise_steps=5,
        seed=0,
        solver='euler',
        fp_per_step=5,
        adam_steps=None,  # legacy alias for refine_steps
    ):
        self.model = model
        self.method = method
        self._rng = jax.random.PRNGKey(seed)
        self.num_denoise_steps = num_denoise_steps
        self.refine_steps = adam_steps if adam_steps is not None else refine_steps
        self.adam_lr = adam_lr
        self.regularization_weight = regularization_weight
        self.grad_clip = grad_clip
        self.adam_denoise_steps = adam_denoise_steps

        assert method in ('euler_reverse', 'adam', 'hybrid', 'fixed_point', 'perstep_fp', 'tiled_adam', 'tiled_focused_adam'), \
            f"Unknown method: {method}"
        assert solver in ('euler', 'midpoint'), f"Unknown solver: {solver}"
        self.solver = solver

        # Split model for JIT; shares device arrays with Policy, no extra memory.
        graphdef, state = nnx.split(model)
        self._graphdef = graphdef
        self._state = state

        build_fn = _build_midpoint_fn if solver == 'midpoint' else _build_euler_fn
        self._euler_reverse_jit = build_fn(
            graphdef, num_denoise_steps, 'reverse'
        )
        self._denoise_jit = build_fn(
            graphdef, num_denoise_steps, 'forward'
        )

        # Adam refinement keeps its own Euler loop for gradients.
        self._adam_refine_jit = _build_adam_refine_fn(
            graphdef, adam_denoise_steps, num_denoise_steps,
            self.refine_steps, adam_lr, grad_clip, regularization_weight,
        )

        self.fp_per_step = fp_per_step
        self._perstep_fp_jit = _build_perstep_fp_reverse_fn(
            graphdef, num_denoise_steps, fp_per_step=fp_per_step,
        )

        ah = model.action_horizon

        # Tiled Adam: optimizes (B,1,32) seed, tiles to (B,ah,32) in loss.
        self._tiled_adam_jit = _build_tiled_adam_refine_fn(
            graphdef, adam_denoise_steps, num_denoise_steps,
            self.refine_steps, adam_lr, grad_clip, regularization_weight,
            action_horizon=ah,
        )

        # Focused tiled Adam: loss on first 20 positions, 7 active dims.
        self._tiled_focused_adam_jit = _build_tiled_focused_adam_refine_fn(
            graphdef, adam_denoise_steps, num_denoise_steps,
            self.refine_steps, adam_lr, grad_clip, regularization_weight,
            action_horizon=ah,
        )

        log.info(
            f"FlowMatchingInverter: method={method}, solver={solver}, "
            f"denoise_steps={num_denoise_steps}, "
            f"refine_steps={self.refine_steps}, adam_lr={adam_lr}, adam_denoise_steps={adam_denoise_steps}"
        )

    def _euler_reverse(self, observation, clean_actions):
        """Euler reverse: t=0->1 (clean->noise). JIT-compiled."""
        return self._euler_reverse_jit(self._state, observation, clean_actions)

    def _denoise(self, observation, noise):
        """Forward denoising: t=1->0 (noise->clean). JIT-compiled."""
        return self._denoise_jit(self._state, observation, noise)

    def _adam_optimize(self, observation, target_actions, init_noise):
        """Optimize noise to reconstruct target actions via Adam (JIT-compiled)."""
        return self._adam_refine_jit(
            self._state, observation, target_actions, init_noise
        )

    def invert(self, obs_pi_zero_processed, target_internal):
        """Find noise w* such that denoise(obs, w*) ~ target_internal.

        target_internal is (B, action_horizon, action_dim) in pi0 internal space.
        Returns the (B, action_horizon, action_dim) inverted noise and per-batch
        reconstruction MSE (B,).
        """
        target_internal = jnp.asarray(target_internal)
        if target_internal.ndim == 2:
            target_internal = target_internal[jnp.newaxis, ...]

        t_start = _time.time()

        if self.method == 'euler_reverse':
            noise = self._euler_reverse(obs_pi_zero_processed, target_internal)
            reconstructed = self._denoise(obs_pi_zero_processed, noise)
            error = jnp.mean(
                jnp.square(reconstructed - target_internal), axis=(-2, -1)
            )

        elif self.method == 'adam':
            self._rng, rng = jax.random.split(self._rng)
            init_noise = jax.random.normal(rng, target_internal.shape)
            noise, error = self._adam_optimize(
                obs_pi_zero_processed, target_internal, init_noise
            )

        elif self.method == 'hybrid':
            init_noise = self._euler_reverse(
                obs_pi_zero_processed, target_internal
            )
            noise, error = self._adam_optimize(
                obs_pi_zero_processed, target_internal, init_noise
            )

        elif self.method == 'fixed_point':
            noise = self._euler_reverse(obs_pi_zero_processed, target_internal)
            corrected_target = target_internal
            for _ in range(self.refine_steps):
                reconstructed = self._denoise(obs_pi_zero_processed, noise)
                residual = target_internal - reconstructed
                corrected_target = target_internal + residual
                noise = self._euler_reverse(obs_pi_zero_processed, corrected_target)
            reconstructed = self._denoise(obs_pi_zero_processed, noise)
            error = jnp.mean(
                jnp.square(reconstructed - target_internal), axis=(-2, -1)
            )

        elif self.method == 'perstep_fp':
            noise = self._perstep_fp_jit(
                self._state, obs_pi_zero_processed, target_internal
            )
            reconstructed = self._denoise(obs_pi_zero_processed, noise)
            error = jnp.mean(
                jnp.square(reconstructed - target_internal), axis=(-2, -1)
            )

        elif self.method == 'tiled_adam':
            # Seed from Euler reverse, mean across rows.
            init_full = self._euler_reverse(obs_pi_zero_processed, target_internal)
            init_seed = jnp.mean(init_full, axis=-2, keepdims=True)  # (B, 1, 32)
            noise, error = self._tiled_adam_jit(
                self._state, obs_pi_zero_processed, target_internal, init_seed
            )

        elif self.method == 'tiled_focused_adam':
            # Seed from Euler reverse, mean across rows.
            init_full = self._euler_reverse(obs_pi_zero_processed, target_internal)
            init_seed = jnp.mean(init_full, axis=-2, keepdims=True)  # (B, 1, 32)
            noise, error = self._tiled_focused_adam_jit(
                self._state, obs_pi_zero_processed, target_internal, init_seed
            )

        else:
            raise ValueError(f"Unknown method: {self.method}")

        elapsed = _time.time() - t_start
        log.debug(
            f"Inversion ({self.method}): error={float(error.mean()):.6f}, "
            f"time={elapsed*1000:.1f}ms"
        )

        return noise, error

    def refine_batch(self, observations, targets, init_noises, max_batch_size=8):
        """Batch Adam refinement of euler_reverse results, in mini-batches.

        Inputs are per-intervention lists (each batch=1): observations, targets
        (1, H, D), and init_noises (1, H, D) to refine. max_batch_size bounds
        peak GPU memory. Returns refined noises (each (1, H, D)) and per-item
        reconstruction MSE.
        """
        n = len(observations)
        if n == 0:
            return [], []

        t0 = _time.time()
        refined_noises = []
        errors = []

        import gc

        for start in range(0, n, max_batch_size):
            end = min(start + max_batch_size, n)
            batch_obs = _stack_observations(observations[start:end])
            batch_targets = jnp.concatenate(targets[start:end], axis=0)
            batch_init = jnp.concatenate(init_noises[start:end], axis=0)

            batch_noise, batch_error = self._adam_optimize(
                batch_obs, batch_targets, batch_init
            )

            # Copy out to numpy immediately to free JAX references.
            for i in range(end - start):
                refined_noises.append(np.array(batch_noise[i:i+1]))
                errors.append(float(batch_error[i]))

            del batch_obs, batch_targets, batch_init, batch_noise, batch_error
            gc.collect()

        elapsed = _time.time() - t0
        log.info(
            f"Batch refinement: {n} inversions in {elapsed:.1f}s "
            f"(avg MSE={np.mean(errors):.6f})"
        )
        print(f"  [Batch refinement] {n} inversions in {elapsed:.1f}s, "
              f"avg MSE={np.mean(errors):.6f}")

        return refined_noises, errors
