#!/usr/bin/env python
"""Flow-matching inversion primitives for GR00T N1.7's action head.

GR00T inference is linear flow matching with Euler integration:
    actions_0 = randn(...)  # noise
    for t in range(N):
        v = predict_velocity(actions_t, t/N)
        actions_{t+1} = actions_t + (1/N) * v

To run noise-space DAgger we need the reverse map: given a target action
chunk, recover the noise w* that the sampler would have integrated into it.
This module provides:
    encode_features      run the backbone+state encoder once, cache features
    predict_velocity     single velocity evaluation at (actions, t)
    forward_euler        noise -> action chunk (the sampler)
    reverse_euler        action chunk -> noise (explicit one-pass inverse)
    perstep_fp_reverse   action chunk -> noise (per-step fixed point, accurate)

Inversion operates in MODEL space (B, action_horizon, action_dim), the padded
tensor the action head integrates, not the runtime-decoded env action.
"""
import torch

from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper


def encode_features(policy: Gr00tPolicy, sim_obs: dict):
    """Run one forward pass, capturing the encoded backbone/state features.

    Monkey-patches action_head.get_action_with_features so we reuse the
    policy's own processor + collator pipeline instead of re-implementing it.
    Returns a dict with backbone_features, state_features, embodiment_id,
    backbone_output, action_input.
    """
    sim_policy = Gr00tSimPolicyWrapper(policy, strict=False)
    captured = {}
    action_head = policy.model.action_head
    orig = action_head.get_action_with_features

    def capturing(backbone_features, state_features, embodiment_id,
                  backbone_output, action_input, options=None):
        captured["backbone_features"] = backbone_features
        captured["state_features"] = state_features
        captured["embodiment_id"] = embodiment_id
        captured["backbone_output"] = backbone_output
        captured["action_input"] = action_input
        return orig(backbone_features, state_features, embodiment_id,
                    backbone_output, action_input, options)

    action_head.get_action_with_features = capturing
    try:
        _ = sim_policy.get_action(sim_obs)  # triggers the patched path
    finally:
        action_head.get_action_with_features = orig
    return captured


@torch.no_grad()
def predict_velocity(action_head, actions, t_cont, backbone_features, state_features,
                     embodiment_id, backbone_output):
    """Predict the flow velocity at a given (actions, t_cont)."""
    B = actions.shape[0]
    device = actions.device
    t_discretized = int(t_cont * action_head.num_timestep_buckets)
    timesteps_tensor = torch.full((B,), t_discretized, device=device)
    action_features = action_head.action_encoder(actions, timesteps_tensor, embodiment_id)
    if action_head.config.add_pos_embed:
        pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
        pos_embs = action_head.position_embedding(pos_ids).unsqueeze(0)
        action_features = action_features + pos_embs
    sa_embs = torch.cat((state_features, action_features), dim=1)
    if action_head.config.use_alternate_vl_dit:
        model_output = action_head.model(
            hidden_states=sa_embs,
            encoder_hidden_states=backbone_features,
            timestep=timesteps_tensor,
            image_mask=backbone_output.image_mask,
            backbone_attention_mask=backbone_output.backbone_attention_mask,
        )
    else:
        model_output = action_head.model(
            hidden_states=sa_embs,
            encoder_hidden_states=backbone_features,
            timestep=timesteps_tensor,
        )
    pred = action_head.action_decoder(model_output, embodiment_id)
    return pred[:, -action_head.action_horizon:]


@torch.no_grad()
def forward_euler(action_head, w_init, ctx):
    """w (noise) -> action chunk. The GR00T flow sampler."""
    N = action_head.num_inference_timesteps
    dt = 1.0 / N
    actions = w_init.clone()
    for t in range(N):
        v = predict_velocity(
            action_head, actions, t / N,
            ctx["backbone_features"], ctx["state_features"],
            ctx["embodiment_id"], ctx["backbone_outputs"],
        )
        actions = actions + dt * v
    return actions


@torch.no_grad()
def reverse_euler(action_head, action_target, ctx):
    """Action chunk -> w (noise) via explicit one-pass reverse Euler.

    Forward step k integrates with velocity evaluated at t=k/N; the inverse
    undoes step (N-1)..0 using velocity at the same t. Coarse; use
    perstep_fp_reverse for BC-quality targets.
    """
    N = action_head.num_inference_timesteps
    dt = 1.0 / N
    actions = action_target.clone()
    for k in range(N - 1, -1, -1):
        v = predict_velocity(
            action_head, actions, k / N,
            ctx["backbone_features"], ctx["state_features"],
            ctx["embodiment_id"], ctx["backbone_outputs"],
        )
        actions = actions - dt * v
    return actions


@torch.no_grad()
def perstep_fp_reverse(action_head, action_target, ctx, fp_per_step=5):
    """Action chunk -> w (noise) via per-step fixed-point reverse Euler.

    Inverts the exact discrete Euler map at each step. Forward step k is
        x_{k+1} = x_k + dt * v(x_k, t_k=k/N)
    Solve the implicit relation x_k = x_{k+1} - dt * v(x_k, t_k) by fixed-point
    iteration, initialized with the explicit reverse step. More accurate than
    the one-pass reverse_euler; cost is roughly linear in fp_per_step.
    """
    N = action_head.num_inference_timesteps
    dt = 1.0 / N
    x = action_target.clone()  # x_N (target)
    for k in range(N - 1, -1, -1):
        t_k = k / N
        # Initial guess (explicit reverse Euler)
        v0 = predict_velocity(
            action_head, x, t_k,
            ctx["backbone_features"], ctx["state_features"],
            ctx["embodiment_id"], ctx["backbone_outputs"],
        )
        x_k = x - dt * v0
        # Fixed-point refinement
        for _ in range(fp_per_step):
            v = predict_velocity(
                action_head, x_k, t_k,
                ctx["backbone_features"], ctx["state_features"],
                ctx["embodiment_id"], ctx["backbone_outputs"],
            )
            x_k = x - dt * v
        x = x_k
    return x


@torch.no_grad()
def fixed_point_reverse(action_head, action_target, ctx, refine_steps=10):
    """Action chunk -> w (noise) via global fixed-point refinement.

    Over-corrects the target then re-inverts each iteration. Usually
    outperformed by perstep_fp_reverse; kept as an alternative.
    """
    noise = reverse_euler(action_head, action_target, ctx)
    for _ in range(refine_steps):
        reconstructed = forward_euler(action_head, noise, ctx)
        corrected = action_target + (action_target - reconstructed)
        noise = reverse_euler(action_head, corrected, ctx)
    return noise
