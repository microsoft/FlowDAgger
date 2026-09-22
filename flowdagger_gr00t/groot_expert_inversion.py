#!/usr/bin/env python
"""Encode/decode between a 7D LIBERO env action chunk and GR00T model space.

The inversion primitives in groot_inversion operate in the model's padded
action space (B, max_action_horizon, max_action_dim). A scripted expert
produces 7D robosuite-frame actions. These helpers bridge the two:

    encode_chunk_to_model_space   (T, 7) env frame -> (1, 40, 132) model space
    decode_chunk_from_model_space (1, 40, 132) model space -> (16, 7) env frame

LIBERO_PANDA action configs are all absolute, so no state is needed for the
conversion. The only nontrivial part is the gripper convention:
    env    convention: +1 = close, -1 = open
    policy convention:  0 = close,  1 = open
"""
import numpy as np
import torch

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy


ACTION_KEYS = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
EMBODIMENT_VALUE = EmbodimentTag.LIBERO_PANDA.value  # "libero_sim"

# Model-space (padded) action shape. Matches the GR00T N1.7 config
# max_action_horizon / max_action_dim.
MODEL_ACTION_HORIZON = 40
MODEL_ACTION_DIM = 132


def env_gripper_to_policy_gripper(env_gripper):
    """env (+1=close, -1=open) -> policy (0=close, 1=open)."""
    return (1.0 - env_gripper) / 2.0


def policy_gripper_to_env_gripper(policy_gripper):
    """policy [0,1] -> env [-1,+1] (no binarization; continuous round-trip)."""
    return 1.0 - 2.0 * policy_gripper


def encode_chunk_to_model_space(policy: Gr00tPolicy, chunk_env_frame: np.ndarray):
    """(T, 7) env-frame chunk -> (1, 40, 132) torch float32 model space.

    Converts the gripper to policy convention, splits into a per-axis dict,
    normalizes via the policy's state_action_processor, concatenates in
    modality-key order, and pads to (40, 132). Output is float32 on CPU; the
    caller casts to device/dtype.
    """
    T = chunk_env_frame.shape[0]
    chunk = chunk_env_frame.copy().astype(np.float32)
    chunk[:, 6] = env_gripper_to_policy_gripper(chunk[:, 6])

    action_dict = {key: chunk[:, i : i + 1] for i, key in enumerate(ACTION_KEYS)}

    sap = policy.processor.state_action_processor
    normalized = sap.apply_action(action_dict, EMBODIMENT_VALUE, state=None)

    parts = [normalized[key] for key in ACTION_KEYS]
    concat = np.concatenate(parts, axis=-1)  # (T, sum_dims)

    out = np.zeros((MODEL_ACTION_HORIZON, MODEL_ACTION_DIM), dtype=np.float32)
    out[:T, : concat.shape[1]] = concat
    return torch.from_numpy(out).unsqueeze(0)  # (1, 40, 132)


def decode_chunk_from_model_space(policy: Gr00tPolicy, model_action: torch.Tensor,
                                  to_env_gripper: bool = True):
    """(1, 40, 132) torch -> (16, 7) np chunk.

    to_env_gripper=True  invert the encode gripper map (policy [0,1] -> env
        [-1,+1]); use for the inversion round-trip test.
    to_env_gripper=False leave the gripper in policy [0,1] convention, which is
        what decode_action natively returns and what the env-side gripper
        postprocess expects; use in the sampling path.
    """
    arr = model_action.detach().float().cpu().numpy()
    decoded = policy.processor.decode_action(arr, EmbodimentTag.LIBERO_PANDA, state=None)
    # decoded keys are bare modality keys ("x", "y", ...), (1, 16, joint_dim).
    parts = [decoded[k][0] for k in ACTION_KEYS]
    chunk = np.concatenate(parts, axis=-1)  # (16, 7), gripper in policy [0,1]
    if to_env_gripper:
        chunk = chunk.copy()
        chunk[:, 6] = policy_gripper_to_env_gripper(chunk[:, 6])
    return chunk
