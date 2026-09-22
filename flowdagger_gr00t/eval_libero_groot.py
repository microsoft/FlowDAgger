#!/usr/bin/env python
"""Evaluate GR00T N1.7 on LIBERO tasks (in-process).

The libero_10 checkpoint is zero-shot on libero_90 tasks. Default focus:
libero_90 task 57 (cream cheese to tray). This script doubles as the obs/action
conversion scaffold reused by train_groot_flowdagger.py (do not reimplement
obs_to_groot / postprocess_gripper there).

Usage:
    python eval_libero_groot.py --episodes 25 --save_video
    python eval_libero_groot.py --suite libero_10 --task_id 2 --episodes 20
"""
import argparse
import json
import math
import os
import pathlib
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
# LIBERO's checked-in init-state tensors predate PyTorch's weights-only default.
# Only use the pinned, trusted LIBERO checkout documented in README.md.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import imageio
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper


# Path to the GR00T-N1.7-LIBERO libero_10 checkpoint. Override with GROOT_CKPT.
DEFAULT_CKPT = os.environ.get(
    "GROOT_CKPT", os.path.expanduser("~/checkpoints/GR00T-N1.7-LIBERO/libero_10")
)

# Per-suite step budgets (match the openpi LIBERO eval conventions).
SUITE_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
SETTLE_STEPS = 10
ACTION_KEYS = ["action.x", "action.y", "action.z", "action.roll", "action.pitch", "action.yaw", "action.gripper"]


def quat2axisangle(quat):
    quat = quat.copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def obs_to_groot(obs, prompt):
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    xyz = obs["robot0_eef_pos"].astype(np.float32)
    rpy = quat2axisangle(obs["robot0_eef_quat"]).astype(np.float32)
    gripper = obs["robot0_gripper_qpos"].astype(np.float32)
    return {
        "video.image": img[None, None, ...].astype(np.uint8),
        "video.wrist_image": wrist[None, None, ...].astype(np.uint8),
        "state.x": np.array([[[xyz[0]]]], dtype=np.float32),
        "state.y": np.array([[[xyz[1]]]], dtype=np.float32),
        "state.z": np.array([[[xyz[2]]]], dtype=np.float32),
        "state.roll": np.array([[[rpy[0]]]], dtype=np.float32),
        "state.pitch": np.array([[[rpy[1]]]], dtype=np.float32),
        "state.yaw": np.array([[[rpy[2]]]], dtype=np.float32),
        "state.gripper": gripper[None, None, :].astype(np.float32),
        "annotation.human.action.task_description": (prompt,),
    }


def action_chunk_to_array(action_dict):
    return np.concatenate([action_dict[k][0] for k in ACTION_KEYS], axis=-1)


def postprocess_gripper(action_seq):
    """LIBERO env wrapper: gripper [0,1]->[-1,1], binarize, invert."""
    a = action_seq.copy()
    a[..., -1] = 2 * a[..., -1] - 1.0
    a[..., -1] = np.sign(a[..., -1])
    a[..., -1] = a[..., -1] * -1.0
    return a


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="libero_90", choices=list(SUITE_MAX_STEPS.keys()))
    parser.add_argument("--task_id", type=int, default=57)
    parser.add_argument("--episodes", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--replan_steps", type=int, default=8)
    parser.add_argument("--ckpt", default=DEFAULT_CKPT)
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--video_dir", default="videos/groot_libero")
    parser.add_argument("--results_path", default="results/groot_libero.json")
    args = parser.parse_args()

    print(f"Loading GR00T N1.7 from {args.ckpt}...")
    policy = Gr00tPolicy(
        embodiment_tag=EmbodimentTag.LIBERO_PANDA,
        model_path=args.ckpt,
        device="cuda",
        strict=False,
    )
    sim_policy = Gr00tSimPolicyWrapper(policy, strict=True)
    print("Loaded.")

    benchmark_dict = benchmark.get_benchmark_dict()
    suite = benchmark_dict[args.suite]()
    task = suite.get_task(args.task_id)
    desc = task.language
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    max_steps = SUITE_MAX_STEPS[args.suite]
    init_states = suite.get_task_init_states(args.task_id)
    label = f"{args.suite}_{args.task_id}_{task.name[:40]}"

    print(f"\n{'='*70}")
    print(f"{args.suite} task_id={args.task_id} ({task.name})")
    print(f"  prompt: {desc}")
    print(f"  max_steps={max_steps}, replan={args.replan_steps}")
    print(f"{'='*70}")

    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    env.seed(args.seed)
    np.random.seed(args.seed)

    if args.save_video:
        task_video_dir = os.path.join(args.video_dir, label)
        os.makedirs(task_video_dir, exist_ok=True)

    per_ep = []
    t_start = time.time()
    for ep in range(args.episodes):
        env.reset()
        init_idx = ep % init_states.shape[0]
        obs = env.set_init_state(init_states[init_idx])
        for _ in range(SETTLE_STEPS):
            obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

        frames = []
        chunk = None
        done = False
        reward = 0.0
        for t in range(max_steps):
            if args.save_video:
                frames.append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
            if t % args.replan_steps == 0:
                action_dict, _ = sim_policy.get_action(obs_to_groot(obs, desc))
                chunk = postprocess_gripper(action_chunk_to_array(action_dict))
            action = chunk[t % args.replan_steps]
            obs, reward, done, _ = env.step(action.tolist())
            if done:
                break

        success = bool(done) and reward > 0
        per_ep.append({"success": success, "steps": t + 1, "init_idx": int(init_idx)})
        tag = "SUCCESS" if success else "FAILURE"
        print(f"  Ep {ep+1}/{args.episodes}: {tag} (steps={t+1})")

        if args.save_video and frames:
            suffix = "success" if success else "failure"
            out_path = os.path.join(task_video_dir, f"ep{ep+1:02d}_{suffix}.mp4")
            imageio.mimwrite(out_path, frames, fps=20)
            print(f"    saved: {out_path}")

    n_succ = sum(int(e["success"]) for e in per_ep)
    sr = n_succ / args.episodes
    elapsed = time.time() - t_start
    result = {
        "checkpoint": "GR00T-N1.7-LIBERO/libero_10",
        "suite": args.suite,
        "task_id": args.task_id,
        "task_name": task.name,
        "prompt": desc,
        "sr": sr,
        "n_success": n_succ,
        "n": args.episodes,
        "episodes": per_ep,
        "elapsed_sec": elapsed,
        "in_distribution": args.suite == "libero_10",
    }
    print(f"\n  -> SR {sr:.0%} ({n_succ}/{args.episodes})  [{elapsed:.0f}s]")
    env.close()

    os.makedirs(os.path.dirname(args.results_path) or ".", exist_ok=True)
    with open(args.results_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results written to {args.results_path}")


if __name__ == "__main__":
    main()
