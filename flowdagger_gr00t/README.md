# FlowDAgger: GR00T N1.7 backend

This backend runs FlowDAgger on top of NVIDIA's GR00T N1.7 policy (PyTorch,
flow-matching action head) for LIBERO manipulation. The base policy weights stay
frozen; FlowDAgger trains a small MLP noise policy that predicts the initial
noise fed to the flow sampler, supervised by inverting scripted-expert action
chunks back into noise space.

The proven configuration is LIBERO-90 task 57 (cream cheese to tray). The
checkpoint used is the libero_10 release, which is zero-shot on libero_90 tasks,
so task 57 is a genuine out-of-the-box generalization target that FlowDAgger
improves.

## External dependencies

This backend uses two source packages that must be set up separately:

- **gr00t** -- NVIDIA Isaac GR00T (the N1.7 base policy + processor).
  https://github.com/NVIDIA/Isaac-GR00T
- **LIBERO** -- the LIBERO benchmark (envs, BDDL task files, init states).
  https://github.com/Lifelong-Robot-Learning/LIBERO

The compatible robosuite and MuJoCo versions are installed by this backend's
`requirements.txt`.

## Install

1. Clone Isaac-GR00T with its submodules and use its supported Python 3.12
   environment. The revision below is the one used for the reference result:

   ```
   git clone --recurse-submodules https://github.com/NVIDIA/Isaac-GR00T
   cd Isaac-GR00T
   git checkout 9c7e746b2cd37a810070a98ef41d290a07e806c2
   git submodule update --init --recursive
   uv sync --python 3.12
   source .venv/bin/activate
   ```

   Isaac-GR00T also requires FFmpeg 4-7 for `torchcodec`; follow its platform
   instructions if your system package is newer.
2. Clone the tested LIBERO revision:

   ```
   git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
   git -C LIBERO checkout 8f1084e3132a39270c3a13ebe37270a43ece2a01
   ```

3. From `flowdagger_gr00t/`, install the compatible LIBERO runtime, install the
   LIBERO source package without its obsolete dependency pins, and configure
   its asset paths:

   ```
   pip install -r requirements.txt
   pip install --no-deps -e /path/to/LIBERO
   python scripts/patch_robosuite_mujoco3.py
   python scripts/configure_libero.py /path/to/LIBERO
   export LIBERO_CONFIG_PATH=~/.libero
   export PYTHONPATH=/path/to/LIBERO:${PYTHONPATH:-}
   ```

   The compatibility script makes robosuite 1.4.1 accept MuJoCo 3's
   mass-matrix API. It is idempotent and refuses to patch other robosuite
   versions.

4. Request access to the gated
   [`nvidia/Cosmos-Reason2-2B`](https://huggingface.co/nvidia/Cosmos-Reason2-2B)
   backbone used by GR00T N1.7, then authenticate with `hf auth login` or set
   `HF_TOKEN`. The GR00T policy cannot load without that access.

## Checkpoint

Download the GR00T-N1.7-LIBERO libero_10 checkpoint:

```
bash scripts/download_groot_libero_ckpt.sh
```

This pulls the inference files from `nvidia/GR00T-N1.7-LIBERO` (the `libero_10`
subfolder) into
`${GROOT_CKPT:-~/checkpoints/GR00T-N1.7-LIBERO/libero_10}`. Export `GROOT_CKPT`
to point the scripts at it, or pass `--ckpt`.

## Run

Minimal launch (proven task 57 config):

```
export GROOT_CKPT=~/checkpoints/GR00T-N1.7-LIBERO/libero_10
python train_groot_flowdagger.py --task_id 57 --episodes 20 --eval_episodes 25
```

Fuller form with the defaults made explicit:

```
python train_groot_flowdagger.py \
    --suite libero_90 \
    --task_id 57 \
    --episodes 20 \
    --seed_expert_episodes 10 \
    --bc_steps_per_episode 100 \
    --bc_batch_size 64 \
    --lr 1e-4 \
    --intervention_probability 1.0 \
    --expert_position_gain 5 \
    --inverter_fp_per_step 10 \
    --replan_steps 8 \
    --eval_interval 10 \
    --eval_episodes 25
```

Sanity-check the frozen base policy with the eval scaffold:

```
python eval_libero_groot.py --suite libero_90 --task_id 57 --episodes 25
```

The verified recipe uses expert control for every collected chunk
(`--intervention_probability 1.0`). Lower values independently mix learned
policy chunks into online collection.

Reference result on LIBERO-90 task 57: the frozen `libero_10` policy is about
0.6 success rate zero-shot at N=25, and this FlowDAgger recipe reaches roughly
0.84-0.96 over 20 online episodes. In our reproduction, gain 5 reached 0.76 at
episode 10 and 0.96 at episode 20; gain 20 reached only 0.48 at best.

## Files

- `train_groot_flowdagger.py` -- entry point: the FlowDAgger training loop
  (noise policy, buffer, rollout, BC update, eval).
- `eval_libero_groot.py` -- evaluation scaffold + obs/action conversion helpers
  reused by the trainer.
- `groot_inversion.py` -- flow-matching inversion primitives (predict_velocity,
  forward_euler, reverse_euler, perstep_fp_reverse, encode_features).
- `groot_expert_inversion.py` -- encode/decode between a 7D LIBERO env action
  chunk and the model's padded action space, including the gripper convention.
- `scripts/download_groot_libero_ckpt.sh` -- checkpoint downloader.

Shared, framework-agnostic pieces (scripted expert, intervention handler, task
registry) live in the repo-level `shared/` package.
