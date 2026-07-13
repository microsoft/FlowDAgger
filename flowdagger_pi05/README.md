# flowdagger_pi05

FlowDAgger with a **pi0.5** base policy (openpi / JAX), running on the
**MetaWorld assembly** task.

A small steering actor predicts the per-chunk Gaussian noise that pi0.5 denoises
into an action chunk. When the policy stalls, a scripted MetaWorld expert takes
over; its executed actions are inverted through pi0.5's flow-matching sampler to
recover the noise that would have produced them, and the steering actor is
trained to predict that noise with a behavior-cloning (MSE) loss. The base
policy weights are never updated.


## Install

Python 3.11 and a CUDA 12 GPU are recommended.

```bash
# from the repo root
python -m venv .venv && source .venv/bin/activate

# 1. openpi submodule (pi0.5 model, sampler, transforms)
git submodule update --init flowdagger_pi05/openpi
pip install -e flowdagger_pi05/openpi

# 2. Reproduction JAX/CUDA stack + envs. Installed AFTER openpi.
pip install -r flowdagger_pi05/requirements.txt
```

The pi0.5 MetaWorld checkpoint is fetched automatically from the Hub
([pi05-metaworld](https://huggingface.co/mmurray-ms/pi05-metaworld))
on the first run and cached under `~/.cache/huggingface`. To use a local
checkpoint instead, set `METAWORLD_CHECKPOINT` to a dir containing `params/`
and `assets/`.

## Run

```bash
cd flowdagger_pi05
python train_flowdagger.py --env metaworld --task_key metaworld_assembly --seed 42
```

On a headless machine (no display), set `MUJOCO_GL=egl` so MuJoCo can render the
camera observations off-screen:

```bash
MUJOCO_GL=egl python train_flowdagger.py --env metaworld --task_key metaworld_assembly --seed 42
```

wandb is opt-in: pass `--prefix <name>` to log a run (project `flowdagger` by
default). Without `--prefix` no wandb run is created (a short id is still
generated for the local output dir). Set `--wandb_project` / `--wandb_entity` to
redirect, or `WANDB_MODE=offline` to log locally. Outputs (eval videos,
`eval_results.jsonl`, steering checkpoints) go to `$EXP/<run-name>` if `EXP` is
set, else `~/flowdagger_runs/<run-name>`.
