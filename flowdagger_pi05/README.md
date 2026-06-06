# flowdagger_pi05

FlowDAgger with a **pi0.5** base policy (openpi / JAX), running on the
**MetaWorld assembly** task.

A small steering actor predicts the per-chunk Gaussian noise that pi0.5 denoises
into an action chunk. When the policy stalls, a scripted MetaWorld expert takes
over; its executed actions are inverted through pi0.5's flow-matching sampler to
recover the noise that would have produced them, and the steering actor is
trained to predict that noise with a behavior-cloning (MSE) loss. The base
policy weights are never updated.

## Layout

```
train_flowdagger.py         entry point (argparse + setup + loop launch)
dagger_loop.py              collect -> buffer -> BC updates -> eval loop
train_utils.py              rollout / buffer / eval helpers, obs preprocessing
steering_policy.py          BC-only noise predictor (PixelMultiplexer + encoder)
flow_matching_inverter.py   inverts pi0.5's flow-matching ODE (action -> noise)
action_converter.py         expert env actions <-> pi0.5 normalized action space
metaworld_pi05_adapter.py   wraps a MetaWorld V3 env in pi0.5's native shape
nets.py                     encoders, PixelMultiplexer, MLP policy heads (flax)
buffer.py                   replay buffer storing (obs, noise) transitions
jax_utils.py                jit action helpers, image augmentations, types
wandb_logger.py             wandb logging helper
launch_util.py              training-arg parsing
openpi/                      git submodule: pi0.5 model + sampler + transforms
```

`shared/` (one level up, a sibling package) provides the scripted MetaWorld
expert, the task registry, and the intervention handler.

## Install

Python 3.11 and a CUDA 12 GPU are recommended.

```bash
# from the repo root
python -m venv .venv && source .venv/bin/activate

# 1. openpi submodule (pi0.5 model, sampler, transforms)
git submodule update --init flowdagger_pi05/openpi
pip install -e flowdagger_pi05/openpi

# 2. JAX stack + envs. Installed AFTER openpi so its pinned versions win
#    (openpi pulls older jax/gymnasium that break the JAX 0.8 / MetaWorld V3 stack).
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

The defaults are the reference recipe (see below), so this command reproduces
the reference result. The first run downloads the checkpoint from the Hub.

Expected on assembly (seed 42, eval N=25): success rate climbs from the pi0.5
base (~0.5) to ~0.9 (peak ~0.96) by the default `--max_steps 2000` (20 online
episodes), with episodes getting shorter as steering improves.

wandb is opt-in: pass `--prefix <name>` to log a run (project `flowdagger` by
default). Without `--prefix` no wandb run is created (a short id is still
generated for the local output dir). Set `--wandb_project` / `--wandb_entity` to
redirect, or `WANDB_MODE=offline` to log locally. Outputs (eval videos,
`eval_results.jsonl`, steering checkpoints) go to `$EXP/<run-name>` if `EXP` is
set, else `~/flowdagger_runs/<run-name>`.

## Recipe (defaults)

The defaults are the configuration that produces the reference result. The
important ones:

- `--action_magnitude 3.0` bounds the steering actor output. It must exceed the
  inverted expert-noise magnitude (p99 ~1.3) or targets get clipped.
- `--noise_basis_k 10` equals the pi0.5 action horizon, so the steering actor
  predicts the full per-step noise directly (the DCT is bypassed when K equals
  the horizon). Lower K compresses the noise onto the first K DCT coefficients
  for smoother, lower-dimensional targets; K=1 tiles a single 32-dim vector.
- `--inversion_method perstep_fp --fp_per_step 5` inverts the exact discrete
  Euler denoising map per step; targets decode back at noise scale 1.0.
- `--encoder_type vlm_pi0` steers on pi0.5's VLM features.
- `--bc_lr 1e-4 --bc_steps_per_episode 100`.
- beta_decay interventions with constant beta=1 (`--beta_start 1 --beta_end 1
  --beta_decay_episodes 1`): the expert takes over at a random step in
  `[--takeover_min 0, --takeover_max 25]` and holds to the end of the episode.
  `--intervention_mode disagreement` instead triggers takeover the first step the
  policy and expert actions diverge by more than `--disagreement_threshold` (a
  state-conditioned alternative to the schedule).
- `--seed_expert_episodes 10` seeds the buffer with expert-only episodes and
  runs a manifold check (is the expert reachable as pi0.5 noise).
- Single replay buffer by default. `--dual_buffer 1 --dual_buffer_auto_frac F`
  instead keeps intervention vs autonomous chunks in separate buffers and mixes
  them at fraction F; single buffer matched or beat dual in our runs.
- `--filter_failures 1` drops failed episodes from the buffer (scripted-expert
  mode); set `0` for human-in-the-loop, where non-intervened chunks are approval.
