#! /usr/bin/env python
"""FlowDAgger for MetaWorld with a pi0.5 base policy.

Noise-space DAgger: a small steering actor predicts the per-chunk noise that
pi0.5 denoises into an action chunk. A scripted MetaWorld expert takes over when
the policy stalls; its executed actions are inverted to noise space (the space
the actor predicts) and become the BC targets. See README.md for details.
"""
import argparse
import os
import pathlib
import sys

# Make local modules, the repo-root `shared` package, and the openpi submodule
# importable regardless of the working directory.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                                  # backend-local modules
sys.path.insert(0, os.path.dirname(_HERE))                 # repo root, for `shared`
sys.path.insert(0, os.path.join(_HERE, "openpi", "src"))   # openpi submodule

# Deterministic GPU ops in JAX.
os.environ['XLA_FLAGS'] = os.environ.get('XLA_FLAGS', '') + ' --xla_gpu_deterministic_ops=true'

import jax
import numpy as np

import gymnasium as gym
from gym.spaces import Dict, Box
import tempfile
from functools import partial
import tensorflow as tf
from jax.experimental.compilation_cache import compilation_cache

from openpi.training import config as openpi_config

from buffer import ReplayBuffer
from jax_utils import add_batch_dim
from wandb_logger import WandBLogger, create_exp_name
from launch_util import parse_training_args

from shared.task_configs import get_task_config

from steering_policy import SteeringPolicy
from dagger_loop import flowdagger_training_loop
from train_utils import _PRECOMPUTED_ENCODER_TYPES, _precomputed_feature_dim
from metaworld_pi05_adapter import MetaworldPi05Adapter

home_dir = os.environ['HOME']
compilation_cache.set_cache_dir(os.path.join(home_dir, 'jax_compilation_cache'))


def shard_batch(batch, sharding):
    return jax.tree_util.tree_map(lambda x: jax.device_put(x, sharding), batch)


class DummyEnv(gym.ObservationWrapper):
    """Provides observation/action space shapes for buffer + actor init."""

    def __init__(self, variant):
        self.variant = variant
        encoder_type = variant.train_kwargs.get('encoder_type', 'small')
        if encoder_type in _PRECOMPUTED_ENCODER_TYPES:
            feature_dim = _precomputed_feature_dim(encoder_type, num_cameras=variant.num_cameras)
            self.image_shape = (feature_dim, 1)
        else:
            self.image_shape = (variant.resize_image, variant.resize_image,
                                3 * variant.num_cameras, 1)
        obs_dict = {}
        if encoder_type in _PRECOMPUTED_ENCODER_TYPES:
            obs_dict['pixels'] = Box(low=-1e6, high=1e6, shape=self.image_shape, dtype=np.float32)
        else:
            obs_dict['pixels'] = Box(low=0, high=255, shape=self.image_shape, dtype=np.uint8)
        if variant.add_states:
            state_dim = 4  # hand_xyz + gripper_norm; pi05_metaworld native
            obs_dict['state'] = Box(low=-1.0, high=1.0, shape=(state_dim, 1), dtype=np.float32)
        self.observation_space = Dict(obs_dict)
        noise_basis_k = variant.get('noise_basis_k', 1)
        self.action_space = Box(low=-1, high=1, shape=(noise_basis_k, 32,), dtype=np.float32)


def main(variant):
    import random
    random.seed(variant.seed)
    np.random.seed(variant.seed)
    import torch
    torch.manual_seed(variant.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Pin JAX to the requested GPU.
    device_str = getattr(variant, 'device', 'cuda:0')
    gpu_id = int(device_str.split(':')[-1]) if ':' in device_str else 0
    all_gpus = jax.devices('gpu')
    assert gpu_id < len(all_gpus), f'Requested GPU {gpu_id} but only {len(all_gpus)} available'
    target_device = all_gpus[gpu_id]
    devices = [target_device]
    print(f'Using JAX device: {target_device} (gpu_id={gpu_id})')

    mesh = jax.sharding.Mesh(np.array(devices), axis_names=('batch',))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('batch'))
    shard_fn = partial(shard_batch, sharding=sharding)

    tf.config.set_visible_devices([], "GPU")

    kwargs = variant['train_kwargs']

    # wandb is opt-in: only an explicit --prefix enables it. A missing prefix
    # still gets an auto id for the local output dir, but no wandb run.
    wandb_enabled = bool(variant.prefix)
    if not variant.prefix:
        import uuid
        variant.prefix = str(uuid.uuid4().fields[-1])[:5]

    if variant.suffix:
        expname = create_exp_name(variant.prefix, seed=variant.seed) + f"_{variant.suffix}"
    else:
        expname = create_exp_name(variant.prefix, seed=variant.seed)

    # Output dir: EXP env var if set, else ~/flowdagger_runs.
    exp_root = os.environ.get('EXP', os.path.join(home_dir, 'flowdagger_runs'))
    outputdir = os.path.join(exp_root, expname)
    variant.outputdir = outputdir
    os.makedirs(outputdir, exist_ok=True)
    print('writing to output dir', outputdir)

    # --- Environment setup (MetaWorld) ---
    task_key = variant.get('task_key', '') or 'metaworld_assembly'
    task_cfg = get_task_config(task_key)
    env = MetaworldPi05Adapter(
        env_name=task_cfg["env_id"],
        seed=variant.seed,
        camera_name=task_cfg.get("camera_name", "corner3"),
        resolution=task_cfg.get("resolution", 256),
    )
    eval_env = MetaworldPi05Adapter(
        env_name=task_cfg["env_id"],
        seed=variant.seed + 1000,
        camera_name=task_cfg.get("camera_name", "corner3"),
        resolution=task_cfg.get("resolution", 256),
    )
    init_states = None
    variant.task_description = task_cfg["prompt"]
    variant.env_max_reward = 1
    variant.max_timesteps = variant.get('max_timesteps', 0) or task_cfg.get("max_timesteps", 200)
    variant._metaworld_task_cfg = task_cfg

    # --- WandB ---
    group_name = variant.prefix + '_' + variant.launch_group_id
    wandb_output_dir = tempfile.mkdtemp()
    wandb_entity = getattr(variant, 'wandb_entity', '') or None
    wandb_logger = WandBLogger(
        wandb_enabled, variant, variant.wandb_project,
        experiment_id=expname, output_dir=wandb_output_dir,
        group_name=group_name, team=wandb_entity,
    )

    # --- Dummy env for shapes ---
    variant.num_cameras = kwargs.get('num_cameras', 1)
    dummy_env = DummyEnv(variant)
    sample_obs = add_batch_dim(dummy_env.observation_space.sample())
    sample_action = add_batch_dim(dummy_env.action_space.sample())
    print('sample obs shapes', [(k, v.shape) for k, v in sample_obs.items()])
    print('sample action shape', sample_action.shape)

    # --- Base policy (pi0.5) loading ---
    encoder_type = kwargs.get('encoder_type', 'small')
    inversion_method = variant.get('inversion_method', 'perstep_fp')

    cfg_name = variant.get('openpi_config', '') or task_cfg["openpi_config"]
    config = openpi_config.get_config(cfg_name)
    ckpt = variant.get('openpi_checkpoint', '') or task_cfg.get("openpi_checkpoint", "")
    ckpt_path = pathlib.Path(os.path.expanduser(ckpt)) if ckpt else None
    if ckpt_path is not None and (ckpt_path / "params").exists():
        checkpoint_dir = ckpt_path
    elif task_cfg.get("hf_checkpoint"):
        # Pull the public checkpoint from the HuggingFace Hub on first run
        # (cached under ~/.cache/huggingface). Point openpi_checkpoint at a
        # local checkpoint dir to use your own instead.
        from huggingface_hub import snapshot_download
        print(f"Fetching checkpoint {task_cfg['hf_checkpoint']} from HuggingFace Hub...")
        checkpoint_dir = pathlib.Path(snapshot_download(task_cfg["hf_checkpoint"]))
    else:
        raise FileNotFoundError(
            f"No checkpoint for task {task_key!r}: set --openpi_checkpoint to a "
            f"local checkpoint dir, or add an 'hf_checkpoint' to its task config."
        )

    from openpi.models import model as _model
    from openpi.training import checkpoints as _checkpoints
    from openpi.policies import policy as _policy
    import openpi.transforms as _transforms

    # Free PyTorch's CUDA cache before loading pi0.5.
    try:
        import gc as _gc
        import torch as _torch
        _gc.collect()
        if _torch.cuda.is_available():
            _torch.cuda.empty_cache()
            _torch.cuda.synchronize()
    except Exception as _e:
        print(f"[flowdagger] torch cache release skipped: {_e}")

    # Load pi0.5 params once and share between raw_model (inversion) and
    # agent_dp (sampling); loading twice would double the bfloat16 footprint.
    raw_model = config.model.load(
        _model.restore_params(checkpoint_dir / "params", dtype=jax.numpy.bfloat16)
    )
    data_config = config.data.create(config.assets_dirs, config.model)
    norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)
    agent_dp = _policy.Policy(
        raw_model,
        transforms=[
            _transforms.InjectDefaultPrompt(None),
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            _transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
        ],
        metadata=config.policy_metadata,
    )
    print(f"Loaded pi0.5 policy from {checkpoint_dir}")

    # --- Steering policy ---
    steering_kwargs = {
        'lr': kwargs.pop('actor_lr', 3e-4),
        'hidden_dims': kwargs.get('hidden_dims', (256, 256, 256)),
        'latent_dim': kwargs.get('latent_dim', 256),
        'dropout_rate': kwargs.get('dropout_rate', None) or None,
        'encoder_type': kwargs.get('encoder_type', 'small'),
        'encoder_norm': kwargs.get('encoder_norm', 'group'),
        'color_jitter': False,  # precomputed features do not need augmentation
        'use_spatial_softmax': kwargs.get('use_spatial_softmax', True),
        'softmax_temperature': kwargs.get('softmax_temperature', 1),
        'use_bottleneck': kwargs.get('use_bottleneck', True),
        'action_magnitude': kwargs.get('action_magnitude', 1.0),
        'num_cameras': kwargs.get('num_cameras', 1),
        'output_bound': kwargs.get('output_bound', 'tanh'),
    }
    if encoder_type in _PRECOMPUTED_ENCODER_TYPES:
        steering_kwargs['color_jitter'] = False

    agent = SteeringPolicy(variant.seed, sample_obs, sample_action, **steering_kwargs)

    # --- Replay buffer ---
    buffer_size = variant.max_steps // max(variant.get('bc_steps_per_episode', 50), 1) * 50
    buffer_size = max(buffer_size, 10000)
    replay_buffer = ReplayBuffer(dummy_env.observation_space, dummy_env.action_space, int(buffer_size))
    replay_buffer.seed(variant.seed)

    # Optional second buffer for dual-buffer sampling (intervention vs autonomous).
    autonomous_buffer = None
    if variant.get('dual_buffer', 0):
        autonomous_buffer = ReplayBuffer(
            dummy_env.observation_space, dummy_env.action_space, int(buffer_size)
        )
        autonomous_buffer.seed(variant.seed + 1)

    # --- Intervention setup ---
    intervention_handler = None
    if variant.get('use_interventions', False):
        from shared.intervention_handler import InterventionHandler
        from shared.experts.metaworld_expert import MetaworldScriptedExpert

        mw_cfg = variant._metaworld_task_cfg
        expert = MetaworldScriptedExpert(env_id=mw_cfg["env_id"])
        print(f"Using expert: MetaworldScriptedExpert (env_id={mw_cfg['env_id']})")

        # Inverter + action converter (noise <-> expert-action mapping).
        inverter = None
        action_converter = None
        if inversion_method != 'none':
            from flow_matching_inverter import FlowMatchingInverter
            from action_converter import ActionConverter

            inverter_method = 'euler_reverse' if inversion_method == 'hybrid' else inversion_method
            inverter = FlowMatchingInverter(
                raw_model,
                method=inverter_method,
                refine_steps=variant.get('inversion_refine_steps', 5),
                adam_lr=variant.get('adam_inversion_lr', 0.01),
                regularization_weight=variant.get('inversion_regularization', 0.01),
                seed=variant.seed,
                solver=variant.get('inversion_solver', 'euler'),
                fp_per_step=variant.get('fp_per_step', 5),
            )
            # pi05_metaworld trained on raw deltas: the data config does NOT
            # apply DeltaActions, so delta_mask must be None, otherwise the
            # converter subtracts state from already-delta actions.
            action_converter = ActionConverter(
                norm_stats_actions=norm_stats["actions"],
                norm_stats_state=norm_stats.get("state"),
                pre_norm_scale=float(variant.get('pre_norm_scale', 1.0)),
                action_horizon=agent_dp.action_horizon,
                action_dim_env=4,
                delta_mask=None,
            )
            print(f'[Inversion] method={inversion_method}, '
                  f'pre_norm_scale={action_converter.pre_norm_scale:.1f}')

        intervention_handler = InterventionHandler(
            expert=expert,
            query_freq=variant.query_freq,
            intervention_probability=variant.get('intervention_probability', 1.0),
            reward_takeover_penalty=variant.get('reward_takeover_penalty', 0.0),
            reward_expert_bonus=variant.get('reward_expert_bonus', 0.0),
            inverter=inverter,
            action_converter=action_converter,
            seed=variant.seed,
            intervention_mode=variant.get('intervention_mode', 'beta_decay'),
            disagreement_threshold=variant.get('disagreement_threshold', 0.5),
            beta_start=variant.get('beta_start', 1.0),
            beta_end=variant.get('beta_end', 0.1),
            beta_decay_episodes=variant.get('beta_decay_episodes', 2000),
            takeover_min=variant.get('takeover_min', 5),
            takeover_max=variant.get('takeover_max', 60),
            takeover_max_start=(
                variant.get('takeover_max_start')
                if variant.get('takeover_max_start', -1) >= 0 else None
            ),
            takeover_max_curriculum_episodes=(
                variant.get('takeover_max_curriculum_episodes')
                if variant.get('takeover_max_curriculum_episodes', -1) > 0 else None
            ),
        )
        print(f'[Interventions] Enabled: mode={variant.get("intervention_mode", "beta_decay")}')

    # --- Eval-only mode: restore steering policy and run a single eval ---
    if variant.get('eval_only_ckpt', ''):
        print(f"[Eval-only] Restoring steering policy from {variant.eval_only_ckpt}")
        agent.restore_checkpoint(variant.eval_only_ckpt)
        from train_utils import perform_control_eval
        perform_control_eval(
            agent, eval_env, 1, variant, wandb_logger,
            agent_dp=agent_dp, init_states=init_states, raw_model=raw_model,
        )
        print(f"[Eval-only] Done on task_key={variant.task_key}")
        return

    flowdagger_training_loop(
        variant, agent, env, eval_env, replay_buffer, wandb_logger,
        shard_fn=shard_fn, agent_dp=agent_dp,
        intervention_handler=intervention_handler,
        init_states=init_states,
        raw_model=raw_model,
        autonomous_buffer=autonomous_buffer,
    )


def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default=42, type=int, help='Random seed.')
    parser.add_argument('--launch_group_id', default='', help='group id for wandb runs.')
    parser.add_argument('--eval_episodes', default=25, type=int, help='Eval episodes.')
    parser.add_argument('--env', default='metaworld', help='environment family (metaworld).')
    parser.add_argument('--log_interval', default=1000, type=int, help='Logging interval.')
    parser.add_argument('--eval_interval', default=500, type=int, help='Eval interval.')
    parser.add_argument('--checkpoint_interval', default=-1, type=int, help='Checkpoint interval.')
    parser.add_argument('--max_steps', default=4000, type=int,
                        help='Number of BC steps (100 per episode, so 4000 = 40 episodes).')
    parser.add_argument('--add_states', default=1, type=int, help='Add low-dim state to obs.')
    parser.add_argument('--wandb_project', default='flowdagger', help='wandb project')
    parser.add_argument('--wandb_entity', default='', help='wandb team/entity')
    parser.add_argument('--start_online_updates', default=1, type=int,
                        help='buffer size before starting BC updates')
    parser.add_argument('--prefix', default='', help='prefix for wandb / run dir')
    parser.add_argument('--suffix', default='', help='suffix for wandb / run dir')
    parser.add_argument('--resize_image', default=128, type=int, help='resize size (raw pixels only)')
    parser.add_argument('--query_freq', default=10, type=int, help='query frequency (-1 = action horizon)')
    parser.add_argument('--task_key', default='metaworld_assembly', type=str,
                        help='Task key in shared.task_configs.TASK_CONFIGS')
    parser.add_argument('--openpi_config', default='', type=str,
                        help='Override openpi config name (default from task config)')
    parser.add_argument('--openpi_checkpoint', default='', type=str,
                        help='Override openpi checkpoint path (default from task config / METAWORLD_CHECKPOINT)')
    parser.add_argument('--max_timesteps', default=300, type=int,
                        help='Max env steps per episode (0 = env default)')
    parser.add_argument('--device', default='cuda:0', type=str, help='GPU device for JAX')

    # BC arguments
    parser.add_argument('--bc_lr', default=1e-4, type=float, help='Steering policy learning rate')
    parser.add_argument('--bc_steps_per_episode', default=100, type=int,
                        help='BC gradient steps per collected episode')
    parser.add_argument('--bc_batch_size', default=256, type=int, help='BC batch size')

    # Intervention arguments
    parser.add_argument('--use_interventions', default=1, type=int, help='Enable expert interventions (0/1)')
    parser.add_argument('--intervention_mode', default='beta_decay',
                        choices=['beta_decay', 'disagreement'],
                        help='Takeover trigger: scheduled (beta_decay) or '
                             'policy/expert action divergence (disagreement)')
    parser.add_argument('--intervention_probability', default=1.0, type=float)
    parser.add_argument('--beta_start', default=1.0, type=float,
                        help='Takeover probability at the start of training')
    parser.add_argument('--beta_end', default=1.0, type=float,
                        help='Takeover probability after beta_decay_episodes')
    parser.add_argument('--beta_decay_episodes', default=1, type=int)
    parser.add_argument('--takeover_min', default=0, type=int,
                        help='Earliest step at which the expert can take over')
    parser.add_argument('--takeover_max', default=25, type=int,
                        help='Latest step at which the expert can take over')
    parser.add_argument('--takeover_max_start', default=-1, type=int,
                        help='If >=0, ramp takeover_max from this up to takeover_max')
    parser.add_argument('--takeover_max_curriculum_episodes', default=-1, type=int)
    parser.add_argument('--disagreement_threshold', default=0.5, type=float,
                        help='intervention_mode=disagreement: L2 action gap that '
                             'triggers takeover')

    # Noise inversion arguments
    parser.add_argument('--inversion_method', default='perstep_fp',
                        choices=['none', 'euler_reverse', 'adam', 'hybrid', 'fixed_point', 'perstep_fp'])
    parser.add_argument('--inversion_refine_steps', default=5, type=int)
    parser.add_argument('--pre_norm_scale', default=1.0, type=float,
                        help='Action scaling for inversion. With perstep_fp, scale=1.0 works.')
    parser.add_argument('--inversion_solver', default='euler', type=str, choices=['euler', 'midpoint'])
    parser.add_argument('--adam_inversion_lr', default=0.01, type=float)
    parser.add_argument('--inversion_regularization', default=0.01, type=float)
    parser.add_argument('--hybrid_batch_size', default=8, type=int)
    parser.add_argument('--fp_per_step', default=5, type=int,
                        help='Per-step fixed-point iterations for inversion accuracy')
    parser.add_argument('--inversion_mse_threshold', default=0.001, type=float,
                        help='Drop inversion chunks with MSE above this from the BC buffer')

    # Noise dimensionality
    parser.add_argument('--noise_basis_k', default=10, type=int,
                        help='Noise basis K: 1 = tiled (32-dim), action_horizon = full noise')

    # Eval-only mode
    parser.add_argument('--eval_only_ckpt', default='', type=str,
                        help='If non-empty: skip training, restore steering policy from this dir, run one eval.')

    # Rendering
    parser.add_argument('--render', default=0, type=int, help='Show live cv2 window during training (0/1)')

    # Buffer / curriculum
    parser.add_argument('--seed_expert_episodes', default=10, type=int,
                        help='Expert-only episodes to seed the buffer before training')
    parser.add_argument('--debug_replay_inversion', default=0, type=int,
                        help='After each intervention episode, replay inverted w* with rendering (slow)')
    parser.add_argument('--filter_failures', default=1, type=int,
                        help='If 1, skip buffer-add for failed episodes (scripted-expert mode). '
                             'Set 0 for real human-in-the-loop.')
    parser.add_argument('--filter_autonomous', default=0, type=int,
                        help='If 1, only buffer-add EXPERT (intervened) chunks.')
    parser.add_argument('--dual_buffer', default=0, type=int,
                        help='If 1, keep two parallel success-gated buffers (intervention vs '
                             'autonomous) and mix them at the --dual_buffer_auto_frac ratio.')
    parser.add_argument('--dual_buffer_auto_frac', default=0.5, type=float,
                        help='Fraction of each dual-buffer BC batch from the AUTONOMOUS buffer.')
    parser.add_argument('--reinvert_autonomous', default=0, type=int,
                        help='If 1, also re-invert autonomous chunks (perstep_fp) instead of '
                             'storing the steering actor raw output.')

    # Intervention weighting (used by add_online_data_to_buffer)
    parser.add_argument('--intervention_weight', default=1.0, type=float)
    parser.add_argument('--pre_intervention_weight', default=1.0, type=float)

    # Base policy selection
    parser.add_argument('--base_policy', default='pi05', type=str, choices=['pi05'],
                        help='Base policy type (pi0.5).')

    # Reward scheme (BC ignores rewards; kept for replay-buffer format compatibility)
    parser.add_argument('--reward_scheme', default='0_1', type=str, choices=['neg1_0', '0_1'])
    parser.add_argument('--reward_takeover_penalty', default=0.0, type=float)
    parser.add_argument('--reward_expert_bonus', default=0.0, type=float)
    parser.add_argument('--save_eval_video', default=1, type=int, help='Log eval videos to wandb (0/1)')
    return parser


if __name__ == '__main__':
    parser = build_parser()

    # Train kwargs for the steering policy (PixelMultiplexer + encoder).
    train_args_dict = dict(
        actor_lr=3e-4,
        hidden_dims=(256, 256, 256),
        cnn_features=(32, 32, 32, 32),
        cnn_strides=(2, 1, 1, 1),
        cnn_padding='VALID',
        latent_dim=256,
        dropout_rate=0.0,
        use_bottleneck=True,
        encoder_type='vlm_pi0',
        encoder_norm='group',
        use_spatial_softmax=True,
        softmax_temperature=-1,
        action_magnitude=3.0,
        num_cameras=1,
        # 'tanh' = bounded steering output (+/- action_magnitude);
        # 'linear' = unbounded raw MLP head.
        output_bound='tanh',
    )

    variant, args = parse_training_args(train_args_dict, parser)

    # Map bc_lr to actor_lr in train_kwargs.
    variant['train_kwargs']['actor_lr'] = variant.bc_lr

    # Buffer and batching defaults used by add_online_data_to_buffer.
    variant.setdefault('discount', 0.99)
    variant.setdefault('multi_grad_step', 1)
    variant.setdefault('batch_size', variant.bc_batch_size)

    print(variant)
    main(variant)
    sys.exit()
