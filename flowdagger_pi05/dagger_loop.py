"""FlowDAgger training loop: collect episode -> add to buffer -> BC updates -> eval.

Each iteration runs BC updates on the steering policy (noise predictor). The
scripted MetaWorld expert takes over when the intervention handler triggers; its
executed actions are inverted to noise space and become the BC targets.
"""

from collections import deque

import jax
import jax.numpy as jnp
import numpy as np
import cv2
from tqdm import tqdm

from train_utils import (
    collect_traj,
    add_online_data_to_buffer,
    perform_control_eval,
    obs_to_pi_zero_input,
    obs_to_qpos,
    _expand_noise_basis,
    _render_hud,
)


def _debug_replay_inversion(traj, env, agent_dp, variant, intervention_handler):
    """Replay an episode by decoding the inverted w* through agent_dp.infer().

    Visual check that inverted expert actions decode back to the expert
    trajectory through pi0.5.
    """
    full_noise = traj.get('full_noise_list', {})
    actions = traj.get('actions', [])
    intervention_flags = traj.get('intervention_flags', [])
    metaworld_seed = traj.get('initial_metaworld_seed')

    if not full_noise and not actions:
        print('  [Debug replay] No noise data available, skipping')
        return

    n_chunks = len(actions)
    query_freq = variant.query_freq

    executed_chunks = traj.get('executed_actions_per_chunk', [])
    chunk_obs_raw = traj.get('chunk_obs_raw_list', [])

    print(f'  [Debug replay] {n_chunks} chunks, '
          f'{len(full_noise)} in full_noise, '
          f'{len(chunk_obs_raw)} stored obs, '
          f'{len(executed_chunks)} executed chunks, '
          f'intervention_flags={intervention_flags}')

    # MetaWorld V3: deterministic reset via the same seed used in the rollout.
    if metaworld_seed is not None:
        obs = env.reset(seed=int(metaworld_seed))
        print(f'  [Debug replay] reset metaworld with seed={metaworld_seed}')
    else:
        obs = env.reset()

    replay_reward = 0
    for ci in range(n_chunks):
        noise_source = 'full_noise' if ci in full_noise else 'actions'
        w = full_noise[ci] if ci in full_noise else actions[ci]
        w = np.asarray(w)
        noise = _expand_noise_basis(w, action_horizon=agent_dp.action_horizon)

        if ci < len(chunk_obs_raw):
            obs_pi_zero = obs_to_pi_zero_input(chunk_obs_raw[ci], variant)
        else:
            obs_pi_zero = obs_to_pi_zero_input(obs, variant)
        chunk_actions = agent_dp.infer(obs_pi_zero, noise=noise)["actions"]

        expert_chunk = executed_chunks[ci] if ci < len(executed_chunks) else None
        if expert_chunk is not None:
            expert_arr = np.array(expert_chunk)
            n_cmp = min(len(expert_arr), len(chunk_actions))
            adim = min(expert_arr.shape[1], chunk_actions.shape[1])
            mae = np.mean(np.abs(expert_arr[:n_cmp, :adim] - chunk_actions[:n_cmp, :adim]))
            print(f'    chunk {ci}: src={noise_source} noise_shape={w.shape} '
                  f'expert_steps={len(expert_arr)} replay_steps={len(chunk_actions)} '
                  f'adim={adim} MAE={mae:.4f}')
        else:
            print(f'    chunk {ci}: src={noise_source} noise_shape={w.shape} (no expert ref)')

        is_int = ci < len(intervention_flags) and intervention_flags[ci]
        label = "REPLAY-EXPERT" if is_int else "REPLAY-POLICY"

        for t_in_chunk in range(query_freq):
            if t_in_chunk >= len(chunk_actions):
                break
            action_t = chunk_actions[t_in_chunk]
            global_t = ci * query_freq + t_in_chunk

            if variant.get('render', 0):
                frame = _render_hud(obs, global_t, 0, is_int, None, None,
                                    variant, label=label)
                cv2.imshow('FlowDAgger Training', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                key = cv2.waitKey(1) & 0xFF
                if key == ord('s'):
                    print('  [Debug replay] Skipped')
                    return

            # MetaworldPi05Adapter expects a 4D action.
            a4 = np.asarray(action_t, dtype=np.float32).reshape(-1)[:4]
            obs, reward, done, _ = env.step(a4)
            replay_reward = max(replay_reward, reward)
            if reward >= variant.env_max_reward:
                break
        if replay_reward >= variant.env_max_reward:
            break

    tag = 'SUCCESS' if replay_reward >= variant.env_max_reward else 'FAIL'
    print(f'  [Debug replay] {tag} (reward={replay_reward:.1f}, {n_chunks} chunks)')


def flowdagger_training_loop(
    variant,
    steering_policy,
    env,
    eval_env,
    replay_buffer,
    wandb_logger,
    shard_fn=None,
    agent_dp=None,
    intervention_handler=None,
    init_states=None,
    raw_model=None,
    autonomous_buffer=None,
):
    """FlowDAgger training loop.

    steering_policy is the BC noise predictor; agent_dp is the pi0.5 policy that
    decodes noise to actions; raw_model serves precomputed features.
    autonomous_buffer is the optional second buffer for dual-buffer sampling.
    """
    bc_steps_per_episode = variant.get('bc_steps_per_episode', 50)
    bc_batch_size = variant.get('bc_batch_size', 256)

    dual_buffer = variant.get('dual_buffer', 0) and autonomous_buffer is not None
    if dual_buffer:
        print('[dual_buffer] enabled: intervention chunks -> online_replay_buffer, '
              'autonomous chunks -> autonomous_buffer; mixed sampling at BC step.')

    total_env_steps = 0
    i = 0
    episode_count = 0
    seed_expert_episodes = variant.get('seed_expert_episodes', 0)

    # Sliding window of (had_intervention, succeeded) per rollout, for logging
    # expert-recovery quality (wInt).
    rollout_history = deque(maxlen=100)

    # --- Seed buffer with expert-only episodes ---
    if seed_expert_episodes > 0 and intervention_handler is not None:
        print(f'Seeding buffer with {seed_expert_episodes} expert-only episodes...')
        intervention_handler._force_immediate_takeover = True
        seed_p99s = []
        for seed_ep in range(seed_expert_episodes):
            traj = collect_traj(
                variant, steering_policy, env, 0, agent_dp,
                intervention_handler=intervention_handler,
                init_states=init_states,
                raw_model=raw_model,
            )
            had_intervention = sum(traj.get('intervention_flags', [])) > 0
            is_success = bool(traj['is_success'])
            skip_buffer = variant.get('filter_failures', 1) and not is_success
            if skip_buffer:
                print(f'  Seed ep {seed_ep+1}: episode failed, skipping buffer add')
            else:
                add_online_data_to_buffer(variant, traj, replay_buffer,
                                          autonomous_buffer=autonomous_buffer)
            total_env_steps += traj['env_steps']
            episode_count += 1
            rollout_history.append((had_intervention, is_success))
            if traj.get('inversion_w_p99') is not None:
                seed_p99s.append(traj['inversion_w_p99'])
            print(f'  Seed ep {seed_ep+1}/{seed_expert_episodes}: '
                  f'buffer={len(replay_buffer)}, success={traj["is_success"]}')
            if variant.get('debug_replay_inversion', 0) and variant.get('render', 0):
                _debug_replay_inversion(traj, env, agent_dp, variant, intervention_handler)
        intervention_handler._force_immediate_takeover = False
        print(f'Seeded {len(replay_buffer)} transitions from {seed_expert_episodes} episodes')

        # Manifold check: if seed-expert inversions push the actor's noise past
        # its bound, BC plateaus on tail targets and likely won't converge.
        # p99 < action_magnitude means the expert is on-manifold for pi0.5 here.
        if seed_p99s:
            seed_p99_mean = float(np.mean(seed_p99s))
            seed_p99_max = float(np.max(seed_p99s))
            actor_bound = float(variant.get('action_magnitude', 3.0))
            on_manifold = seed_p99_mean < actor_bound
            tag = 'OK' if on_manifold else 'WARN'
            msg = (f'[Manifold check / {tag}] seed-expert inv_w p99: '
                   f'mean={seed_p99_mean:.2f} max={seed_p99_max:.2f} '
                   f'(actor bound={actor_bound:.2f})')
            if not on_manifold:
                msg += (' : expert appears OFF-manifold for pi0.5 on this task. '
                        'BC will plateau on tail targets; consider SFT-pretraining '
                        'pi0.5 on expert demos before FlowDAgger.')
            print(msg)
            wandb_logger.log({
                'inversion/seed_p99_mean': seed_p99_mean,
                'inversion/seed_p99_max': seed_p99_max,
                'inversion/seed_on_manifold': int(on_manifold),
            }, step=0)

    wandb_logger.log({
        'num_online_samples': len(replay_buffer),
        'num_online_trajs': episode_count,
        'env_steps': total_env_steps,
    }, step=0)

    with tqdm(total=variant.max_steps, initial=0, desc='FlowDAgger') as pbar:
        while i <= variant.max_steps:
            # --- Collect episode ---
            traj = collect_traj(
                variant, steering_policy, env, i, agent_dp,
                intervention_handler=intervention_handler,
                init_states=init_states,
                raw_model=raw_model,
            )
            n_interventions = sum(traj.get('intervention_flags', []))
            n_chunks = len(traj.get('intervention_flags', []))
            intervention_rate = n_interventions / max(n_chunks, 1)
            had_intervention = n_interventions > 0
            is_success = bool(traj['is_success'])

            # Skip buffer-add for failed episodes when --filter_failures is on
            # (scripted-expert modes, where "no intervention" is not approval).
            # Use --filter_failures 0 for real human-in-the-loop, where
            # non-intervened chunks ARE implicit approval.
            skip_buffer = variant.get('filter_failures', 1) and not is_success
            if skip_buffer:
                print(f'  [skip-buffer] episode failed, '
                      f'discarding ep {episode_count + 1} from buffer')
            else:
                add_online_data_to_buffer(variant, traj, replay_buffer,
                                          autonomous_buffer=autonomous_buffer)
            total_env_steps += traj['env_steps']
            episode_count += 1
            rollout_history.append((had_intervention, is_success))

            print(f'FlowDAgger step {i} | episode {episode_count} | '
                  f'buffer: {len(replay_buffer)} | env_steps: {total_env_steps} | '
                  f'interventions: {n_interventions}/{n_chunks} ({intervention_rate:.1%})')

            # --- Debug: replay inverted w* in env with live rendering ---
            if (variant.get('debug_replay_inversion', 0)
                    and n_interventions > 0
                    and variant.get('render', 0)):
                _debug_replay_inversion(traj, env, agent_dp, variant, intervention_handler)

            # --- Initial evaluation (before any updates) ---
            if i == 0:
                print('Performing initial evaluation...')
                perform_control_eval(
                    steering_policy, eval_env, i, variant, wandb_logger,
                    agent_dp, init_states=init_states, raw_model=raw_model,
                )

            # --- BC updates ---
            # In dual_buffer mode wait for BOTH buffers to have data so the mix
            # is well-defined.
            min_buf = (min(len(replay_buffer), len(autonomous_buffer))
                       if dual_buffer else len(replay_buffer))
            if min_buf >= max(variant.get('start_online_updates', 1), 1):
                if dual_buffer:
                    auto_frac = float(variant.get('dual_buffer_auto_frac', 0.5))
                    n_aut = int(round(bc_batch_size * auto_frac))
                    n_aut = min(max(n_aut, 1), bc_batch_size - 1)
                    n_int = bc_batch_size - n_aut
                    iter_int = replay_buffer.get_iterator(n_int)
                    iter_aut = autonomous_buffer.get_iterator(n_aut)

                    def _concat_batch(b_int, b_aut):
                        from flax.core import frozen_dict as _fd

                        def merge_leaf(a, b):
                            return jnp.concatenate([a, b], axis=0)
                        flat_a = dict(_fd.unfreeze(b_int))
                        flat_b = dict(_fd.unfreeze(b_aut))
                        out = {}
                        for k in flat_a:
                            if isinstance(flat_a[k], dict):
                                out[k] = {kk: merge_leaf(flat_a[k][kk], flat_b[k][kk])
                                          for kk in flat_a[k]}
                            else:
                                out[k] = merge_leaf(flat_a[k], flat_b[k])
                        return _fd.freeze(out)

                    def _dual_iter():
                        while True:
                            yield _concat_batch(next(iter_int), next(iter_aut))
                    replay_iterator = _dual_iter()
                else:
                    replay_iterator = replay_buffer.get_iterator(bc_batch_size)
                if shard_fn is not None:
                    replay_iterator = map(shard_fn, replay_iterator)

                for _ in range(bc_steps_per_episode):
                    batch = next(replay_iterator)
                    info = steering_policy.update(batch)
                    pbar.update()
                    i += 1

                    if i % variant.log_interval == 0:
                        info_np = {k: float(jax.device_get(v)) for k, v in info.items()}
                        for k, v in info_np.items():
                            wandb_logger.log({f'training/{k}': v}, step=i)
                        # wInt: of recent rollouts where the expert intervened
                        # at least once, what fraction succeeded?
                        wint_t = sum(1 for had, _ in rollout_history if had)
                        wint_s = sum(1 for had, ok in rollout_history if had and ok)
                        wint_rate = wint_s / wint_t if wint_t else float('nan')
                        rollout_succ_rate = (
                            sum(1 for _, ok in rollout_history if ok) / len(rollout_history)
                            if rollout_history else float('nan')
                        )

                        log_dict = {
                            'replay_buffer_size': len(replay_buffer),
                            'autonomous_buffer_size': (
                                len(autonomous_buffer) if dual_buffer else -1
                            ),
                            'episode_return (exploration)': traj['episode_return'],
                            'is_success (exploration)': int(traj['is_success']),
                            'num_online_trajs': episode_count,
                            'num_online_samples': len(replay_buffer),
                            'env_steps': total_env_steps,
                            'intervention_rate': intervention_rate,
                            'rollout/wInt_succ_rate_w100': wint_rate,
                            'rollout/wInt_count_w100': wint_t,
                            'rollout/all_succ_rate_w100': rollout_succ_rate,
                        }
                        if traj.get('inversion_7d_mse') is not None:
                            log_dict['inversion/7d_mse'] = traj['inversion_7d_mse']
                        if traj.get('inversion_w_mean_abs') is not None:
                            log_dict['inversion/inv_w_mean_abs'] = traj['inversion_w_mean_abs']
                        if traj.get('inversion_w_max_abs') is not None:
                            log_dict['inversion/inv_w_max'] = traj['inversion_w_max_abs']
                        if traj.get('inversion_w_p99') is not None:
                            log_dict['inversion/inv_w_p99'] = traj['inversion_w_p99']
                            log_dict['inversion/inv_w_p99_threshold'] = float(
                                variant.get('action_magnitude', 3.0)
                            )
                        wandb_logger.log(log_dict, step=i)

                    if i % variant.eval_interval == 0:
                        if intervention_handler is not None:
                            wandb_logger.log(intervention_handler.get_stats(), step=i)
                        perform_control_eval(
                            steering_policy, eval_env, i, variant, wandb_logger,
                            agent_dp, init_states=init_states, raw_model=raw_model,
                        )

                    if variant.checkpoint_interval != -1 and i % variant.checkpoint_interval == 0:
                        steering_policy.save_checkpoint(
                            variant.outputdir, i, variant.checkpoint_interval
                        )

                    if i > variant.max_steps:
                        break
