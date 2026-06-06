"""Rollout / buffer / eval helpers for the MetaWorld pi0.5 FlowDAgger loop.

Trimmed to the MetaWorld code path:
  collect_traj            roll out one episode (policy + scripted-expert takeover),
                          invert executed expert actions to noise space.
  add_online_data_to_buffer  push (obs, noise) transitions into the replay buffer.
  perform_control_eval    deterministic eval rollouts, logs success rate.
  obs_to_*                observation preprocessing for pi0.5.
  noise-basis helpers     compress / expand the per-chunk noise via a DCT basis.
"""

import logging
import os
import threading

import numpy as np
import wandb
import jax
import PIL
import cv2
from tqdm import tqdm
from openpi_client import image_tools

log = logging.getLogger(__name__)


# Press 's' anywhere (pynput hooks the global keyboard) to mark the current eval
# rollout as FAIL and move on.
_eval_skip_episode = threading.Event()
_eval_skip_listener_started = False


def _start_eval_skip_listener():
    global _eval_skip_listener_started
    if _eval_skip_listener_started:
        return
    try:
        from pynput import keyboard
    except Exception as e:
        print(f"[eval_skip] pynput unavailable, skip-key disabled ({e})")
        _eval_skip_listener_started = True
        return

    def on_press(key):
        try:
            if hasattr(key, "char") and key.char and key.char.lower() == "s":
                if not _eval_skip_episode.is_set():
                    print("[eval_skip] 's' pressed, current eval rollout will be skipped")
                _eval_skip_episode.set()
        except Exception:
            pass

    keyboard.Listener(on_press=on_press, daemon=True).start()
    _eval_skip_listener_started = True
    print("[eval_skip] Press 's' anywhere during eval to abandon the current rollout")


# --- DCT basis for noise compression ---------------------------------------

def _build_poly_basis(K, N=50):
    """Orthonormal DCT-II basis matrix Phi of shape (N, K).

    The first K DCT functions are orthonormal, stable at any K, and yield
    coefficients with natural magnitude (fit the steering actor's tanh range
    without rescaling).
    """
    Phi = np.zeros((N, K), dtype=np.float32)
    for k in range(K):
        for n in range(N):
            Phi[n, k] = np.cos(np.pi * k * (2 * n + 1) / (2 * N))
        if k == 0:
            Phi[:, k] *= np.sqrt(1.0 / N)
        else:
            Phi[:, k] *= np.sqrt(2.0 / N)
    return Phi


_POLY_BASIS_CACHE = {}


def _get_poly_basis(K, N=50):
    key = (K, N)
    if key not in _POLY_BASIS_CACHE:
        _POLY_BASIS_CACHE[key] = _build_poly_basis(K, N)
    return _POLY_BASIS_CACHE[key]


def _expand_noise_basis(coefficients, action_horizon=50):
    """Expand basis coefficients (K, D) or (B, K, D) to a full noise tensor
    (1, action_horizon, D) or (B, action_horizon, D).

    When K == action_horizon (no compression) the coefficients are returned
    directly, skipping the DCT transform.
    """
    if coefficients.ndim == 2:
        K = coefficients.shape[0]
        if K == action_horizon:
            return coefficients[np.newaxis]
        Phi = _get_poly_basis(K, action_horizon)
        noise = Phi @ coefficients
        return noise[np.newaxis]
    elif coefficients.ndim == 3:
        B, K, D = coefficients.shape
        if K == action_horizon:
            return coefficients
        Phi = _get_poly_basis(K, action_horizon)
        noise = np.einsum('nk,bkd->bnd', Phi, coefficients)
        return noise
    else:
        raise ValueError(f"Expected 2D or 3D coefficients, got {coefficients.ndim}D")


def _project_noise_to_basis(noise_full, K):
    """Project full (N, 32) noise to basis coefficients (K, 32).

    When K == N (no compression) the noise is returned directly.
    """
    N = noise_full.shape[0]
    if K == N:
        return noise_full.astype(np.float32)
    Phi = _get_poly_basis(K, N)  # (N, K)
    C, _, _, _ = np.linalg.lstsq(Phi, noise_full, rcond=None)
    return C.astype(np.float32)  # (K, 32)


# --- Live rendering helpers ------------------------------------------------
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_RENDER_SIZE = 512  # each camera panel is scaled to this


def _render_hud(obs, t, training_step, is_intervention, off_nominal_dist,
                _unused_threshold, variant, label=None):
    """Composite the scene image with a status overlay; returns an RGB frame
    (caller converts to BGR for cv2.imshow)."""
    # MetaworldPi05Adapter emits a single right-side-up image at "image".
    agent_img = np.asarray(obs["image"], dtype=np.uint8).copy()
    agent_img = cv2.resize(agent_img, (_RENDER_SIZE, _RENDER_SIZE),
                           interpolation=cv2.INTER_NEAREST)
    frame = agent_img
    total_w = frame.shape[1]

    banner_h = 50
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (total_w, banner_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    if label is not None:
        tag = label
        tag_color = (80, 255, 80)
    elif is_intervention:
        tag = "EXPERT"
        tag_color = (255, 80, 80)
    else:
        tag = "POLICY"
        tag_color = (80, 180, 255)
    cv2.putText(frame, tag, (10, 32), _FONT, 1.0, tag_color, 2)

    dist_text = f"  d={off_nominal_dist:.3f}" if off_nominal_dist is not None else ""
    step_text = f"t={t}  step={training_step}{dist_text}"
    cv2.putText(frame, step_text, (total_w - 450, 32), _FONT, 0.65,
                (255, 255, 255), 1)

    if off_nominal_dist is not None:
        bar_y = frame.shape[0] - 26
        bar_x0 = 16
        bar_max_w = total_w - 32
        cv2.rectangle(frame, (bar_x0, bar_y), (bar_x0 + bar_max_w, bar_y + 16),
                      (60, 60, 60), -1)
        bar_w = int(min(off_nominal_dist / 0.3, 1.0) * bar_max_w)
        if off_nominal_dist < 0.05:
            bar_color = (80, 220, 80)
        elif off_nominal_dist < 0.15:
            bar_color = (220, 200, 60)
        else:
            bar_color = (255, 80, 80)
        cv2.rectangle(frame, (bar_x0, bar_y), (bar_x0 + bar_w, bar_y + 16),
                      bar_color, -1)
        cv2.putText(frame, f"d={off_nominal_dist:.3f}", (bar_x0, bar_y - 6),
                    _FONT, 0.5, (200, 200, 200), 1)

    return frame


# --- Observation preprocessing (MetaWorld pi0.5) ---------------------------

def obs_to_img(obs, variant):
    """Convert raw observation to a (possibly resized) image for the actor."""
    # MetaworldPi05Adapter emits a single right-side-up image at "image".
    curr_image = obs["image"]
    if variant.resize_image > 0:
        curr_image = np.array(
            PIL.Image.fromarray(curr_image).resize(
                (variant.resize_image, variant.resize_image)
            )
        )
    return curr_image


def obs_to_pi_zero_input(obs, variant):
    """Build the pi0.5 input dict from a MetaWorld observation.

    pi05_metaworld uses a single 224x224 image, 4D state, no wrist cam. The
    openpi input transform zero-pads the wrist slots, so only the scene image
    and state are provided.
    """
    img = np.ascontiguousarray(obs["image"])
    img = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(img, 224, 224)
    )
    obs_pi_zero = {
        "observation/image": img,
        "observation/state": np.asarray(obs["state"], dtype=np.float32),
        "prompt": str(variant.task_description),
    }
    return obs_pi_zero


def obs_to_qpos(obs, variant):
    """Robot state used by the ActionConverter (4D native for MetaWorld)."""
    return np.asarray(obs["state"], dtype=np.float32)


# Encoder types that consume precomputed pi0.5 features (stored in the replay
# buffer as float vectors) rather than raw pixels.
_SIGLIP_FEATURE_DIM = 2048   # SigLIP patches projected to Gemma width, mean-pooled
_VLM_FEATURE_DIM = 2048      # Gemma prefix hidden states, mean-pooled
_PRECOMPUTED_ENCODER_TYPES = {'siglip_pi0', 'vlm_pi0'}


def _extract_siglip_features(raw_model, obs_pi_zero, agent_dp):
    """Mean-pooled SigLIP features from pi0.5's vision encoder.

    Returns (2048 * num_cameras,): each camera's patches mean-pooled
    independently then concatenated. Uses the loaded pi0.5 model, no extra GPU
    memory.
    """
    import jax.numpy as jnp

    features_list = []
    for img_key in ["observation/image", "observation/wrist_image"]:
        if img_key not in obs_pi_zero:
            continue
        image = obs_pi_zero[img_key]  # (224, 224, 3) uint8
        image_jax = jnp.asarray(image, dtype=jnp.float32)[jnp.newaxis]
        image_tokens, _ = raw_model.PaliGemma.img(image_jax, train=False)
        pooled = jax.lax.stop_gradient(image_tokens.mean(axis=1))  # (1, 2048)
        features_list.append(np.asarray(pooled[0]))
    return np.concatenate(features_list, axis=0)


def _extract_vlm_features(agent_dp, obs_pi_zero):
    """Mean-pooled VLM features from pi0.5's Gemma prefix (images + language)."""
    import jax.numpy as jnp
    hidden_state, _ = agent_dp.get_prefix_rep(obs_pi_zero)
    pooled = jax.lax.stop_gradient(hidden_state.mean(axis=1))  # (1, 2048)
    return np.asarray(pooled[0])


def _extract_precomputed_features(encoder_type, raw_model, agent_dp, obs_pi_zero,
                                  obs=None, variant=None):
    if encoder_type == 'siglip_pi0':
        return _extract_siglip_features(raw_model, obs_pi_zero, agent_dp)
    elif encoder_type == 'vlm_pi0':
        return _extract_vlm_features(agent_dp, obs_pi_zero)
    else:
        raise ValueError(f'Unknown precomputed encoder type: {encoder_type}')


def _precomputed_feature_dim(encoder_type, num_cameras=1):
    if encoder_type == 'siglip_pi0':
        # _extract_siglip_features iterates over scene + wrist keys and skips
        # whichever is absent; with num_cameras=1 only the scene is fed.
        return _SIGLIP_FEATURE_DIM * num_cameras
    elif encoder_type == 'vlm_pi0':
        return _VLM_FEATURE_DIM  # already fused
    else:
        raise ValueError(f'Unknown precomputed encoder type: {encoder_type}')


# --- Buffer insertion -------------------------------------------------------

def add_online_data_to_buffer(variant, traj, online_replay_buffer,
                              expert_buffer=None, autonomous_buffer=None):
    discount_horizon = variant.query_freq
    actions = np.array(traj['actions'])  # (T, chunk_size, action_dim)
    episode_len = len(actions)
    rewards = np.array(traj['rewards'])
    masks = np.array(traj['masks'])
    intervention_flags = traj.get('intervention_flags', [])

    intervention_weight = variant.get('intervention_weight', 1.0)
    pre_intervention_weight = variant.get('pre_intervention_weight', 1.0)
    # dual_buffer: route intervention chunks to online_replay_buffer and
    # autonomous chunks to autonomous_buffer; the BC step samples a fixed
    # fraction from each.
    dual_buffer = variant.get('dual_buffer', 0) and autonomous_buffer is not None

    # Per-step weights from intervention flags.
    weights = np.ones(episode_len, dtype=np.float32)
    for t in range(episode_len):
        if t < len(intervention_flags):
            is_int = intervention_flags[t]
            if is_int:
                weights[t] = intervention_weight
            # Pre-intervention: last policy step before a takeover.
            is_next_int = (t + 1 < len(intervention_flags) and intervention_flags[t + 1])
            is_new_next = is_next_int and not is_int
            if is_new_next:
                weights[t] = pre_intervention_weight

    has_autonomous_insertions = False
    filter_autonomous = variant.get('filter_autonomous', 0)
    skipped_autonomous = 0
    n_to_intervention, n_to_autonomous = 0, 0
    for t in range(episode_len):
        # filter_autonomous: keep only expert (intervened) chunks, dropping
        # policy-autonomous ones. Mutually exclusive with dual_buffer.
        if (not dual_buffer and filter_autonomous and t < len(intervention_flags)
                and not intervention_flags[t]):
            skipped_autonomous += 1
            continue
        obs = traj['observations'][t]
        next_obs = traj['observations'][t + 1]
        obs = {k: v[0] for k, v in obs.items()}
        next_obs = {k: v[0] for k, v in next_obs.items()}
        if not variant.add_states:
            obs.pop('state', None)
            next_obs.pop('state', None)

        insert_dict = dict(
            observations=obs,
            next_observations=next_obs,
            actions=actions[t],
            next_actions=actions[t + 1] if t < episode_len - 1 else actions[t],
            rewards=rewards[t],
            masks=masks[t],
            discount=variant.discount ** discount_horizon,
        )

        if dual_buffer:
            is_int = t < len(intervention_flags) and intervention_flags[t]
            if is_int:
                online_replay_buffer.insert(insert_dict, weight=weights[t])
                n_to_intervention += 1
            else:
                autonomous_buffer.insert(insert_dict, weight=weights[t])
                n_to_autonomous += 1
                has_autonomous_insertions = True
        else:
            online_replay_buffer.insert(insert_dict, weight=weights[t])

    online_replay_buffer.increment_traj_counter()
    if filter_autonomous and skipped_autonomous > 0:
        print(f'  [filter_autonomous] dropped {skipped_autonomous} policy-chunk '
              f'transitions; added {episode_len - skipped_autonomous} expert-chunk transitions')
    if dual_buffer:
        print(f'  [dual_buffer] +{n_to_intervention} intervention chunks -> '
              f'online_replay_buffer; +{n_to_autonomous} autonomous chunks -> autonomous_buffer')
        if has_autonomous_insertions:
            autonomous_buffer.increment_traj_counter()


def _denoise_to_env_actions(noise, obs_pi_zero, agent_dp, variant, obs,
                            inverter=None, action_converter=None):
    """Denoise a noise vector through pi0.5 and convert to env-space actions.

    With an ActionConverter, uses the direct denoise + converter path;
    otherwise falls back to agent_dp.infer() (applies the openpi output
    transform).
    """
    if action_converter is not None and inverter is not None:
        import jax.numpy as jnp
        from openpi.models import model as _model
        inputs = agent_dp._input_transform(dict(obs_pi_zero))
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[jnp.newaxis, ...], inputs)
        pi0_obs = _model.Observation.from_dict(inputs)
        noise_jax = jnp.asarray(noise)
        recon_internal = np.asarray(
            inverter._denoise_jit(inverter._state, pi0_obs, noise_jax)[0]
        )
        state = obs_to_qpos(obs, variant)
        return action_converter.internal_to_env(recon_internal, state)
    else:
        return agent_dp.infer(obs_pi_zero, noise=noise)["actions"]


# --- Rollout ---------------------------------------------------------------

def collect_traj(variant, agent, env, i, agent_dp=None, intervention_handler=None,
                 init_states=None, raw_model=None):
    query_frequency = variant.query_freq
    if query_frequency is None or query_frequency <= 0:
        query_frequency = agent_dp.action_horizon if agent_dp is not None else 1
    max_timesteps = variant.max_timesteps
    env_max_reward = variant.env_max_reward
    inversion_7d_mse = None
    inversion_w_mean_abs = None
    inversion_w_max_abs = None
    inversion_w_p99 = None

    agent._rng, rng, reset_rng = jax.random.split(agent._rng, 3)

    reset_seed = int(jax.random.randint(reset_rng, (), 0, 2**30))
    obs = env.reset(seed=reset_seed)

    # MetaWorld V3 does not expose MjSim; reproducible reset is via the seed.
    initial_metaworld_seed = int(getattr(env, '_seed', reset_seed))

    if intervention_handler is not None:
        intervention_handler.on_episode_reset(env)

    image_list = []
    rewards = []
    action_list = []
    obs_list = []
    intervention_flags = []          # per query-step: was this chunk intervened?
    executed_actions_per_chunk = []  # env-space actions per chunk (for inversion)
    chunk_obs_raw_list = []          # raw env obs at each query point
    chunk_state_list = []            # robot state at each query point
    current_chunk_executed = []      # accumulates actions for current chunk
    render = variant.get('render', 0)
    current_is_intervention = False
    current_off_nominal_dist = None

    # Deferred inversion: collect expert actions during chunk execution, then
    # invert the full chunk retroactively at the next query point.
    pending_inv_chunk_actions = []
    pending_inv_obs = None
    pending_inv_state = None
    pending_inv_idx = None

    def _finalize_pending_inversion():
        """Invert the collected expert action chunk and update action_list."""
        nonlocal pending_inv_chunk_actions, pending_inv_obs, pending_inv_state, pending_inv_idx
        if (pending_inv_idx is not None
                and pending_inv_obs is not None
                and len(pending_inv_chunk_actions) > 0
                and intervention_handler is not None
                and intervention_handler.inverter is not None
                and intervention_handler.action_converter is not None
                # perstep_fp inverts everything in one batched pass at episode end.
                and getattr(intervention_handler.inverter, 'method', '') != 'perstep_fp'):
            import jax.numpy as jnp
            expert_chunk = np.array(pending_inv_chunk_actions)
            target_internal = intervention_handler.action_converter.expert_to_internal(
                expert_chunk, pending_inv_state
            )
            target_jax = jnp.asarray(target_internal)[jnp.newaxis, ...]

            try:
                w_star, error = intervention_handler.inverter.invert(
                    pending_inv_obs, target_jax
                )
                inversion_error = float(error.mean())
                intervention_handler._inversion_errors.append(inversion_error)
                w_star_np = np.asarray(w_star[0])
                chunk_len = agent.action_chunk_shape[0]
                chunk_dim = (agent.action_chunk_shape[1]
                             if len(agent.action_chunk_shape) > 1
                             else w_star_np.shape[-1])
                coeffs = _project_noise_to_basis(w_star_np[..., :chunk_dim], chunk_len)
                action_list[pending_inv_idx] = coeffs
                print(f'  [Inversion] chunk {pending_inv_idx}: '
                      f'{len(expert_chunk)} expert actions, err={inversion_error:.6f}')
            except Exception as e:
                log.warning(f"Deferred noise inversion failed: {e}")

        pending_inv_chunk_actions = []
        pending_inv_obs = None
        pending_inv_state = None
        pending_inv_idx = None

    encoder_type = variant.train_kwargs.get('encoder_type', 'small')
    use_precomputed = (encoder_type in _PRECOMPUTED_ENCODER_TYPES)

    for t in tqdm(range(max_timesteps)):
        curr_image = obs_to_img(obs, variant)
        qpos = obs_to_qpos(obs, variant)

        if use_precomputed:
            obs_dict = {}  # replaced with features at query points
        elif variant.add_states:
            obs_dict = {
                'pixels': curr_image[np.newaxis, ..., np.newaxis],
                'state': qpos[np.newaxis, ..., np.newaxis],
            }
        else:
            obs_dict = {'pixels': curr_image[np.newaxis, ..., np.newaxis]}

        if t % query_frequency == 0:
            # Retroactively mark previous chunk as intervention if the expert
            # took over mid-chunk (flag was False at the query point).
            if (intervention_handler is not None
                    and intervention_handler._intervening
                    and intervention_flags
                    and not intervention_flags[-1]):
                intervention_flags[-1] = True
                print(f'  [Intervention] retroactive flag for chunk {len(intervention_flags)-1}')

            if current_chunk_executed:
                executed_actions_per_chunk.append(list(current_chunk_executed))
            current_chunk_executed = []
            chunk_obs_raw_list.append(
                {k: v.copy() if hasattr(v, 'copy') else v for k, v in obs.items()}
            )
            chunk_state_list.append(qpos.copy())

            # Finalize inversion from the previous intervention chunk (if any).
            _finalize_pending_inversion()

            assert agent_dp is not None
            rng, key = jax.random.split(rng)
            obs_pi_zero = obs_to_pi_zero_input(obs, variant)

            if use_precomputed:
                precomputed = _extract_precomputed_features(
                    encoder_type, raw_model, agent_dp, obs_pi_zero, obs=obs, variant=variant
                )
                obs_dict = {'pixels': precomputed[np.newaxis, ..., np.newaxis]}
                if variant.add_states:
                    obs_dict['state'] = qpos[np.newaxis, ..., np.newaxis]

            # Steering actor predicts noise coefficients; pi0.5 decodes them to
            # an action chunk.
            if i == 0:
                actions_noise = jax.random.normal(key, agent.action_chunk_shape)
            else:
                actions_noise = agent.sample_actions(obs_dict)
                actions_noise = np.reshape(actions_noise, agent.action_chunk_shape)
            noise = _expand_noise_basis(actions_noise, action_horizon=agent_dp.action_horizon)
            actions = agent_dp.infer(obs_pi_zero, noise=noise)["actions"]

            # Intervention check at the query point.
            is_intervention = False
            if intervention_handler is not None:
                is_intervention = intervention_handler._intervening
                intervention_handler._query_step_idx += 1
                if is_intervention:
                    print(f'  [Intervention] t={t}')
                    # Prepare pi0.5 observation for deferred inversion.
                    if (intervention_handler.inverter is not None
                            and intervention_handler.action_converter is not None):
                        import jax.numpy as jnp
                        from openpi.models import model as _model
                        inv_inputs = jax.tree.map(lambda x: x, obs_pi_zero)
                        inv_inputs = agent_dp._input_transform(inv_inputs)
                        inv_inputs = jax.tree.map(
                            lambda x: jnp.asarray(x)[jnp.newaxis, ...], inv_inputs
                        )
                        pending_inv_obs = _model.Observation.from_dict(inv_inputs)
                        pending_inv_state = qpos.copy()
                        pending_inv_idx = len(action_list)
                        pending_inv_chunk_actions = []

            action_list.append(actions_noise)
            obs_list.append(obs_dict)
            intervention_flags.append(is_intervention)
            current_is_intervention = is_intervention

        # --- Live rendering ---
        if render:
            frame = _render_hud(obs, t, i, current_is_intervention,
                                current_off_nominal_dist, None, variant)
            cv2.imshow('FlowDAgger Training', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(1) & 0xFF
            if key == ord('p'):
                print('[Paused] Press p again to resume...')
                while True:
                    key2 = cv2.waitKey(100) & 0xFF
                    if key2 == ord('p'):
                        print('[Resumed]')
                        break

        # During expert intervention, query the expert every env step for
        # responsive P-control and collect actions for deferred inversion.
        _pi0_action_t = actions[t % query_frequency]
        if current_is_intervention and intervention_handler is not None:
            action_t = intervention_handler.expert.act(env)
        else:
            action_t = _pi0_action_t
        if pending_inv_idx is not None:
            pending_inv_chunk_actions.append(action_t.copy())
        current_chunk_executed.append(np.array(action_t, dtype=np.float32).copy())

        # MetaworldPi05Adapter expects a 4D action.
        a4 = np.asarray(action_t, dtype=np.float32).reshape(-1)[:4]
        obs, reward, done, _ = env.step(a4)

        # Progress check every env step (updates intervention state).
        if intervention_handler is not None:
            _, current_off_nominal_dist = intervention_handler.check_progress(
                env, policy_action=_pi0_action_t)
            intervention_handler.step()
            current_is_intervention = intervention_handler._intervening

        rewards.append(reward)
        image_list.append(curr_image)
        if done:
            break
        if reward >= env_max_reward:
            break

    # Finalize any remaining pending inversion from the last chunk.
    deferred_inv_chunks = []
    if pending_inv_idx is not None and len(pending_inv_chunk_actions) > 0:
        if pending_inv_obs is not None:
            deferred_inv_chunks.append({
                'obs': pending_inv_obs,
                'state': pending_inv_state,
                'actions': list(pending_inv_chunk_actions),
                'idx': pending_inv_idx,
            })
        pending_inv_chunk_actions = []
        pending_inv_obs = None
        pending_inv_state = None
        pending_inv_idx = None

    _use_batched_perstep_global = (
        intervention_handler is not None
        and hasattr(getattr(intervention_handler, 'inverter', None), '_perstep_fp_jit')
        and getattr(getattr(intervention_handler, 'inverter', None), 'method', '') == 'perstep_fp'
        and len(executed_actions_per_chunk) > 0
    )
    if (deferred_inv_chunks
            and intervention_handler is not None
            and intervention_handler.inverter is not None
            and intervention_handler.action_converter is not None
            and not _use_batched_perstep_global):
        # Non-perstep methods: invert the stashed chunk inline.
        import jax.numpy as jnp
        for chunk_data in deferred_inv_chunks:
            expert_chunk = np.array(chunk_data['actions'])
            target_internal = intervention_handler.action_converter.expert_to_internal(
                expert_chunk, chunk_data['state']
            )
            target_jax = jnp.asarray(target_internal)[jnp.newaxis, ...]
            try:
                w_star, error = intervention_handler.inverter.invert(
                    chunk_data['obs'], target_jax
                )
                inversion_error = float(error.mean())
                intervention_handler._inversion_errors.append(inversion_error)
                w_star_np = np.asarray(w_star[0])
                chunk_len = agent.action_chunk_shape[0]
                chunk_dim = (agent.action_chunk_shape[1]
                             if len(agent.action_chunk_shape) > 1
                             else w_star_np.shape[-1])
                coeffs = _project_noise_to_basis(w_star_np[..., :chunk_dim], chunk_len)
                action_list[chunk_data['idx']] = coeffs
            except Exception as e:
                log.warning(f"Deferred noise inversion failed: {e}")
        deferred_inv_chunks = []

    # Add last observation.
    curr_image = obs_to_img(obs, variant)
    qpos = obs_to_qpos(obs, variant)
    if use_precomputed:
        obs_pi_zero_last = obs_to_pi_zero_input(obs, variant)
        precomputed = _extract_precomputed_features(
            encoder_type, raw_model, agent_dp, obs_pi_zero_last, obs=obs, variant=variant
        )
        obs_dict = {
            'pixels': precomputed[np.newaxis, ..., np.newaxis],
            'state': qpos[np.newaxis, ..., np.newaxis],
        }
    else:
        obs_dict = {
            'pixels': curr_image[np.newaxis, ..., np.newaxis],
            'state': qpos[np.newaxis, ..., np.newaxis],
        }
    obs_list.append(obs_dict)
    image_list.append(curr_image)

    if current_chunk_executed:
        executed_actions_per_chunk.append(list(current_chunk_executed))

    full_noise_list = {}  # full noise per chunk (filled by per-step FP)

    # --- Per-step FP inversion: re-invert intervention chunks (batched) ---
    # Per-step FP inverts the exact discrete Euler denoising map, so w* decodes
    # correctly through agent_dp.infer at scale 1.0.
    if (intervention_handler is not None
            and intervention_handler.inverter is not None
            and intervention_handler.action_converter is not None
            and getattr(intervention_handler.inverter, 'method', '') == 'perstep_fp'
            and len(executed_actions_per_chunk) > 0):
        import jax.numpy as jnp
        import time as _time
        from openpi.models import model as _model

        n_chunks = len(action_list)
        n_executed = len(executed_actions_per_chunk)
        if n_executed == n_chunks:
            inv = intervention_handler.inverter
            conv = intervention_handler.action_converter

            # Phase 1: build observations, targets, and deferred features.
            t_prep = _time.time()
            obs_batch_list = []
            target_batch_list = []
            valid_indices = []
            n_features_computed = 0
            for ci in range(n_chunks):
                executed = executed_actions_per_chunk[ci]
                if not executed:
                    continue

                if use_precomputed and isinstance(obs_list[ci], tuple):
                    obs_pi_zero_feat = obs_to_pi_zero_input(chunk_obs_raw_list[ci], variant)
                    precomputed = _extract_precomputed_features(
                        encoder_type, raw_model, agent_dp, obs_pi_zero_feat,
                        obs=chunk_obs_raw_list[ci], variant=variant
                    )
                    obs_entry = {'pixels': precomputed[np.newaxis, ..., np.newaxis]}
                    if variant.add_states:
                        obs_entry['state'] = chunk_state_list[ci][np.newaxis, ..., np.newaxis]
                    obs_list[ci] = obs_entry
                    n_features_computed += 1

                # Only invert intervention chunks by default. Autonomous chunks
                # keep the actor's predicted coefficients; re-inverting them
                # risks corruption through the lossy round-trip.
                # --reinvert_autonomous=1 re-inverts autonomous chunks too.
                is_intervention_chunk = (ci < len(intervention_flags)
                                         and intervention_flags[ci])
                reinvert_auto = bool(variant.get('reinvert_autonomous', 0))
                if not is_intervention_chunk and not reinvert_auto:
                    continue

                obs_pi_zero = obs_to_pi_zero_input(chunk_obs_raw_list[ci], variant)
                inv_inputs = jax.tree.map(lambda x: x, obs_pi_zero)
                inv_inputs = agent_dp._input_transform(inv_inputs)
                inv_inputs = jax.tree.map(
                    lambda x: jnp.asarray(x)[jnp.newaxis, ...], inv_inputs
                )
                obs_batch_list.append(_model.Observation.from_dict(inv_inputs))

                executed_arr = np.array(executed)
                target = conv.expert_to_internal(executed_arr, chunk_state_list[ci])
                target_batch_list.append(target)
                valid_indices.append(ci)
            t_prep_done = _time.time()

            if obs_batch_list:
                # Phase 2+3: invert in mini-batches to bound peak GPU memory.
                t_stack = _time.time()
                INV_BATCH_SIZE = 4
                all_w_list = []
                for mb_start in range(0, len(obs_batch_list), INV_BATCH_SIZE):
                    mb_end = min(mb_start + INV_BATCH_SIZE, len(obs_batch_list))
                    mb_obs = jax.tree.map(
                        lambda *xs: jnp.concatenate(xs, axis=0),
                        *obs_batch_list[mb_start:mb_end]
                    )
                    mb_targets = jnp.stack(
                        [jnp.asarray(t) for t in target_batch_list[mb_start:mb_end]], axis=0
                    )
                    mb_w = inv._perstep_fp_jit(inv._state, mb_obs, mb_targets)
                    mb_w.block_until_ready()
                    all_w_list.append(np.asarray(mb_w))
                t_stack_done = _time.time()

                t_inv = _time.time()
                batched_w = np.concatenate(all_w_list, axis=0)
                t_inv_done = _time.time()

                # Phase 4: per-chunk inversion-quality eval (MSE on env dims).
                t_eval = _time.time()
                per_chunk_mse = np.empty(len(valid_indices), dtype=np.float32)
                for mb_start in range(0, len(obs_batch_list), INV_BATCH_SIZE):
                    mb_end = min(mb_start + INV_BATCH_SIZE, len(obs_batch_list))
                    mb_obs = jax.tree.map(
                        lambda *xs: jnp.concatenate(xs, axis=0),
                        *obs_batch_list[mb_start:mb_end]
                    )
                    mb_w = jnp.asarray(batched_w[mb_start:mb_end])
                    mb_targets = jnp.stack(
                        [jnp.asarray(t) for t in target_batch_list[mb_start:mb_end]], axis=0
                    )
                    mb_recon = inv._denoise_jit(inv._state, mb_obs, mb_w)
                    mb_mse = np.asarray(jnp.mean(
                        jnp.square(mb_recon[..., :7] - mb_targets[..., :7]),
                        axis=(-2, -1),
                    ))
                    per_chunk_mse[mb_start:mb_end] = mb_mse
                t_eval_done = _time.time()
                err_7d = float(np.mean(per_chunk_mse))

                # Drop high-MSE chunks from the BC buffer: their inverted w* is
                # unreliable and would train the actor on noise that does not
                # decode to the expert action.
                mse_threshold = float(variant.get('inversion_mse_threshold', 0.001))
                excluded_chunks = set()

                K = agent.action_chunk_shape[0]
                for bi, ci in enumerate(valid_indices):
                    if per_chunk_mse[bi] > mse_threshold:
                        excluded_chunks.add(int(ci))
                        continue
                    w_np = np.asarray(batched_w[bi])
                    coeffs = _project_noise_to_basis(w_np, K)
                    action_list[ci] = coeffs
                    full_noise_list[ci] = w_np

                inversion_7d_mse = err_7d
                w_abs = np.abs(np.asarray(batched_w))
                inversion_w_mean_abs = float(np.mean(w_abs))
                inversion_w_max_abs = float(np.max(w_abs))
                inversion_w_p99 = float(np.percentile(w_abs, 99))
                n_kept = len(valid_indices) - len(excluded_chunks)
                print(f'  [Per-step FP] {len(valid_indices)} chunks | '
                      f'prep={t_prep_done-t_prep:.1f}s '
                      f'(features={n_features_computed}) | '
                      f'stack={t_stack_done-t_stack:.1f}s | '
                      f'invert={t_inv_done-t_inv:.1f}s | '
                      f'eval={t_eval_done-t_eval:.1f}s | '
                      f'7D MSE={err_7d:.6f} | '
                      f'inv_w_mean_abs={inversion_w_mean_abs:.3f} '
                      f'max={inversion_w_max_abs:.2f} p99={inversion_w_p99:.2f} | '
                      f'kept={n_kept}/{len(valid_indices)} '
                      f'(thresh={mse_threshold:.4f}, max_chunk_mse={float(per_chunk_mse.max()):.6f})')

                del obs_batch_list, target_batch_list, all_w_list, batched_w
                del per_chunk_mse
                import gc
                gc.collect()
        else:
            log.warning(f'Per-step FP skipped: {n_executed} executed chunks '
                        f'!= {n_chunks} action_list entries')

    # --- Episode summary + reward shaping ---
    rewards = np.array(rewards)
    episode_return = np.sum(rewards[rewards != None])
    is_success = (reward == env_max_reward)

    n_interventions = sum(intervention_flags)
    n_chunks = len(intervention_flags)
    int_str = f', interventions: {n_interventions}/{n_chunks}' if intervention_handler else ''
    print(f'Rollout Done: {episode_return=}, Success: {is_success}{int_str}')

    query_steps = len(action_list)
    reward_scheme = variant.get('reward_scheme', '0_1')
    if reward_scheme == '0_1':
        if is_success:
            rewards = np.concatenate([np.zeros(query_steps - 1), [1.0]])
            masks = np.concatenate([np.ones(query_steps - 1), [0]])
        else:
            rewards = np.zeros(query_steps)
            masks = np.ones(query_steps)
    else:  # neg1_0
        if is_success:
            rewards = np.concatenate([-np.ones(query_steps - 1), [0]])
            masks = np.concatenate([np.ones(query_steps - 1), [0]])
        else:
            rewards = -np.ones(query_steps)
            masks = np.ones(query_steps)

    # Reward shaping (BC ignores rewards; stored only for replay-buffer format
    # compatibility).
    if intervention_handler is not None and len(intervention_flags) == len(rewards):
        for idx, is_int in enumerate(intervention_flags):
            if is_int:
                is_new = (idx == 0 or not intervention_flags[idx - 1])
                rewards[idx] = intervention_handler.shape_reward(
                    rewards[idx], is_intervention=True, is_new_takeover=is_new,
                )

    return {
        'observations': obs_list,
        'actions': action_list,
        'rewards': rewards,
        'masks': masks,
        'is_success': is_success,
        'episode_return': episode_return,
        'images': image_list,
        'env_steps': t + 1,
        'full_noise_list': full_noise_list,
        'chunk_obs_raw_list': chunk_obs_raw_list,
        'chunk_state_list': chunk_state_list,
        'initial_metaworld_seed': initial_metaworld_seed,
        'executed_actions_per_chunk': executed_actions_per_chunk,
        'intervention_flags': intervention_flags,
        'inversion_7d_mse': inversion_7d_mse,
        'inversion_w_mean_abs': inversion_w_mean_abs,
        'inversion_w_max_abs': inversion_w_max_abs,
        'inversion_w_p99': inversion_w_p99,
    }


# --- Evaluation ------------------------------------------------------------

def perform_control_eval(agent, env, i, variant, wandb_logger, agent_dp=None,
                         init_states=None, raw_model=None):
    _start_eval_skip_listener()  # safe to call repeatedly
    query_frequency = variant.query_freq
    if query_frequency is None or query_frequency <= 0:
        query_frequency = agent_dp.action_horizon if agent_dp is not None else 1
    print('query frequency', query_frequency)
    max_timesteps = variant.max_timesteps
    env_max_reward = variant.env_max_reward
    episode_returns = []
    highest_rewards = []
    success_rates = []
    episode_lens = []

    # Save and replace the agent RNG so eval is deterministic.
    saved_agent_rng = agent._rng
    agent._rng = jax.random.PRNGKey(variant.seed + 789)
    rng = jax.random.PRNGKey(variant.seed + 456)

    for rollout_id in range(variant.eval_episodes):
        rng, eval_rng = jax.random.split(rng)
        eval_seed = int(jax.random.randint(eval_rng, (), 0, 2**30))
        env.seed(eval_seed)
        obs = env.reset()

        image_list = []
        rewards = []
        eval_action_log = []

        eval_encoder_type = variant.train_kwargs.get('encoder_type', 'small')
        eval_use_precomputed = (eval_encoder_type in _PRECOMPUTED_ENCODER_TYPES)

        for t in tqdm(range(max_timesteps)):
            curr_image = obs_to_img(obs, variant)

            if t % query_frequency == 0:
                qpos = obs_to_qpos(obs, variant)
                obs_pi_zero = obs_to_pi_zero_input(obs, variant)

                if eval_use_precomputed:
                    precomputed = _extract_precomputed_features(
                        eval_encoder_type, raw_model, agent_dp, obs_pi_zero,
                        obs=obs, variant=variant
                    )
                    obs_dict = {'pixels': precomputed[np.newaxis, ..., np.newaxis]}
                    if variant.add_states:
                        obs_dict['state'] = qpos[np.newaxis, ..., np.newaxis]
                elif variant.add_states:
                    obs_dict = {
                        'pixels': curr_image[np.newaxis, ..., np.newaxis],
                        'state': qpos[np.newaxis, ..., np.newaxis],
                    }
                else:
                    obs_dict = {'pixels': curr_image[np.newaxis, ..., np.newaxis]}

                rng, key = jax.random.split(rng)
                assert agent_dp is not None

                if i == 0:
                    noise = jax.random.normal(
                        rng, (1, agent_dp.action_horizon, agent_dp.action_dim)
                    )
                else:
                    actions_noise = agent.sample_actions(obs_dict)
                    actions_noise = np.reshape(actions_noise, agent.action_chunk_shape)
                    noise = _expand_noise_basis(
                        actions_noise, action_horizon=agent_dp.action_horizon
                    )
                actions = agent_dp.infer(obs_pi_zero, noise=noise)["actions"]

                # Noise + action chunk stats.
                noise_np = np.asarray(noise).reshape(-1, 32)
                chunk_actions = np.asarray(actions)
                pos_mag = np.linalg.norm(chunk_actions[:, :3], axis=1).mean()
                grip_idx = chunk_actions.shape[1] - 1
                grip_mean = chunk_actions[:, grip_idx].mean()
                noise_mag = np.abs(noise_np).mean()
                if t == 0 or (rollout_id == 0 and t % query_frequency == 0):
                    print(f'    [Eval t={t}] pos_mag={pos_mag:.4f} '
                          f'grip={grip_mean:.3f} noise_mag={noise_mag:.3f} '
                          f'action[0]={chunk_actions[0]}')

            action_t = actions[t % query_frequency]
            eval_action_log.append({
                't': t,
                'action': action_t.copy(),
                'pos_mag': float(np.linalg.norm(action_t[:3])),
            })

            if variant.get('render', 0):
                frame = _render_hud(obs, t, i, False, None, None, variant,
                                    label=f'EVAL {rollout_id}')
                cv2.imshow('FlowDAgger Training', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                key = cv2.waitKey(1) & 0xFF
                if key == ord('s'):
                    print(f'  [Eval] Skipping rollout {rollout_id} at step {t}')
                    break
                elif key == ord('p'):
                    print('[Paused] Press p again to resume...')
                    while True:
                        key2 = cv2.waitKey(100) & 0xFF
                        if key2 == ord('p'):
                            print('[Resumed]')
                            break

            a4 = np.asarray(action_t, dtype=np.float32).reshape(-1)[:4]
            obs, reward, done, _ = env.step(a4)

            rewards.append(reward)
            image_list.append(curr_image)
            if _eval_skip_episode.is_set():
                _eval_skip_episode.clear()
                print(f"  [Eval] 's'-skipped rollout {rollout_id} at step {t}")
                break
            if done:
                break
            if reward >= env_max_reward:
                break

        if len(rewards) == 0:
            episode_lens.append(0)
            episode_returns.append(0.0)
            highest_rewards.append(0.0)
            success_rates.append(False)
            print(f'Rollout {rollout_id} : SKIPPED')
            continue
        episode_lens.append(t + 1)
        rewards = np.array(rewards)
        episode_return = np.sum(rewards)
        episode_returns.append(episode_return)
        highest_rewards.append(np.max(rewards))
        is_success = (reward == env_max_reward)
        success_rates.append(is_success)
        print(f'Rollout {rollout_id} : {episode_return=}, Success: {is_success}')

        if not variant.get('save_eval_video', 1):
            continue
        video_frames = np.stack(image_list)  # (T, H, W, C)
        video = video_frames.transpose(0, 3, 1, 2)  # (T, C, H, W)
        wandb_logger.log({f'eval_video/{rollout_id}': wandb.Video(video, fps=50)}, step=i)

        if hasattr(variant, 'outputdir') and variant.outputdir:
            video_dir = os.path.join(variant.outputdir, 'videos')
            os.makedirs(video_dir, exist_ok=True)
            video_path = os.path.join(
                video_dir, f'eval_step{i}_rollout{rollout_id}.mp4'
            )
            frames = video.transpose(0, 2, 3, 1)  # (T, H, W, C)
            h, w = frames.shape[1], frames.shape[2]
            writer = cv2.VideoWriter(
                video_path, cv2.VideoWriter_fourcc(*'mp4v'), 50, (w, h)
            )
            for frame in frames:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer.release()
            print(f'  Saved video: {video_path}')

    success_rate = np.mean(np.array(success_rates))
    avg_return = np.mean(episode_returns)
    avg_episode_len = np.mean(episode_lens)
    summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'
    wandb_logger.log({'evaluation/avg_return': avg_return}, step=i)
    wandb_logger.log({'evaluation/success_rate': success_rate}, step=i)
    wandb_logger.log({'evaluation/avg_episode_len': avg_episode_len}, step=i)
    for r in range(env_max_reward + 1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / variant.eval_episodes
        wandb_logger.log({f'evaluation/Reward >= {r}': more_or_equal_r_rate}, step=i)
        summary_str += (f'Reward >= {r}: {more_or_equal_r}/{variant.eval_episodes} '
                        f'= {more_or_equal_r_rate*100}%\n')
    print(summary_str)

    # Restore the agent RNG so training continues from its original state.
    agent._rng = saved_agent_rng

    if hasattr(variant, 'outputdir') and variant.outputdir:
        import json
        results_path = os.path.join(variant.outputdir, 'eval_results.jsonl')
        result_entry = {
            'step': i,
            'success_rate': float(success_rate),
            'avg_return': float(avg_return),
            'avg_episode_len': float(avg_episode_len),
            'prefix': getattr(variant, 'prefix', ''),
            'seed': getattr(variant, 'seed', None),
            'launch_group_id': getattr(variant, 'launch_group_id', ''),
        }
        with open(results_path, 'a') as f:
            f.write(json.dumps(result_entry) + '\n')
