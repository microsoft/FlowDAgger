"""Decides when a scripted expert takes over during a single-env rollout.

Two trigger modes (intervention_mode):

- "beta_decay" (default): with probability beta, the expert takes over at a
  random step in [takeover_min, takeover_max] and holds for the rest of the
  episode. beta decays linearly from beta_start to beta_end over
  beta_decay_episodes. takeover_max can ramp over training (curriculum) so early
  takeovers happen at near-pristine states where the expert is reliable,
  expanding to later, more off-distribution states as the policy improves.
- "disagreement": the expert takes over the first step the policy and expert
  actions diverge by more than disagreement_threshold (L2 over the 4D action),
  and holds for the rest of the episode. A state-conditioned trigger that does
  not depend on a per-task off-nominal distance.

Once triggered (either mode), takeover holds until the episode ends.
"""

import logging
import numpy as np

log = logging.getLogger(__name__)


class InterventionHandler:
    """Orchestrates expert intervention during rollout collection.

    Key args:
      expert: implements act(), reset(), compute_off_nominal_distance().
      query_freq: chunk length (steps per policy/expert action chunk).
      intervention_probability: probability intervention is enabled per episode.
      reward_takeover_penalty: added on the first step of takeover.
      reward_expert_bonus: added on every expert step.
      inverter, action_converter: map expert actions into policy noise space.
      beta_start/beta_end/beta_decay_episodes: takeover-probability schedule.
      takeover_min/takeover_max: range of steps at which takeover can start.
      takeover_max_start/takeover_max_curriculum_episodes: optional ramp of
        takeover_max (None = static takeover_max).
      seed: RNG seed for per-episode sampling.
    """

    def __init__(
        self,
        expert,
        query_freq=20,
        intervention_probability=1.0,
        reward_takeover_penalty=0.0,
        reward_expert_bonus=0.0,
        inverter=None,
        action_converter=None,
        seed=None,
        intervention_mode="beta_decay",
        beta_start=1.0,
        beta_end=0.1,
        beta_decay_episodes=2000,
        takeover_min=5,
        takeover_max=60,
        takeover_max_start=None,
        takeover_max_curriculum_episodes=None,
        disagreement_threshold=0.5,
        **_ignored,
    ):
        if intervention_mode not in ("beta_decay", "disagreement"):
            raise ValueError(f"Unknown intervention_mode: {intervention_mode!r}")
        self.intervention_mode = intervention_mode
        self.disagreement_threshold = disagreement_threshold
        self.expert = expert
        self.query_freq = query_freq
        self.intervention_probability = intervention_probability
        self.reward_takeover_penalty = reward_takeover_penalty
        self.reward_expert_bonus = reward_expert_bonus
        self.inverter = inverter
        self.action_converter = action_converter
        self._rng = np.random.RandomState(seed)

        self._beta_start = beta_start
        self._beta_end = beta_end
        self._beta_decay_episodes = beta_decay_episodes
        self._takeover_min = takeover_min
        self._takeover_max = takeover_max
        self._takeover_max_start = takeover_max_start
        self._takeover_max_curriculum_episodes = (
            takeover_max_curriculum_episodes
            if takeover_max_curriculum_episodes is not None
            else beta_decay_episodes
        )
        self._beta_takeover_step = None  # sampled per episode

        # Set by the trainer to force expert-only seed episodes.
        self._force_immediate_takeover = False

        # Per-episode state
        self._intervening = False
        self._allow_intervention = True
        self._step = 0
        self._query_step_idx = 0

        # Logging counters
        self.total_steps = 0
        self.intervention_steps = 0
        self.total_episodes = 0
        self.intervened_episodes = 0
        self._inversion_errors = []
        self._trigger_distances = []  # distance at moment of takeover

    def on_episode_reset(self, env):
        """Reset state and sample this episode's takeover step."""
        self._intervening = False
        self._step = 0
        self._query_step_idx = 0
        self._allow_intervention = (
            self._rng.random() < self.intervention_probability
        )

        self._beta_takeover_step = None
        if self._force_immediate_takeover:
            # Seed episodes: force expert from step 0.
            self._beta_takeover_step = 0
            self._intervening = True
        elif self.intervention_mode == "beta_decay":
            beta = self._beta_start + (self._beta_end - self._beta_start) * min(
                1.0, self.total_episodes / max(self._beta_decay_episodes, 1)
            )
            if self._rng.random() < beta:
                if self._takeover_max_start is not None:
                    progress = min(
                        1.0,
                        self.total_episodes
                        / max(self._takeover_max_curriculum_episodes, 1),
                    )
                    current_takeover_max = int(
                        self._takeover_max_start
                        + (self._takeover_max - self._takeover_max_start) * progress
                    )
                else:
                    current_takeover_max = self._takeover_max
                self._beta_takeover_step = self._rng.randint(
                    self._takeover_min, min(current_takeover_max, 400) + 1
                )

        self.expert.reset(env)
        self.total_episodes += 1

    def check_progress(self, env, policy_action=None, expert_action=None):
        """Decide whether the expert should take over this step.

        Call once per env step. Returns (should_take_over, off_nominal_distance).
        """
        dist = self.expert.compute_off_nominal_distance(env)
        if self._intervening:
            return True, dist
        if not self._allow_intervention:
            return False, dist

        if self.intervention_mode == "beta_decay":
            triggered = (self._beta_takeover_step is not None
                         and self._step >= self._beta_takeover_step)
        else:  # disagreement
            triggered = False
            if policy_action is not None:
                if expert_action is None:
                    expert_action = self.expert.act(env)
                pa = np.asarray(policy_action, dtype=np.float32).reshape(-1)[:4]
                ea = np.asarray(expert_action, dtype=np.float32).reshape(-1)[:4]
                triggered = (np.linalg.norm(pa - ea)
                             >= self.disagreement_threshold)

        if triggered:
            self._intervening = True
            self.intervened_episodes += 1
            self._trigger_distances.append(dist)
            log.info(f"{self.intervention_mode} takeover at step "
                     f"{self._step}: dist={dist:.4f}")
            return True, dist
        return False, dist

    def step(self):
        """Advance the step counter. Call once per env step."""
        self._step += 1
        self.total_steps += 1
        if self._intervening:
            self.intervention_steps += 1

    def shape_reward(self, reward, is_intervention, is_new_takeover):
        """Apply reward shaping for intervention steps."""
        shaped = reward
        if is_new_takeover:
            shaped += self.reward_takeover_penalty
        if is_intervention:
            shaped += self.reward_expert_bonus
        return shaped

    def get_stats(self):
        """Return intervention statistics for logging."""
        stats = {}
        if self.total_steps > 0:
            stats["intervention/step_rate"] = (
                self.intervention_steps / self.total_steps
            )
        if self.total_episodes > 0:
            stats["intervention/episode_rate"] = (
                self.intervened_episodes / self.total_episodes
            )
        stats["intervention/total_steps"] = self.total_steps
        stats["intervention/expert_steps"] = self.intervention_steps
        stats["intervention/total_episodes"] = self.total_episodes
        stats["intervention/intervened_episodes"] = self.intervened_episodes
        if self._inversion_errors:
            stats["intervention/inversion_error_mean"] = np.mean(self._inversion_errors)
            stats["intervention/inversion_error_max"] = np.max(self._inversion_errors)
            stats["intervention/num_inversions"] = len(self._inversion_errors)
        if self._trigger_distances:
            stats["intervention/trigger_distance_mean"] = np.mean(self._trigger_distances)
        return stats

    def reset_stats(self):
        """Reset logging counters."""
        self.total_steps = 0
        self.intervention_steps = 0
        self.total_episodes = 0
        self.intervened_episodes = 0
        self._inversion_errors = []
        self._trigger_distances = []
