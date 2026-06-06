"""Base class for scripted experts that supply corrective actions."""

from abc import ABC, abstractmethod
import numpy as np


class BaseExpert(ABC):
    """Interface for experts that read privileged simulator state and return
    corrective actions, used by the intervention handler.
    """

    @abstractmethod
    def reset(self, env):
        """Reset internal state at the start of an episode."""

    @abstractmethod
    def act(self, env):
        """Return an action matching the env's action space."""

    @abstractmethod
    def compute_off_nominal_distance(self, env):
        """Return a non-negative scalar; higher means more off-nominal."""

    def partial_reset(self, env):
        """Alias for reset (single-env case)."""
        self.reset(env)

    def save_state(self):
        """Return internal state for the save/restore pattern."""
        return {}

    def restore_state(self, state):
        """Restore internal state from save_state()."""
        pass
