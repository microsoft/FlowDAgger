"""Heuristic expert for LIBERO tasks using privileged simulator state.

Designed for pick-and-place tasks (e.g. task 57: "pick up the book on
the left and place it on top of the shelf").  Uses a state-machine with
P-control, reading positions directly from MuJoCo sim data.

Action layout (OSC_POSE, 7D):
    [dx, dy, dz, wx, wy, wz, gripper]
    Actions in [-1, 1].  OSC maps to position deltas (+/-0.05 m)
    and rotation deltas (+/-0.5 rad).  Gripper: +1 = close, -1 = open.
"""

import numpy as np
from shared.experts.base_expert import BaseExpert


# State machine phases
ORIENT = 0   # correct wrist rotation (rotation-only, no position)
SCOUT = 1    # move to high vantage point to observe object (optional)
REACH = 2    # move above object
LOWER = 3    # descend to object
GRASP = 4    # close gripper
LIFT = 5     # lift object
MOVE = 6     # move to place target
PLACE = 7    # lower to place target
RELEASE = 8  # open gripper
DONE = 9     # task complete


class LiberoExpert(BaseExpert):
    """Heuristic pick-and-place expert for LIBERO tasks.

    Reads privileged state from the MuJoCo sim to identify object and
    target positions, then executes a state machine:
        reach -> lower -> grasp -> lift -> move -> place -> release

    Parameters
    ----------
    object_name : str or None
        MuJoCo body name for the object to pick.  If None, auto-detected
        from common LIBERO object names on first reset.
    place_target_name : str or None
        MuJoCo body name for the place target.  If None, auto-detected.
    place_target_offset : np.ndarray
        Offset from place target body position to desired placement (meters).
        Default is [0, 0, 0.05] (5 cm above target body center).
    reach_height_offset : float
        Height above object during approach (meters).
    grasp_xy_threshold : float
        XY distance to transition from reach -> lower (meters).
    grasp_z_threshold : float
        Z distance to begin grasping (meters).
    lift_height : float
        Height to lift object to before moving to target (meters above table).
    place_xy_threshold : float
        XY distance to transition from move -> place (meters).
    place_z_threshold : float
        Z distance to release (meters).
    position_gain : float
        P-control gain (gain=20 -> 5cm error = full speed).
    grasp_steps : int
        Steps to hold gripper closed before lifting.
    """

    def __init__(
        self,
        object_name=None,
        place_target_name=None,
        place_target_offset=None,
        reach_height_offset=0.06,
        grasp_xy_threshold=0.02,
        grasp_z_threshold=0.02,
        lift_height=0.62,
        place_xy_threshold=0.03,
        place_z_threshold=0.015,
        position_gain=20.0,
        grasp_steps=15,
        grasp_offset=0.0,
        scout_position=None,
        scout_threshold=0.03,
    ):
        self.object_name = object_name
        self.place_target_name = place_target_name
        self.place_target_offset = (
            np.array(place_target_offset, dtype=np.float64)
            if place_target_offset is not None
            else np.array([0.0, 0.0, 0.03])
        )
        self.reach_height_offset = reach_height_offset
        self.grasp_xy_threshold = grasp_xy_threshold
        self.grasp_z_threshold = grasp_z_threshold
        self.lift_height = lift_height
        self.place_xy_threshold = place_xy_threshold
        self.place_z_threshold = place_z_threshold
        self.position_gain = position_gain
        self.grasp_steps = grasp_steps
        self.grasp_offset = grasp_offset
        self.scout_position = (
            np.array(scout_position, dtype=np.float64)
            if scout_position is not None
            else None
        )
        self.scout_threshold = scout_threshold

        # Internal state
        self._phase = ORIENT
        self._grasp_counter = 0
        self._orient_counter = 0
        self._grasp_attempts = 0  # number of failed grasp cycles
        self._grip_delta = None  # EE-to-object offset measured at grasp
        self._initial_obj_z = None  # recorded on reset for state-independent distance
        self.max_grasp_attempts = 10  # give up after this many failed grasps
        self.max_orient_steps = 30  # skip ORIENT if it can't converge

        # Stuck detection: if EE doesn't move for N steps, retreat upward
        self._stuck_window = 20
        self._stuck_threshold = 0.003  # 3mm over window = stuck
        self._retreat_steps = 15  # how many steps to retreat upward
        self._ee_history = []
        self._retreating = False
        self._retreat_counter = 0

        # Cached body IDs (resolved on first reset)
        self._object_body_id = None
        self._place_target_body_id = None
        self._resolved = False

    def _resolve_bodies(self, env):
        """Resolve MuJoCo body IDs from names, with auto-detection."""
        model = env.sim.model

        if self.object_name is not None:
            self._object_body_id = model.body_name2id(self.object_name)
        else:
            self._object_body_id = self._find_object_body(env)

        if self.place_target_name is not None:
            self._place_target_body_id = model.body_name2id(self.place_target_name)
        else:
            self._place_target_body_id = self._find_place_target_body(env)

        self._resolved = True
        print(f"[LiberoExpert] Object body ID: {self._object_body_id} "
              f"({model.body_id2name(self._object_body_id)})")
        print(f"[LiberoExpert] Place target body ID: {self._place_target_body_id} "
              f"({model.body_id2name(self._place_target_body_id)})")

    def _find_object_body(self, env):
        """Auto-detect the graspable object body from MuJoCo model."""
        model = env.sim.model
        # Look for common LIBERO object body names
        candidates = []
        for i in range(model.nbody):
            name = model.body_id2name(i)
            name_lower = name.lower()
            # Skip robot parts, world, and fixtures
            if any(skip in name_lower for skip in [
                'robot', 'gripper', 'world', 'base', 'mount',
                'shelf', 'table', 'cabinet', 'drawer', 'counter',
                'fixture', 'wall', 'floor', 'bin', 'box_0',
            ]):
                continue
            # Look for graspable objects
            if any(obj in name_lower for obj in [
                'book', 'cube', 'can', 'bowl', 'mug', 'plate',
                'bottle', 'cup', 'pan', 'pot', 'object',
                'cheese', 'cream', 'butter', 'milk', 'bread',
                'food', 'snack', 'fruit', 'vegetable',
            ]):
                candidates.append((i, name))

        if not candidates:
            # Fallback: find bodies with free joints (graspable objects)
            for i in range(model.nbody):
                name = model.body_id2name(i)
                if name and 'robot' not in name.lower() and 'world' not in name.lower():
                    # Check if body has a free joint
                    jnt_start = model.body_jntadr[i]
                    if jnt_start >= 0 and model.jnt_type[jnt_start] == 0:  # 0 = free joint
                        candidates.append((i, name))

        if len(candidates) == 0:
            raise RuntimeError(
                "Could not auto-detect object body. "
                "Set object_name explicitly. Available bodies: "
                + str([model.body_id2name(i) for i in range(model.nbody)])
            )

        # If task mentions "left", prefer leftmost (most negative y in robosuite)
        # For now just take first candidate
        print(f"[LiberoExpert] Object candidates: {candidates}")
        return candidates[0][0]

    def _find_place_target_body(self, env):
        """Auto-detect the place target body from MuJoCo model."""
        model = env.sim.model
        candidates = []
        for i in range(model.nbody):
            name = model.body_id2name(i)
            name_lower = name.lower()
            if any(target in name_lower for target in [
                'shelf', 'cabinet', 'bin', 'box', 'tray', 'plate',
                'counter', 'target',
            ]):
                candidates.append((i, name))

        if len(candidates) == 0:
            raise RuntimeError(
                "Could not auto-detect place target body. "
                "Set place_target_name explicitly. Available bodies: "
                + str([model.body_id2name(i) for i in range(model.nbody)])
            )

        print(f"[LiberoExpert] Place target candidates: {candidates}")
        # Prefer tray > shelf > others
        for keyword in ['tray', 'shelf', 'bin', 'box']:
            for bid, name in candidates:
                if keyword in name.lower():
                    return bid
        return candidates[0][0]

    def _is_holding_object(self, env, distance_threshold=0.05):
        """Check if the gripper is actually holding the object.

        Uses gripper joint position AND proximity to the object body.
        When gripping an object the fingers can't fully close, so qpos
        settles around 0.020-0.025 (vs ~0.005 for empty close, ~0.039 open).
        We check: gripper is commanded closed (qpos < 0.035) AND something
        is preventing full closure (qpos > 0.012) AND object is nearby.
        """
        # Check gripper closure via joint qpos
        joint_name = env.robots[0].gripper.joints[0]
        jid = env.sim.model.joint_name2id(joint_name)
        qpos = env.sim.data.qpos[env.sim.model.jnt_qposadr[jid]]

        # Gripper must be commanded closed but not fully closed (object between fingers)
        gripper_holding = 0.012 < qpos < 0.035

        if not gripper_holding:
            return False

        # Check object proximity to EE
        ee_pos = np.array(env.sim.data.site_xpos[env.robots[0].eef_site_id])
        # Use raw object pos (no grasp_offset) for this check
        obj_pos = np.array(env.sim.data.body_xpos[self._object_body_id])
        distance = np.linalg.norm(ee_pos - obj_pos)
        return distance < distance_threshold

    def _get_ee_pos(self, env):
        """Get end-effector position from privileged state."""
        return np.array(env.sim.data.site_xpos[env.robots[0].eef_site_id])

    def _get_object_pos(self, env):
        """Get object position from privileged state.

        During pre-grasp phases (REACH/LOWER/GRASP), applies grasp_offset in
        +y to shift the EE toward the object rim for rim-gripping wide objects
        like bowls.
        """
        pos = np.array(env.sim.data.body_xpos[self._object_body_id])
        if self.grasp_offset != 0.0 and self._phase <= GRASP:
            pos = pos.copy()
            pos[1] += self.grasp_offset
        return pos

    def _get_place_target_pos(self, env):
        """Get place target position from privileged state.

        When grasp_offset is active, compensates for the EE-to-object offset
        measured at grasp time so the *object* (not the EE) lands on target.
        """
        pos = np.array(env.sim.data.body_xpos[self._place_target_body_id])
        pos = pos + self.place_target_offset
        if self._grip_delta is not None:
            pos = pos + self._grip_delta
        return pos

    def _rotation_error(self, env):
        """Compute axis-angle rotation from current EE orientation to home.

        Returns the rotation vector (axis * angle) that would rotate the
        current EE orientation back to the home orientation recorded at reset.
        """
        R_cur = np.array(
            env.sim.data.site_xmat[env.robots[0].eef_site_id]
        ).reshape(3, 3)
        R_err = self._home_rotmat @ R_cur.T
        cos_angle = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        if angle < 1e-7:
            return np.zeros(3)
        axis = np.array([
            R_err[2, 1] - R_err[1, 2],
            R_err[0, 2] - R_err[2, 0],
            R_err[1, 0] - R_err[0, 1],
        ]) / (2.0 * np.sin(angle))
        return axis * angle

    def reset(self, env):
        """Reset expert state for new episode."""
        if not self._resolved:
            self._resolve_bodies(env)
        self._phase = ORIENT
        self._grasp_counter = 0
        self._orient_counter = 0
        self._grasp_attempts = 0
        self._grip_delta = None
        self._initial_obj_z = self._get_object_pos(env)[2]
        self._home_rotmat = np.array(
            env.sim.data.site_xmat[env.robots[0].eef_site_id]
        ).reshape(3, 3).copy()
        self._ee_history = []
        self._retreating = False
        self._retreat_counter = 0

    def act(self, env):
        """Compute expert action using privileged sim state.

        Returns
        -------
        np.ndarray (7,)
            OSC_POSE action: [dx, dy, dz, wx, wy, wz, gripper]
        """
        ee_pos = self._get_ee_pos(env)
        obj_pos = self._get_object_pos(env)
        place_pos = self._get_place_target_pos(env)

        action = np.zeros(7, dtype=np.float64)

        # Stuck detection: if EE barely moves over a window, retreat upward
        # to escape singularity/joint-limit lockups (only during pre-grasp phases)
        if self._phase <= GRASP:
            self._ee_history.append(ee_pos.copy())
            if len(self._ee_history) > self._stuck_window:
                self._ee_history.pop(0)

            if self._retreating:
                # Move up and slightly toward robot base to escape singularity
                action[2] = 1.0   # up
                action[1] = -0.3  # toward robot
                action[6] = -1.0  # open gripper
                self._retreat_counter += 1
                if self._retreat_counter >= self._retreat_steps:
                    print("[LiberoExpert] Retreat done, restarting from ORIENT")
                    self._retreating = False
                    self._retreat_counter = 0
                    self._ee_history = []
                    self._phase = ORIENT
                    self._orient_counter = 0
                # Clip and return early
                action[:3] = np.clip(action[:3], -1.0, 1.0)
                action[3:6] = 0.0
                action[6] = np.clip(action[6], -1.0, 1.0)
                return action.astype(np.float32)

            if len(self._ee_history) >= self._stuck_window:
                displacement = np.linalg.norm(
                    self._ee_history[-1] - self._ee_history[0]
                )
                if displacement < self._stuck_threshold:
                    print(f"[LiberoExpert] Stuck detected (displacement={displacement:.4f}), retreating")
                    self._retreating = True
                    self._retreat_counter = 0
                    self._ee_history = []
                    action[2] = 1.0
                    action[1] = -0.3
                    action[6] = -1.0
                    action[:3] = np.clip(action[:3], -1.0, 1.0)
                    action[3:6] = 0.0
                    action[6] = np.clip(action[6], -1.0, 1.0)
                    return action.astype(np.float32)

        if self._phase == ORIENT:
            # Correct wrist rotation before approaching - rotation only, no position.
            # This avoids the OSC coupling issue where simultaneous position +
            # rotation commands destabilize each other.
            rot_err = self._rotation_error(env)
            rot_mag = np.linalg.norm(rot_err)
            self._orient_counter += 1
            if rot_mag < 0.15 or self._orient_counter >= self.max_orient_steps:
                if self.scout_position is not None and self._grasp_attempts == 0:
                    self._phase = SCOUT
                else:
                    self._phase = REACH
                self._orient_counter = 0
            else:
                action[3:6] = rot_err * 1.0  # moderate gain
                action[6] = -1.0  # open gripper

        elif self._phase == SCOUT:
            # Move to a high vantage point so the wrist camera can observe the
            # object before descending.  Only used on the first grasp attempt;
            # retries skip straight to REACH since the object location is known.
            target = self.scout_position
            action[:3] = (target - ee_pos) * self.position_gain
            action[6] = -1.0  # open gripper

            dist = np.linalg.norm(ee_pos - target)
            if dist < self.scout_threshold:
                print(f"[LiberoExpert] Scout position reached (dist={dist:.4f}), proceeding to REACH")
                self._phase = REACH

        elif self._phase == REACH:
            target = obj_pos.copy()
            target[2] += self.reach_height_offset
            action[:3] = (target - ee_pos) * self.position_gain
            action[6] = -1.0  # open gripper

            xy_dist = np.linalg.norm(ee_pos[:2] - obj_pos[:2])
            if xy_dist < self.grasp_xy_threshold:
                self._phase = LOWER

        elif self._phase == LOWER:
            target = obj_pos.copy()
            action[:3] = (target - ee_pos) * self.position_gain
            action[6] = -1.0  # open gripper

            z_dist = abs(ee_pos[2] - obj_pos[2])
            if z_dist < self.grasp_z_threshold:
                self._phase = GRASP
                self._grasp_counter = 0

        elif self._phase == GRASP:
            target = obj_pos.copy()
            action[:3] = (target - ee_pos) * self.position_gain
            action[6] = 1.0  # close gripper

            self._grasp_counter += 1
            if self._grasp_counter >= self.grasp_steps:
                # Verify we actually have the object before advancing
                if self._is_holding_object(env):
                    self._phase = LIFT
                else:
                    # Failed grasp - open gripper and retry from REACH
                    self._grasp_attempts += 1
                    if self._grasp_attempts >= self.max_grasp_attempts:
                        # Give up - go to DONE to stop producing wild actions
                        print(f"[LiberoExpert] Grasp failed {self._grasp_attempts} times, giving up")
                        self._phase = DONE
                    else:
                        print(f"[LiberoExpert] Grasp failed (attempt {self._grasp_attempts}), retrying")
                        self._phase = ORIENT
                        self._grasp_counter = 0

        elif self._phase == LIFT:
            # Verify we still have the object (could have slipped)
            if not self._is_holding_object(env):
                print("[LiberoExpert] Object lost during LIFT, retrying")
                self._grasp_attempts += 1
                self._grip_delta = None
                if self._grasp_attempts >= self.max_grasp_attempts:
                    self._phase = DONE
                else:
                    self._phase = ORIENT
                    self._grasp_counter = 0
            else:
                # Measure grip delta on first LIFT step for placement compensation.
                # When the object is grasped off-center (e.g. the policy moved it),
                # we need to offset the place target so the OBJECT lands on
                # target, not the EE.
                if self._grip_delta is None:
                    raw_obj = np.array(env.sim.data.body_xpos[self._object_body_id])
                    self._grip_delta = ee_pos - raw_obj
                    raw_tray = np.array(env.sim.data.body_xpos[self._place_target_body_id])
                    print(f"[LiberoExpert] grip_delta={self._grip_delta} ee={ee_pos} obj={raw_obj} tray={raw_tray}")

                target = ee_pos.copy()
                target[2] = self.lift_height
                action[:3] = (target - ee_pos) * self.position_gain
                action[6] = 1.0  # close gripper

                if ee_pos[2] >= self.lift_height - 0.02:
                    self._phase = MOVE

        elif self._phase == MOVE:
            # Verify we still have the object
            if not self._is_holding_object(env):
                print("[LiberoExpert] Object lost during MOVE, retrying")
                self._grasp_attempts += 1
                if self._grasp_attempts >= self.max_grasp_attempts:
                    self._phase = DONE
                else:
                    self._phase = ORIENT
                    self._grasp_counter = 0
                    self._grip_delta = None
            else:
                # Move to place target at fixed transit height
                target = place_pos.copy()
                target[2] = self.lift_height
                action[:3] = (target - ee_pos) * self.position_gain
                action[6] = 1.0  # close gripper

                xy_dist = np.linalg.norm(ee_pos[:2] - place_pos[:2])
                if xy_dist < self.place_xy_threshold:
                    raw_tray = np.array(env.sim.data.body_xpos[self._place_target_body_id])
                    raw_obj = np.array(env.sim.data.body_xpos[self._object_body_id])
                    print(f"[LiberoExpert] MOVE->PLACE ee={ee_pos} place_target={place_pos} "
                          f"raw_tray={raw_tray} raw_obj={raw_obj} grip_delta={self._grip_delta}")
                    self._phase = PLACE

        elif self._phase == PLACE:
            # Descend straight down to place target
            target = place_pos.copy()
            # Only command z movement - hold xy position to avoid drift
            action[0] = (target[0] - ee_pos[0]) * self.position_gain
            action[1] = (target[1] - ee_pos[1]) * self.position_gain
            action[2] = (target[2] - ee_pos[2]) * self.position_gain
            action[6] = 1.0  # close gripper

            z_dist = abs(ee_pos[2] - place_pos[2])
            if z_dist < self.place_z_threshold:
                raw_tray = np.array(env.sim.data.body_xpos[self._place_target_body_id])
                raw_obj = np.array(env.sim.data.body_xpos[self._object_body_id])
                print(f"[LiberoExpert] PLACE->RELEASE ee={ee_pos} obj={raw_obj} "
                      f"raw_tray={raw_tray} place_target={place_pos}")
                self._phase = RELEASE
                self._grasp_counter = 0

        elif self._phase == RELEASE:
            action[6] = -1.0  # open gripper
            # Hold open for a few steps so gripper has time to release
            self._grasp_counter += 1
            if self._grasp_counter >= 10:
                self._phase = DONE

        elif self._phase == DONE:
            action[6] = -1.0  # keep open

        # Clip position and gripper; allow rotation only during ORIENT
        action[:3] = np.clip(action[:3], -1.0, 1.0)
        if self._phase == ORIENT:
            action[3:6] = np.clip(action[3:6], -1.0, 1.0)
        else:
            action[3:6] = 0.0
        action[6] = np.clip(action[6], -1.0, 1.0)

        return action.astype(np.float32)

    def compute_off_nominal_distance(self, env):
        """State-independent distance between EE and current target.

        Determines the task phase from env observations (object height +
        gripper state) rather than the expert's internal state machine,
        so it works correctly even when the learned policy is in control.
        """
        ee_pos = self._get_ee_pos(env)
        obj_pos = self._get_object_pos(env)

        # Detect whether the object has been picked up: object is above
        # its initial resting height and close to the EE
        obj_lifted = (obj_pos[2] > self._initial_obj_z + 0.03
                      if self._initial_obj_z is not None
                      else self._phase > GRASP)

        if not obj_lifted:
            # Still need to pick up the object
            return float(np.linalg.norm(ee_pos - obj_pos))
        else:
            # Object is lifted, target is the place location
            place_pos = self._get_place_target_pos(env)
            return float(np.linalg.norm(ee_pos - place_pos))

    def save_state(self):
        return {
            "phase": self._phase,
            "grasp_counter": self._grasp_counter,
            "grasp_attempts": self._grasp_attempts,
            "grip_delta": self._grip_delta.copy() if self._grip_delta is not None else None,
        }

    def restore_state(self, state):
        self._phase = state["phase"]
        self._grasp_counter = state["grasp_counter"]
        self._grasp_attempts = state.get("grasp_attempts", 0)
        self._grip_delta = state.get("grip_delta")
