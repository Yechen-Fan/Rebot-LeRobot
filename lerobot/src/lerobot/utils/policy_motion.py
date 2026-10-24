"""Opt-in action decoding and joint-reference shaping for hardware policies."""

import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np


@dataclass
class HandoverPosePair:
    donor: list[float]  # xyz in metres, followed by rotation-6d
    receiver: list[float]


@dataclass
class HandoverReleaseConfig:
    donor_gripper: str
    receiver_gripper: str
    pose_pairs: list[HandoverPosePair]
    position_tolerance: float = 0.04
    relative_position_tolerance: float = 0.025
    rotation_tolerance: float = 20.0
    settle_time_s: float = 1.0
    settle_position_tolerance: float = 0.004
    settle_rotation_tolerance: float = 3.0
    max_observation_gap_s: float = 0.2
    # After entering a demonstrated pose, allow small boundary fluctuations.
    # Entry thresholds remain unchanged; exiting this larger band cancels it.
    match_exit_scale: float = 1.2

    def __post_init__(self):
        if {self.donor_gripper, self.receiver_gripper} != {"left_gripper.pos", "right_gripper.pos"}:
            raise ValueError("Handover release requires distinct left and right grippers.")
        values = [self.position_tolerance, self.relative_position_tolerance, self.rotation_tolerance,
                  self.settle_time_s, self.settle_position_tolerance, self.settle_rotation_tolerance,
                  self.max_observation_gap_s]
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError("Handover release tolerances and times must be finite and positive.")
        if not np.isfinite(self.match_exit_scale) or self.match_exit_scale < 1:
            raise ValueError("Handover match exit scale must be finite and at least 1.")
        if not self.pose_pairs:
            raise ValueError("Handover release needs paired demonstration poses.")
        for pair in self.pose_pairs:
            for pose in (pair.donor, pair.receiver):
                if len(pose) != 9 or not np.isfinite(pose).all():
                    raise ValueError("Handover poses must contain finite xyz and rotation-6d values.")
                _rotation_6d_matrix(pose[3:])


@dataclass
class PolicyMotionConfig:
    max_joint_velocity: float = 45.0  # deg/s
    max_joint_acceleration: float = 180.0  # deg/s^2
    max_gripper_velocity: float = 180.0
    max_gripper_acceleration: float = 720.0
    # Only for grippers whose demonstration commands are binary (0=open, 1=closed).
    binary_gripper_actions: list[str] = field(default_factory=list)
    binary_gripper_threshold: float = 0.5
    synchronized_gripper_actions: list[str] = field(default_factory=list)
    gripper_position_tolerance: float = 0.003  # TCP metres
    gripper_rotation_tolerance: float = 5.0  # degrees
    gripper_closed_feedback: float = 0.9
    # Optional single-handover release rule for a fixed task and robot mounting.
    handover_release: HandoverReleaseConfig | None = None

    def __post_init__(self):
        values = [
            self.max_joint_velocity, self.max_joint_acceleration,
            self.max_gripper_velocity, self.max_gripper_acceleration,
            self.gripper_position_tolerance, self.gripper_rotation_tolerance,
        ]
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError("Policy motion limits must be finite and positive.")
        if not np.isfinite(self.binary_gripper_threshold) or not 0 < self.binary_gripper_threshold < 1:
            raise ValueError("Binary gripper threshold must be finite and strictly between 0 and 1.")
        allowed = {"gripper.pos", "left_gripper.pos", "right_gripper.pos"}
        if any(key not in allowed for key in self.binary_gripper_actions):
            raise ValueError("Binary gripper actions must name normalized gripper.pos actions.")
        if any(key not in self.binary_gripper_actions for key in self.synchronized_gripper_actions):
            raise ValueError("Synchronized grippers must also be configured as binary grippers.")
        if len(set(self.synchronized_gripper_actions)) != len(self.synchronized_gripper_actions):
            raise ValueError("Synchronized gripper actions must not contain duplicates.")
        if not np.isfinite(self.gripper_closed_feedback) or not 0.5 < self.gripper_closed_feedback <= 1:
            raise ValueError("Closed gripper feedback must be finite and in (0.5, 1].")
        if self.handover_release is not None and self.handover_release.donor_gripper not in self.binary_gripper_actions:
            raise ValueError("The handover donor must be configured as a binary gripper.")


def decode_policy_grippers(action, config: PolicyMotionConfig | None):
    """Decode selected binary demonstration commands before joint motion limiting.

    Regression noise near zero must not start closing an open binary gripper.
    The existing reference limiter still shapes the resulting open/close motion.
    Other action values, including continuous grippers, are preserved exactly.
    """
    if config is None or not config.binary_gripper_actions:
        return action
    decoded = dict(action)
    for key in config.binary_gripper_actions:
        if key not in action:
            raise ValueError(f"Configured binary gripper action is missing: {key}.")
        value = float(action[key])
        if not np.isfinite(value):
            raise ValueError(f"Non-finite policy gripper target: {key}.")
        decoded[key] = float(value >= config.binary_gripper_threshold)
    return decoded


def _rotation_6d_matrix(values):
    first, second = np.asarray(values[:3], dtype=float), np.asarray(values[3:], dtype=float)
    norm = np.linalg.norm(first)
    if not np.isfinite(norm) or norm < 1e-6:
        raise ValueError("Invalid TCP rotation in gripper synchronization.")
    first = first / norm
    second = second - first * np.dot(first, second)
    norm = np.linalg.norm(second)
    if not np.isfinite(norm) or norm < 1e-6:
        raise ValueError("Invalid TCP rotation in gripper synchronization.")
    second = second / norm
    return np.column_stack((first, second, np.cross(first, second)))


class PolicyHandoverRelease:
    """Release once at demonstrated paired poses; hold both arms while opening.

    This task-specific rule assumes the demonstrated grasp offsets and robot
    mounting. Pose/closed feedback does not prove that a receiving object is
    present. A new policy session owns fresh state; no global elapsed-time trigger.
    """

    def __init__(self, config: HandoverReleaseConfig | None, closed_feedback: float = 0.9):
        self.config = config
        self.closed_feedback = closed_feedback
        self.phase = "waiting"
        self.anchor = None
        self.since = None
        self.last_time = None
        self.held_action = {}
        self.matching_profile = None
        if config is not None:
            self.keys = [
                [gripper.removesuffix("gripper.pos") + "tcp." + axis
                 for axis in ("x", "y", "z", "r1", "r2", "r3", "r4", "r5", "r6")]
                for gripper in (config.donor_gripper, config.receiver_gripper)
            ]
            self.profiles = np.array([[p.donor, p.receiver] for p in config.pose_pairs])
            self.rotations = np.array([
                [_rotation_6d_matrix(pose[3:]) for pose in pair] for pair in self.profiles
            ])

    @staticmethod
    def _angles(first, second):
        return np.degrees(np.arccos(np.clip(
            (np.sum(first * second, axis=(-2, -1)) - 1) / 2, -1, 1
        )))

    def apply(self, action, observation, *, now=None):
        if self.config is None:
            return action
        cfg = self.config
        feedback = np.array([observation[cfg.donor_gripper], observation[cfg.receiver_gripper]], dtype=float)
        if not np.isfinite(feedback).all() or np.any((feedback < 0) | (feedback > 1)):
            raise ValueError("Handover release requires finite normalized gripper feedback.")
        if self.phase == "released":
            return {**action, cfg.donor_gripper: 0.0}
        if self.phase == "opening":
            result = {**action, **self.held_action, cfg.donor_gripper: 0.0}
            if feedback[0] <= 0.05:
                self.phase = "released"
            return result

        now = time.monotonic() if now is None else float(now)
        if not np.isfinite(now) or (self.last_time is not None and now <= self.last_time):
            raise ValueError("Handover release requires a strictly increasing clock.")
        if self.last_time is not None and now - self.last_time > cfg.max_observation_gap_s:
            self.anchor = self.since = None
            self.matching_profile = None
        self.last_time = now
        poses = np.array([[observation[key] for key in keys] for keys in self.keys], dtype=float)
        if not np.isfinite(poses).all():
            raise ValueError("Handover release requires finite Cartesian feedback.")
        rotations = np.array([_rotation_6d_matrix(pose[3:]) for pose in poses])
        distance = np.linalg.norm(poses[None, :, :3] - self.profiles[:, :, :3], axis=2)
        angles = self._angles(rotations[None], self.rotations)
        relative_error = np.linalg.norm(
            (poses[0, :3] - poses[1, :3])
            - (self.profiles[:, 0, :3] - self.profiles[:, 1, :3]), axis=1,
        )
        # Both arms must match the SAME demonstration. Enter using the strict
        # band; retain that same match within a slightly larger exit band.
        # This must not turn borderline samples into repeated new arrivals.
        score = np.maximum.reduce([
            np.max(distance, axis=1) / cfg.position_tolerance,
            np.max(angles, axis=1) / cfg.rotation_tolerance,
            relative_error / cfg.relative_position_tolerance,
        ])
        if self.matching_profile is not None and score[self.matching_profile] > cfg.match_exit_scale:
            self.matching_profile = None
            self.anchor = self.since = None
        if self.matching_profile is None and np.min(score) <= 1:
            self.matching_profile = int(np.argmin(score))
        commands = np.array([action[cfg.donor_gripper], action[cfg.receiver_gripper]], dtype=float)
        if not np.isfinite(commands).all():
            raise ValueError("Handover release requires finite gripper commands.")
        if (
            self.matching_profile is None
            or np.any(feedback < self.closed_feedback)
            or np.any(commands < self.closed_feedback)
        ):
            self.anchor = self.since = None
            self.matching_profile = None
            return action
        if self.anchor is not None:
            anchor_poses, anchor_rotations = self.anchor
            if (
                np.any(np.linalg.norm(poses[:, :3] - anchor_poses[:, :3], axis=1) > cfg.settle_position_tolerance)
                or np.any(self._angles(rotations, anchor_rotations) > cfg.settle_rotation_tolerance)
            ):
                self.anchor = self.since = None
        if self.anchor is None:
            self.anchor = (poses.copy(), rotations.copy())
            self.since = now
        if now - self.since < cfg.settle_time_s:
            return action
        self.phase = "opening"
        self.held_action = {key: float(observation[key]) for keys in self.keys for key in keys}
        # Keep the receiver closed until donor feedback confirms opening.
        self.held_action[cfg.receiver_gripper] = float(commands[1])
        return {**action, **self.held_action, cfg.donor_gripper: 0.0}


class PolicyGripperCoordinator:
    """Reach a predicted grasp pose, close there, then resume arm predictions.

    This coordinates execution; it cannot identify an egg or correct a model
    that selects the wrong grasp pose. Each recording loop owns fresh state.
    """

    def __init__(self, config: PolicyMotionConfig | None):
        self.config = config
        self.targets: dict[str, dict] = {}
        self.arrival_frames: dict[str, int] = {}
        self.closing: set[str] = set()
        self.completed: set[str] = set()

    def apply(self, action, observation):
        if self.config is None or not self.config.synchronized_gripper_actions:
            return action
        result = dict(action)
        for key in self.config.synchronized_gripper_actions:
            if key not in action or key not in observation:
                raise ValueError(f"Gripper synchronization requires action and feedback for {key}.")
            command, feedback = float(action[key]), float(observation[key])
            if not np.isfinite([command, feedback]).all() or not 0 <= feedback <= 1:
                raise ValueError(f"Invalid gripper command or feedback for {key}.")
            if command < self.config.binary_gripper_threshold:
                if key in self.closing and feedback > 0.05:
                    # A policy reversal may reopen the gripper, but must not
                    # lift the arm while that reversal is still in progress.
                    result.update(self.targets[key])
                    result[key] = 0.0
                    continue
                self.targets.pop(key, None)
                self.arrival_frames.pop(key, None)
                self.closing.discard(key)
                self.completed.discard(key)
                continue
            if key in self.completed:
                continue
            if key not in self.targets and feedback >= self.config.gripper_closed_feedback:
                self.completed.add(key)
                continue
            prefix = key.removesuffix("gripper.pos")
            pose_keys = [prefix + "tcp." + axis for axis in ("x", "y", "z", "r1", "r2", "r3", "r4", "r5", "r6")]
            if any(k not in action or k not in observation for k in pose_keys):
                raise ValueError(f"Gripper synchronization requires Cartesian pose and feedback for {key}.")
            requested = np.array([action[k] for k in pose_keys], dtype=float)
            present = np.array([observation[k] for k in pose_keys], dtype=float)
            if not np.isfinite(requested).all() or not np.isfinite(present).all():
                raise ValueError("Non-finite TCP pose in gripper synchronization.")
            goal = self.targets.setdefault(key, dict(zip(pose_keys, map(float, requested), strict=True)))
            target = np.array([goal[k] for k in pose_keys])
            rotation = _rotation_6d_matrix(present[3:]).T @ _rotation_6d_matrix(target[3:])
            angle = np.degrees(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1)))
            arrived = (
                np.linalg.norm(present[:3] - target[:3]) <= self.config.gripper_position_tolerance
                and angle <= self.config.gripper_rotation_tolerance
            )
            self.arrival_frames[key] = self.arrival_frames.get(key, 0) + 1 if arrived else 0
            if self.arrival_frames[key] >= 2:
                self.closing.add(key)
            if key in self.closing and arrived and feedback >= self.config.gripper_closed_feedback:
                self.completed.add(key)
                self.targets.pop(key, None)
                self.arrival_frames.pop(key, None)
                self.closing.discard(key)
                continue
            result.update(goal)
            result[key] = command if key in self.closing else 0.0
        return result


class JointReferenceLimiter:
    """Bound finite-difference reference velocity/acceleration, including startup.

    Braking starts before reaching a target, with one worst-case update period
    of margin. Target reversals decelerate instead of resetting to feedback.
    Slow inference never accumulates time that could cause a catch-up jump.
    These are reference constraints, not a model of the physical motor response.
    """

    def __init__(self, names, positions, joint_limits, native_velocity_limits, config, period, now):
        if not np.isfinite(period) or period <= 0:
            raise ValueError("Policy update period must be finite and positive.")
        self.names = tuple(names)
        self.lower = np.array([joint_limits[name][0] for name in names], dtype=float)
        self.upper = np.array([joint_limits[name][1] for name in names], dtype=float)
        self.position = np.clip(
            np.array([positions[name] for name in names], dtype=float), self.lower, self.upper
        )
        if not np.isfinite(self.position).all():
            raise ValueError("Policy motion requires finite initial joint feedback.")
        self.vmax = np.minimum(
            native_velocity_limits,
            [
                config.max_gripper_velocity if name == "gripper" else config.max_joint_velocity
                for name in names
            ],
        )
        self.amax = np.array(
            [
                config.max_gripper_acceleration if name == "gripper" else config.max_joint_acceleration
                for name in names
            ]
        )
        if not np.isfinite(self.vmax).all() or np.min(self.vmax) <= 0:
            raise ValueError("Native policy velocity limits must be finite and positive.")
        self.velocity = np.zeros(len(names))
        self.speed_cap = np.minimum(1.0, self.vmax)
        self.period = float(period)
        self.last_time = float(now)

    def step(self, desired, feedback, now):
        elapsed = float(now) - self.last_time
        if not np.isfinite(elapsed) or elapsed <= 0:
            raise ValueError("Policy motion requires a strictly increasing monotonic clock.")
        dt = min(elapsed, self.period)
        target = np.array([desired[name] for name in self.names], dtype=float)
        present = np.array([feedback[name] for name in self.names], dtype=float)
        if not np.isfinite(target).all() or not np.isfinite(present).all():
            raise ValueError("Non-finite policy joint target or feedback.")
        target = np.clip(target, self.lower, self.upper)
        error = target - self.position
        margin = self.amax * self.period
        braking_speed = np.sqrt(margin**2 + 2.0 * self.amax * abs(error)) - margin
        desired_velocity = np.sign(error) * np.minimum(self.vmax, braking_speed)
        change = self.amax * dt
        velocity = np.clip(desired_velocity, self.velocity - change, self.velocity + change)
        position = self.position + velocity * dt
        if np.any(position < self.lower - 1e-8) or np.any(position > self.upper + 1e-8):
            raise RuntimeError(
                "Policy reference cannot respect a joint limit; stopping instead of clipping its path."
            )
        # Motor caps must not remain at the former 150 deg/s for tiny reference
        # steps. Allow bounded correction for tracking lag, and ramp caps too.
        desired_cap = np.minimum(
            self.vmax, np.maximum(1.0, np.maximum(abs(velocity), abs(position - present) / 0.1))
        )
        self.speed_cap = np.clip(desired_cap, self.speed_cap - change, self.speed_cap + change)
        self.speed_cap = np.clip(self.speed_cap, np.minimum(1.0, self.vmax), self.vmax)
        self.position, self.velocity, self.last_time = position, velocity, float(now)
        return dict(zip(self.names, map(float, position), strict=True)), np.radians(self.speed_cap).tolist()


@contextmanager
def policy_motion_session(robot, config, fps):
    """Always release opt-in inference state on stop, failure or interruption."""
    if config is None:
        yield
        return
    robot.set_policy_motion_limits(config, period=1.0 / fps)
    try:
        yield
    finally:
        # Leave the last bounded reference held; do not command feedback back
        # over it or keep integrating toward a stale model destination.
        robot.set_policy_motion_limits(None)
