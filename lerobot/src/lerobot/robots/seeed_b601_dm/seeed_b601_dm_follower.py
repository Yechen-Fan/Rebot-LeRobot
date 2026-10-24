import logging
import math
import os
import time
from contextlib import suppress
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.motors import MotorCalibration
from lerobot.processor import RobotAction, RobotObservation
from lerobot.robots.robot import Robot
from lerobot.utils.errors import (
    DeviceAlreadyConnectedError,
    DeviceNotConnectedError,
    ReturnToInitialPositionError,
)
from lerobot.utils.policy_motion import JointReferenceLimiter, PolicyMotionConfig

from .config_seeed_b601_dm_follower import GripperType, SeeedB601DMFollowerConfig
from .flexitac import COLS, ROWS, SINGLE_TACTILE_NAMES, FlexiTacManager

logger = logging.getLogger(__name__)

KINEMATIC_MOTORS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_yaw",
    "wrist_roll",
]
TCP_POSE_KEYS = (
    "tcp.x",
    "tcp.y",
    "tcp.z",
    "tcp.r1",
    "tcp.r2",
    "tcp.r3",
    "tcp.r4",
    "tcp.r5",
    "tcp.r6",
)


def _silent_safe_goal_position(
    goal_present_pos: dict[str, tuple[float, float]],
    max_relative_target: float | dict[str, float],
) -> dict[str, float]:
    safe_goal_pos: dict[str, float] = {}
    for motor_name, (goal_pos, present_pos) in goal_present_pos.items():
        max_delta = (
            float(max_relative_target[motor_name])
            if isinstance(max_relative_target, dict)
            else float(max_relative_target)
        )
        delta = float(goal_pos) - float(present_pos)
        safe_goal_pos[motor_name] = float(present_pos) + max(-max_delta, min(max_delta, delta))
    return safe_goal_pos

DAMIAO_STATUS_NAMES = {
    0x0: "DISABLED",
    0x1: "ENABLED",
    0x8: "OVER_VOLTAGE",
    0x9: "UNDER_VOLTAGE",
    0xA: "OVER_CURRENT",
    0xB: "MOS_OVER_TEMP",
    0xC: "ROTOR_OVER_TEMP",
    0xD: "LOST_COMM",
    0xE: "OVERLOAD",
    0xFF: "NO_FEEDBACK",
}


class SeeedB601DMFollower(Robot):
    """Seeed B601 follower backed by rebotarm_control_dm's Rust actuator loop."""

    config_class = SeeedB601DMFollowerConfig
    name = "seeed_b601_follower"

    def __init__(self, config: SeeedB601DMFollowerConfig):
        super().__init__(config)
        self.config = config
        self.arm = None
        self._calibration_only_connection = False
        self.serial_gripper = None
        self.cameras = make_cameras_from_configs(config.cameras)
        tactile_ports = {
            "inner": config.flexitac_inner_port,
            "outer": config.flexitac_outer_port,
        }
        self.flexitac = (
            FlexiTacManager(
                tactile_ports,
                baudrate=config.flexitac_baudrate,
                timeout_s=config.flexitac_serial_timeout_s,
                max_frame_age_s=config.flexitac_max_frame_age_s,
                auto_zero=config.flexitac_auto_zero,
            )
            if all(tactile_ports.values())
            else None
        )
        self._uses_serial_gripper = bool(
            config.control_gripper and config.gripper_type == GripperType.SERIAL
        )
        self._uses_builtin_gripper = bool(
            config.control_gripper and config.gripper_type == GripperType.REBOTARMB601
        )
        self.motor_names = [
            name
            for name in config.motor_can_ids
            if self._uses_builtin_gripper or name != "gripper"
        ]
        self._generated_cfg_path: Path | None = None
        self._initial_positions_deg: dict[str, float] = {}
        self._last_goal_deg: dict[str, float] = {}
        self._policy_motion_limiter: JointReferenceLimiter | None = None
        self._teleop_motion_enabled: bool | None = None
        self._last_teleop_tcp_target_z: float | None = None
        self._coordinated_tcp_step = False
        self._teleop_translation = False
        self._teleop_wrist_rotation = False
        self._teleop_locked_proximal_deg: dict[str, float] = {}
        self._release_from: dict[str, float] = {}
        self._release_to: dict[str, float] = {}
        self._release_start_s = 0.0
        self._release_duration_s = 0.15
        self._last_positions_deg: dict[str, float] = {}
        self._last_velocities_deg: dict[str, float] = {}
        self._last_torques: dict[str, float] = {}
        self._last_state_read_s: float | None = None
        self._translation_reference_deg: dict[str, float] = {}
        self._translation_reference_s: float | None = None
        self._translation_brake_deg: dict[str, float] = {}
        self._last_valid_feedback_s: dict[str, float] = {}
        self._last_feedback_warning_s = 0.0
        self._last_ik_accepted = True
        self.last_return_to_initial_succeeded: bool | None = None
        self._kinematic_model = None
        self._kinematic_frame_id: int | None = None
        self._validate_config()

    def _validate_config(self) -> None:
        ids = set(self.config.motor_can_ids)
        required_maps = {
            "motor_models": set(self.config.motor_models),
            "joint_limits": set(self.config.joint_limits),
            "mit_gains": set(self.config.mit_gains),
            "pos_vel_gains": set(self.config.pos_vel_gains),
        }
        for field_name, keys in required_maps.items():
            missing = ids - keys
            if missing:
                raise ValueError(f"{field_name} missing keys: {sorted(missing)}")
        for motor_name in ids:
            mit_gains = self.config.mit_gains[motor_name]
            if len(mit_gains) != 2:
                raise ValueError(f"mit_gains[{motor_name!r}] must contain (kp, kd).")
            if any(float(value) < 0.0 for value in mit_gains):
                raise ValueError(f"mit_gains[{motor_name!r}] values must be >= 0.")

            gains = self.config.pos_vel_gains[motor_name]
            if len(gains) != 4:
                raise ValueError(
                    f"pos_vel_gains[{motor_name!r}] must contain "
                    "(vel_kp, vel_ki, pos_kp, pos_ki)."
                )
            if any(float(value) < 0.0 for value in gains):
                raise ValueError(f"pos_vel_gains[{motor_name!r}] values must be >= 0.")
        mode = self.config.control_mode.lower()
        if mode not in {"pos_vel", "mit"}:
            raise ValueError("control_mode must be 'pos_vel' or 'mit'.")
        action_mode = self.config.action_mode.lower()
        if action_mode not in {"joint", "cartesian"}:
            raise ValueError("action_mode must be 'joint' or 'cartesian'.")
        if isinstance(self.config.gripper_type, str):
            self.config.gripper_type = GripperType(self.config.gripper_type)
        if self.config.can_adapter not in {"damiao", "socketcan", "robstride"}:
            raise ValueError("can_adapter must be 'damiao', 'socketcan', or 'robstride'.")
        if isinstance(self.config.pos_vel_velocity, list) and len(self.config.pos_vel_velocity) not in {
            len(self.motor_names),
            len(self.config.motor_can_ids),
        }:
            raise ValueError(
                "pos_vel_velocity list length must match the controlled joint count or the full motor count."
            )
        self._return_to_initial_vlim_rad()
        if self.config.rt_command_gap_us < 0:
            raise ValueError("rt_command_gap_us must be >= 0.")
        if self.config.damiao_tx_debug < 0:
            raise ValueError("damiao_tx_debug must be >= 0.")
        if self.config.ik_accept_error < 0:
            raise ValueError("ik_accept_error must be >= 0.")
        if self.config.ik_reject_error < self.config.ik_accept_error:
            raise ValueError("ik_reject_error must be >= ik_accept_error.")
        self._ik_joint_step_limits_deg()
        if not self.config.ik_position_backoff_scales:
            raise ValueError("ik_position_backoff_scales must not be empty.")
        if any(not 0.0 < float(scale) < 1.0 for scale in self.config.ik_position_backoff_scales):
            raise ValueError("ik_position_backoff_scales values must be between 0 and 1.")
        if self.config.runtime_gripper_feedback_grace_s < 0:
            raise ValueError("runtime_gripper_feedback_grace_s must be >= 0.")
        if (
            self._uses_builtin_gripper
            and not 0.0 <= self.config.gripper_force_pos_torque_ratio <= 1.0
        ):
            raise ValueError("gripper_force_pos_torque_ratio must be in [0, 1].")
    @property
    def _action_motors_ft(self) -> dict[str, type]:
        features = {f"{motor}.pos": float for motor in self.motor_names}
        if self._uses_serial_gripper:
            features["gripper.pos"] = float
        return features

    @property
    def _action_tcp_ft(self) -> dict[str, type]:
        features = dict.fromkeys(TCP_POSE_KEYS, float)
        if self.config.control_gripper:
            features["gripper.pos"] = float
        return features

    def _observation_motor_key(self, motor_name: str) -> str:
        if motor_name not in KINEMATIC_MOTORS:
            return motor_name
        return f"joint_{KINEMATIC_MOTORS.index(motor_name) + 1}"

    @property
    def _observation_motors_ft(self) -> dict[str, type]:
        features: dict[str, type] = {}
        if self._uses_serial_gripper:
            features["gripper.pos"] = float
        for motor in self.motor_names:
            obs_key = self._observation_motor_key(motor)
            if motor == "gripper":
                features[f"{obs_key}.pos"] = float
                if self.config.enable_observation_gripper_vel:
                    features[f"{obs_key}.vel"] = float
                if self.config.enable_observation_gripper_torque:
                    features[f"{obs_key}.torque"] = float
                continue

            if self.config.enable_observation_joint_pos:
                features[f"{obs_key}.pos"] = float
            if self.config.enable_observation_joint_vel:
                features[f"{obs_key}.vel"] = float
            if self.config.enable_observation_joint_torque:
                features[f"{obs_key}.torque"] = float
        return features

    @property
    def _observation_tcp_ft(self) -> dict[str, type]:
        if self.config.action_mode.lower() != "cartesian":
            return {}
        return dict.fromkeys(TCP_POSE_KEYS, float)

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
            for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        tactile_features = (
            {
                f"tactile.{name}": PolicyFeature(FeatureType.TACTILE, (ROWS, COLS))
                for name in SINGLE_TACTILE_NAMES
            }
            | {
                f"{kind}.{name}": PolicyFeature(FeatureType.AUXILIARY, (ROWS, COLS))
                for kind in ("tactile_raw", "tactile_raw_baseline")
                for name in SINGLE_TACTILE_NAMES
            }
            | {
                f"tactile_baseline.{name}": PolicyFeature(
                    FeatureType.AUXILIARY, (ROWS, COLS)
                )
                for name in SINGLE_TACTILE_NAMES
            }
            if self.flexitac
            else {}
        )
        return {
            **self._observation_motors_ft,
            **self._observation_tcp_ft,
            **self._cameras_ft,
            **tactile_features,
        }

    @cached_property
    def action_features(self) -> dict[str, type]:
        if self.config.action_mode.lower() == "cartesian":
            return self._action_tcp_ft
        return self._action_motors_ft

    @property
    def is_connected(self) -> bool:
        serial_gripper_connected = (
            not self._uses_serial_gripper
            or (self.serial_gripper is not None and self.serial_gripper.is_connected)
        )
        return (
            self.arm is not None
            and serial_gripper_connected
            and all(cam.is_connected for cam in self.cameras.values())
        )

    @property
    def is_calibrated(self) -> bool:
        return bool(self.calibration)

    @property
    def last_ik_accepted(self) -> bool:
        return self._last_ik_accepted

    def calibrate(self) -> None:
        if self.arm is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self.calibration:
            user_input = input(
                f"Press ENTER to keep the calibration associated with id {self.id}, "
                "or type 'c' and press ENTER to recalibrate: "
            )
            if user_input.strip().lower() != "c":
                logger.info("Keeping calibration associated with id %s.", self.id)
                return

        logger.info("Running full 7-axis calibration for %s.", self)
        self.arm.disable(retries=10)
        print(
            "\nCalibration: Set Zero Position\n"
            "Support the arm, then MANUALLY move all six joints to the B601 ZERO POSITION "
            "and fully CLOSE the built-in gripper.\n"
            "All motor torque is disabled. Keep clear of pinch points.\n"
        )
        input("Press ENTER only when all 7 axes are at their zero positions...")

        self.arm.set_zero()
        time.sleep(0.5)
        positions_deg = self._read_positions_deg_blocking()
        not_zero = {name: value for name, value in positions_deg.items() if abs(value) > 5.0}
        if not_zero:
            raise RuntimeError(f"B601 zero verification failed: {not_zero}")

        self.calibration = {
            motor_name: MotorCalibration(
                id=int(self.config.motor_can_ids[motor_name][0]),
                drive_mode=0,
                homing_offset=0,
                range_min=int(round(self.config.joint_limits[motor_name][0])),
                range_max=int(round(self.config.joint_limits[motor_name][1])),
            )
            for motor_name in self.motor_names
        }
        self._save_calibration()
        logger.info("Full B601 calibration saved to %s.", self.calibration_fpath)

    def _default_vendor(self, motor_name: str) -> str:
        if motor_name in self.config.motor_vendors:
            return self.config.motor_vendors[motor_name].lower()
        if self.config.motor_vendor:
            return self.config.motor_vendor.lower()
        if self.config.can_adapter == "robstride":
            return "robstride"
        return "damiao"

    def _velocity_limits_rad(self) -> list[float]:
        velocity = self.config.pos_vel_velocity
        if isinstance(velocity, list):
            if len(velocity) == len(self.motor_names):
                values = velocity
            else:
                all_motor_names = list(self.config.motor_can_ids)
                values = [velocity[all_motor_names.index(motor_name)] for motor_name in self.motor_names]
        else:
            values = [velocity] * len(self.motor_names)
        return [math.radians(float(v)) for v in values]

    def _ik_joint_step_limits_deg(self) -> dict[str, float]:
        spec = self.config.max_ik_joint_step_deg
        if isinstance(spec, dict):
            missing = [name for name in KINEMATIC_MOTORS if name not in spec]
            if missing:
                raise ValueError(f"max_ik_joint_step_deg missing keys: {missing}")
            values = {name: float(spec[name]) for name in KINEMATIC_MOTORS}
        elif isinstance(spec, (list, tuple)):
            if len(spec) != len(KINEMATIC_MOTORS):
                raise ValueError("max_ik_joint_step_deg list length must match the 6 kinematic joints.")
            values = {name: float(spec[idx]) for idx, name in enumerate(KINEMATIC_MOTORS)}
        else:
            values = {name: float(spec) for name in KINEMATIC_MOTORS}

        if any(value <= 0.0 for value in values.values()):
            raise ValueError("max_ik_joint_step_deg values must be > 0.")
        return values

    @staticmethod
    def _rotation_6d_to_matrix(action: RobotAction) -> np.ndarray:
        a1 = np.array([action["tcp.r1"], action["tcp.r2"], action["tcp.r3"]], dtype=float)
        a2 = np.array([action["tcp.r4"], action["tcp.r5"], action["tcp.r6"]], dtype=float)
        b1_norm = np.linalg.norm(a1)
        if b1_norm < 1e-9:
            raise ValueError("Invalid tcp.r1-r3 rotation vector from Quest3 action.")
        b1 = a1 / b1_norm

        a2_orth = a2 - np.dot(b1, a2) * b1
        b2_norm = np.linalg.norm(a2_orth)
        if b2_norm < 1e-9:
            raise ValueError("Invalid tcp.r4-r6 rotation vector from Quest3 action.")
        b2 = a2_orth / b2_norm
        b3 = np.cross(b1, b2)
        return np.column_stack([b1, b2, b3])

    @staticmethod
    def _matrix_to_rotation_6d(rot: np.ndarray) -> np.ndarray:
        return np.array(
            [
                rot[0, 0],
                rot[1, 0],
                rot[2, 0],
                rot[0, 1],
                rot[1, 1],
                rot[2, 1],
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _matrix_to_quaternion_wxyz(rot: np.ndarray) -> np.ndarray:
        trace = float(np.trace(rot))
        if trace > 0:
            s = 0.5 / math.sqrt(trace + 1.0)
            qw = 0.25 / s
            qx = (rot[2, 1] - rot[1, 2]) * s
            qy = (rot[0, 2] - rot[2, 0]) * s
            qz = (rot[1, 0] - rot[0, 1]) * s
        elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
            s = 2.0 * math.sqrt(max(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2], 0.0))
            qw = (rot[2, 1] - rot[1, 2]) / s
            qx = 0.25 * s
            qy = (rot[0, 1] + rot[1, 0]) / s
            qz = (rot[0, 2] + rot[2, 0]) / s
        elif rot[1, 1] > rot[2, 2]:
            s = 2.0 * math.sqrt(max(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2], 0.0))
            qw = (rot[0, 2] - rot[2, 0]) / s
            qx = (rot[0, 1] + rot[1, 0]) / s
            qy = 0.25 * s
            qz = (rot[1, 2] + rot[2, 1]) / s
        else:
            s = 2.0 * math.sqrt(max(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1], 0.0))
            qw = (rot[1, 0] - rot[0, 1]) / s
            qx = (rot[0, 2] + rot[2, 0]) / s
            qy = (rot[1, 2] + rot[2, 1]) / s
            qz = 0.25 * s

        quat = np.array([qw, qx, qy, qz], dtype=np.float32)
        norm = np.linalg.norm(quat)
        if norm < 1e-9:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        return quat / norm

    def _kinematic_joint_rad(self, positions_deg: dict[str, float]) -> np.ndarray:
        return np.array([math.radians(float(positions_deg[name])) for name in KINEMATIC_MOTORS], dtype=float)

    def _load_kinematic_model(self):
        if self._kinematic_model is None:
            from rebotarm_control_dm.kinematics import load_robot_model

            self._kinematic_model = load_robot_model(self.config.kinematic_urdf_path)
            self._kinematic_frame_id = self._kinematic_model.end_effector_frame_id()
        return self._kinematic_model

    def _tcp_pose_matrix_from_positions(self, positions_deg: dict[str, float]) -> np.ndarray:
        model = self._load_kinematic_model()
        _, _, end_pose = model.fk(self._kinematic_joint_rad(positions_deg), "")
        return np.asarray(end_pose, dtype=float)

    def _gripper_pos_to_norm(self, gripper_pos_deg: float) -> float:
        open_deg, closed_deg = self.config.joint_limits["gripper"]
        span = closed_deg - open_deg
        if abs(span) < 1e-9:
            return 1.0
        gripper_norm = (float(gripper_pos_deg) - open_deg) / span
        return max(0.0, min(1.0, gripper_norm))

    def _current_gripper_norm(self) -> float:
        if self._uses_serial_gripper:
            if self.serial_gripper is None:
                return 1.0
            # LeRobot/B601 convention: 0=open, 1=closed.
            # SerialGripper convention: 0=closed, 1=open.
            return 1.0 - float(self.serial_gripper.get_gripper_position())
        if "gripper" in self._last_positions_deg:
            return self._gripper_pos_to_norm(self._last_positions_deg["gripper"])
        return 1.0

    def get_current_tcp_pose_quat(self) -> np.ndarray:
        """Return current TCP pose as [x, y, z, qw, qx, qy, qz, gripper_norm]."""
        if self.arm is None or not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if not self._last_positions_deg:
            positions_deg, _, _ = self._read_state_deg(request=False)
        else:
            positions_deg = dict(self._last_positions_deg)

        tcp_pose = self._tcp_pose_matrix_from_positions(positions_deg)
        quat = self._matrix_to_quaternion_wxyz(tcp_pose[:3, :3])

        gripper_norm = self._current_gripper_norm()

        return np.array(
            [
                float(tcp_pose[0, 3]),
                float(tcp_pose[1, 3]),
                float(tcp_pose[2, 3]),
                float(quat[0]),
                float(quat[1]),
                float(quat[2]),
                float(quat[3]),
                float(gripper_norm),
            ],
            dtype=np.float32,
        )

    def set_teleop_wrist_rotation(self, enabled: bool) -> None:
        if enabled and not self._teleop_wrist_rotation:
            self._teleop_locked_proximal_deg = {
                name: self._last_positions_deg[name] for name in KINEMATIC_MOTORS[:3]
            }
        elif not enabled:
            self._teleop_locked_proximal_deg = {}
        self._teleop_wrist_rotation = enabled

    def _solve_wrist_rotation(self, target: np.ndarray, seed: np.ndarray):
        """Advance along a bounded rotation using joints 4--6 only."""
        model = self._load_kinematic_model()
        limits = self._ik_joint_step_limits_deg()
        q = seed.copy()
        for i, name in enumerate(KINEMATIC_MOTORS[:3]):
            q[i] = math.radians(self._teleop_locked_proximal_deg[name])
        physical = np.radians([self.config.joint_limits[name] for name in KINEMATIC_MOTORS[3:]])
        # Encoder quantization can put feedback just outside a soft limit.
        q[3:] = np.clip(q[3:], physical[:, 0], physical[:, 1])
        bounds = physical.copy()
        for i, name in enumerate(KINEMATIC_MOTORS[3:]):
            step = limits[name]
            relative = self.config.max_relative_target
            if relative is not None:
                step = min(step, relative[name] if isinstance(relative, dict) else relative)
            bounds[i, 0] = min(physical[i, 1], max(bounds[i, 0], seed[i + 3] - math.radians(step)))
            bounds[i, 1] = max(physical[i, 0], min(bounds[i, 1], seed[i + 3] + math.radians(step)))

        def joint_step(jacobian, error):
            # Keep the existing damping away from singularities, and avoid
            # large cancelling wrist motions when two wrist axes align.
            smallest = np.linalg.svd(jacobian, compute_uv=False)[-1]
            damping = 1e-5 + max(0.0, 0.05 - smallest) ** 2
            return jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + damping * np.eye(3), error)

        def step_fraction(position, delta):
            fraction = 1.0
            for i, value in enumerate(delta):
                if abs(value) > 1e-12:
                    bound = bounds[i, 1 if value > 0 else 0]
                    fraction = min(fraction, max(0.0, (bound - position[i]) / value))
            return fraction

        pose = np.asarray(model.fk(q, "")[2])
        error = Rotation.from_matrix(pose[:3, :3].T @ target[:3, :3]).as_rotvec()
        jacobian = np.asarray(model.frame_jacobian(q, self._kinematic_frame_id))[3:, 3:6]
        delta = joint_step(jacobian, error)
        blocked = ((q[3:] <= physical[:, 0] + 1e-8) & (delta < 0)) | (
            (q[3:] >= physical[:, 1] - 1e-8) & (delta > 0)
        )
        delta[blocked] = 0.0
        # Shorten the ROTATION before solving it. Clipping each joint against
        # the full target changes the rotation axis and lets the other joints
        # keep compensating for a joint which has already exhausted its step.
        error_step = (jacobian @ delta if np.any(blocked) else error) * step_fraction(q[3:], delta)
        step_target = pose[:3, :3] @ Rotation.from_rotvec(error_step).as_matrix()
        for _ in range(20):
            pose = np.asarray(model.fk(q, "")[2])
            error = Rotation.from_matrix(pose[:3, :3].T @ step_target).as_rotvec()
            if np.linalg.norm(error) < 1e-4:
                break
            jacobian = np.asarray(model.frame_jacobian(q, self._kinematic_frame_id))[3:, 3:6]
            delta = joint_step(jacobian, error)
            outward = ((q[3:] <= physical[:, 0] + 1e-8) & (delta < 0)) | (
                (q[3:] >= physical[:, 1] - 1e-8) & (delta > 0)
            )
            delta[outward] = 0.0
            fraction = step_fraction(q[3:], delta)
            if fraction <= 1e-9:
                break
            target_error = Rotation.from_matrix(pose[:3, :3].T @ target[:3, :3]).magnitude()
            improved = False
            for scale in (1.0, 0.5, 0.25, 0.125):
                trial = q.copy()
                trial[3:] += scale * fraction * delta
                trial_pose = np.asarray(model.fk(trial, "")[2])
                trial_error = Rotation.from_matrix(trial_pose[:3, :3].T @ step_target).magnitude()
                remaining = Rotation.from_matrix(trial_pose[:3, :3].T @ target[:3, :3]).magnitude()
                if trial_error < np.linalg.norm(error) - 1e-10 and remaining < target_error - 1e-10:
                    q = trial
                    improved = True
                    break
            if not improved:
                break
        return q

    def set_teleop_translation(self, enabled: bool) -> None:
        if not enabled:
            self._translation_reference_deg = {}
            self._translation_reference_s = None
            self._translation_brake_deg = {}
        self._teleop_translation = enabled

    def _solve_translation_position(self, target: np.ndarray, seed: np.ndarray) -> np.ndarray:
        """Relax attitude only when anchored Switch translation is unreachable."""
        model = self._load_kinematic_model()
        limits = np.radians([self.config.joint_limits[n] for n in KINEMATIC_MOTORS])
        q = seed.copy()
        for _ in range(16):
            pose = np.asarray(model.fk(q, "")[2])
            error = target[:3, 3] - pose[:3, 3]
            if np.linalg.norm(error) < 1e-4:
                break
            jacobian = pose[:3, :3] @ np.asarray(
                model.frame_jacobian(q, self._kinematic_frame_id)
            )[:3]
            delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + 1e-6 * np.eye(3), error)
            delta *= min(1., .1 / max(float(np.max(np.abs(delta))), 1e-12))
            improved = False
            for fraction in (1., .5, .25, .125):
                candidate = np.clip(q + fraction * delta, limits[:, 0], limits[:, 1])
                position = np.asarray(model.fk(candidate, "")[2])[:3, 3]
                if np.linalg.norm(target[:3, 3] - position) < np.linalg.norm(error) - 1e-10:
                    q = candidate
                    improved = True
                    break
            if not improved:
                break
        return q

    def tcp_action_to_joint_action(self, action: RobotAction) -> RobotAction:
        """Convert a Quest3 Cartesian TCP action to B601 joint-space action."""
        self._coordinated_tcp_step = False
        required = set(TCP_POSE_KEYS)
        missing = required - set(action)
        if missing:
            raise ValueError(f"Quest3 TCP action missing keys: {sorted(missing)}")

        if not self._last_positions_deg:
            self._read_state_deg(request=False)

        model = self._load_kinematic_model()
        target_tcp = np.eye(4, dtype=float)
        target_tcp[:3, :3] = self._rotation_6d_to_matrix(action)
        target_tcp[:3, 3] = [float(action["tcp.x"]), float(action["tcp.y"]), float(action["tcp.z"])]

        q_seed = self._kinematic_joint_rad(self._last_positions_deg)
        if self._teleop_wrist_rotation:
            result_q = self._solve_wrist_rotation(target_tcp, q_seed)
            self._last_ik_accepted = True
            joints = {f"{name}.pos": math.degrees(float(result_q[i]))
                      for i, name in enumerate(KINEMATIC_MOTORS)}
            joints.update({f"{name}.pos": value for name, value in self._teleop_locked_proximal_deg.items()})
            return self._add_gripper_action(joints, action)
        result = model.solve_ik(target_tcp, q_seed, self._kinematic_frame_id)
        result_q = np.asarray(result.q, dtype=float)
        result_error = float(result.error)
        backoff_scale = 1.0

        if (
            self._teleop_translation and self._teleop_motion_enabled is True
            and self.config.control_mode.lower() == "pos_vel"
            and not result.success and np.all(np.isfinite(result_q))
        ):
            # Keep the translation attitude anchor while it is reachable. At
            # the workspace edge, allow necessary attitude change rather than
            # letting the new anchor reduce the old lifting range.
            candidate_q = self._solve_translation_position(target_tcp, q_seed)
            candidate_error = float(np.linalg.norm(
                target_tcp[:3, 3] - np.asarray(model.fk(candidate_q, "")[2])[:3, 3]
            ))
            original_position_error = float(np.linalg.norm(
                target_tcp[:3, 3] - np.asarray(model.fk(result_q, "")[2])[:3, 3]
            ))
            if candidate_error < original_position_error - 1e-6:
                result_q, result_error = candidate_q, candidate_error

        if np.all(np.isfinite(result_q)) and result_error > float(self.config.ik_reject_error):
            current_tcp = self._tcp_pose_matrix_from_positions(self._last_positions_deg)
            target_pos = target_tcp[:3, 3].copy()
            current_pos = current_tcp[:3, 3].copy()
            for scale in self.config.ik_position_backoff_scales:
                candidate_tcp = target_tcp.copy()
                candidate_tcp[:3, 3] = current_pos + (target_pos - current_pos) * float(scale)
                candidate = model.solve_ik(candidate_tcp, q_seed, self._kinematic_frame_id)
                candidate_q = np.asarray(candidate.q, dtype=float)
                candidate_error = float(candidate.error)
                if np.all(np.isfinite(candidate_q)) and candidate_error <= float(self.config.ik_reject_error):
                    logger.debug(
                        "Quest3 IK backoff accepted: scale=%.2f success=%s error=%.5f iterations=%s.",
                        float(scale),
                        candidate.success,
                        candidate_error,
                        getattr(candidate, "iterations", None),
                    )
                    result = candidate
                    result_q = candidate_q
                    result_error = candidate_error
                    backoff_scale = float(scale)
                    break

        if not np.all(np.isfinite(result_q)) or result_error > float(self.config.ik_reject_error):
            self._last_ik_accepted = False
            logger.debug(
                "Quest3 IK rejected: success=%s error=%.5f iterations=%s; holding previous target.",
                result.success,
                result_error,
                getattr(result, "iterations", None),
            )
            return {
                f"{motor_name}.pos": self._last_goal_deg.get(
                    motor_name, self._last_positions_deg[motor_name]
                )
                for motor_name in KINEMATIC_MOTORS
            }
        if not result.success and result_error > float(self.config.ik_accept_error):
            logger.debug(
                "Quest3 IK best-effort accepted: success=%s error=%.5f iterations=%s.",
                result.success,
                result_error,
                getattr(result, "iterations", None),
            )

        self._last_ik_accepted = True

        joint_action: RobotAction = {}
        step_limits = self._ik_joint_step_limits_deg()
        if self.config.ik_reject_error > self.config.ik_accept_error and result_error > self.config.ik_accept_error:
            edge_ratio = (result_error - self.config.ik_accept_error) / (
                self.config.ik_reject_error - self.config.ik_accept_error
            )
            edge_scale = 1.0 - 0.65 * max(0.0, min(1.0, float(edge_ratio)))
        else:
            edge_scale = 1.0
        step_scale = min(backoff_scale, edge_scale)
        # Independently clipping the IK step destroys its joint coordination:
        # the height guard then repeatedly steers individual joints elsewhere.
        # Preserve the IK direction whenever a teleoperation step reaches a
        # bound, including horizontal motion and intentional descent. Leave
        # ordinary, unclipped steps on the existing path.
        coordinated_step = (
            (self._teleop_motion_enabled is True or self._policy_motion_limiter is not None)
            and self.config.control_mode.lower() == "pos_vel"
        )
        if coordinated_step:
            current = np.degrees(q_seed)
            candidate = np.array([
                self._clip(math.degrees(float(result_q[i])), self.config.joint_limits[name])
                for i, name in enumerate(KINEMATIC_MOTORS)
            ])
            delta = candidate - current
            fraction = 1.0
            for i, name in enumerate(KINEMATIC_MOTORS):
                cap = float(step_limits[name]) * step_scale
                relative = self.config.max_relative_target
                if relative is not None:
                    cap = min(cap, float(relative[name] if isinstance(relative, dict) else relative))
                if abs(delta[i]) > 1e-12:
                    fraction = min(fraction, cap / abs(delta[i]))
            if fraction < 1.0:
                self._coordinated_tcp_step = True
                return self._add_gripper_action({
                    f"{name}.pos": float(current[i] + fraction * delta[i])
                    for i, name in enumerate(KINEMATIC_MOTORS)
                }, action)
        for idx, motor_name in enumerate(KINEMATIC_MOTORS):
            candidate_deg = math.degrees(float(result_q[idx]))
            current_deg = float(self._last_positions_deg[motor_name])
            max_step = float(step_limits[motor_name]) * step_scale
            joint_action[f"{motor_name}.pos"] = current_deg + max(
                -max_step, min(max_step, candidate_deg - current_deg)
            )

        return self._add_gripper_action(joint_action, action)

    def _add_gripper_action(self, joint_action: RobotAction, action: RobotAction) -> RobotAction:
        if "gripper" in self.motor_names and "gripper.pos" in action:
            gripper_norm = max(0.0, min(1.0, float(action["gripper.pos"])))
            open_deg, closed_deg = self.config.joint_limits["gripper"]
            joint_action["gripper.pos"] = open_deg + gripper_norm * (closed_deg - open_deg)
        elif self._uses_serial_gripper and "gripper.pos" in action:
            joint_action["gripper.pos"] = max(0.0, min(1.0, float(action["gripper.pos"])))

        return joint_action

    def _rt_cfg_path(self) -> Path:
        if self.config.arm_cfg_path is not None:
            return Path(self.config.arm_cfg_path).expanduser()

        path = self.calibration_dir / f"{self.id or 'default'}_rt_arm.yaml"
        path.write_text(self._render_rt_arm_yaml(), encoding="utf-8")
        self._generated_cfg_path = path
        return path

    def _rt_channel(self) -> str:
        channel = str(self.config.port)
        if self.config.can_adapter != "damiao":
            return channel

        try:
            resolved = Path(channel).expanduser().resolve(strict=True)
        except OSError:
            return channel

        resolved_str = str(resolved)
        if resolved_str.startswith("/dev/tty"):
            return resolved_str
        return channel

    def _render_rt_arm_yaml(self) -> str:
        lines = [
            "# Generated by lerobot_robot_seeed_b601_dm.",
            "name: reBotArmB601DM",
            f"channel: {self._rt_channel()}",
            f"rate: {float(self.config.rt_rate)}",
            "",
            "joints:",
        ]
        vlim = self._velocity_limits_rad()
        for idx, motor_name in enumerate(self.motor_names):
            motor_id, feedback_id = self.config.motor_can_ids[motor_name]
            kp, kd = self.config.mit_gains[motor_name]
            vel_kp, vel_ki, pos_kp, pos_ki = self.config.pos_vel_gains[motor_name]
            lines.extend(
                [
                    f"  - name: {motor_name}",
                    f"    motor_id: 0x{int(motor_id):02X}",
                    f"    feedback_id: 0x{int(feedback_id):02X}",
                    f"    model: \"{self.config.motor_models[motor_name]}\"",
                    f"    vendor: \"{self._default_vendor(motor_name)}\"",
                    "    MIT:",
                    f"      kp: {float(kp)}",
                    f"      kd: {float(kd)}",
                    "    POS_VEL:",
                    f"      vel_kp: {float(vel_kp)}",
                    f"      vel_ki: {float(vel_ki)}",
                    f"      pos_kp: {float(pos_kp)}",
                    f"      pos_ki: {float(pos_ki)}",
                    f"      vlim: {float(vlim[idx])}",
                    "",
                ]
            )
        return "\n".join(lines)

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self._teleop_motion_enabled = None
        self.set_teleop_translation(False)
        self.set_teleop_wrist_rotation(False)
        self._last_teleop_tcp_target_z = None

        from rebotarm_control_dm.actuator import RobotArm

        cfg_path = self._rt_cfg_path()
        if self.config.damiao_tx_debug:
            tx_debug = int(self.config.damiao_tx_debug)
            os.environ["MOTORBRIDGE_DAMIAO_TX_DEBUG"] = str(tx_debug)
            logger.info("Configured MOTORBRIDGE_DAMIAO_TX_DEBUG=%s.", tx_debug)

        logger.info("Connecting arm on %s with rebotarm_control_dm config %s...", self.config.port, cfg_path)
        arm = RobotArm(str(cfg_path))
        try:
            arm.connect()
            self.arm = arm
            if self._uses_serial_gripper:
                from .serial_gripper import SerialGripper

                if self.config.serial_gripper is None:
                    raise ValueError(
                        "gripper_type=serial requires serial gripper config. "
                        "Set serial_gripper_sn or serial_gripper_port in the robot config."
                    )
                self.serial_gripper = SerialGripper(self.config.serial_gripper)
                self.serial_gripper.connect()
            if not self.cameras:
                logger.debug("No cameras configured; skipping camera connect.")
            else:
                logger.info("Connecting %d camera(s): %s...", len(self.cameras), ", ".join(self.cameras))
                for cam in self.cameras.values():
                    cam.connect()
                logger.info("All cameras connected.")
            if self.flexitac is not None:
                logger.info("Connecting and calibrating two single-arm FlexiTac sensors with no contact...")
                self.flexitac.connect(
                    calibration_frames=self.config.flexitac_calibration_frames,
                    calibration_timeout_s=self.config.flexitac_calibration_timeout_s,
                    calibration_max_span=self.config.flexitac_calibration_max_span,
                    calibration_mean_min=self.config.flexitac_calibration_mean_min,
                    calibration_mean_max=self.config.flexitac_calibration_mean_max,
                )
            if not calibrate:
                self._calibration_only_connection = True
                logger.info("%s connected with torque disabled for calibration.", self)
                return
            if not self.is_calibrated:
                logger.info("No calibration file found for id %s; calibration is required.", self.id)
                self.calibrate()
            self.configure()
        except Exception:
            with suppress(Exception):
                arm.disconnect()
            if self.serial_gripper is not None:
                with suppress(Exception):
                    self.serial_gripper.disconnect()
                self.serial_gripper = None
            self.arm = None
            for cam in self.cameras.values():
                try:
                    if cam.is_connected:
                        cam.disconnect()
                except Exception:
                    logger.debug("Failed to disconnect camera during cleanup.", exc_info=True)
            if self.flexitac is not None:
                self.flexitac.disconnect()
            raise

        logger.debug("%s connected.", self)

    def configure(self) -> None:
        if self.arm is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.arm.enable()
        current_positions_deg = self._read_positions_deg_blocking()
        self._clear_possible_mit_torque(current_positions_deg)

        mode = self.config.control_mode.lower()
        if mode == "pos_vel":
            if not self.arm.mode_pos_vel(vlim=self._velocity_limits_rad()):
                raise RuntimeError("One or more B601 joints failed to enter POS_VEL mode.")
            self._verify_pos_vel_wrist_gains()
            self._configure_gripper_force_pos_mode()
        else:
            kp, kd = self._mit_gains_lists()
            if not self.arm.mode_mit(kp=kp, kd=kd):
                raise RuntimeError("One or more B601 joints failed to enter MIT mode.")

        self._initial_positions_deg = dict(current_positions_deg)
        # Install a measured-position hold before re-enabling after mode
        # configuration. Never enable against an old/default position target.
        self._last_goal_deg = dict(self._initial_positions_deg)
        self._set_rt_target_deg(self._initial_positions_deg)
        self.arm.start_rt_loop(
            rate=float(self.config.rt_rate),
            rt_priority=int(self.config.rt_priority),
            cpu=self.config.rt_cpu,
            command_gap_us=int(self.config.rt_command_gap_us),
            request_feedback=False,
        )
        # enable() at the beginning is needed to read initial positions, but
        # it does not establish readiness after the subsequent mode changes
        # (the built-in gripper enters FORCE_POS last).
        self.arm.enable(retries=3, poll_interval=0.02)
        self._wait_for_enabled_feedback()
        self._apply_initial_gripper_pose(self._initial_positions_deg)
        self._last_goal_deg = dict(self._initial_positions_deg)
        self._set_rt_target_deg(self._initial_positions_deg)
        logger.debug(
            "Started RT target loop at %.1f Hz (command gap %s us).",
            float(self.config.rt_rate),
            int(self.config.rt_command_gap_us),
        )

    def _wait_for_enabled_feedback(self, timeout_s: float = 0.5) -> None:
        """Require two valid startup samples; never waive disabled/fault states."""
        deadline = time.monotonic() + timeout_s
        valid_samples = 0
        last_error = None
        while True:
            try:
                self._read_state_deg(request=True)
                valid_samples += 1
                if valid_samples >= 2:
                    return
            except RuntimeError as error:
                valid_samples = 0
                last_error = error
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"{self}: motors did not become ready after mode configuration: {last_error}"
                ) from last_error
            time.sleep(0.02)

    def _read_positions_deg_blocking(self, attempts: int = 60) -> dict[str, float]:
        if self.arm is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if not hasattr(self.arm, "get_positions_blocking"):
            raise RuntimeError(
                "rebotarm_control_dm is missing get_positions_blocking(); reinstall/rebuild it before "
                "running the B601 RT follower safely."
            )

        positions = np.asarray(self.arm.get_positions_blocking(attempts=attempts), dtype=float)

        if positions.size != len(self.motor_names):
            raise RuntimeError(
                f"Expected {len(self.motor_names)} joint positions, got {positions.size}; "
                "refusing to build an initial RT target."
            )

        pos_deg = {
            motor_name: math.degrees(float(positions[idx]))
            for idx, motor_name in enumerate(self.motor_names)
        }
        self._last_positions_deg = dict(pos_deg)
        now = time.monotonic()
        self._last_velocities_deg = dict.fromkeys(self.motor_names, 0.0)
        self._last_torques = dict.fromkeys(self.motor_names, 0.0)
        self._last_valid_feedback_s = dict.fromkeys(self.motor_names, now)
        return pos_deg

    def _clear_possible_mit_torque(self, positions_deg: dict[str, float], frames: int = 5, dt_s: float = 0.02) -> None:
        if self.arm is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        pos_rad = [math.radians(positions_deg[motor_name]) for motor_name in self.motor_names]
        zeros = [0.0] * len(self.motor_names)
        try:
            self.arm.mode_mit(kp=zeros, kd=zeros, stabilize_delay=0.0)
            for _ in range(frames):
                self.arm.mit(
                    pos=pos_rad,
                    vel=zeros,
                    kp=zeros,
                    kd=zeros,
                    tau=zeros,
                    request_feedback=False,
                )
                time.sleep(dt_s)
        except Exception:
            logger.warning("Failed to clear possible MIT torque before taking control.", exc_info=True)
            raise

    def _verify_pos_vel_wrist_gains(self) -> None:
        if self.config.arm_cfg_path is not None:
            # A prebuilt actuator file, rather than these config fields,
            # owns its PID parameters.
            return
        # The SDK's mode_pos_vel() does not propagate individual gain-write
        # errors. Check the actuator before starting RT/teleoperation so a
        # missed write cannot silently leave the previous J4 response active.
        reader = getattr(self.arm, "get_damiao_register_f32", None)
        if not callable(reader):
            raise RuntimeError("rebotarm_control_dm needs get_damiao_register_f32(); rebuild the RT package.")
        expected = self.config.pos_vel_gains["wrist_flex"]
        actual = tuple(float(reader("wrist_flex", rid, timeout_ms=200)) for rid in (25, 26, 27, 28))
        if not all(math.isclose(a, float(e), rel_tol=1e-5, abs_tol=1e-8) for a, e in zip(actual, expected)):
            raise RuntimeError(f"J4 POS_VEL gains were not applied: expected {expected}, read back {actual}.")

    def _configure_gripper_force_pos_mode(self) -> None:
        if (
            self.config.control_mode.lower() != "pos_vel"
            or not self._uses_builtin_gripper
            or not self.config.enabled_gripper_force
            or "gripper" not in self.motor_names
        ):
            return

        try:
            self.arm.ensure_mode("gripper", 4, timeout_ms=1000)
        except Exception:
            logger.exception("Failed to switch gripper to FORCE_POS mode.")
            raise
        logger.debug(
            "gripper ensure mode FORCE_POS with torque ratio %.3f.",
            float(self.config.gripper_force_pos_torque_ratio),
        )

    def _read_state_deg(
        self, request: bool, *, allow_transient_gripper: bool = False
    ) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
        if self.arm is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        pos_rad, vel_rad, torque = self.arm.get_state(request=request)
        state_read_s = time.monotonic()
        positions = np.asarray(pos_rad, dtype=float)
        velocities = np.asarray(vel_rad, dtype=float)
        torques = np.asarray(torque, dtype=float)
        if not hasattr(self.arm, "get_status_codes"):
            raise RuntimeError(
                "rebotarm_control_dm is missing get_status_codes(); rebuild the RT package."
            )
        # get_state already requested and polled every motor. Read the same
        # cached status instead of issuing a second round of CAN requests.
        statuses = np.asarray(self.arm.get_status_codes(request=False), dtype=np.uint8)
        expected = len(self.motor_names)
        if (
            positions.size != expected
            or velocities.size != expected
            or torques.size != expected
            or statuses.size != expected
        ):
            raise RuntimeError(
                f"Invalid B601 feedback sizes: pos={positions.size}, vel={velocities.size}, "
                f"torque={torques.size}, status={statuses.size}, expected={expected}."
            )
        now = time.monotonic()
        invalid = []
        for idx, motor_name in enumerate(self.motor_names):
            valid = bool(
                statuses[idx] == 1
                and np.isfinite(positions[idx])
                and np.isfinite(velocities[idx])
                and np.isfinite(torques[idx])
            )
            if valid:
                self._last_valid_feedback_s[motor_name] = now
                continue

            can_reuse_gripper = bool(
                allow_transient_gripper
                and motor_name == "gripper"
                and motor_name in self._last_positions_deg
                and now - self._last_valid_feedback_s.get(motor_name, float("-inf"))
                <= float(self.config.runtime_gripper_feedback_grace_s)
            )
            if can_reuse_gripper:
                positions[idx] = math.radians(self._last_positions_deg[motor_name])
                velocities[idx] = math.radians(self._last_velocities_deg.get(motor_name, 0.0))
                torques[idx] = self._last_torques.get(motor_name, 0.0)
                if now - self._last_feedback_warning_s >= 1.0:
                    status = int(statuses[idx])
                    logger.warning(
                        "Transient gripper feedback status=%s (%s); holding last valid feedback for %.2fs.",
                        status,
                        DAMIAO_STATUS_NAMES.get(status, "UNKNOWN"),
                        float(self.config.runtime_gripper_feedback_grace_s),
                    )
                    self._last_feedback_warning_s = now
                continue

            status = int(statuses[idx])
            invalid.append(
                f"{motor_name}=status {status} ({DAMIAO_STATUS_NAMES.get(status, 'UNKNOWN')})"
            )
        if invalid:
            raise RuntimeError(f"B601 feedback invalid or motor disabled: {', '.join(invalid)}")

        pos_deg = {
            motor_name: math.degrees(float(positions[idx]))
            for idx, motor_name in enumerate(self.motor_names)
        }
        vel_deg = {
            motor_name: math.degrees(float(velocities[idx]))
            for idx, motor_name in enumerate(self.motor_names)
        }
        torq = {motor_name: float(torques[idx]) for idx, motor_name in enumerate(self.motor_names)}
        self._last_positions_deg = pos_deg
        self._last_velocities_deg = vel_deg
        self._last_torques = torq
        self._last_state_read_s = state_read_s
        return pos_deg, vel_deg, torq

    def get_observation(self) -> RobotObservation:
        start = time.perf_counter()

        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Sample external sensors before the joints used by the control loop.
        # A slow camera must not make us solve IK from pre-camera feedback.
        obs_dict: dict[str, Any] = {
            cam_key: cam.read_latest(max_age_ms=200) for cam_key, cam in self.cameras.items()
        }
        if self._uses_serial_gripper:
            obs_dict["gripper.pos"] = self._current_gripper_norm()

        pos_deg, vel_deg, torque = self._read_state_deg(
            request=True, allow_transient_gripper=True
        )

        for motor_name in self.motor_names:
            obs_key = self._observation_motor_key(motor_name)
            if motor_name == "gripper":
                obs_dict[f"{obs_key}.pos"] = self._gripper_pos_to_norm(pos_deg[motor_name])
                if self.config.enable_observation_gripper_vel:
                    obs_dict[f"{obs_key}.vel"] = vel_deg[motor_name]
                if self.config.enable_observation_gripper_torque:
                    obs_dict[f"{obs_key}.torque"] = torque[motor_name]
                continue

            if self.config.enable_observation_joint_pos:
                obs_dict[f"{obs_key}.pos"] = pos_deg[motor_name]
            if self.config.enable_observation_joint_vel:
                obs_dict[f"{obs_key}.vel"] = vel_deg[motor_name]
            if self.config.enable_observation_joint_torque:
                obs_dict[f"{obs_key}.torque"] = torque[motor_name]

        if self.flexitac is not None:
            gripper = obs_dict.get("gripper.pos")
            obs_dict.update(self.flexitac.get_contact_maps({"single": gripper}))

        if self.config.action_mode.lower() == "cartesian":
            tcp_pose = self._tcp_pose_matrix_from_positions(pos_deg)
            r6d = self._matrix_to_rotation_6d(tcp_pose[:3, :3])
            obs_dict["tcp.x"] = float(tcp_pose[0, 3])
            obs_dict["tcp.y"] = float(tcp_pose[1, 3])
            obs_dict["tcp.z"] = float(tcp_pose[2, 3])
            for idx, key in enumerate(TCP_POSE_KEYS[3:]):
                obs_dict[key] = float(r6d[idx])

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug("%s get_observation took: %.1fms", self, dt_ms)
        return obs_dict

    def get_episode_metadata(self) -> dict[str, Any]:
        if self.flexitac is None:
            return {}
        metadata: dict[str, Any] = {"tactile/preprocessing_version": "fast-32-16-release-zero-u8-v2"}
        for name, calibration in self.flexitac.calibration_metadata().items():
            prefix = f"tactile/{name}"
            metadata[f"{prefix}/port"] = calibration["port"]
            metadata[f"{prefix}/shape"] = calibration["shape"]
            metadata[f"{prefix}/protocol"] = calibration["protocol"]
            metadata[f"{prefix}/baseline"] = calibration["baseline"]
            metadata[f"{prefix}/raw_baseline"] = calibration["raw_baseline"]
            metadata[f"{prefix}/preprocessing"] = calibration["preprocessing"]
        return metadata

    def recalibrate_tactile(self) -> None:
        if self.flexitac is None:
            raise ValueError("No FlexiTac sensors configured")
        self.flexitac.recalibrate(
            calibration_frames=self.config.flexitac_calibration_frames,
            calibration_timeout_s=self.config.flexitac_calibration_timeout_s,
            calibration_max_span=self.config.flexitac_calibration_max_span,
            calibration_mean_min=self.config.flexitac_calibration_mean_min,
            calibration_mean_max=self.config.flexitac_calibration_mean_max,
        )

    def recover_observation(self) -> None:
        if self.flexitac is None:
            return
        logger.warning("Recovering two single-arm FlexiTac readers without changing their baselines")
        self.flexitac.recover(timeout_s=self.config.flexitac_calibration_timeout_s)

    @staticmethod
    def _clip(value: float, limits: tuple[float, float]) -> float:
        return max(float(limits[0]), min(float(limits[1]), value))

    @staticmethod
    def _nearest_equivalent_degrees(target_deg: float, reference_deg: float) -> float:
        """Return target_deg plus whole turns so it is closest to reference_deg."""
        turns = round((float(reference_deg) - float(target_deg)) / 360.0)
        return float(target_deg) + 360.0 * turns

    def _apply_initial_gripper_pose(self, positions_deg: dict[str, float]) -> None:
        if "gripper" not in positions_deg:
            return

        # Calibration defines the fully closed pose as 0 deg; opening moves
        # toward the negative soft limit.
        positions_deg["gripper"] = self._clip(
            self.config.joint_limits["gripper"][0],
            self.config.joint_limits["gripper"],
        )

    def _force_pos_flags(self) -> tuple[list[bool] | None, list[float] | None]:
        if (
            self.config.control_mode.lower() != "pos_vel"
            or not self._uses_builtin_gripper
            or not self.config.enabled_gripper_force
            or "gripper" not in self.motor_names
        ):
            return None, None

        force_pos = [motor_name == "gripper" for motor_name in self.motor_names]
        torque_ratio = [
            float(self.config.gripper_force_pos_torque_ratio) if motor_name == "gripper" else 0.0
            for motor_name in self.motor_names
        ]
        return force_pos, torque_ratio

    def _mit_gains_lists(self) -> tuple[list[float], list[float]]:
        kp = [float(self.config.mit_gains[motor_name][0]) for motor_name in self.motor_names]
        kd = [float(self.config.mit_gains[motor_name][1]) for motor_name in self.motor_names]
        return kp, kd

    def _set_rt_target_deg(self, goal_pos: dict[str, float], vlim_rad: list[float] | None = None) -> None:
        target_rad = [math.radians(goal_pos[motor_name]) for motor_name in self.motor_names]
        force_pos, force_pos_torque_ratio = self._force_pos_flags()
        mit_kp = None
        mit_kd = None
        if self.config.control_mode.lower() == "mit":
            mit_kp, mit_kd = self._mit_gains_lists()
        self.arm.set_targets(
            pos=target_rad,
            kp=mit_kp,
            kd=mit_kd,
            vlim=vlim_rad if vlim_rad is not None else self._velocity_limits_rad(),
            force_pos=force_pos,
            force_pos_torque_ratio=force_pos_torque_ratio,
        )

    def _return_to_initial_vlim_rad(self) -> list[float]:
        """Return per-joint reset velocity limits in rad/s."""
        spec = self.config.return_to_initial_vlim_deg_s
        normal_vlim = self._velocity_limits_rad()
        if isinstance(spec, (list, tuple)) and len(spec) not in {len(self.motor_names), len(self.config.motor_can_ids)}:
            raise ValueError(
                "return_to_initial_vlim_deg_s list length must match the controlled joint count or the full motor count."
            )

        values: list[float] = []
        for idx, motor_name in enumerate(self.motor_names):
            if isinstance(spec, dict):
                if motor_name not in spec:
                    raise ValueError(f"return_to_initial_vlim_deg_s missing key: {motor_name}")
                deg_s = spec[motor_name]
            elif isinstance(spec, (list, tuple)):
                if len(spec) == len(self.motor_names):
                    deg_s = spec[idx]
                else:
                    all_motor_names = list(self.config.motor_can_ids)
                    deg_s = spec[all_motor_names.index(motor_name)]
            else:
                deg_s = spec

            deg_s = float(deg_s)
            if deg_s <= 0:
                raise ValueError("return_to_initial_vlim_deg_s values must be > 0.")
            values.append(min(math.radians(deg_s), normal_vlim[idx]))
        return values

    def _return_to_initial_vlim_deg(self) -> list[float]:
        return [math.degrees(value) for value in self._return_to_initial_vlim_rad()]

    def _complete_goal(self, action: RobotAction) -> dict[str, float]:
        goal_pos = dict(self._last_goal_deg)
        if not goal_pos and self._last_positions_deg:
            goal_pos = dict(self._last_positions_deg)

        for key, val in action.items():
            if not key.endswith(".pos"):
                continue
            motor_name = key.removesuffix(".pos")
            if motor_name in self.motor_names:
                goal_pos[motor_name] = float(val)

        if "wrist_yaw" in self.motor_names and "wrist_yaw" not in goal_pos:
            goal_pos["wrist_yaw"] = 0.0

        for motor_name in self.motor_names:
            goal_pos.setdefault(motor_name, self._last_positions_deg.get(motor_name, 0.0))
        return {motor_name: goal_pos[motor_name] for motor_name in self.motor_names}

    def _complete_tcp_action(self, action: RobotAction) -> RobotAction:
        missing = set(TCP_POSE_KEYS) - set(action)
        if missing:
            raise ValueError(f"Cartesian action missing keys: {sorted(missing)}")

        tcp_action: RobotAction = {key: float(action[key]) for key in TCP_POSE_KEYS}
        if self.config.control_gripper:
            if "gripper.pos" in action:
                gripper_norm = float(action["gripper.pos"])
            else:
                gripper_norm = self._current_gripper_norm()
            tcp_action["gripper.pos"] = max(0.0, min(1.0, gripper_norm))
        return tcp_action

    def _send_serial_gripper_action(self, action: RobotAction) -> dict[str, float]:
        if not self._uses_serial_gripper:
            return {}
        if "gripper.pos" not in action:
            return {}
        if self.serial_gripper is None:
            raise DeviceNotConnectedError("Serial gripper is not connected.")

        gripper_norm = max(0.0, min(1.0, float(action["gripper.pos"])))
        self.serial_gripper.set_gripper_position(1.0 - gripper_norm)
        return {"gripper.pos": gripper_norm}

    def _guard_tcp_height(
        self, goal: dict[str, float], floor: float, *, ceiling: float | None = None,
        rising_target: bool = False, tcp_target: RobotAction | None = None,
    ) -> dict[str, float]:
        """Guard TCP path height while preserving ordinary commands exactly."""
        model = self._load_kinematic_model()
        q = self._kinematic_joint_rad(goal)
        current = self._kinematic_joint_rad(self._last_positions_deg)
        radius = np.abs(q - current)
        limits = np.radians([self.config.joint_limits[name] for name in KINEMATIC_MOTORS])
        lower = np.maximum(current - radius, limits[:, 0])
        upper = np.minimum(current + radius, limits[:, 1])
        velocity = dict(zip(self.motor_names, self._velocity_limits_rad()))
        speed = np.maximum([abs(velocity[name]) for name in KINEMATIC_MOTORS], 1e-9)
        if (
            self._teleop_motion_enabled is True and not self._teleop_wrist_rotation
            and self.config.control_mode.lower() == "pos_vel"
            and not rising_target
        ):
            def within_height_range(candidate, synchronized):
                delta = candidate - current
                arrival = np.abs(delta) / speed
                if synchronized:
                    samples = current + np.linspace(0.0, 1.0, 17)[:, None] * delta
                else:
                    times = np.unique(np.r_[arrival, np.linspace(0.0, float(max(arrival)), 9)])
                    samples = current + np.clip(delta, -times[:, None] * speed, times[:, None] * speed)
                for sample in samples:
                    height = float(np.asarray(model.fk(sample, "")[2])[2, 3])
                    if height < floor - 1e-10 or (ceiling is not None and height > ceiling + 1e-10):
                        return False
                return True

            if within_height_range(q, self._coordinated_tcp_step):
                return goal
            # Try coordinating arrival before changing any joint target. Never
            # repair a dip by steering individual joints above the TCP target:
            # that creates a return command on the following frame.
            self._coordinated_tcp_step = True
            if within_height_range(q, True):
                return goal
            if tcp_target is not None and self._last_ik_accepted:
                # A joint-space chord can bend out of the height band even
                # when both TCP endpoints are valid. Re-solve a nearer point
                # on the requested Cartesian path instead of repeatedly
                # shortening that same chord until motion stalls.
                start_pose = np.asarray(model.fk(current, "")[2])
                position_delta = np.array([
                    tcp_target[k] for k in ("tcp.x", "tcp.y", "tcp.z")
                ]) - start_pose[:3, 3]
                rotation_delta = Rotation.from_matrix(
                    start_pose[:3, :3].T @ self._rotation_6d_to_matrix(tcp_target)
                ).as_rotvec()
                caps = np.radians([self._ik_joint_step_limits_deg()[n] for n in KINEMATIC_MOTORS])
                relative = self.config.max_relative_target
                if relative is not None:
                    caps = np.minimum(caps, np.radians([
                        relative[n] if isinstance(relative, dict) else relative for n in KINEMATIC_MOTORS
                    ]))
                for attempt in range(1, 17):
                    fraction = 0.5 ** attempt
                    waypoint = start_pose.copy()
                    waypoint[:3, 3] += fraction * position_delta
                    waypoint[:3, :3] = start_pose[:3, :3] @ Rotation.from_rotvec(
                        fraction * rotation_delta
                    ).as_matrix()
                    result = model.solve_ik(waypoint, current, self._kinematic_frame_id)
                    candidate = np.asarray(result.q)
                    if not result.success or not np.all(np.isfinite(candidate)):
                        continue
                    if np.any(candidate < limits[:, 0]) or np.any(candidate > limits[:, 1]):
                        continue
                    if np.any(np.abs(candidate - current) > caps + 1e-10):
                        continue
                    if within_height_range(candidate, True):
                        return {**goal, **dict(zip(KINEMATIC_MOTORS, map(float, np.degrees(candidate))))}
            low, high = 0.0, 1.0
            for _ in range(24):
                fraction = (low + high) * 0.5
                if within_height_range(current + fraction * (q - current), True):
                    low = fraction
                else:
                    high = fraction
            q = current + low * (q - current)
            return {**goal, **dict(zip(KINEMATIC_MOTORS, map(float, np.degrees(q))))}

        # Keep the verified increasing-Z repair, including zero-pose lift.
        coordinated_lift = ceiling is not None and self._coordinated_tcp_step

        def lowest_pose(candidate):
            # Preview independently speed-limited axes, including their arrival
            # times. Checking only the endpoint misses transient descent.
            arrival = np.abs(candidate - current) / speed
            times = (np.linspace(0.0, 1.0, 17) if coordinated_lift else
                     np.unique(np.r_[arrival, np.linspace(0.0, float(max(arrival)), 9)]))
            worst = None
            for t in times:
                # Bounded upward steps receive proportional velocity limits;
                # preview their synchronized, speed-limited reference path.
                sampled = (current + t * (candidate - current) if coordinated_lift else
                           current + np.clip(candidate - current, -speed * t, speed * t))
                pose = np.asarray(model.fk(sampled, "")[2])
                if worst is None or pose[2, 3] < worst[0]:
                    worst = (float(pose[2, 3]), t, sampled, pose)
            return worst

        def cap_endpoint(candidate):
            # Height repair must not overshoot an upward TCP target. Otherwise
            # the next frame commands a descent back to the accumulated target.
            if ceiling is None or float(np.asarray(model.fk(candidate, "")[2])[2, 3]) <= ceiling:
                return candidate
            low, high = 0.0, 1.0
            for _ in range(24):
                fraction = (low + high) * 0.5
                trial = current + fraction * (candidate - current)
                if float(np.asarray(model.fk(trial, "")[2])[2, 3]) > ceiling:
                    high = fraction
                else:
                    low = fraction
            return current + low * (candidate - current)

        if np.all(lower <= upper):
            for attempt in range(16):
                height, t, sampled, pose = lowest_pose(q)
                if height >= floor - 0.001:
                    if attempt == 0:
                        return goal
                    q = cap_endpoint(q)
                    height, t, sampled, pose = lowest_pose(q)
                    if height >= floor - 0.001:
                        return {**goal, **dict(zip(KINEMATIC_MOTORS, map(float, np.degrees(q))))}
                # Convert the native LOCAL linear Jacobian to the base frame.
                gradient = (pose[:3, :3] @ np.asarray(
                    model.frame_jacobian(sampled, self._kinematic_frame_id)
                )[:3])[2]
                arrived = (np.ones(len(q), dtype=bool) if coordinated_lift else
                           np.abs(q - current) <= speed * t + 1e-10)
                if coordinated_lift:
                    gradient = gradient * t
                blocked = ((gradient > 0) & (q >= upper - 1e-10)) | ((gradient < 0) & (q <= lower + 1e-10))
                free = np.where(arrived & ~blocked, gradient, 0.0)
                norm_sq = float(free @ free)
                proposal = q
                if norm_sq > 1e-12:
                    proposal = np.clip(q + (floor + 0.0001 - height) * free / norm_sq, lower, upper)
                if np.max(np.abs(proposal - q)) < 1e-9:
                    # An axis still at its speed cap cannot respond to a tiny
                    # target correction: shorten the most downward contribution.
                    contribution = gradient * (sampled - current)
                    index = int(np.argmin(contribution))
                    if contribution[index] >= -1e-12:
                        break
                    proposal = q.copy()
                    proposal[index] = current[index] + 0.5 * (q[index] - current[index])
                q = proposal
            # Only for an unresolved abnormal transition, retain a smaller step
            # instead of introducing a persistent pause or clutch reset.
            for _ in range(8):
                q = cap_endpoint(current + 0.5 * (q - current))
                if lowest_pose(q)[0] >= floor - 0.001:
                    return {**goal, **dict(zip(KINEMATIC_MOTORS, map(float, np.degrees(q))))}
        return {**goal, **{name: self._last_positions_deg[name] for name in KINEMATIC_MOTORS}}

    def _tcp_step_velocity_limits_rad(self, goal: dict[str, float]) -> list[float]:
        """Coordinate a bounded/repaired TCP step within normal speed caps."""
        limits = self._velocity_limits_rad()
        travel = {
            name: abs(math.radians(goal[name] - self._last_positions_deg[name]))
            for name in KINEMATIC_MOTORS
        }
        duration = max(
            travel[name] / max(limits[i], 1e-9)
            for i, name in enumerate(self.motor_names) if name in travel
        )
        if duration > 1e-9:
            for i, name in enumerate(self.motor_names):
                # Keep normal holding authority below one feedback quantum,
                # as for IMU wrist coordination. A tiny IK correction must not
                # reduce a stationary motor's speed cap almost to zero.
                if name in travel and travel[name] > math.radians(0.02):
                    limits[i] = min(limits[i], travel[name] / duration)
        return limits

    def _wrist_velocity_limits_rad(self, goal: dict[str, float]) -> list[float]:
        """Let the moving wrist joints reach their goals together."""
        limits = self._velocity_limits_rad()
        travel = {
            name: abs(math.radians(goal[name] - self._last_positions_deg[name]))
            for name in KINEMATIC_MOTORS[3:]
        }
        duration = max(
            travel[name] / max(limits[i], 1e-9)
            for i, name in enumerate(self.motor_names) if name in travel
        )
        if duration > 1e-9:
            for i, name in enumerate(self.motor_names):
                # Preserve normal holding authority for stationary joints;
                # differences below one feedback quantum need no coordination.
                if name in travel and travel[name] > math.radians(0.02):
                    limits[i] = min(limits[i], travel[name] / duration)
        return limits

    def _brake_translation_wrist(self, goal: dict[str, float]) -> dict[str, float]:
        """Damp wrist speed in excess of the moving IK reference, not its motion.

        A position-only arrival cap assumes instantaneous acceleration/braking.
        In the recorded lift J4 kept moving past its reference for 200--250 ms.
        Advance braking by 50 ms of excess velocity while retaining the full
        reference for a stationary/lagging wrist. Never feed the corrected
        target back into the reference derivative (that creates a return loop).
        Filter only this derivative correction over 100 ms: directly feeding
        sampled velocity back into position can oscillate at 15--30 Hz.
        """
        now = time.monotonic()
        elapsed = (now - self._translation_reference_s
                   if self._translation_reference_s is not None else 1. / 30.)
        corrected = dict(goal)
        caps = self._ik_joint_step_limits_deg()
        alpha = 1. - math.exp(-max(elapsed, 1e-3) / .1)
        for name in KINEMATIC_MOTORS[3:]:
            present = self._last_positions_deg[name]
            previous = self._translation_reference_deg.get(name, present)
            reference_velocity = (goal[name] - previous) / max(elapsed, 1e-3)
            velocity = self._last_velocities_deg.get(name, 0.)
            if not math.isfinite(velocity) or not math.isfinite(reference_velocity):
                continue
            # Motor feedback has ~0.84 deg/s velocity quantization. Leave
            # stationary noise and joints slower than their reference alone.
            raw = (-.05 * (velocity - reference_velocity)
                   if abs(velocity) > max(1., abs(reference_velocity)) else 0.)
            previous_brake = self._translation_brake_deg.get(name, 0.)
            correction = previous_brake + alpha * (raw - previous_brake)
            cap = float(caps[name])
            relative = self.config.max_relative_target
            if relative is not None:
                cap = min(cap, float(relative[name] if isinstance(relative, dict) else relative))
            corrected[name] = self._clip(
                self._clip(goal[name] + correction, (present - cap, present + cap)),
                self.config.joint_limits[name],
            )
            # Do not retain a correction that the joint/step limit prevented
            # us from applying; otherwise it would linger after braking.
            self._translation_brake_deg[name] = corrected[name] - goal[name]
        self._translation_reference_deg = dict(goal)
        self._translation_reference_s = now
        return corrected

    def _send_joint_action(
        self, action: RobotAction, *, tcp_height_floor: float | None = None,
        tcp_height_ceiling: float | None = None, tcp_height_rising: bool = False,
        tcp_target: RobotAction | None = None,
    ) -> RobotAction:
        if self.arm is None or not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        serial_gripper_action = self._send_serial_gripper_action(action)
        goal_pos = self._complete_goal(action)
        for motor_name, position in list(goal_pos.items()):
            goal_pos[motor_name] = self._clip(position, self.config.joint_limits[motor_name])
            if motor_name in KINEMATIC_MOTORS and motor_name in self._last_positions_deg:
                goal_pos[motor_name] = self._nearest_equivalent_degrees(
                    goal_pos[motor_name],
                    self._last_positions_deg[motor_name],
                )

        if self.config.max_relative_target is not None:
            if not self._last_positions_deg:
                self._read_state_deg(request=True)
            limited_names = [motor_name for motor_name in self.motor_names if motor_name != "gripper"]
            goal_present_pos = {
                motor_name: (goal_pos[motor_name], self._last_positions_deg.get(motor_name, goal_pos[motor_name]))
                for motor_name in limited_names
            }
            safe_goal_pos = _silent_safe_goal_position(goal_present_pos, self.config.max_relative_target)
            if "gripper" in goal_pos:
                safe_goal_pos["gripper"] = goal_pos["gripper"]
            goal_pos = safe_goal_pos

        if tcp_height_rising and self._teleop_translation and self.config.control_mode.lower() == "pos_vel":
            # Coordinate lift from its first small step, before joint clipping
            # is reached. Otherwise the wrist starts with the full 150 deg/s
            # cap while the loaded shoulder/elbow are still accelerating.
            self._coordinated_tcp_step = True
        if tcp_height_floor is not None:
            goal_pos = self._guard_tcp_height(
                goal_pos, tcp_height_floor, ceiling=tcp_height_ceiling, rising_target=tcp_height_rising,
                tcp_target=tcp_target,
            )
            if (
                self._teleop_translation and self._teleop_motion_enabled is True
                and not self._teleop_wrist_rotation and self._last_ik_accepted
                and self.config.control_mode.lower() == "pos_vel"
            ):
                # Apply dynamic damping after the geometric path check: a
                # braking reference is not an instantaneous physical pose.
                goal_pos = self._brake_translation_wrist(goal_pos)
        vlim_rad = None
        if self._policy_motion_limiter is not None:
            goal_pos, vlim_rad = self._policy_motion_limiter.step(
                goal_pos, self._last_positions_deg, time.monotonic()
            )
            for name in KINEMATIC_MOTORS:
                relative = self.config.max_relative_target
                cap = relative[name] if isinstance(relative, dict) else relative
                if cap is not None and abs(goal_pos[name] - self._last_positions_deg[name]) > cap:
                    raise RuntimeError(f"Policy joint tracking error exceeds the configured bound for {name}.")
            self._set_rt_target_deg(goal_pos, vlim_rad=vlim_rad)
        elif tcp_height_floor is not None and self._coordinated_tcp_step:
            vlim_rad = self._tcp_step_velocity_limits_rad(goal_pos)
            self._set_rt_target_deg(goal_pos, vlim_rad=vlim_rad)
        elif (
            self._teleop_wrist_rotation and self._teleop_motion_enabled is True
            and self.config.control_mode.lower() == "pos_vel"
        ):
            self._set_rt_target_deg(goal_pos, vlim_rad=self._wrist_velocity_limits_rad(goal_pos))
        else:
            self._set_rt_target_deg(goal_pos)
        self._last_goal_deg = dict(goal_pos)
        return {**{f"{motor}.pos": val for motor, val in goal_pos.items()}, **serial_gripper_action}

    def set_policy_motion_limits(self, config: PolicyMotionConfig | None, *, period: float = 1. / 30.) -> None:
        if config is None:
            self._policy_motion_limiter = None
            return
        if not self.is_connected or self.config.control_mode.lower() != "pos_vel":
            raise RuntimeError("Policy motion constraints require a connected B601 in POS_VEL mode.")
        self._policy_motion_limiter = JointReferenceLimiter(
            self.motor_names, self._last_positions_deg, self.config.joint_limits,
            [math.degrees(value) for value in self._velocity_limits_rad()],
            config, period, time.monotonic(),
        )

    def set_teleop_motion_enabled(self, enabled: bool, *, immediate_stop: bool = False) -> None:
        if not enabled:
            self._coordinated_tcp_step = False
            self._translation_reference_deg = {}
            self._translation_reference_s = None
            self._translation_brake_deg = {}
        if not enabled and self._teleop_motion_enabled is not False:
            was_active = self._teleop_motion_enabled is True and not immediate_stop
            self._release_start_s = time.monotonic()
            self._release_duration_s = 0.15
            limits = dict(zip(self.motor_names, self._velocity_limits_rad()))
            for name in KINEMATIC_MOTORS:
                present = self._last_positions_deg[name]
                start = self._last_goal_deg.get(name, present) if was_active else present
                speed = math.degrees(limits[name])
                velocity = self._clip(self._last_velocities_deg.get(name, 0.0), (-speed, speed)) if was_active else 0.0
                # Freeze a nearby stopping point, rather than chasing the old
                # TCP. Bound predicted travel from feedback to three degrees.
                target = self._clip(present + self._clip(velocity * 0.075, (-3.0, 3.0)),
                                    self.config.joint_limits[name])
                self._release_from[name] = start
                self._release_to[name] = target
                # Quintic blend has a maximum slope of 1.875.
                self._release_duration_s = max(self._release_duration_s,
                                              1.875 * abs(target - start) / max(speed, 1e-6))
        self._teleop_motion_enabled = enabled

    def _release_joint_action(self, tcp_action: RobotAction) -> RobotAction:
        u = min(max((time.monotonic() - self._release_start_s) / self._release_duration_s, 0.0), 1.0)
        blend = u**3 * (10.0 + u * (-15.0 + 6.0 * u))
        action = {f"{name}.pos": self._release_from[name] + blend * (self._release_to[name] - self._release_from[name])
                  for name in KINEMATIC_MOTORS}
        if "gripper.pos" in tcp_action:
            value = tcp_action["gripper.pos"]
            if self._uses_builtin_gripper:
                opened, closed = self.config.joint_limits["gripper"]
                value = opened + value * (closed - opened)
            action["gripper.pos"] = value
        self._last_ik_accepted = True
        return action

    def send_action(self, action: RobotAction) -> RobotAction:
        if self.arm is None or not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        action_mode = self.config.action_mode.lower()
        is_tcp_action = any(key in action for key in TCP_POSE_KEYS)
        if action_mode == "cartesian" or is_tcp_action:
            if (
                self._teleop_translation and self._teleop_motion_enabled is True
                and not self._teleop_wrist_rotation and self.config.control_mode.lower() == "pos_vel"
                and self._last_state_read_s is not None
                and time.monotonic() - self._last_state_read_s > 1. / self.config.rt_rate
            ):
                # Poll existing RX feedback immediately before IK. Do not add
                # CAN queries or wait for a new packet/camera frame.
                self._read_state_deg(request=False, allow_transient_gripper=True)
            tcp_action = self._complete_tcp_action(action)
            joint_action = (self._release_joint_action(tcp_action) if self._teleop_motion_enabled is False
                            else self.tcp_action_to_joint_action(tcp_action))
            height_floor = None
            height_ceiling = None
            height_rising = False
            if self._teleop_motion_enabled is True and not self._teleop_wrist_rotation:
                current_z = float(self._tcp_pose_matrix_from_positions(self._last_positions_deg)[2, 3])
                height_floor = min(current_z, tcp_action["tcp.z"])
                # Horizontal/descent commands must not acquire an upward kick
                # from height protection. Allow the same 1 mm path tolerance
                # as the floor so constant-height joint paths can progress.
                height_ceiling = max(current_z, tcp_action["tcp.z"] + 0.001)
                previous_z = self._last_teleop_tcp_target_z
                # Preserve the increasing-Z repair's strict target ceiling.
                # A constant-Z target above lagging feedback is not a new lift.
                if tcp_action["tcp.z"] > current_z and (
                    previous_z is None or tcp_action["tcp.z"] > previous_z + 1e-9
                ):
                    height_ceiling = tcp_action["tcp.z"]
                    height_rising = True
                elif self.config.control_mode.lower() == "pos_vel":
                    height_floor = min(current_z, tcp_action["tcp.z"] - 0.001)
                self._last_teleop_tcp_target_z = tcp_action["tcp.z"]
            else:
                self._last_teleop_tcp_target_z = None
            sent_joint_action = self._send_joint_action(
                joint_action, tcp_height_floor=height_floor, tcp_height_ceiling=height_ceiling,
                tcp_height_rising=height_rising, tcp_target=tcp_action,
            )
            if action_mode == "cartesian":
                if self._teleop_motion_enabled is False or self._policy_motion_limiter is not None:
                    pose = self._tcp_pose_matrix_from_positions(
                        {name: sent_joint_action[f"{name}.pos"] for name in KINEMATIC_MOTORS}
                    )
                    tcp_action.update(dict(zip(("tcp.x", "tcp.y", "tcp.z"), map(float, pose[:3, 3]))))
                    tcp_action.update({f"tcp.r{i}": float(value) for i, value in enumerate(
                        self._matrix_to_rotation_6d(pose[:3, :3]), 1
                    )})
                    if self._policy_motion_limiter is not None and self._uses_builtin_gripper and "gripper.pos" in sent_joint_action:
                        tcp_action["gripper.pos"] = self._gripper_pos_to_norm(sent_joint_action["gripper.pos"])
                return tcp_action
            return sent_joint_action

        return self._send_joint_action(action)

    def _return_to_initial_position(self) -> bool:
        if self.arm is None:
            self.last_return_to_initial_succeeded = True
            return True
        if not self._initial_positions_deg:
            logger.error("Cannot return %s to zero: no initial joint positions were captured.", self)
            self.last_return_to_initial_succeeded = False
            return False

        try:
            current, _, _ = self._read_state_deg(request=False)
        except Exception:
            logger.error("Failed to read state before returning %s to zero.", self, exc_info=True)
            self.last_return_to_initial_succeeded = False
            return False

        target = {
            name: self._clip(self._initial_positions_deg[name], self.config.joint_limits[name])
            for name in self.motor_names
        }
        if self.config.control_mode.lower() == "mit":
            result = self._return_to_initial_position_mit(current, target)
            self.last_return_to_initial_succeeded = result
            return result

        return_started_s = time.monotonic()
        return_vlim_rad = self._return_to_initial_vlim_rad()

        try:
            self._set_rt_target_deg(target, vlim_rad=return_vlim_rad)
            self._last_goal_deg = dict(target)
        except Exception:
            logger.error("Failed to command %s toward zero before disconnect.", self, exc_info=True)
            self.last_return_to_initial_succeeded = False
            return False

        expected_s = max(
            abs(target[name] - current.get(name, target[name])) / math.degrees(return_vlim_rad[idx])
            for idx, name in enumerate(self.motor_names)
        )
        wait_until = return_started_s + expected_s + 1.0
        remaining_s = wait_until - time.monotonic()
        if remaining_s > 0:
            time.sleep(remaining_s)
        logger.info("Completed the timed return-to-initial command for %s.", self)
        self.last_return_to_initial_succeeded = True
        return True

    def _return_to_initial_position_mit(
        self, current: dict[str, float], target: dict[str, float]
    ) -> bool:
        step_dt_s = 0.02
        return_vlim_deg = self._return_to_initial_vlim_deg()
        commanded = {
            name: self._last_goal_deg.get(name, current.get(name, target[name]))
            for name in self.motor_names
        }

        while True:
            done = True
            for idx, name in enumerate(self.motor_names):
                error = target[name] - commanded[name]
                max_step = return_vlim_deg[idx] * step_dt_s
                if abs(error) > max_step:
                    commanded[name] += math.copysign(max_step, error)
                    done = False
                else:
                    commanded[name] = target[name]

            try:
                self._set_rt_target_deg(commanded)
                self._last_goal_deg = dict(commanded)
            except Exception:
                logger.error("Failed to command MIT zero target for %s.", self, exc_info=True)
                return False

            if done:
                logger.info("Completed the return-to-initial command sequence for %s.", self)
                return True

            time.sleep(step_dt_s)

    def reset_to_initial_position(self) -> bool:
        self._teleop_motion_enabled = None
        self.set_teleop_translation(False)
        self.set_teleop_wrist_rotation(False)
        self._last_teleop_tcp_target_z = None
        succeeded = self._return_to_initial_position()
        if succeeded:
            # Homing updates targets without polling observations. Refresh the
            # cached joints before VR uses them as its new Cartesian reference.
            self._read_state_deg(request=False)
        return succeeded

    def disconnect(self) -> None:
        if self.arm is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self._calibration_only_connection:
            try:
                self.arm.disconnect(disable=True)
            finally:
                self.arm = None
                self._calibration_only_connection = False
            logger.info("%s calibration connection closed with torque disabled.", self)
            return

        try:
            returned_to_zero = self._return_to_initial_position()
            self.arm.disconnect(disable=bool(self.config.disable_torque_on_disconnect))
        finally:
            self.arm = None
            if self.serial_gripper is not None:
                try:
                    self.serial_gripper.disconnect()
                finally:
                    self.serial_gripper = None
            for cam in self.cameras.values():
                cam.disconnect()
            if self.flexitac is not None:
                self.flexitac.disconnect()

        if not returned_to_zero:
            logger.error("%s disconnected after failing to return to zero.", self)
            raise ReturnToInitialPositionError(
                f"{self} disconnected, but failed to return to its initial position"
            )
        else:
            logger.info("%s disconnected.", self)
