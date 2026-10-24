import logging
import time
from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as ScipyRotation

from lerobot.processor import RobotAction
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from .config_switch_joycon import SwitchJoyconConfig

logger = logging.getLogger(__name__)

DEFAULT_TCP_POSE_QUAT = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)


def _normalize_quaternion(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32).reshape(-1)
    if len(q) != 4:
        raise ValueError(f"Quaternion must have 4 components, got {len(q)}.")
    norm = float(np.linalg.norm(q))
    if norm < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return (q / norm).astype(np.float32)


def _quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    qw1, qx1, qy1, qz1 = q1
    qw2, qx2, qy2, qz2 = q2
    return np.array(
        [
            qw1 * qw2 - qx1 * qx2 - qy1 * qy2 - qz1 * qz2,
            qw1 * qx2 + qx1 * qw2 + qy1 * qz2 - qz1 * qy2,
            qw1 * qy2 - qx1 * qz2 + qy1 * qw2 + qz1 * qx2,
            qw1 * qz2 + qx1 * qy2 - qy1 * qx2 + qz1 * qw2,
        ],
        dtype=np.float32,
    )


def _rotvec_to_quaternion(rotvec: np.ndarray) -> np.ndarray:
    quat_xyzw = ScipyRotation.from_rotvec(rotvec).as_quat()
    return _normalize_quaternion(
        np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32)
    )


def _quaternion_to_rotation_6d(q: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = _normalize_quaternion(q)
    return np.array(
        [
            1.0 - 2.0 * (qy * qy + qz * qz),
            2.0 * (qx * qy + qz * qw),
            2.0 * (qx * qz - qy * qw),
            2.0 * (qx * qy - qz * qw),
            1.0 - 2.0 * (qx * qx + qz * qz),
            2.0 * (qy * qz + qx * qw),
        ],
        dtype=np.float32,
    )


def _wrap_angle_delta(delta: np.ndarray) -> np.ndarray:
    return (delta + np.pi) % (2.0 * np.pi) - np.pi


class SwitchJoycon(Teleoperator):
    """Switch Joy-Con teleoperator that emits Quest3-compatible Cartesian TCP actions."""

    config_class = SwitchJoyconConfig
    name = "switch_joycon"

    def __init__(self, config: SwitchJoyconConfig):
        super().__init__(config)
        self.config = config
        self.joycon = None
        self._is_connected = False
        self._target_pos = np.zeros(3, dtype=np.float32)
        self._target_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self._target_gripper_pos = float(config.gripper_open_pos)
        self._current_tcp_pose_quat = DEFAULT_TCP_POSE_QUAT.copy()
        self._enabled = False
        self._was_enabled = False
        self._motion_enabled = False
        self._translation_enabled = False
        self._orientation_ref: np.ndarray | None = None
        self._orientation_delta_continuous = np.zeros(3)
        self._last_action_time: float | None = None
        self._prev_target_pos: np.ndarray | None = None
        self._prev_target_quat: np.ndarray | None = None
        self._was_reset_button_pressed = False
        self._reset_just_pressed = False
        self._was_recalibrate_button_pressed = False

    @property
    def requires_current_tcp_pose(self) -> bool:
        return True

    @property
    def motion_enabled(self) -> bool:
        return self._motion_enabled

    @property
    def stop_immediately_on_release(self) -> bool:
        return True

    @property
    def prefer_wrist_rotation(self) -> bool:
        return self._enabled and self.config.control_orientation

    @property
    def prefer_translation_position(self) -> bool:
        return self._translation_enabled

    @property
    def action_features(self) -> dict[str, Any]:
        return {
            "dtype": "float32",
            "shape": (10,),
            "names": {
                "tcp.x": 0,
                "tcp.y": 1,
                "tcp.z": 2,
                "tcp.r1": 3,
                "tcp.r2": 4,
                "tcp.r3": 5,
                "tcp.r4": 6,
                "tcp.r5": 7,
                "tcp.r6": 8,
                "gripper.pos": 9,
            },
        }

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        return self._is_connected

    def connect(self, calibrate: bool = True) -> None:
        if self._is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        try:
            from .evdev_joycon import EvdevJoycon
            self.joycon = EvdevJoycon(self.config)
        except Exception as exc:
            if self.joycon is not None:
                self.joycon.disconnect()
                self.joycon = None
            raise DeviceNotConnectedError(
                f"Failed to initialize {self.config.device!r} Joy-Con through evdev. "
                f"Check /dev/input/event* permissions and Bluetooth pairing. Details: {exc}"
            ) from exc

        try:
            if calibrate and self.config.control_orientation and self.config.imu_calibrate_on_connect:
                self.calibrate()
        except Exception:
            self.joycon.disconnect()
            self.joycon = None
            raise

        self._sync_target_to_current_tcp_pose()
        self._enabled = False
        self._was_enabled = False
        self._motion_enabled = False
        self._translation_enabled = False
        self._orientation_ref = None
        self._orientation_delta_continuous = np.zeros(3)
        self._last_action_time = None
        self._prev_target_pos = None
        self._prev_target_quat = None
        self._is_connected = True
        logger.info("%s connected.", self)

    def configure(self) -> None:
        pass

    def calibrate(self) -> None:
        if self.joycon is not None:
            self.joycon.reset_joycon()
            self._orientation_ref = self._joycon_orientation().copy()
            self._orientation_delta_continuous = np.zeros(3)

    def set_current_tcp_pose(self, current_tcp_pose_quat: np.ndarray) -> None:
        current_tcp_pose_quat = np.asarray(current_tcp_pose_quat, dtype=np.float32)
        if current_tcp_pose_quat.shape != (8,):
            raise ValueError(
                "Switch Joy-Con current TCP pose must be [x, y, z, qw, qx, qy, qz, gripper_norm]."
            )
        self._current_tcp_pose_quat = current_tcp_pose_quat.copy()

    def reset_to_current_tcp_pose(self) -> None:
        self._sync_target_to_current_tcp_pose()
        self._enabled = False
        self._was_enabled = False
        self._motion_enabled = False
        self._translation_enabled = False
        self._orientation_ref = None
        self._orientation_delta_continuous = np.zeros(3)
        self._last_action_time = None
        self._prev_target_pos = None
        self._prev_target_quat = None

    def handle_robot_action_rejected(self) -> None:
        self.reset_to_current_tcp_pose()

    def _sync_target_to_current_tcp_pose(self) -> None:
        self._target_pos = self._current_tcp_pose_quat[:3].copy()
        self._target_quat = _normalize_quaternion(self._current_tcp_pose_quat[3:7])
        self._target_gripper_pos = float(self._current_tcp_pose_quat[7])

    def _joycon_device(self):
        if self.joycon is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        return getattr(self.joycon, "joycon", self.joycon)

    def _button_getter(self, name: str) -> Callable[[], int]:
        joycon = self._joycon_device()
        prefix = "r" if self.config.device == "right" else "l"
        getters = {
            "zr": joycon.get_button_zr,
            "zl": joycon.get_button_zl,
            "a": joycon.get_button_a,
            "b": joycon.get_button_b,
            "x": joycon.get_button_x,
            "y": joycon.get_button_y,
            "plus": joycon.get_button_plus,
            "minus": joycon.get_button_minus,
            "home": joycon.get_button_home,
            "capture": joycon.get_button_capture,
            "sr": joycon.get_button_right_sr if prefix == "r" else joycon.get_button_left_sr,
            "sl": joycon.get_button_right_sl if prefix == "r" else joycon.get_button_left_sl,
            "stick": joycon.get_button_r_stick if prefix == "r" else joycon.get_button_l_stick,
        }
        if name not in getters:
            raise ValueError(f"Unsupported Joy-Con button {name!r}.")
        return getters[name]

    def _pressed(self, name: str) -> bool:
        return bool(self._button_getter(name)())

    def _stick_axis(self, value: int) -> float:
        normalized = (float(value) - 2048.0) / 2048.0
        if abs(normalized) < self.config.stick_deadzone:
            return 0.0
        return max(-1.0, min(1.0, normalized))

    def _stick_xy(self) -> tuple[float, float]:
        joycon = self._joycon_device()
        if self.config.device == "right":
            horizontal = joycon.get_stick_right_horizontal()
            vertical = joycon.get_stick_right_vertical()
        else:
            horizontal = joycon.get_stick_left_horizontal()
            vertical = joycon.get_stick_left_vertical()
        return self._stick_axis(horizontal), self._stick_axis(vertical)

    def _joycon_orientation(self) -> np.ndarray:
        if self.joycon is None:
            return np.zeros(3, dtype=np.float32)
        return np.asarray(self.joycon.orientation_rad, dtype=np.float32)

    def _relative_orientation_delta(self, current: np.ndarray) -> np.ndarray:
        """Choose the equivalent rotation vector nearest the previous input.

        The principal SO(3) logarithm changes sign at pi. It cannot be used
        directly as a continuous angular-rate command across that branch.
        """
        reference = ScipyRotation.from_rotvec(self._orientation_ref)
        principal = (reference.inv() * ScipyRotation.from_rotvec(current)).as_rotvec()
        angle = float(np.linalg.norm(principal))
        previous = self._orientation_delta_continuous
        if angle > 1e-8:
            axis = principal / angle
            turns = round((float(previous @ axis) - angle) / (2 * np.pi))
            continuous = principal + (2 * np.pi * turns) * axis
        else:
            length = float(np.linalg.norm(previous))
            if length > 1e-8:
                continuous = (previous / length) * (2 * np.pi * round(length / (2 * np.pi)))
            else:
                continuous = principal
        self._orientation_delta_continuous = continuous
        return continuous.copy()

    def _apply_rate_limit(self, dt: float) -> None:
        if self._prev_target_pos is not None and self.config.max_pos_velocity > 0 and dt > 0:
            delta_pos = self._target_pos - self._prev_target_pos
            max_delta = float(self.config.max_pos_velocity) * dt
            delta_norm = float(np.linalg.norm(delta_pos))
            if delta_norm > max_delta > 0:
                self._target_pos = self._prev_target_pos + delta_pos * (max_delta / delta_norm)

        if (
            self._prev_target_quat is not None
            and self.config.max_rot_velocity > 0
            and dt > 0
        ):
            dot = float(np.clip(abs(np.dot(self._target_quat, self._prev_target_quat)), 0.0, 1.0))
            angle = float(2.0 * np.arccos(dot))
            max_angle = float(self.config.max_rot_velocity) * dt
            if angle > max_angle > 0:
                scale = max_angle / angle
                prev = ScipyRotation.from_quat(
                    [
                        self._prev_target_quat[1],
                        self._prev_target_quat[2],
                        self._prev_target_quat[3],
                        self._prev_target_quat[0],
                    ]
                )
                cur = ScipyRotation.from_quat(
                    [self._target_quat[1], self._target_quat[2], self._target_quat[3], self._target_quat[0]]
                )
                delta = cur * prev.inv()
                limited = ScipyRotation.from_rotvec(delta.as_rotvec() * scale) * prev
                q = limited.as_quat()
                self._target_quat = _normalize_quaternion(np.array([q[3], q[0], q[1], q[2]], dtype=np.float32))

        if self.prefer_wrist_rotation:
            actual = ScipyRotation.from_quat(self._current_tcp_pose_quat[[4, 5, 6, 3]])
            target = ScipyRotation.from_quat(self._target_quat[[1, 2, 3, 0]])
            error = (target * actual.inv()).as_rotvec()
            angle = float(np.linalg.norm(error))
            limit = self.config.max_orientation_target_lead_rad
            if angle > limit:
                # Discard excess accumulated rotation when the wrist cannot keep
                # up. Never let its shortest-rotation error reach the pi branch
                # flip. Below the limit, preserve the original target exactly.
                q = (ScipyRotation.from_rotvec(error * (limit / angle)) * actual).as_quat()
                self._target_quat = _normalize_quaternion(q[[3, 0, 1, 2]])

        self._prev_target_pos = self._target_pos.copy()
        self._prev_target_quat = self._target_quat.copy()

    def _update_buttons(self) -> None:
        reset_pressed = self._pressed(self.config.reset_button)
        self._reset_just_pressed = reset_pressed and not self._was_reset_button_pressed
        self._was_reset_button_pressed = reset_pressed

        recalibrate_pressed = self._pressed(self.config.recalibrate_button)
        if recalibrate_pressed and not self._was_recalibrate_button_pressed:
            self.calibrate()
        self._was_recalibrate_button_pressed = recalibrate_pressed

        if self._pressed(self.config.gripper_close_button):
            self._target_gripper_pos = float(self.config.gripper_close_pos)
        elif self._pressed(self.config.gripper_open_button):
            self._target_gripper_pos = float(self.config.gripper_open_pos)

    def get_action(self) -> RobotAction:
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        now = time.perf_counter()
        dt = 0.0 if self._last_action_time is None else max(0.0, min(0.1, now - self._last_action_time))
        self._last_action_time = now

        self._update_buttons()
        enabled = self._pressed(self.config.enable_button)
        stick_x, stick_y = self._stick_xy()
        z_axis = float(self._pressed(self.config.z_up_button)) - float(
            self._pressed(self.config.z_down_button)
        )
        axis = np.asarray([stick_y, stick_x, z_axis], dtype=np.float32)
        axis *= np.asarray(self.config.position_axes, dtype=np.float32)
        if enabled and self.config.control_orientation:
            # While using the IMU, the arm's first three joints are locked;
            # do not queue translation commands to execute after releasing ZR.
            axis[:] = 0
        speed = float(self.config.base_translation_speed_m_s) * float(self.config.position_sensitivity)
        translating = bool(np.any(axis)) and speed > 0
        moving = translating or (enabled and self.config.control_orientation)
        # Discard unexecuted targets on release, and rebase before resuming.
        # Gripper commands remain independent of arm motion.
        if not self._motion_enabled or not moving:
            self._target_pos = self._current_tcp_pose_quat[:3].copy()
            self._target_quat = _normalize_quaternion(self._current_tcp_pose_quat[3:7])
            self._prev_target_pos = None
            self._prev_target_quat = None
        if translating != self._translation_enabled:
            self._target_pos = self._current_tcp_pose_quat[:3].copy()
            self._prev_target_pos = None
        self._motion_enabled = moving
        self._translation_enabled = translating
        just_enabled = enabled and not self._was_enabled
        just_disabled = self._was_enabled and not enabled
        if just_enabled:
            self._orientation_ref = self._joycon_orientation().copy()
            self._orientation_delta_continuous = np.zeros(3)
            self._prev_target_quat = None
        if just_disabled:
            self._orientation_ref = None
            self._orientation_delta_continuous = np.zeros(3)
            self._target_quat = _normalize_quaternion(self._current_tcp_pose_quat[3:7])
            self._prev_target_quat = None
        self._enabled = enabled
        self._was_enabled = enabled

        self._target_pos = self._target_pos + axis * speed * dt

        if enabled:
            if self.config.control_orientation:
                if self._orientation_ref is None:
                    self._orientation_ref = self._joycon_orientation().copy()
                    self._orientation_delta_continuous = np.zeros(3)
                orientation_delta = self._relative_orientation_delta(self._joycon_orientation())
                orientation_delta *= np.asarray(self.config.rotation_axes, dtype=np.float32)
                rot_speed = (
                    orientation_delta
                    * float(self.config.base_rotation_speed_rad_s)
                    * float(self.config.rotation_sensitivity)
                )
                delta_rotvec = rot_speed * dt
                if float(np.linalg.norm(delta_rotvec)) > 1e-9:
                    delta_q = _rotvec_to_quaternion(delta_rotvec)
                    self._target_quat = _normalize_quaternion(
                        _quaternion_multiply(delta_q, self._target_quat)
                    )
        elif not translating:
            # Anchor attitude for the whole translation gesture. Copying each
            # measured tilt here accepts wrist/shoulder tracking error as a new
            # desired attitude, so a startup dip can reinforce itself.
            self._target_quat = _normalize_quaternion(self._current_tcp_pose_quat[3:7])

        self._apply_rate_limit(dt)
        r6d = _quaternion_to_rotation_6d(self._target_quat)
        return {
            "tcp.x": float(self._target_pos[0]),
            "tcp.y": float(self._target_pos[1]),
            "tcp.z": float(self._target_pos[2]),
            "tcp.r1": float(r6d[0]),
            "tcp.r2": float(r6d[1]),
            "tcp.r3": float(r6d[2]),
            "tcp.r4": float(r6d[3]),
            "tcp.r5": float(r6d[4]),
            "tcp.r6": float(r6d[5]),
            "gripper.pos": float(self._target_gripper_pos),
        }

    def get_reset_button(self) -> bool:
        just_pressed = self._reset_just_pressed
        self._reset_just_pressed = False
        return just_pressed

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    def disconnect(self) -> None:
        if not self._is_connected or self.joycon is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        if hasattr(self.joycon, "running"):
            self.joycon.running = False
        if hasattr(self.joycon, "thread") and self.joycon.thread.is_alive():
            self.joycon.thread.join(timeout=1.0)
        self.joycon.disconnect()
        self.joycon = None
        self._is_connected = False
        self._enabled = False
        self._was_enabled = False
        self._motion_enabled = False
        self._translation_enabled = False
        self._orientation_ref = None
        self._orientation_delta_continuous = np.zeros(3)
