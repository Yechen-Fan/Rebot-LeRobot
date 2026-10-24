import logging
from typing import Any

import numpy as np

from lerobot.processor import RobotAction
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from .config_switch_joycon import BiSwitchJoyconConfig, SwitchJoyconConfig
from .switch_joycon import SwitchJoycon

logger = logging.getLogger(__name__)


class BiSwitchJoycon(Teleoperator):
    """Bimanual Switch Joy-Con teleoperator with one reader per controller."""

    config_class = BiSwitchJoyconConfig
    name = "bi_switch_joycon"

    def __init__(self, config: BiSwitchJoyconConfig):
        super().__init__(config)
        self.config = config
        self._is_connected = False
        self.joycon = None
        self._was_reset_button_pressed = False

        self._left_switch = SwitchJoycon(
            SwitchJoyconConfig(
                id=f"{config.id}_left",
                device=config.left_device,
                input_backend=config.input_backend,
                enable_button=config.left_enable_button,
                reset_button=config.left_reset_button,
                gripper_close_button=config.left_gripper_close_button,
                gripper_open_button=config.left_gripper_open_button,
                recalibrate_button=config.recalibrate_button,
                z_up_button=config.z_up_button,
                z_down_button=config.z_down_button,
                base_translation_speed_m_s=config.base_translation_speed_m_s,
                position_sensitivity=config.position_sensitivity,
                base_rotation_speed_rad_s=config.base_rotation_speed_rad_s,
                rotation_sensitivity=config.rotation_sensitivity,
                stick_deadzone=config.stick_deadzone,
                max_pos_velocity=config.max_pos_velocity,
                max_rot_velocity=config.max_rot_velocity,
                max_orientation_target_lead_rad=config.max_orientation_target_lead_rad,
                control_orientation=config.control_orientation,
                gripper_open_pos=config.left_gripper_open_pos,
                gripper_close_pos=config.left_gripper_close_pos,
                imu_calibrate_on_connect=config.imu_calibrate_on_connect,
                position_axes=config.position_axes,
                rotation_axes=config.rotation_axes,
                evdev_device_path=config.left_evdev_device_path or config.evdev_device_path,
                evdev_stick_x_axis=config.evdev_stick_x_axis,
                evdev_stick_y_axis=config.evdev_stick_y_axis,
                evdev_invert_stick_x=config.evdev_invert_stick_x,
                evdev_invert_stick_y=config.evdev_invert_stick_y,
                evdev_zr_button=config.evdev_zr_button,
                evdev_zl_button=config.evdev_zl_button,
                evdev_a_button=config.evdev_a_button,
                evdev_b_button=config.evdev_b_button,
                evdev_x_button=config.evdev_x_button,
                evdev_y_button=config.evdev_y_button,
                evdev_plus_button=config.evdev_plus_button,
                evdev_minus_button=config.evdev_minus_button,
                evdev_sr_button=config.evdev_sr_button,
                evdev_sl_button=config.evdev_sl_button,
                evdev_stick_button=config.evdev_stick_button,
                evdev_orientation_axes=config.evdev_orientation_axes,
            )
        )
        self._right_switch = SwitchJoycon(
            SwitchJoyconConfig(
                id=f"{config.id}_right",
                device=config.right_device,
                input_backend=config.input_backend,
                enable_button=config.right_enable_button,
                reset_button=config.right_reset_button,
                gripper_close_button=config.right_gripper_close_button,
                gripper_open_button=config.right_gripper_open_button,
                recalibrate_button=config.recalibrate_button,
                z_up_button=config.z_up_button,
                z_down_button=config.z_down_button,
                base_translation_speed_m_s=config.base_translation_speed_m_s,
                position_sensitivity=config.position_sensitivity,
                base_rotation_speed_rad_s=config.base_rotation_speed_rad_s,
                rotation_sensitivity=config.rotation_sensitivity,
                stick_deadzone=config.stick_deadzone,
                max_pos_velocity=config.max_pos_velocity,
                max_rot_velocity=config.max_rot_velocity,
                max_orientation_target_lead_rad=config.max_orientation_target_lead_rad,
                control_orientation=config.control_orientation,
                gripper_open_pos=config.right_gripper_open_pos,
                gripper_close_pos=config.right_gripper_close_pos,
                imu_calibrate_on_connect=config.imu_calibrate_on_connect,
                position_axes=config.position_axes,
                rotation_axes=config.rotation_axes,
                evdev_device_path=config.right_evdev_device_path or config.evdev_device_path,
                evdev_stick_x_axis=config.evdev_stick_x_axis,
                evdev_stick_y_axis=config.evdev_stick_y_axis,
                evdev_invert_stick_x=config.evdev_invert_stick_x,
                evdev_invert_stick_y=config.evdev_invert_stick_y,
                evdev_zr_button=config.evdev_zr_button,
                evdev_zl_button=config.evdev_zl_button,
                evdev_a_button=config.evdev_a_button,
                evdev_b_button=config.evdev_b_button,
                evdev_x_button=config.evdev_x_button,
                evdev_y_button=config.evdev_y_button,
                evdev_plus_button=config.evdev_plus_button,
                evdev_minus_button=config.evdev_minus_button,
                evdev_sr_button=config.evdev_sr_button,
                evdev_sl_button=config.evdev_sl_button,
                evdev_stick_button=config.evdev_stick_button,
                evdev_orientation_axes=config.evdev_orientation_axes,
            )
        )

    @property
    def requires_current_tcp_pose(self) -> bool:
        return True

    @property
    def motion_enabled(self) -> tuple[bool, bool]:
        return self._left_switch.motion_enabled, self._right_switch.motion_enabled

    @property
    def stop_immediately_on_release(self) -> bool:
        return True

    @property
    def prefer_wrist_rotation(self) -> tuple[bool, bool]:
        return self._left_switch.prefer_wrist_rotation, self._right_switch.prefer_wrist_rotation

    @property
    def prefer_translation_position(self) -> tuple[bool, bool]:
        return self._left_switch.prefer_translation_position, self._right_switch.prefer_translation_position

    @property
    def action_features(self) -> dict[str, Any]:
        return {
            "dtype": "float32",
            "shape": (20,),
            "names": {
                "left_tcp.x": 0,
                "left_tcp.y": 1,
                "left_tcp.z": 2,
                "left_tcp.r1": 3,
                "left_tcp.r2": 4,
                "left_tcp.r3": 5,
                "left_tcp.r4": 6,
                "left_tcp.r5": 7,
                "left_tcp.r6": 8,
                "left_gripper.pos": 9,
                "right_tcp.x": 10,
                "right_tcp.y": 11,
                "right_tcp.z": 12,
                "right_tcp.r1": 13,
                "right_tcp.r2": 14,
                "right_tcp.r3": 15,
                "right_tcp.r4": 16,
                "right_tcp.r5": 17,
                "right_tcp.r6": 18,
                "right_gripper.pos": 19,
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

    def configure(self) -> None:
        pass

    def calibrate(self) -> None:
        pass

    def set_current_tcp_poses(self, left_tcp_pose_quat: np.ndarray, right_tcp_pose_quat: np.ndarray) -> None:
        self._left_switch.set_current_tcp_pose(left_tcp_pose_quat)
        self._right_switch.set_current_tcp_pose(right_tcp_pose_quat)

    def set_current_tcp_pose(self, current_tcp_pose_quat) -> None:
        left_pose, right_pose = current_tcp_pose_quat
        self.set_current_tcp_poses(left_pose, right_pose)

    def reset_to_current_tcp_pose(self) -> None:
        self._left_switch.reset_to_current_tcp_pose()
        self._right_switch.reset_to_current_tcp_pose()

    def handle_robot_action_rejected(self) -> None:
        self._left_switch.handle_robot_action_rejected()
        self._right_switch.handle_robot_action_rejected()

    def connect(
        self,
        calibrate: bool = True,
        left_tcp_pose_quat: np.ndarray | None = None,
        right_tcp_pose_quat: np.ndarray | None = None,
    ) -> None:
        if self._is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        left_tcp_pose_quat = (
            self._left_switch._current_tcp_pose_quat.copy()
            if left_tcp_pose_quat is None
            else np.asarray(left_tcp_pose_quat, dtype=np.float32)
        )
        right_tcp_pose_quat = (
            self._right_switch._current_tcp_pose_quat.copy()
            if right_tcp_pose_quat is None
            else np.asarray(right_tcp_pose_quat, dtype=np.float32)
        )

        logger.info("Connecting to Joy-Con input for both controllers...")
        try:
            self._left_switch.connect(calibrate=calibrate)
            self._right_switch.connect(calibrate=calibrate)
            self._left_switch._target_gripper_pos = float(self.config.left_gripper_open_pos)
            self._right_switch._target_gripper_pos = float(self.config.right_gripper_open_pos)
            self._left_switch.set_current_tcp_pose(left_tcp_pose_quat)
            self._right_switch.set_current_tcp_pose(right_tcp_pose_quat)
            self._left_switch.reset_to_current_tcp_pose()
            self._right_switch.reset_to_current_tcp_pose()
            self.joycon = self._right_switch.joycon
            self._is_connected = True
            self._was_reset_button_pressed = False
            logger.info("%s connected.", self)
        except Exception:
            self._is_connected = False
            try:
                if self._left_switch.is_connected:
                    self._left_switch.disconnect()
            except Exception:
                logger.debug("Failed to disconnect left SwitchJoycon after connect error.", exc_info=True)
            try:
                if self._right_switch.is_connected:
                    self._right_switch.disconnect()
            except Exception:
                logger.debug("Failed to disconnect right SwitchJoycon after connect error.", exc_info=True)
            self.joycon = None
            raise

    def get_action(self) -> RobotAction:
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        return {
            **{f"left_{k}": v for k, v in self._left_switch.get_action().items()},
            **{f"right_{k}": v for k, v in self._right_switch.get_action().items()},
        }

    def poll_buttons(self) -> None:
        pass

    def get_reset_button(self) -> bool:
        current_pressed = self._right_switch._pressed(self.config.right_reset_button)
        just_pressed = current_pressed and not self._was_reset_button_pressed
        self._was_reset_button_pressed = current_pressed
        return just_pressed

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        raise NotImplementedError("BiSwitchJoycon does not support feedback.")

    def disconnect(self) -> None:
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        try:
            self._left_switch.disconnect()
        finally:
            self._right_switch.disconnect()
            self._is_connected = False
            self.joycon = None
        logger.info("%s disconnected.", self)
