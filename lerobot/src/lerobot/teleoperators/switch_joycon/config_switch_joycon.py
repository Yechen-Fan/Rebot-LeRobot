import math
from dataclasses import dataclass, field

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("switch_joycon")
@dataclass
class SwitchJoyconConfig(TeleoperatorConfig):
    """Switch Joy-Con Cartesian TCP teleoperator for the B601 RT stack."""

    id: str = "switch_joycon"
    device: str = "right"
    input_backend: str = "evdev"

    enable_button: str = "zr"
    reset_button: str = "a"
    gripper_close_button: str = "x"
    gripper_open_button: str = "b"
    recalibrate_button: str = "plus"
    z_up_button: str = "sr"
    z_down_button: str = "sl"

    base_translation_speed_m_s: float = 0.05
    position_sensitivity: float = 1.5
    base_rotation_speed_rad_s: float = 1.0
    rotation_sensitivity: float = 2.0
    stick_deadzone: float = 0.18
    max_pos_velocity: float = 1.0
    max_rot_velocity: float = 6.28
    # Bound unexecuted wrist rotation relative to feedback, without changing tilt-to-rate mapping.
    max_orientation_target_lead_rad: float = math.radians(45.0)
    control_orientation: bool = True

    gripper_open_pos: float = 0.0
    gripper_close_pos: float = 1.0

    imu_calibrate_on_connect: bool = True
    position_axes: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    rotation_axes: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])

    evdev_device_path: str | None = None
    evdev_stick_x_axis: str = "ABS_RX"
    evdev_stick_y_axis: str = "ABS_RY"
    evdev_invert_stick_x: bool = True
    evdev_invert_stick_y: bool = True
    evdev_zr_button: str = "BTN_TR2"
    evdev_zl_button: str = "BTN_TL2"
    evdev_a_button: str = "BTN_EAST"
    evdev_b_button: str = "BTN_SOUTH"
    evdev_x_button: str = "BTN_NORTH"
    evdev_y_button: str = "BTN_WEST"
    evdev_plus_button: str = "BTN_START"
    evdev_minus_button: str = "BTN_SELECT"
    evdev_sr_button: str = "BTN_TL2"
    evdev_sl_button: str = "BTN_TL"
    evdev_stick_button: str = "BTN_THUMBR"
    evdev_orientation_axes: list[str] = field(
        default_factory=lambda: ["ABS_RX", "ABS_RY", "ABS_RZ"]
    )

    def __post_init__(self) -> None:
        if self.device not in {"right", "left"}:
            raise ValueError("device must be 'right' or 'left'.")
        if self.input_backend != "evdev":
            raise ValueError("input_backend must be 'evdev'.")
        if not 0.0 <= self.stick_deadzone < 1.0:
            raise ValueError("stick_deadzone must be in [0, 1).")
        if self.base_translation_speed_m_s < 0.0 or self.position_sensitivity < 0.0:
            raise ValueError("Translation speed and sensitivity must be non-negative.")
        if self.base_rotation_speed_rad_s < 0.0 or self.rotation_sensitivity < 0.0:
            raise ValueError("Rotation speed and sensitivity must be non-negative.")
        if self.max_pos_velocity < 0.0 or self.max_rot_velocity < 0.0:
            raise ValueError("Velocity limits must be non-negative.")
        if not 0.0 < self.max_orientation_target_lead_rad < math.pi:
            raise ValueError("max_orientation_target_lead_rad must be between 0 and pi radians.")
        if len(self.position_axes) != 3:
            raise ValueError("position_axes must contain exactly 3 values.")
        if len(self.rotation_axes) != 3:
            raise ValueError("rotation_axes must contain exactly 3 values.")
        if len(self.evdev_orientation_axes) != 3:
            raise ValueError("evdev_orientation_axes must contain exactly 3 values.")


@TeleoperatorConfig.register_subclass("bi_switch_joycon")
@dataclass
class BiSwitchJoyconConfig(TeleoperatorConfig):
    """Configuration for bimanual Switch Joy-Con teleoperation."""

    id: str = "bi_switch_joycon"
    left_device: str = "left"
    right_device: str = "right"
    input_backend: str = "evdev"

    left_enable_button: str = "zr"
    right_enable_button: str = "zr"
    left_reset_button: str = "a"
    right_reset_button: str = "a"
    left_gripper_close_button: str = "x"
    left_gripper_open_button: str = "b"
    right_gripper_close_button: str = "x"
    right_gripper_open_button: str = "b"
    recalibrate_button: str = "plus"
    z_up_button: str = "sr"
    z_down_button: str = "sl"

    base_translation_speed_m_s: float = 0.05
    position_sensitivity: float = 1.5
    base_rotation_speed_rad_s: float = 1.0
    rotation_sensitivity: float = 2.0
    stick_deadzone: float = 0.18
    max_pos_velocity: float = 1.0
    max_rot_velocity: float = 6.28
    max_orientation_target_lead_rad: float = math.radians(45.0)
    control_orientation: bool = True

    left_gripper_open_pos: float = 0.0
    left_gripper_close_pos: float = 1.0
    right_gripper_open_pos: float = 0.0
    right_gripper_close_pos: float = 1.0

    imu_calibrate_on_connect: bool = True
    position_axes: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    rotation_axes: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])

    evdev_device_path: str | None = None
    left_evdev_device_path: str | None = None
    right_evdev_device_path: str | None = None
    evdev_stick_x_axis: str = "ABS_RX"
    evdev_stick_y_axis: str = "ABS_RY"
    evdev_invert_stick_x: bool = True
    evdev_invert_stick_y: bool = True
    evdev_zr_button: str = "BTN_TR2"
    evdev_zl_button: str = "BTN_TL2"
    evdev_a_button: str = "BTN_EAST"
    evdev_b_button: str = "BTN_SOUTH"
    evdev_x_button: str = "BTN_NORTH"
    evdev_y_button: str = "BTN_WEST"
    evdev_plus_button: str = "BTN_START"
    evdev_minus_button: str = "BTN_SELECT"
    evdev_sr_button: str = "BTN_TL2"
    evdev_sl_button: str = "BTN_TL"
    evdev_stick_button: str = "BTN_THUMBR"
    evdev_orientation_axes: list[str] = field(
        default_factory=lambda: ["ABS_RX", "ABS_RY", "ABS_RZ"]
    )

    def __post_init__(self) -> None:
        if self.left_device not in {"right", "left"} or self.right_device not in {"right", "left"}:
            raise ValueError("left_device and right_device must be 'right' or 'left'.")
        if self.input_backend != "evdev":
            raise ValueError("input_backend must be 'evdev'.")
        if not 0.0 <= self.stick_deadzone < 1.0:
            raise ValueError("stick_deadzone must be in [0, 1).")
        if self.base_translation_speed_m_s < 0.0 or self.position_sensitivity < 0.0:
            raise ValueError("Translation speed and sensitivity must be non-negative.")
        if self.base_rotation_speed_rad_s < 0.0 or self.rotation_sensitivity < 0.0:
            raise ValueError("Rotation speed and sensitivity must be non-negative.")
        if self.max_pos_velocity < 0.0 or self.max_rot_velocity < 0.0:
            raise ValueError("Velocity limits must be non-negative.")
        if not 0.0 < self.max_orientation_target_lead_rad < math.pi:
            raise ValueError("max_orientation_target_lead_rad must be between 0 and pi radians.")
        if len(self.position_axes) != 3:
            raise ValueError("position_axes must contain exactly 3 values.")
        if len(self.rotation_axes) != 3:
            raise ValueError("rotation_axes must contain exactly 3 values.")
        if len(self.evdev_orientation_axes) != 3:
            raise ValueError("evdev_orientation_axes must contain exactly 3 values.")
