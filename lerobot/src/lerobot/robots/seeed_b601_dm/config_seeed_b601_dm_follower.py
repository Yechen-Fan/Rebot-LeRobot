from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from lerobot.cameras import CameraConfig
from lerobot.robots.robot import RobotConfig

from .config_serial_gripper import SerialGripperConfig


class ActionMode(str, Enum):
    """Control mode for reBot Arm B601.

    Note: vel and torque are not observed.
    JOINT:
        Action: joint positions (6D) + gripper (1D) = 7D
        Observation: joint positions (6D) + gripper (1D) = 7D

    CARTESIAN:
        Action: TCP position (3D) + rotation (6D) + gripper (1D) = 10D
        Observation: TCP position (3D) + rotation (6D) + gripper (1D) = 10D
    """

    JOINT = "joint"
    CARTESIAN = "cartesian"

class ControlMode(str, Enum):
    """Control mode for reBot Arm B601.

    POS_VEL:
        Position velocity control.
    MIT:
        MIT control.
    """
    POS_VEL = "pos_vel"
    MIT = "mit"

class GripperType(str, Enum):
    """Gripper type for reBot Arm B601."""
    SERIAL = "serial" # Serial gripper.
    REBOTARMB601 = "rebotarm_b601" # reBotArm B601 gripper.

@RobotConfig.register_subclass("seeed_b601_dm_follower")
@dataclass
class SeeedB601DMFollowerConfig(RobotConfig):
    """Configuration for the Seeed B601 follower driven by rebotarm_control_dm."""

    # Communication channel for rebotarm_control_dm: SocketCAN (can0) or Damiao serial bridge (/dev/ttyACM0).
    port: str
    can_adapter: str = "damiao"

    flexitac_inner_port: str = ""
    flexitac_outer_port: str = ""
    flexitac_baudrate: int = 2_000_000
    flexitac_serial_timeout_s: float = 0.05
    flexitac_calibration_frames: int = 30
    flexitac_calibration_timeout_s: float = 5.0
    flexitac_calibration_max_span: float = 30.0
    flexitac_calibration_mean_min: float = 0.0
    flexitac_calibration_mean_max: float = 200.0
    flexitac_max_frame_age_s: float = 0.2
    flexitac_auto_zero: bool = True

    # Optional prebuilt rebotarm_control_dm actuator config.
    # If omitted, this LeRobot plugin generates one from the B601 mapping below.
    arm_cfg_path: str | Path | None = None
    # URDF used by the Cartesian kinematics model. Relative paths are resolved
    # by rebotarm_control_dm; None uses its built-in default URDF.
    kinematic_urdf_path: str | Path | None = None

    disable_torque_on_disconnect: bool = False
    max_relative_target: float | dict[str, float] | None = 6.0
    ik_accept_error: float = 0.02
    ik_reject_error: float = 0.08
    ik_position_backoff_scales: list[float] = field(default_factory=lambda: [0.5, 0.25, 0.1])
    max_ik_joint_step_deg: float | list[float] | dict[str, float] = field(
        default_factory=lambda: {
            "shoulder_pan": 8.0,
            "shoulder_lift": 8.0,
            "elbow_flex": 8.0,
            "wrist_flex": 4.0,
            "wrist_yaw": 4.0,
            "wrist_roll": 4.0,
        }
    )
    runtime_gripper_feedback_grace_s: float = 0.25

    # LeRobot-facing action and control modes.
    action_mode: ActionMode = ActionMode.JOINT
    control_mode: ControlMode = ControlMode.POS_VEL

    # LeRobot-facing action mode. "joint" exposes joint targets; "cartesian" exposes TCP targets.
    control_gripper: bool = True
    gripper_type: GripperType = GripperType.REBOTARMB601

    # Serial gripper parameters.
    serial_gripper_sn: str = "000033"
    serial_gripper_port: str = ""
    serial_gripper_baudrate: int = 115200
    serial_gripper_serial_timeout: float = 1.0
    serial_gripper_device_id: int = 1
    serial_gripper_min_pos: float = 0.0
    serial_gripper_max_pos: float = 85.0
    serial_gripper_v_max: float = 80.0
    serial_gripper_f_max: float = 27.0
    serial_gripper_init_open: bool = True
    serial_gripper: SerialGripperConfig | None = field(default=None, init=False)

    # reBotArm B601 gripper force control parameters.
    enabled_gripper_force: bool = True # Whether to enable gripper force control.
    gripper_force_pos_torque_ratio: float = 0.1  # Matches the previously validated B601 plugin.

    # Get observation.
    enable_observation_joint_pos: bool = False
    enable_observation_joint_vel: bool = False
    enable_observation_joint_torque: bool = False
    enable_observation_gripper_vel: bool = False
    enable_observation_gripper_torque: bool = False

    # RT control loop parameters.
    rt_rate: float = 150.0
    rt_command_gap_us: int = 0
    rt_priority: int = 0
    rt_cpu: int | None = 3

    # Debugging parameters.
    damiao_tx_debug: int = 0 # print debug Damiao TX information.

    # Maps LeRobot joint names to (command id, feedback id).
    motor_can_ids: dict[str, tuple[int, int]] = field(
        default_factory=lambda: {
            "shoulder_pan": (0x01, 0x11),
            "shoulder_lift": (0x02, 0x12),
            "elbow_flex": (0x03, 0x13),
            "wrist_flex": (0x04, 0x14),
            "wrist_yaw": (0x05, 0x15),
            "wrist_roll": (0x06, 0x16),
            "gripper": (0x07, 0x17),
        }
    )
    motor_models: dict[str, str] = field(
        default_factory=lambda: {
            "shoulder_pan": "4340P",
            "shoulder_lift": "4340P",
            "elbow_flex": "4340P",
            "wrist_flex": "4310",
            "wrist_yaw": "4310",
            "wrist_roll": "4310",
            "gripper": "4310",
        }
    )
    motor_vendor: str | None = None
    motor_vendors: dict[str, str] = field(default_factory=dict)

    # Position velocity limits in degrees/s. These are converted to rad/s for rebotarm_control_dm.
    pos_vel_velocity: float | list[float] = field(
        default_factory=lambda: [150, 150, 150, 150, 150, 150, 300]
    )
    # MIT gains used by rebotarm_control_dm in ControlMode.MIT.
    # Tuple order: (kp, kd).
    mit_gains: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "shoulder_pan": (120.0, 8.0),
            "shoulder_lift": (120.0, 8.0),
            "elbow_flex": (120.0, 8.0),
            "wrist_flex": (18.0, 2.0),
            "wrist_yaw": (18.0, 2.0),
            "wrist_roll": (18.0, 2.0),
            "gripper": (8.0, 1.0),
        }
    )
    # Damiao POS_VEL register gains written before starting the RT loop.
    # Tuple order: (vel_kp, vel_ki, pos_kp, pos_ki).
    pos_vel_gains: dict[str, tuple[float, float, float, float]] = field(
        default_factory=lambda: {
            "shoulder_pan": (0.0125, 0.004, 150.0, 0.5),
            "shoulder_lift": (0.013, 0.004, 200.0, 10.0),
            "elbow_flex": (0.013, 0.004, 200.0, 10.0),
            "wrist_flex": (0.0008, 0.002, 50.0, 1.0),
            "wrist_yaw": (0.0008, 0.004, 50.0, 1.0),
            "wrist_roll": (0.0008, 0.002, 50.0, 1.0),
            "gripper": (0.0008, 0.002, 50.0, 1.0),
        }
    )

    # Per-joint velocity limit (deg/s) used when returning to the initial pose
    # (reset / disconnect). Accepts a scalar (all joints), a list (per motor in
    # the motor_can_ids order), or a dict keyed by joint name. Each value is
    # clamped to <= the corresponding pos_vel_velocity at runtime.
    return_to_initial_vlim_deg_s: float | list[float] | dict[str, float] = field(
        default_factory=lambda: {
            "shoulder_pan": 45.0,
            "shoulder_lift": 45.0,
            "elbow_flex": 45.0,
            "wrist_flex": 45.0,
            "wrist_yaw": 45.0,
            "wrist_roll": 45.0,
            "gripper": 150.0,
        }
    )

    # Soft limits in degrees. The calibrated built-in gripper opens toward the minimum.
    joint_limits: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "shoulder_pan": (-145.0, 145.0),
            "shoulder_lift": (-170.0, 1.0),
            "elbow_flex": (-200.0, 1.0),
            "wrist_flex": (-80.0, 90.0),
            "wrist_yaw": (-90.0, 90.0),
            "wrist_roll": (-90.0, 90.0),
            "gripper": (-270.0, 0.0),
        }
    )

    # Cameras for the follower robot.
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            # "head": RealSenseCameraConfig(
            #     serial_number_or_name="021422060263",
            #     fps=30,
            #     width=640,
            #     height=480,
            #     warmup_s=1.0,
            # ),
            # "head": OpenCVCameraConfig(
            #     index_or_path="/dev/video6",
            #     fourcc="YUYV",
            #     fps=60,
            #     width=640,
            #     height=480,
            #     warmup_s=1.0,
            # ),
        }
    )

    def __post_init__(self):
        super().__post_init__()

        tactile_ports = [self.flexitac_inner_port, self.flexitac_outer_port]
        if any(tactile_ports) and not all(tactile_ports):
            raise ValueError("Configure both single-arm FlexiTac ports or leave both empty")
        if self.flexitac_baudrate <= 0 or self.flexitac_calibration_frames <= 0:
            raise ValueError("FlexiTac baudrate and calibration frame count must be positive")
        if self.flexitac_serial_timeout_s <= 0 or self.flexitac_max_frame_age_s <= 0:
            raise ValueError("FlexiTac timeouts must be positive")
        if not 0 <= self.flexitac_calibration_mean_min <= self.flexitac_calibration_mean_max <= 255:
            raise ValueError("FlexiTac calibration mean range must satisfy 0 <= min <= max <= 255")

        if isinstance(self.gripper_type, str):
            self.gripper_type = GripperType(self.gripper_type)

        if self.gripper_type == GripperType.SERIAL:
            self.serial_gripper = SerialGripperConfig(
                sn=str(self.serial_gripper_sn) if self.serial_gripper_sn else None,
                port=self.serial_gripper_port,
                baudrate=self.serial_gripper_baudrate,
                serial_timeout=self.serial_gripper_serial_timeout,
                device_id=self.serial_gripper_device_id,
                gripper_min_pos=self.serial_gripper_min_pos,
                gripper_max_pos=self.serial_gripper_max_pos,
                gripper_v_max=self.serial_gripper_v_max,
                gripper_f_max=self.serial_gripper_f_max,
                init_open=self.serial_gripper_init_open,
            )
        else:
            self.serial_gripper = None
