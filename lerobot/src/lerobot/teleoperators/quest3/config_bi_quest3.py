from dataclasses import dataclass, field

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("bi_quest3")
@dataclass
class BiQuest3Config(TeleoperatorConfig):
    """Configuration for bimanual Quest3 controller teleoperation."""

    id: str = "bi_quest3"
    adb_ip: str | None = None
    adb_port: int = 5555

    pos_sensitivity: float = 1.5
    pos_sensitivity_xyz: list[float] = field(default_factory=lambda: [1.5, 1.5, 1.5])
    ori_sensitivity: float = 1.5
    filter_window_size: int = 3
    grip_deadband_frames: int = 2

    left_gripper_width: float = 1.0
    right_gripper_width: float = 1.0

    grip_enable_threshold: float = 0.5
    grip_disable_threshold: float = 0.3
    orientation_offset_warning_deg: float = 180.0
    target_tcp_drift_max_deg: float = 45.0
    position_jump_threshold: float = 0.1
    max_pos_velocity: float = 1.0
    max_rot_velocity: float = 6.28

    def __post_init__(self) -> None:
        if not 0.0 <= self.grip_disable_threshold < self.grip_enable_threshold <= 1.0:
            raise ValueError("Grip thresholds must satisfy 0 <= disable < enable <= 1.")
        if self.filter_window_size < 1:
            raise ValueError("filter_window_size must be >= 1.")
        if len(self.pos_sensitivity_xyz) != 3:
            raise ValueError("pos_sensitivity_xyz must contain exactly 3 values.")
        if self.pos_sensitivity < 0 or self.ori_sensitivity < 0:
            raise ValueError("Quest 3 sensitivities must be non-negative.")
        if any(value < 0 for value in self.pos_sensitivity_xyz):
            raise ValueError("Quest 3 per-axis sensitivities must be non-negative.")
        if self.grip_deadband_frames < 0:
            raise ValueError("grip_deadband_frames must be >= 0.")
        if self.adb_port <= 0:
            raise ValueError("adb_port must be positive.")
