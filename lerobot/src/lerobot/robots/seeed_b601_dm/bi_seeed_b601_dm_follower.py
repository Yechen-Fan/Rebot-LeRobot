import logging
import time
from concurrent.futures import ThreadPoolExecutor, wait
from functools import cached_property, partial
from typing import Any

import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.processor import RobotAction, RobotObservation
from lerobot.robots.robot import Robot
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.utils.policy_motion import PolicyMotionConfig

from .config_bi_seeed_b601_dm_follower import BiSeeedB601DMFollowerConfig
from .config_seeed_b601_dm_follower import SeeedB601DMFollowerConfig
from .flexitac import COLS, ROWS, TACTILE_NAMES, FlexiTacManager
from .seeed_b601_dm_follower import SeeedB601DMFollower

logger = logging.getLogger(__name__)


class BiSeeedB601DMFollower(Robot):
    """Dual Seeed B601 follower composed from two RT B601 follower instances."""

    config_class = BiSeeedB601DMFollowerConfig
    name = "bi_seeed_b601_follower"

    def __init__(self, config: BiSeeedB601DMFollowerConfig):
        super().__init__(config)
        self.config = config
        self._control_executor: ThreadPoolExecutor | None = None
        self.left = SeeedB601DMFollower(self._single_config("left"))
        self.right = SeeedB601DMFollower(self._single_config("right"))
        self.cameras = make_cameras_from_configs(config.cameras)
        tactile_ports = {
            "left_inner": config.flexitac_left_inner_port,
            "left_outer": config.flexitac_left_outer_port,
            "right_inner": config.flexitac_right_inner_port,
            "right_outer": config.flexitac_right_outer_port,
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

    def _single_config(self, side: str) -> SeeedB601DMFollowerConfig:
        if side not in {"left", "right"}:
            raise ValueError(f"Unknown side {side!r}.")

        return SeeedB601DMFollowerConfig(
            id=self.config.left_id if side == "left" else self.config.right_id,
            calibration_dir=self.calibration_dir,
            port=self.config.left_port if side == "left" else self.config.right_port,
            can_adapter=self.config.can_adapter,
            arm_cfg_path=(
                self.config.left_arm_cfg_path if side == "left" else self.config.right_arm_cfg_path
            ),
            kinematic_urdf_path=(
                self.config.left_kinematic_urdf_path
                if side == "left"
                else self.config.right_kinematic_urdf_path
            ),
            disable_torque_on_disconnect=self.config.disable_torque_on_disconnect,
            max_relative_target=self.config.max_relative_target,
            ik_accept_error=self.config.ik_accept_error,
            ik_reject_error=self.config.ik_reject_error,
            ik_position_backoff_scales=self.config.ik_position_backoff_scales,
            max_ik_joint_step_deg=self.config.max_ik_joint_step_deg,
            runtime_gripper_feedback_grace_s=self.config.runtime_gripper_feedback_grace_s,
            action_mode=self.config.action_mode,
            control_mode=self.config.control_mode,
            control_gripper=self.config.control_gripper,
            gripper_type=(
                self.config.left_gripper_type if side == "left" else self.config.right_gripper_type
            ),
            serial_gripper_sn=(
                self.config.left_serial_gripper_sn
                if side == "left"
                else self.config.right_serial_gripper_sn
            ),
            serial_gripper_port=(
                self.config.left_serial_gripper_port
                if side == "left"
                else self.config.right_serial_gripper_port
            ),
            serial_gripper_baudrate=(
                self.config.left_serial_gripper_baudrate
                if side == "left"
                else self.config.right_serial_gripper_baudrate
            ),
            serial_gripper_serial_timeout=(
                self.config.left_serial_gripper_serial_timeout
                if side == "left"
                else self.config.right_serial_gripper_serial_timeout
            ),
            serial_gripper_device_id=(
                self.config.left_serial_gripper_device_id
                if side == "left"
                else self.config.right_serial_gripper_device_id
            ),
            serial_gripper_min_pos=(
                self.config.left_serial_gripper_min_pos
                if side == "left"
                else self.config.right_serial_gripper_min_pos
            ),
            serial_gripper_max_pos=(
                self.config.left_serial_gripper_max_pos
                if side == "left"
                else self.config.right_serial_gripper_max_pos
            ),
            serial_gripper_v_max=(
                self.config.left_serial_gripper_v_max
                if side == "left"
                else self.config.right_serial_gripper_v_max
            ),
            serial_gripper_f_max=(
                self.config.left_serial_gripper_f_max
                if side == "left"
                else self.config.right_serial_gripper_f_max
            ),
            serial_gripper_init_open=(
                self.config.left_serial_gripper_init_open
                if side == "left"
                else self.config.right_serial_gripper_init_open
            ),
            enabled_gripper_force=self.config.enabled_gripper_force,
            gripper_force_pos_torque_ratio=self.config.gripper_force_pos_torque_ratio,
            enable_observation_joint_pos=self.config.enable_observation_joint_pos,
            enable_observation_joint_vel=self.config.enable_observation_joint_vel,
            enable_observation_joint_torque=self.config.enable_observation_joint_torque,
            enable_observation_gripper_vel=self.config.enable_observation_gripper_vel,
            enable_observation_gripper_torque=self.config.enable_observation_gripper_torque,
            rt_rate=self.config.rt_rate,
            rt_command_gap_us=self.config.rt_command_gap_us,
            rt_priority=(
                self.config.left_rt_priority if side == "left" else self.config.right_rt_priority
            ),
            rt_cpu=self.config.left_rt_cpu if side == "left" else self.config.right_rt_cpu,
            damiao_tx_debug=self.config.damiao_tx_debug,
            motor_can_ids=self.config.motor_can_ids,
            motor_models=self.config.motor_models,
            motor_vendor=self.config.motor_vendor,
            motor_vendors=self.config.motor_vendors,
            pos_vel_velocity=self.config.pos_vel_velocity,
            mit_gains=(
                self.config.left_mit_gains if side == "left" else self.config.right_mit_gains
            ),
            pos_vel_gains=(
                self.config.left_pos_vel_gains if side == "left" else self.config.right_pos_vel_gains
            ),
            return_to_initial_vlim_deg_s=self.config.return_to_initial_vlim_deg_s,
            joint_limits=self.config.joint_limits,
            cameras={},
        )

    @staticmethod
    def _prefix_key(side: str, key: str) -> str:
        return f"{side}_{key}"

    @staticmethod
    def _strip_side_prefix(key: str, side: str) -> str | None:
        prefix = f"{side}_"
        if key.startswith(prefix):
            return key[len(prefix):]
        return None

    @staticmethod
    def _prefix_dict(side: str, data: dict[str, Any]) -> dict[str, Any]:
        return {BiSeeedB601DMFollower._prefix_key(side, key): value for key, value in data.items()}

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        camera_features = {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
            for cam in self.cameras
        }
        return {
            **self._prefix_dict("left", self.left.observation_features),
            **self._prefix_dict("right", self.right.observation_features),
            **camera_features,
            **(
                {
                    f"tactile.{name}": PolicyFeature(FeatureType.TACTILE, (ROWS, COLS))
                    for name in TACTILE_NAMES
                }
                | {
                    f"tactile_baseline.{name}": PolicyFeature(
                        FeatureType.AUXILIARY, (ROWS, COLS)
                    )
                    for name in TACTILE_NAMES
                }
                | {
                    f"{kind}.{name}": PolicyFeature(FeatureType.AUXILIARY, (ROWS, COLS))
                    for kind in ("tactile_raw", "tactile_raw_baseline")
                    for name in TACTILE_NAMES
                }
                if self.flexitac
                else {}
            ),
        }

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {
            **self._prefix_dict("left", self.left.action_features),
            **self._prefix_dict("right", self.right.action_features),
        }

    @property
    def is_connected(self) -> bool:
        return (
            self.left.is_connected
            and self.right.is_connected
            and all(cam.is_connected for cam in self.cameras.values())
        )

    @property
    def is_calibrated(self) -> bool:
        return self.left.is_calibrated and self.right.is_calibrated

    @property
    def last_ik_accepted(self) -> bool:
        return self.left.last_ik_accepted and self.right.last_ik_accepted

    def calibrate(self) -> None:
        self.left.calibrate()
        self.right.calibrate()

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        logger.info(
            "Connecting dual B601 arms: left=%s right=%s...",
            self.config.left_port,
            self.config.right_port,
        )
        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [
                    executor.submit(self.left.connect, calibrate),
                    executor.submit(self.right.connect, calibrate),
                ]
                for future in futures:
                    future.result()

            if not self.cameras:
                logger.debug("No cameras configured; skipping camera connect.")
            else:
                logger.info("Connecting %d camera(s): %s...", len(self.cameras), ", ".join(self.cameras))
                for cam in self.cameras.values():
                    cam.connect()
                logger.info("All cameras connected.")
            if self.flexitac is not None:
                logger.info("Connecting and calibrating four FlexiTac sensors with no contact...")
                self.flexitac.connect(
                    calibration_frames=self.config.flexitac_calibration_frames,
                    calibration_timeout_s=self.config.flexitac_calibration_timeout_s,
                    calibration_max_span=self.config.flexitac_calibration_max_span,
                    calibration_mean_min=self.config.flexitac_calibration_mean_min,
                    calibration_mean_max=self.config.flexitac_calibration_mean_max,
                )
        except Exception:
            self._disconnect_connected_children()
            raise

        logger.debug("%s connected.", self)

    def configure(self) -> None:
        self.left.configure()
        self.right.configure()

    def _parallel_control(self, left, right):
        # Observation, reference sync and action each used to create and join
        # two new threads on every frame. Keep the same two workers instead.
        if self._control_executor is None:
            self._control_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="b601-control")
        futures = [self._control_executor.submit(call) if call is not None else None
                   for call in (left, right)]
        # Preserve the old context manager's guarantee: both sides finish
        # before propagating an exception and starting robot cleanup.
        wait([future for future in futures if future is not None])
        return tuple(future.result() if future is not None else {} for future in futures)

    def _close_control_executor(self) -> None:
        if self._control_executor is not None:
            self._control_executor.shutdown(wait=True)
            self._control_executor = None

    def get_observation(self) -> RobotObservation:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start = time.perf_counter()
        # Read cached images without waiting for each camera's next frame,
        # then obtain the joint feedback used to generate the next action.
        obs: RobotObservation = {
            cam_key: cam.read_latest(max_age_ms=200) for cam_key, cam in self.cameras.items()
        }
        left_obs, right_obs = self._parallel_control(self.left.get_observation, self.right.get_observation)

        obs.update({
            **self._prefix_dict("left", left_obs),
            **self._prefix_dict("right", right_obs),
        })
        if self.flexitac is not None:
            obs.update(self.flexitac.get_contact_maps({
                "left": left_obs.get("gripper.pos"), "right": right_obs.get("gripper.pos"),
            }))

        logger.debug("%s get_observation took: %.1fms", self, (time.perf_counter() - start) * 1e3)
        return obs

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
        logger.warning("Recovering four FlexiTac readers without changing their baselines")
        self.flexitac.recover(timeout_s=self.config.flexitac_calibration_timeout_s)

    def _split_action(self, action: RobotAction) -> tuple[RobotAction, RobotAction]:
        left_action: RobotAction = {}
        right_action: RobotAction = {}
        for key, value in action.items():
            left_key = self._strip_side_prefix(key, "left")
            if left_key is not None:
                left_action[left_key] = value
                continue

            right_key = self._strip_side_prefix(key, "right")
            if right_key is not None:
                right_action[right_key] = value

        return left_action, right_action

    def set_teleop_translation(self, enabled: bool | tuple[bool, bool]) -> None:
        left, right = enabled if isinstance(enabled, tuple) else (enabled, enabled)
        self.left.set_teleop_translation(left)
        self.right.set_teleop_translation(right)

    def set_teleop_wrist_rotation(self, enabled: bool | tuple[bool, bool]) -> None:
        left_enabled, right_enabled = enabled if isinstance(enabled, tuple) else (enabled, enabled)
        self.left.set_teleop_wrist_rotation(left_enabled)
        self.right.set_teleop_wrist_rotation(right_enabled)

    def set_teleop_motion_enabled(self, enabled: tuple[bool, bool], *, immediate_stop: bool = False) -> None:
        self.left.set_teleop_motion_enabled(enabled[0], immediate_stop=immediate_stop)
        self.right.set_teleop_motion_enabled(enabled[1], immediate_stop=immediate_stop)

    def send_action(self, action: RobotAction) -> RobotAction:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        left_action, right_action = self._split_action(action)
        if not left_action and not right_action:
            raise ValueError("Dual B601 action must contain left_* or right_* keys.")

        sent_left, sent_right = self._parallel_control(
            partial(self.left.send_action, left_action) if left_action else None,
            partial(self.right.send_action, right_action) if right_action else None,
        )

        return {
            **self._prefix_dict("left", sent_left),
            **self._prefix_dict("right", sent_right),
        }

    def set_policy_motion_limits(self, config: PolicyMotionConfig | None, *, period: float = 1. / 30.) -> None:
        try:
            self.left.set_policy_motion_limits(config, period=period)
            self.right.set_policy_motion_limits(config, period=period)
        except Exception:
            self.left.set_policy_motion_limits(None)
            self.right.set_policy_motion_limits(None)
            raise

    def get_current_tcp_pose_quat(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        return self._parallel_control(self.left.get_current_tcp_pose_quat, self.right.get_current_tcp_pose_quat)

    @property
    def last_return_to_initial_succeeded(self) -> bool | None:
        results = (
            self.left.last_return_to_initial_succeeded,
            self.right.last_return_to_initial_succeeded,
        )
        if any(result is None for result in results):
            return None
        return all(bool(result) for result in results)

    def reset_to_initial_position(self) -> bool:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(self.left.reset_to_initial_position),
                executor.submit(self.right.reset_to_initial_position),
            ]
            return all(future.result() for future in futures)

    def _disconnect_connected_children(self) -> None:
        self._close_control_executor()
        for arm in (self.left, self.right):
            try:
                if arm.arm is not None:
                    arm.disconnect()
            except Exception:
                logger.debug("Failed to disconnect %s during cleanup.", arm, exc_info=True)
        for cam in self.cameras.values():
            try:
                if cam.is_connected:
                    cam.disconnect()
            except Exception:
                logger.debug("Failed to disconnect camera during cleanup.", exc_info=True)
        if self.flexitac is not None:
            self.flexitac.disconnect()

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        try:
            self._close_control_executor()
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [
                    executor.submit(self.left.disconnect),
                    executor.submit(self.right.disconnect),
                ]
                for future in futures:
                    future.result()
        finally:
            for cam in self.cameras.values():
                cam.disconnect()
            if self.flexitac is not None:
                self.flexitac.disconnect()

        logger.info("%s disconnected.", self)
