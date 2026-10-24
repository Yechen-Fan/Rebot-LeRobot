# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Simple script to control a robot from teleoperation.

Example:

```shell
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --robot.id=black \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem58760431551 \
    --teleop.id=blue \
    --display_data=true
```

Example teleoperation with bimanual so100:

```shell
lerobot-teleoperate \
  --robot.type=bi_so_follower \
  --robot.left_arm_config.port=/dev/tty.usbmodem5A460822851 \
  --robot.right_arm_config.port=/dev/tty.usbmodem5A460814411 \
  --robot.id=bimanual_follower \
  --robot.left_arm_config.cameras='{
    wrist: {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30},
  }' --robot.right_arm_config.cameras='{
    wrist: {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30},
  }' \
  --teleop.type=bi_so_leader \
  --teleop.left_arm_config.port=/dev/tty.usbmodem5A460852721 \
  --teleop.right_arm_config.port=/dev/tty.usbmodem5A460819811 \
  --teleop.id=bimanual_leader \
  --display_data=true
```

"""

import logging
import time
from dataclasses import dataclass

import rerun as rr

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.processor import (
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
)
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    make_robot_from_config,
    seeed_b601_dm,
)
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    make_teleoperator_from_config,
    quest3,
    switch_joycon,
)
from lerobot.utils.control_utils import (
    connect_teleoperator,
    handle_rejected_robot_action,
    install_cleanup_signal_handlers,
    reset_robot_from_teleoperator,
    sync_teleoperator_control_reference,
    sync_teleoperator_clutch,
    teleoperator_reset_requested,
)
from lerobot.utils.errors import RecoverableObservationError
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, move_cursor_up
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

COMPACT_TELEOP_TYPES = {"quest3", "bi_quest3", "switch_joycon", "bi_switch_joycon"}


@dataclass
class TeleoperateConfig:
    # TODO: pepijn, steven: if more robots require multiple teleoperators (like lekiwi) its good to make this possibele in teleop.py and record.py with List[Teleoperator]
    teleop: TeleoperatorConfig
    robot: RobotConfig
    # Limit the maximum frames per second.
    fps: int = 60
    teleop_time_s: float | None = None
    # Display all cameras on screen
    display_data: bool = False
    # Display data on a remote Rerun server
    display_ip: str | None = None
    # Port of the remote Rerun server
    display_port: int | None = None
    # Whether to  display compressed images in Rerun
    display_compressed_images: bool = False


def teleop_loop(
    teleop: Teleoperator,
    robot: Robot,
    fps: int,
    teleop_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_observation_processor: RobotProcessorPipeline[RobotObservation, RobotObservation],
    display_data: bool = False,
    duration: float | None = None,
    display_compressed_images: bool = False,
):
    """
    This function continuously reads actions from a teleoperation device, processes them through optional
    pipelines, sends them to a robot, and optionally displays the robot's state. The loop runs at a
    specified frequency until a set duration is reached or it is manually interrupted.

    Args:
        teleop: The teleoperator device instance providing control actions.
        robot: The robot instance being controlled.
        fps: The target frequency for the control loop in frames per second.
        display_data: If True, fetches robot observations and displays them in the console and Rerun.
        display_compressed_images: If True, compresses images before sending them to Rerun for display.
        duration: The maximum duration of the teleoperation loop in seconds. If None, the loop runs indefinitely.
        teleop_action_processor: An optional pipeline to process raw actions from the teleoperator.
        robot_action_processor: An optional pipeline to process actions before they are sent to the robot.
        robot_observation_processor: An optional pipeline to process raw observations from the robot.
    """

    show_console = teleop.config.type not in COMPACT_TELEOP_TYPES
    display_len = max(len(key) for key in robot.action_features) if show_console else 0
    start = time.perf_counter()
    last_rate_report = start - 0.2

    while True:
        loop_start = time.perf_counter()

        # Get robot observation
        # Not really needed for now other than for visualization
        # teleop_action_processor can take None as an observation
        # given that it is the identity processor as default
        try:
            obs = robot.get_observation()
        except RecoverableObservationError as error:
            recover = getattr(robot, "recover_observation", None)
            if recover is None:
                raise
            logging.error("Observation failed during teleoperation: %s", error)
            while True:
                try:
                    recover()
                    logging.info("Observation source recovered; resuming teleoperation.")
                    sync_teleoperator_control_reference(teleop, robot, reset_target=True)
                    break
                except (RecoverableObservationError, OSError) as recovery_error:
                    logging.warning("Observation recovery failed; retrying in 1 second: %s", recovery_error)
                    time.sleep(1.0)
            continue

        sync_teleoperator_control_reference(teleop, robot)
        if teleop.feedback_features:
            teleop.send_feedback(obs)

        if teleoperator_reset_requested(teleop):
            logging.info("Teleoperator requested a return to the initial robot position.")
            reset_robot_from_teleoperator(teleop, robot)
            continue

        # Get teleop action
        raw_action = teleop.get_action()
        sync_teleoperator_clutch(teleop, robot)

        # Process teleop action through pipeline
        teleop_action = teleop_action_processor((raw_action, obs))

        # Process action for robot through pipeline
        robot_action_to_send = robot_action_processor((teleop_action, obs))

        # Send processed action to robot (robot_action_processor.to_output should return RobotAction)
        sent_action = robot.send_action(robot_action_to_send)
        handle_rejected_robot_action(teleop, robot)

        if display_data:
            # Process robot observation through pipeline
            obs_transition = robot_observation_processor(obs)

            log_rerun_data(
                observation=obs_transition,
                action=sent_action,
                compress_images=display_compressed_images,
            )

            if show_console:
                print("\n" + "-" * (display_len + 10))
                print(f"{'NAME':<{display_len}} | {'NORM':>7}")
                # Display the final robot action that was sent
                for motor, value in robot_action_to_send.items():
                    print(f"{motor:<{display_len}} | {value:>7.2f}")
                move_cursor_up(len(robot_action_to_send) + 3)

        dt_s = time.perf_counter() - loop_start
        precise_sleep(max(1 / fps - dt_s, 0.0))
        if show_console:
            loop_s = time.perf_counter() - loop_start
            print(f"Teleop loop time: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")
            move_cursor_up(1)
        else:
            now = time.perf_counter()
            if now - last_rate_report >= 0.2:
                loop_s = now - loop_start
                print(f"\r循环频率: {1 / max(loop_s, 1e-9):.0f} Hz\033[K", end="", flush=True)
                last_rate_report = now

        if duration is not None and time.perf_counter() - start >= duration:
            return


@parser.wrap()
def teleoperate(cfg: TeleoperateConfig):
    compact_console = cfg.teleop.type in COMPACT_TELEOP_TYPES
    init_logging(console_level="ERROR" if compact_console else "INFO")
    install_cleanup_signal_handlers()
    if cfg.display_data:
        init_rerun(session_name="teleoperation", ip=cfg.display_ip, port=cfg.display_port)
    display_compressed_images = (
        True
        if (cfg.display_data and cfg.display_ip is not None and cfg.display_port is not None)
        else cfg.display_compressed_images
    )

    teleop = make_teleoperator_from_config(cfg.teleop)
    robot = make_robot_from_config(cfg.robot)
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()
    teleop_connected = False
    robot_connected = False
    run_error: Exception | None = None
    cleanup_error: Exception | None = None
    try:
        if compact_console:
            print("正在连接机械臂…", flush=True)
        robot.connect()
        robot_connected = True
        if compact_console:
            print("正在连接手柄并校准，请保持手柄静止…", flush=True)
        connect_teleoperator(teleop, robot)
        teleop_connected = True
        if compact_console:
            print("遥操作已就绪。", flush=True)

        teleop_loop(
            teleop=teleop,
            robot=robot,
            fps=cfg.fps,
            display_data=cfg.display_data,
            duration=cfg.teleop_time_s,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            display_compressed_images=display_compressed_images,
        )
    except KeyboardInterrupt:
        pass
    except Exception as error:
        if compact_console:
            print("\r\033[K", end="", flush=True)
        logging.exception("Teleoperation stopped because of an error: %s", error)
        run_error = error
    finally:
        if compact_console:
            print("\r\033[K", end="", flush=True)
        if cfg.display_data:
            rr.rerun_shutdown()
        if teleop_connected and getattr(teleop, "is_connected", False):
            try:
                teleop.disconnect()
            except Exception as error:
                logging.exception("Failed to disconnect the teleoperator cleanly.")
                cleanup_error = cleanup_error or error
        if robot_connected and getattr(robot, "is_connected", False):
            try:
                robot.disconnect()
            except Exception as error:
                logging.exception("Failed to return the robot to zero or disconnect cleanly.")
                cleanup_error = cleanup_error or error

    if run_error is not None:
        raise run_error
    if cleanup_error is not None:
        raise cleanup_error


def main():
    register_third_party_plugins()
    teleoperate()


if __name__ == "__main__":
    main()
