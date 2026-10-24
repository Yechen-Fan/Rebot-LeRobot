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

########################################################################################
# Utilities
########################################################################################


import logging
import signal
import traceback
from contextlib import nullcontext, suppress
from copy import copy
from functools import cache
from typing import Any

import numpy as np
import torch
from deepdiff import DeepDiff

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import DEFAULT_FEATURES, dataset_to_policy_features
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.processor import PolicyAction, PolicyProcessorPipeline
from lerobot.robots import Robot
from lerobot.teleoperators import Teleoperator
from lerobot.utils.errors import ReturnToInitialPositionError


def install_cleanup_signal_handlers() -> None:
    """Route termination signals through the CLI's normal KeyboardInterrupt cleanup path."""
    for signum_name in ("SIGTERM", "SIGHUP"):
        signum = getattr(signal, signum_name, None)
        if signum is None:
            continue
        with suppress(OSError, ValueError):
            signal.signal(signum, _raise_keyboard_interrupt)


def _raise_keyboard_interrupt(_signum, _frame) -> None:
    raise KeyboardInterrupt


def teleoperator_requires_control_reference(teleop: Teleoperator) -> bool:
    """Return whether a teleoperator must be anchored to the robot's current pose."""
    return bool(getattr(teleop, "requires_current_tcp_pose", False))


def sync_teleoperator_control_reference(
    teleop: Teleoperator, robot: Robot, *, reset_target: bool = False
) -> None:
    if not teleoperator_requires_control_reference(teleop):
        return
    get_pose = getattr(robot, "get_current_tcp_pose_quat", None)
    set_pose = getattr(teleop, "set_current_tcp_pose", None)
    if get_pose is None or set_pose is None:
        raise ValueError(
            f"{teleop} requires a current TCP pose, but {robot} does not provide a compatible reference"
        )
    set_pose(get_pose())
    if reset_target:
        reset_reference = getattr(teleop, "reset_to_current_tcp_pose", None)
        if reset_reference is not None:
            reset_reference()


def connect_teleoperator(teleop: Teleoperator, robot: Robot) -> None:
    """Connect a teleoperator after seeding any robot-relative control reference."""
    sync_teleoperator_control_reference(teleop, robot)
    teleop.connect()
    sync_teleoperator_control_reference(teleop, robot, reset_target=True)


def teleoperator_reset_requested(teleop: Teleoperator) -> bool:
    poll = getattr(teleop, "poll_buttons", None)
    if poll is not None:
        poll()
    get_reset = getattr(teleop, "get_reset_button", None)
    return bool(get_reset is not None and get_reset())


def reset_robot_from_teleoperator(teleop: Teleoperator, robot: Robot) -> None:
    reset_robot = getattr(robot, "reset_to_initial_position", None)
    if reset_robot is None:
        raise ValueError(f"{robot} does not support reset_to_initial_position()")
    result = reset_robot()
    if result is False:
        raise ReturnToInitialPositionError(f"{robot} failed to return to its initial position")
    sync_teleoperator_control_reference(teleop, robot, reset_target=True)
    on_reset = getattr(teleop, "on_robot_reset", None)
    if on_reset is not None:
        on_reset()


def sync_teleoperator_clutch(teleop: Teleoperator, robot: Robot) -> None:
    """Forward trigger state outside the action schema and recorded data."""
    set_translation_priority = getattr(robot, "set_teleop_translation", None)
    if set_translation_priority is not None:
        set_translation_priority(getattr(teleop, "prefer_translation_position", False))
    set_rotation_priority = getattr(robot, "set_teleop_wrist_rotation", None)
    if set_rotation_priority is not None:
        set_rotation_priority(getattr(teleop, "prefer_wrist_rotation", False))
    enabled = getattr(teleop, "motion_enabled", None)
    setter = getattr(robot, "set_teleop_motion_enabled", None)
    if setter is not None and enabled is not None:
        if getattr(teleop, "stop_immediately_on_release", False):
            setter(enabled, immediate_stop=True)
        else:
            setter(enabled)


def handle_rejected_robot_action(teleop: Teleoperator, robot: Robot) -> bool:
    accepted = bool(getattr(robot, "last_action_accepted", getattr(robot, "last_ik_accepted", True)))
    if accepted:
        return False
    sync_teleoperator_control_reference(teleop, robot, reset_target=True)
    rejected = getattr(teleop, "handle_robot_action_rejected", None)
    if rejected is not None:
        rejected()
    return True


def validate_policy_robot_features(
    policy_cfg: PreTrainedConfig,
    dataset_features: dict[str, dict],
    rename_map: dict[str, str] | None = None,
) -> None:
    """Fail before hardware connection when a checkpoint and robot schema are incompatible."""
    if not policy_cfg.input_features or not policy_cfg.output_features:
        raise ValueError(
            "Hardware inference requires the pretrained policy config to declare complete "
            "input_features and output_features"
        )

    available = dataset_to_policy_features(dataset_features)
    if rename_map:
        available = {rename_map.get(key, key): feature for key, feature in available.items()}
    mismatches: list[str] = []
    for section, expected_features in (
        ("input", policy_cfg.input_features),
        ("output", policy_cfg.output_features),
    ):
        for key, expected in expected_features.items():
            actual = available.get(key)
            if actual is None:
                mismatches.append(f"{section} {key!r}: missing from robot features")
                continue
            if actual.type is not expected.type:
                mismatches.append(
                    f"{section} {key!r}: type {actual.type.value} != {expected.type.value}"
                )
            if tuple(actual.shape) != tuple(expected.shape):
                mismatches.append(
                    f"{section} {key!r}: shape {tuple(actual.shape)} != {tuple(expected.shape)}"
                )

    if mismatches:
        raise ValueError("Policy/robot feature compatibility check failed:\n" + "\n".join(mismatches))


@cache
def is_headless():
    """
    Detects if the Python script is running in a headless environment (e.g., without a display).

    This function attempts to import `pynput`, a library that requires a graphical environment.
    If the import fails, it assumes the environment is headless. The result is cached to avoid
    re-running the check.

    Returns:
        True if the environment is determined to be headless, False otherwise.
    """
    try:
        import pynput  # noqa

        return False
    except Exception:
        print(
            "Error trying to import pynput. Switching to headless mode. "
            "As a result, the video stream from the cameras won't be shown, "
            "and you won't be able to change the control flow with keyboards. "
            "For more info, see traceback below.\n"
        )
        traceback.print_exc()
        print()
        return True


def predict_action(
    observation: dict[str, np.ndarray],
    policy: PreTrainedPolicy,
    device: torch.device,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    use_amp: bool,
    task: str | None = None,
    robot_type: str | None = None,
):
    """
    Performs a single-step inference to predict a robot action from an observation.

    This function encapsulates the full inference pipeline:
    1. Prepares the observation by converting it to PyTorch tensors and adding a batch dimension.
    2. Runs the preprocessor pipeline on the observation.
    3. Feeds the processed observation to the policy to get a raw action.
    4. Runs the postprocessor pipeline on the raw action.
    5. Formats the final action by removing the batch dimension and moving it to the CPU.

    Args:
        observation: A dictionary of NumPy arrays representing the robot's current observation.
        policy: The `PreTrainedPolicy` model to use for action prediction.
        device: The `torch.device` (e.g., 'cuda' or 'cpu') to run inference on.
        preprocessor: The `PolicyProcessorPipeline` for preprocessing observations.
        postprocessor: The `PolicyProcessorPipeline` for postprocessing actions.
        use_amp: A boolean to enable/disable Automatic Mixed Precision for CUDA inference.
        task: An optional string identifier for the task.
        robot_type: An optional string identifier for the robot type.

    Returns:
        A `torch.Tensor` containing the predicted action, ready for the robot.
    """
    observation = copy(observation)
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
    ):
        # Convert to pytorch format: channel first and float32 in [0,1] with batch dimension
        observation = prepare_observation_for_inference(observation, device, task, robot_type)
        observation = preprocessor(observation)

        # Compute the next action with the policy
        # based on the current observation
        action = policy.select_action(observation)

        action = postprocessor(action)

    return action


def init_keyboard_listener(*, enable_home: bool = False):
    """
    Initializes a non-blocking keyboard listener for real-time user interaction.

    This function sets up a listener for specific keys (right arrow, left arrow, escape) to control
    the program flow during execution, such as stopping recording or exiting loops. It gracefully
    handles headless environments where keyboard listening is not possible.

    Returns:
        A tuple containing:
        - The `pynput.keyboard.Listener` instance, or `None` if in a headless environment.
        - A dictionary of event flags (e.g., `exit_early`) that are set by key presses.
    """
    # Allow to exit early while recording an episode or resetting the environment,
    # by tapping the right arrow key '->'. This might require a sudo permission
    # to allow your terminal to monitor keyboard events.
    events = {}
    events["exit_early"] = False
    events["rerecord_episode"] = False
    events["stop_recording"] = False
    events["start_episode"] = False
    events["stop_episode"] = False
    events["delete_last_episode"] = False
    events["home_requested"] = False
    events["home_in_progress"] = False

    if is_headless():
        logging.warning(
            "Headless environment detected. On-screen cameras display and keyboard inputs will not be available."
        )
        listener = None
        return listener, events

    # Only import pynput if not in a headless environment
    from pynput import keyboard

    home_pressed = False
    start_pressed = False

    def on_press(key):
        nonlocal home_pressed, start_pressed
        try:
            if key == keyboard.Key.right:
                logging.debug("Right arrow key pressed. Exiting loop...")
                events["exit_early"] = True
            elif key == keyboard.Key.left:
                logging.debug("Left arrow key pressed. Exiting loop and rerecord the last episode...")
                events["rerecord_episode"] = True
                events["exit_early"] = True
            elif key == keyboard.Key.backspace:
                logging.debug("Backspace key pressed. Stopping and discarding the current episode...")
                events["delete_last_episode"] = True
                events["exit_early"] = True
            elif key == keyboard.Key.esc:
                logging.debug("Escape key pressed. Stopping data recording...")
                events["stop_recording"] = True
                events["exit_early"] = True
            elif hasattr(key, "char") and key.char is not None:
                key_char = key.char.lower()
                if key_char == "h" and enable_home:
                    if not home_pressed and not events["home_in_progress"]:
                        events["home_in_progress"] = True
                        events["home_requested"] = True
                    home_pressed = True
                elif key_char == "c":
                    if (
                        not events["home_in_progress"]
                        and (not enable_home or not start_pressed)
                    ):
                        logging.debug("'c' key pressed. Starting episode recording...")
                        events["start_episode"] = True
                    start_pressed = True
                elif key_char == "s" and not events["home_in_progress"]:
                    logging.debug("'s' key pressed. Stopping current episode...")
                    events["stop_episode"] = True
                    events["exit_early"] = True
        except Exception as e:
            print(f"Error handling key press: {e}")

    def on_release(key):
        nonlocal home_pressed, start_pressed
        if getattr(key, "char", None) in ("h", "H"):
            home_pressed = False
        elif getattr(key, "char", None) in ("c", "C"):
            start_pressed = False
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    return listener, events


def sanity_check_dataset_name(repo_id, policy_cfg):
    """
    Validates the dataset repository name against the presence of a policy configuration.

    This function enforces a naming convention: a dataset repository ID should start with "eval_"
    if and only if a policy configuration is provided for evaluation purposes.

    Args:
        repo_id: The Hugging Face Hub repository ID of the dataset.
        policy_cfg: The configuration object for the policy, or `None`.

    Raises:
        ValueError: If the naming convention is violated.
    """
    _, dataset_name = repo_id.split("/")
    # either repo_id doesnt start with "eval_" and there is no policy
    # or repo_id starts with "eval_" and there is a policy

    # Check if dataset_name starts with "eval_" but policy is missing
    if dataset_name.startswith("eval_") and policy_cfg is None:
        raise ValueError(
            f"Your dataset name begins with 'eval_' ({dataset_name}), but no policy is provided ({policy_cfg.type})."
        )

    # Check if dataset_name does not start with "eval_" but policy is provided
    if not dataset_name.startswith("eval_") and policy_cfg is not None:
        raise ValueError(
            f"Your dataset name does not begin with 'eval_' ({dataset_name}), but a policy is provided ({policy_cfg.type})."
        )


def sanity_check_dataset_robot_compatibility(
    dataset: LeRobotDataset, robot: Robot, fps: int, features: dict
) -> None:
    """
    Checks if a dataset's metadata is compatible with the current robot and recording setup.

    This function compares key metadata fields (`robot_type`, `fps`, and `features`) from the
    dataset against the current configuration to ensure that appended data will be consistent.

    Args:
        dataset: The `LeRobotDataset` instance to check.
        robot: The `Robot` instance representing the current hardware setup.
        fps: The current recording frequency (frames per second).
        features: The dictionary of features for the current recording session.

    Raises:
        ValueError: If any of the checked metadata fields do not match.
    """
    fields = [
        ("robot_type", dataset.meta.robot_type, robot.robot_type),
        ("fps", dataset.fps, fps),
        ("features", dataset.features, {**features, **DEFAULT_FEATURES}),
    ]

    mismatches = []
    for field, dataset_value, present_value in fields:
        diff = DeepDiff(dataset_value, present_value, exclude_regex_paths=[r".*\['info'\]$"])
        if diff:
            mismatches.append(f"{field}: expected {present_value}, got {dataset_value}")

    if mismatches:
        raise ValueError(
            "Dataset metadata compatibility check failed with mismatches:\n" + "\n".join(mismatches)
        )
