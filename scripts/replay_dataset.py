#!/usr/bin/env python3
"""Replay measured B601 joint trajectories without crossing recording gaps."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import draccus
import numpy as np
import yaml
from scipy.interpolate import CubicHermiteSpline, PchipInterpolator

try:
    from datasets import disable_progress_bars
except (
    ImportError
):  # pragma: no cover - datasets is a LeRobot dependency in normal use.
    disable_progress_bars = None

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION
from lerobot.utils.robot_utils import precise_sleep
from lerobot.robots.seeed_b601_dm import (
    BiSeeedB601DMFollower,
    BiSeeedB601DMFollowerConfig,
)


DEFAULT_DATASET_ROOT = Path(__file__).resolve().parents[1] / "data/bread_joint"
DEFAULT_REPO_ID = "yechen/b601-bi-quest3-demo"
DEFAULT_LEFT_PORT = "/dev/serial/by-path/pci-0000:0b:00.3-usb-0:2.2:1.0"
DEFAULT_RIGHT_PORT = "/dev/serial/by-path/pci-0000:0b:00.3-usb-0:2.1:1.0"
DEFAULT_ROBOT_CONFIG = (
    Path(__file__).resolve().parents[1] / "lerobot/configs/tele_switch_bi.yaml"
)
LEGACY_REPLAY_CALIBRATION_DIR = Path("/tmp/lerobot-b601-replay-calibration")

JOINT_MOTORS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_yaw",
    "wrist_roll",
)
EXPECTED_ACTION_NAMES = tuple(
    [f"left_{motor}.pos" for motor in (*JOINT_MOTORS, "gripper")]
    + [f"right_{motor}.pos" for motor in (*JOINT_MOTORS, "gripper")]
)
EXPECTED_TCP_ACTION_NAMES = tuple(
    f"{side}_{name}"
    for side in ("left", "right")
    for name in (
        "tcp.x",
        "tcp.y",
        "tcp.z",
        *(f"tcp.r{i}" for i in range(1, 7)),
        "gripper.pos",
    )
)
GRIPPER_ACTION_NAMES = {"left_gripper.pos", "right_gripper.pos"}


@dataclass(frozen=True)
class ReplayPlan:
    action_names: tuple[str, ...]
    actions: np.ndarray
    dataset_fps: float
    playback_speed: float
    dataset_root: Path
    repo_id: str
    episode: int
    source: str = "action"
    segment_ranges: tuple[tuple[int, int], ...] = ()
    segment_index: int = 0
    limit_clipped_samples: int = 0

    @property
    def replay_fps(self) -> float:
        return self.dataset_fps * self.playback_speed

    @property
    def has_gripper(self) -> bool:
        return any(name in GRIPPER_ACTION_NAMES for name in self.action_names)

    @property
    def action_mode(self) -> str:
        return (
            "cartesian"
            if any("tcp." in name for name in self.action_names)
            else "joint"
        )


def _to_numpy_action(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _validate_action_names(action_names: Iterable[str]) -> tuple[str, ...]:
    names = tuple(action_names)
    expected = (
        EXPECTED_TCP_ACTION_NAMES
        if any("tcp." in name for name in names)
        else EXPECTED_ACTION_NAMES
    )
    if names != expected:
        missing = [name for name in expected if name not in names]
        extra = [name for name in names if name not in expected]
        details = []
        if missing:
            details.append(f"missing={missing}")
        if extra:
            details.append(f"extra={extra}")
        if not missing and not extra:
            details.append("incorrect column order or duplicate names")
        raise ValueError(
            "Action names do not match the B601 bimanual schema: " + "; ".join(details)
        )
    return names


def _episode_frames(dataset: LeRobotDataset, episode: int):
    if disable_progress_bars is not None:
        disable_progress_bars()
    episode_frames = dataset.hf_dataset.filter(
        lambda row: row["episode_index"] == episode
    )
    if len(episode_frames) == 0:
        raise ValueError(f"Episode {episode} has no frames.")
    return episode_frames


def _measured_joint_actions(frames, state_names: Iterable[str], *, no_gripper: bool):
    """Map numbered observation joints to actuator names; grippers are normalized."""
    state_names = tuple(state_names)
    if len(set(state_names)) != len(state_names):
        raise ValueError("Duplicate observation.state column names.")
    names, columns = [], []
    for side in ("left", "right"):
        for number, motor in enumerate(JOINT_MOTORS, 1):
            key = f"{side}_joint_{number}.pos"
            if key not in state_names:
                key = f"{side}_{motor}.pos"
            if key not in state_names:
                raise ValueError(
                    f"Measured joint replay requires observation.state column {key}."
                )
            names.append(f"{side}_{motor}.pos")
            columns.append(state_names.index(key))
        if not no_gripper:
            key = f"{side}_gripper.pos"
            if key not in state_names:
                raise ValueError(
                    f"Measured joint replay requires observation.state column {key}."
                )
            names.append(key)
            columns.append(state_names.index(key))
    selected = frames.select_columns("observation.state")
    state = np.stack([_to_numpy_action(row["observation.state"]) for row in selected])
    if state.ndim != 2 or state.shape[1] != len(state_names):
        raise ValueError("observation.state dimensions do not match its column names.")
    actions = state[:, columns].copy()
    for index, name in enumerate(names):
        if name in GRIPPER_ACTION_NAMES:
            values = actions[:, index]
            if np.any((values < -1e-6) | (values > 1.0 + 1e-6)):
                raise ValueError(f"{name} observations must be normalized to [0, 1].")
            actions[:, index] = -270.0 + 270.0 * np.clip(values, 0.0, 1.0)
    return tuple(names), actions


def load_replay_plan(
    dataset_root: str | Path,
    episode: int,
    repo_id: str = DEFAULT_REPO_ID,
    playback_speed: float = 1.0,
    filter_static: bool = False,
    no_gripper: bool = False,
    segment: int = 0,
) -> ReplayPlan:
    root = Path(dataset_root).expanduser()
    if not (root / "meta" / "info.json").exists():
        raise FileNotFoundError(
            f"Not a local LeRobot dataset: {root} (missing meta/info.json)"
        )
    if episode < 0:
        raise ValueError("episode must be >= 0.")
    if not 0.0 < playback_speed <= 1.0:
        raise ValueError("--playback-speed must be in (0, 1].")
    if filter_static:
        raise ValueError(
            "--filter-static would remove recorded dwell time; replay now preserves all frames."
        )

    dataset = LeRobotDataset(
        repo_id, root=root, episodes=[episode], download_videos=False
    )
    if not np.isfinite(dataset.fps) or dataset.fps <= 0:
        raise ValueError("Dataset fps must be finite and positive.")
    frames = _episode_frames(dataset, episode)
    if "observation.state" in dataset.features:
        action_names, actions = _measured_joint_actions(
            frames,
            dataset.features["observation.state"]["names"],
            no_gripper=no_gripper,
        )
        source = "observation.state (measured joints)"
    else:
        action_names = _validate_action_names(dataset.features[ACTION]["names"])
        if any("tcp." in name for name in action_names):
            raise ValueError(
                "Cartesian targets are not a measured trajectory. This dataset has no joint observations."
            )
        selected = frames.select_columns(ACTION)
        actions = np.stack([_to_numpy_action(row[ACTION]) for row in selected])
        source = "action (legacy joint-only dataset)"

    if actions.ndim != 2 or actions.shape[1] != len(action_names):
        raise ValueError(
            f"Expected action shape (frames, {len(action_names)}), got {actions.shape}."
        )
    if not np.isfinite(actions).all():
        bad = np.argwhere(~np.isfinite(actions))
        frame, dim = bad[0]
        raise ValueError(
            f"Action contains NaN/Inf at frame={frame}, dim={dim} ({action_names[dim]})."
        )

    if no_gripper:
        keep = [
            idx
            for idx, name in enumerate(action_names)
            if name not in GRIPPER_ACTION_NAMES
        ]
        action_names = tuple(action_names[idx] for idx in keep)
        actions = actions[:, keep]

    joint_columns = [
        i for i, name in enumerate(action_names) if name not in GRIPPER_ACTION_NAMES
    ]
    # A Home operation is blocking and absent from the recorded frames. Never
    # invent a path across that gap, even when the stored timestamps look regular.
    gaps = np.max(np.abs(np.diff(actions[:, joint_columns], axis=0)), axis=1) > 10.0
    if "timestamp" in frames.column_names:
        timestamps = np.asarray(
            frames.select_columns("timestamp")["timestamp"], dtype=float
        )
        if not np.isfinite(timestamps).all():
            raise ValueError("Non-finite dataset timestamps.")
        dt = np.diff(timestamps)
        gaps |= (dt <= 0.0) | (dt > 1.5 / dataset.fps)
    if "frame_index" in frames.column_names:
        indices = np.asarray(
            frames.select_columns("frame_index")["frame_index"], dtype=int
        )
        gaps |= np.diff(indices) != 1
    boundaries = [0, *(np.flatnonzero(gaps) + 1).tolist(), len(actions)]
    ranges = tuple(zip(boundaries[:-1], boundaries[1:], strict=True))
    if not 0 <= segment < len(ranges):
        raise ValueError(f"--segment must be between 0 and {len(ranges) - 1}.")
    start, end = ranges[segment]
    actions = actions[start:end]

    return ReplayPlan(
        action_names=action_names,
        actions=actions,
        dataset_fps=float(dataset.fps),
        playback_speed=float(playback_speed),
        dataset_root=root,
        repo_id=repo_id,
        episode=episode,
        source=source,
        segment_ranges=ranges,
        segment_index=segment,
    )


@dataclass(frozen=True)
class JointTrajectory:
    values: np.ndarray
    curve: CubicHermiteSpline | None
    base_duration: float
    time_scale: float

    @property
    def duration(self) -> float:
        return self.base_duration * self.time_scale

    def sample(self, elapsed: float) -> np.ndarray:
        if self.curve is None or elapsed <= 0.0:
            return self.values[0].copy()
        if elapsed >= self.duration:
            return self.values[-1].copy()
        return self.curve(np.clip(elapsed / self.time_scale, 0.0, self.base_duration))


def motion_limits(names, max_velocity: float, max_acceleration: float):
    if (
        not np.isfinite([max_velocity, max_acceleration]).all()
        or min(max_velocity, max_acceleration) <= 0
    ):
        raise ValueError(
            "Joint velocity and acceleration limits must be finite and positive."
        )
    return (
        np.array(
            [180.0 if name in GRIPPER_ACTION_NAMES else max_velocity for name in names]
        ),
        np.array(
            [
                2000.0 if name in GRIPPER_ACTION_NAMES else max_acceleration
                for name in names
            ]
        ),
    )


def build_joint_trajectory(
    values, period: float, velocity_limits, acceleration_limits
) -> JointTrajectory:
    """Shape-preserving C1 interpolation, stopped endpoints and exact derivative bounds."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Joint trajectory must contain finite rows.")
    if not np.isfinite(period) or period <= 0:
        raise ValueError("Trajectory period must be finite and positive.")
    if len(values) == 1:
        return JointTrajectory(values, None, 0.0, 1.0)
    times = np.arange(len(values), dtype=float) * period
    slopes = PchipInterpolator(times, values).derivative()(times)
    slopes[[0, -1]] = 0.0
    curve = CubicHermiteSpline(times, values, slopes)
    c3, c2, c1, _ = curve.c
    h = np.diff(times)[:, None]
    # Velocity extrema occur at endpoints or at roots of the linear acceleration.
    root = np.divide(-c2, 3.0 * c3, out=np.zeros_like(c2), where=c3 != 0.0)
    root = np.clip(root, 0.0, h)
    peak_velocity = np.max(
        np.stack(
            [
                abs(c1),
                abs((3.0 * c3 * h + 2.0 * c2) * h + c1),
                abs((3.0 * c3 * root + 2.0 * c2) * root + c1),
            ]
        ),
        axis=(0, 1),
    )
    peak_acceleration = np.max(
        np.stack([abs(2.0 * c2), abs(6.0 * c3 * h + 2.0 * c2)]), axis=(0, 1)
    )
    scale = max(
        1.0,
        float(np.max(peak_velocity / velocity_limits)),
        float(np.sqrt(np.max(peak_acceleration / acceleration_limits))),
    )
    return JointTrajectory(values, curve, float(times[-1]), scale)


def observation_joint_vector(obs: dict[str, Any], names: tuple[str, ...]) -> np.ndarray:
    values = []
    for name in names:
        side, motor_key = name.split("_", 1)
        if name in GRIPPER_ACTION_NAMES:
            value = _obs_gripper_deg(obs, name)
        else:
            number = JOINT_MOTORS.index(motor_key.removesuffix(".pos")) + 1
            value = obs.get(f"{side}_joint_{number}.pos", obs.get(name))
        if value is None or not np.isfinite(value):
            raise ValueError(f"Missing or invalid live joint feedback: {name}.")
        values.append(float(value))
    return np.asarray(values)


def validate_robot_limits(robot, plan: ReplayPlan) -> None:
    for i, name in enumerate(plan.action_names):
        side, motor = name.split("_", 1)
        limits = getattr(robot, side).config.joint_limits[motor.removesuffix(".pos")]
        if np.any((plan.actions[:, i] < limits[0]) | (plan.actions[:, i] > limits[1])):
            raise ValueError(
                f"Recorded {name} is outside configured joint limits {limits}; no robot was connected."
            )


def constrain_recorded_positions(plan: ReplayPlan, joint_limits) -> ReplayPlan:
    """Allow sub-degree measured tracking overshoot without expanding soft limits."""
    values = plan.actions.copy()
    clipped = 0
    for i, name in enumerate(plan.action_names):
        _, motor = name.split("_", 1)
        lower, upper = joint_limits[motor.removesuffix(".pos")]
        excess = np.maximum(lower - values[:, i], values[:, i] - upper)
        if np.any(excess > 0.5):
            raise ValueError(
                f"Recorded {name} exceeds configured limits by more than 0.5 degrees; no robot was connected."
            )
        clipped += int(np.count_nonzero(excess > 0.0))
        values[:, i] = np.clip(values[:, i], lower, upper)
    return replace(plan, actions=values, limit_clipped_samples=clipped)


def stream_trajectory(
    robot,
    trajectory,
    names,
    *,
    control_rate,
    max_relative_target,
    label,
    debug_gripper=False,
    debug_gripper_interval=30,
):
    """Do not skip or catch up samples after slow I/O: stretch time instead."""
    steps = max(1, int(np.ceil(trajectory.duration * control_rate)))
    period = trajectory.duration / steps if trajectory.duration else 1.0 / control_rate
    joint_columns = [
        i for i, name in enumerate(names) if name not in GRIPPER_ACTION_NAMES
    ]
    last_print = time.perf_counter()
    for step in range(steps + 1):
        started = time.perf_counter()
        obs = robot.get_observation()
        present = observation_joint_vector(obs, names)
        target = trajectory.sample(step * period)
        if (
            np.max(abs(target[joint_columns] - present[joint_columns]))
            > max_relative_target
        ):
            raise RuntimeError(
                f"{label}: joint tracking error exceeds {max_relative_target:g} degrees; replay stopped."
            )
        action = dict(zip(names, map(float, target), strict=True))
        robot.send_action(action)
        if debug_gripper and step % debug_gripper_interval == 0:
            _print_gripper_debug(step, action, obs)
        if started - last_print >= 1.0:
            print(f"  {label}: {step * period:.1f}/{trajectory.duration:.1f}s")
            last_print = started
        precise_sleep(max(period - (time.perf_counter() - started), 0.0))


def wait_for_joint_arrival(robot, names, target, timeout=5.0):
    joint_columns = [
        i for i, name in enumerate(names) if name not in GRIPPER_ACTION_NAMES
    ]
    deadline = time.perf_counter() + timeout
    while True:
        present = observation_joint_vector(robot.get_observation(), names)
        if np.max(abs(target[joint_columns] - present[joint_columns])) <= 1.0:
            return
        if time.perf_counter() >= deadline:
            raise RuntimeError(
                "Joints did not reach the replay target; stopping instead of advancing."
            )
        robot.send_action(dict(zip(names, map(float, target), strict=True)))
        precise_sleep(0.02)


def action_dict(plan: ReplayPlan, index: int) -> dict[str, float]:
    return {
        name: float(value)
        for name, value in zip(plan.action_names, plan.actions[index], strict=True)
    }


def _obs_gripper_deg(obs: dict[str, Any], key: str) -> float | None:
    if key not in obs:
        return None
    value = float(obs[key])
    # B601 observations use normalized gripper convention: 0=open, 1=closed.
    return -270.0 + max(0.0, min(1.0, value)) * 270.0


def _print_gripper_debug(
    idx: int, action: dict[str, float], obs: dict[str, Any], *, action_mode="joint"
) -> None:
    left_cmd = action.get("left_gripper.pos")
    right_cmd = action.get("right_gripper.pos")
    if action_mode == "cartesian":
        left_cmd = None if left_cmd is None else -270.0 + 270.0 * left_cmd
        right_cmd = None if right_cmd is None else -270.0 + 270.0 * right_cmd
    left_obs = _obs_gripper_deg(obs, "left_gripper.pos")
    right_obs = _obs_gripper_deg(obs, "right_gripper.pos")
    left_err = None if left_cmd is None or left_obs is None else left_cmd - left_obs
    right_err = (
        None if right_cmd is None or right_obs is None else right_cmd - right_obs
    )

    def fmt(value: float | None) -> str:
        return "n/a" if value is None else f"{value:7.1f}"

    print(
        f"[gripper-debug] frame={idx + 1:04d} "
        f"L cmd/obs/err={fmt(left_cmd)}/{fmt(left_obs)}/{fmt(left_err)} "
        f"R cmd/obs/err={fmt(right_cmd)}/{fmt(right_obs)}/{fmt(right_err)}"
    )


def quality_report(plan: ReplayPlan) -> str:
    duration_s = (
        max(0, len(plan.actions) - 1) / plan.replay_fps if plan.replay_fps > 0 else 0.0
    )
    lines = [
        "Replay quality report",
        f"  dataset: {plan.dataset_root}",
        f"  episode: {plan.episode}",
        f"  frames: {len(plan.actions)}",
        f"  dataset_fps: {plan.dataset_fps:.2f}",
        f"  replay_fps: {plan.replay_fps:.2f}",
        f"  duration_s: {duration_s:.2f}",
        f"  action_dim: {plan.actions.shape[1]}",
        f"  mode: bimanual {plan.action_mode}, gripper={'yes' if plan.has_gripper else 'no'}",
        f"  source: {plan.source}",
        f"  measured values clipped to existing soft limits (overshoot <=0.5 deg): {plan.limit_clipped_samples}",
    ]
    if plan.segment_ranges:
        start, end = plan.segment_ranges[plan.segment_index]
        lines.append(
            f"  segment: {plan.segment_index}/{len(plan.segment_ranges) - 1}, source frames {start}-{end - 1}"
        )
        if len(plan.segment_ranges) > 1:
            lines.append(
                f"  recording gaps: {[end for _, end in plan.segment_ranges[:-1]]}"
            )
            lines.append(
                "  Only the selected continuous segment is replayed; gaps are NOT interpolated."
            )
    if len(plan.actions) > 1:
        jumps = np.max(np.abs(np.diff(plan.actions, axis=0)), axis=0)
        order = np.argsort(jumps)[::-1]
        lines.append("  largest per-frame jumps:")
        for idx in order[: min(8, len(order))]:
            name = plan.action_names[idx]
            if plan.action_mode == "joint":
                unit = "deg"
            elif name.endswith(("tcp.x", "tcp.y", "tcp.z")):
                unit = "m"
            else:
                unit = "unitless"
            lines.append(f"    {name}: {jumps[idx]:.6f} {unit}")
    else:
        lines.append("  largest per-frame jumps: n/a (single frame)")
    lines.append("  finite: yes")
    return "\n".join(lines)


def gripper_report(plan: ReplayPlan) -> str:
    if not plan.has_gripper:
        return "Gripper report: no gripper action in this plan."

    lines = ["Gripper report"]
    for name in ("left_gripper.pos", "right_gripper.pos"):
        if name not in plan.action_names:
            lines.append(f"  {name}: missing")
            continue
        values = plan.actions[:, plan.action_names.index(name)]
        close_mask = values > (0.8 if plan.action_mode == "cartesian" else -60.0)
        close_frames = int(close_mask.sum())
        if close_frames:
            close_indices = np.flatnonzero(close_mask)
            close_span = f"{int(close_indices[0]) + 1}-{int(close_indices[-1]) + 1}"
        else:
            close_span = "none"
        lines.append(
            f"  {name}: min={values.min():.1f} max={values.max():.1f} "
            f"start={values[0]:.1f} end={values[-1]:.1f} "
            f"close_frames={close_frames}/{len(values)} close_span={close_span}"
        )
    return "\n".join(lines)


def make_robot_config(
    args: argparse.Namespace, *, action_mode: str = "joint"
) -> BiSeeedB601DMFollowerConfig:
    config = yaml.safe_load(args.robot_config.read_text(encoding="utf-8"))["robot"]
    if config.get("type") != "bi_seeed_b601_dm_follower":
        raise ValueError(
            "--robot-config must describe a bi_seeed_b601_dm_follower robot."
        )
    config = {key: value for key, value in config.items() if key != "type"}
    # Replay does not capture images or tactile data. Retain the current
    # actuator gains, limits and IK settings from the working robot config.
    config["cameras"] = {}
    for key in list(config):
        if key.startswith("flexitac_") and key.endswith("_port"):
            config[key] = ""
    config.update(
        left_port=args.left_port,
        right_port=args.right_port,
        can_adapter=args.can_adapter,
        action_mode=action_mode,
        control_mode="pos_vel",
        control_gripper=not args.no_gripper,
        left_gripper_type="rebotarm_b601",
        right_gripper_type="rebotarm_b601",
        gripper_force_pos_torque_ratio=args.gripper_force_ratio,
        enable_observation_joint_pos=True,
        cameras={},
        max_relative_target=args.max_relative_target,
    )
    # The scheduler's playback speed must not leave 150 deg/s motor caps in
    # place. Bound both replay and the follower's existing disconnect return.
    caps = [args.max_joint_velocity] * 6 + [180.0]
    configured = config.get("pos_vel_velocity", [150.0] * 6 + [300.0])
    if isinstance(configured, (float, int)):
        configured = [configured] * 7
    if len(configured) != 7:
        raise ValueError("Replay expects seven B601 actuator velocity limits.")
    config["pos_vel_velocity"] = [
        min(float(value), cap) for value, cap in zip(configured, caps, strict=True)
    ]
    if (
        not np.isfinite(config["pos_vel_velocity"]).all()
        or min(config["pos_vel_velocity"]) <= 0.0
    ):
        raise ValueError(
            "Configured actuator velocity limits must be finite and positive."
        )
    configured_return = config.get(
        "return_to_initial_vlim_deg_s", dict(zip((*JOINT_MOTORS, "gripper"), caps))
    )
    if isinstance(configured_return, dict):
        return_values = [configured_return[name] for name in (*JOINT_MOTORS, "gripper")]
    elif isinstance(configured_return, (float, int)):
        return_values = [configured_return] * 7
    else:
        return_values = configured_return
    config["return_to_initial_vlim_deg_s"] = [
        min(float(value), cap) for value, cap in zip(return_values, caps, strict=True)
    ]
    config["rt_rate"] = args.control_rate
    return draccus.decode(BiSeeedB601DMFollowerConfig, config)


def make_robot(
    args: argparse.Namespace, *, action_mode: str = "joint"
) -> BiSeeedB601DMFollower:
    return BiSeeedB601DMFollower(make_robot_config(args, action_mode=action_mode))


def require_replay_calibration(robot: BiSeeedB601DMFollower) -> None:
    """Replay must use established calibration and never enter a set-zero flow."""
    for side in ("left", "right"):
        child = getattr(robot, side)
        if not child.is_calibrated:
            raise RuntimeError(
                f"{side} arm has no existing calibration: {child.calibration_fpath}. "
                "Replay will not connect or automatically calibrate. "
                "Complete calibration with the normal robot configuration first."
            )
        current = child.calibration_fpath
        temporary = LEGACY_REPLAY_CALIBRATION_DIR / current.name
        if (
            temporary.is_file()
            and temporary.resolve() != current.resolve()
            and (
                not current.is_file()
                or temporary.stat().st_mtime_ns > current.stat().st_mtime_ns
            )
        ):
            raise RuntimeError(
                f"{side} arm has a newer set-zero record from the old replay directory: {temporary}. "
                "Its hardware zero may have changed. Restore/verify the physical zero and calibrate "
                "with the normal robot configuration before replaying. No robot was connected."
            )


def replay(args: argparse.Namespace) -> None:
    if not np.isfinite(args.control_rate) or not 30.0 <= args.control_rate <= 150.0:
        raise ValueError("--control-rate must be between 30 and 150 Hz.")
    if not np.isfinite(args.max_relative_target) or args.max_relative_target <= 0:
        raise ValueError("--max-relative-target must be finite and positive.")
    if args.debug_gripper_interval <= 0:
        raise ValueError("--debug-gripper-interval must be positive.")
    plan = load_replay_plan(
        args.dataset_root,
        args.episode,
        repo_id=args.repo_id,
        playback_speed=args.playback_speed,
        filter_static=args.filter_static,
        no_gripper=args.no_gripper,
        segment=args.segment,
    )
    if plan.action_mode != "joint":
        raise ValueError(
            "Replay requires a joint trajectory, not Cartesian control targets."
        )
    config = make_robot_config(args, action_mode="joint")
    plan = constrain_recorded_positions(plan, config.joint_limits)
    velocity_limits, acceleration_limits = motion_limits(
        plan.action_names, args.max_joint_velocity, args.max_joint_acceleration
    )
    velocity_by_motor = dict(
        zip((*JOINT_MOTORS, "gripper"), config.pos_vel_velocity, strict=True)
    )
    velocity_limits = np.minimum(
        velocity_limits,
        [
            velocity_by_motor[name.split("_", 1)[1].removesuffix(".pos")]
            for name in plan.action_names
        ],
    )
    trajectory = build_joint_trajectory(
        plan.actions, 1.0 / plan.replay_fps, velocity_limits, acceleration_limits
    )
    print(quality_report(plan))
    print(f"  interpolated control rate: {args.control_rate:g} Hz")
    print(
        f"  limited trajectory duration: {trajectory.duration:.2f}s (retiming factor {trajectory.time_scale:.3f})"
    )
    print("  Startup alignment and normal return use interpolated joint paths.")
    print(
        "  On interruption/error the follower retains its existing speed-limited return to the startup pose."
    )
    if args.report_gripper:
        print(gripper_report(plan))
    if args.dry_run:
        print("Dry-run complete: robot was not connected.")
        return

    robot = make_robot(args, action_mode=plan.action_mode)
    require_replay_calibration(robot)
    validate_robot_limits(robot, plan)
    if not args.yes:
        print("\nThis will move both B601 arms. Type REPLAY to continue:")
        if input("> ").strip() != "REPLAY":
            print("Replay cancelled.")
            return

    connected = False
    try:
        robot.connect()
        connected = True

        present = observation_joint_vector(robot.get_observation(), plan.action_names)
        startup = present.copy()
        alignment = build_joint_trajectory(
            np.stack([present, plan.actions[0]]),
            1.0,
            velocity_limits,
            acceleration_limits,
        )
        print(
            f"Aligning to the first recorded joint pose ({alignment.duration:.1f}s)..."
        )
        stream_options = dict(
            control_rate=args.control_rate,
            max_relative_target=args.max_relative_target,
            debug_gripper=args.debug_gripper,
            debug_gripper_interval=args.debug_gripper_interval,
        )
        stream_trajectory(
            robot, alignment, plan.action_names, label="align", **stream_options
        )
        wait_for_joint_arrival(robot, plan.action_names, plan.actions[0])

        print("Replaying measured joint trajectory...")
        stream_trajectory(
            robot, trajectory, plan.action_names, label="replay", **stream_options
        )
        wait_for_joint_arrival(robot, plan.action_names, plan.actions[-1])
        if plan.segment_ranges and plan.segment_index < len(plan.segment_ranges) - 1:
            print(
                "Stopped before a recording gap; the unrecorded Home path was not replayed."
            )
        present = observation_joint_vector(robot.get_observation(), plan.action_names)
        return_path = build_joint_trajectory(
            np.stack([present, startup]), 1.0, velocity_limits, acceleration_limits
        )
        print(f"Returning to the startup joint pose ({return_path.duration:.1f}s)...")
        stream_trajectory(
            robot, return_path, plan.action_names, label="return", **stream_options
        )
        wait_for_joint_arrival(robot, plan.action_names, startup)
        print("Replay complete.")
    except KeyboardInterrupt:
        print("\nReplay interrupted by Ctrl-C.")
    finally:
        if connected:
            robot.disconnect()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("episode", type=int)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument(
        "--robot-config",
        type=Path,
        default=DEFAULT_ROBOT_CONFIG,
        help="Working dual-arm config used for actuator gains, joint limits and IK settings.",
    )
    parser.add_argument("--left-port", default=DEFAULT_LEFT_PORT)
    parser.add_argument("--right-port", default=DEFAULT_RIGHT_PORT)
    parser.add_argument(
        "--can-adapter", default="damiao", choices=("damiao", "socketcan", "robstride")
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--yes", action="store_true", help="Skip the REPLAY confirmation prompt."
    )
    parser.add_argument("--playback-speed", type=float, default=1.0)
    parser.add_argument("--filter-static", action="store_true")
    parser.add_argument(
        "--segment",
        type=int,
        default=0,
        help="Continuous recording segment to replay (default: 0); never cross Home/gaps.",
    )
    parser.add_argument(
        "--control-rate",
        type=float,
        default=100.0,
        help="Interpolated command rate in Hz.",
    )
    parser.add_argument(
        "--max-joint-velocity",
        type=float,
        default=45.0,
        help="Joint speed ceiling in deg/s.",
    )
    parser.add_argument(
        "--max-joint-acceleration",
        type=float,
        default=360.0,
        help="Joint acceleration ceiling in deg/s^2.",
    )
    parser.add_argument("--max-relative-target", type=float, default=6.0)
    parser.add_argument("--no-gripper", action="store_true")
    parser.add_argument("--gripper-force-ratio", type=float, default=0.1)
    parser.add_argument("--debug-gripper", action="store_true")
    parser.add_argument("--debug-gripper-interval", type=int, default=30)
    parser.add_argument("--report-gripper", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        replay(args)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
