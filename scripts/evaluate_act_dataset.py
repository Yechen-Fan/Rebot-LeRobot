#!/usr/bin/env python3
"""Offline ACT Cartesian prediction errors on recorded images/state; no robot I/O."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np


def scalar_stats(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
    return dict(
        count=len(values),
        mean=float(values.mean()),
        median=float(np.median(values)),
        p95=float(np.quantile(values, 0.95)),
        max=float(values.max()),
    )


def rotation_matrices(values):
    """Same column-based 6D rotation convention as the B601 follower."""
    values = np.asarray(values, dtype=float)
    x, y = values[:, :3], values[:, 3:]
    nx = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / np.maximum(nx, 1e-12)
    y = y - x * np.sum(x * y, axis=1, keepdims=True)
    ny = np.linalg.norm(y, axis=1, keepdims=True)
    y = y / np.maximum(ny, 1e-12)
    valid = (nx[:, 0] > 1e-6) & (ny[:, 0] > 1e-6) & np.isfinite(values).all(axis=1)
    return np.stack([x, y, np.cross(x, y)], axis=2), valid


def cartesian_metrics(predicted, target, names):
    predicted, target = np.asarray(predicted), np.asarray(target)
    if (
        predicted.shape != target.shape
        or predicted.ndim != 2
        or predicted.shape[1] != len(names)
    ):
        raise ValueError(
            "Prediction/target dimensions do not match action column names."
        )
    if not np.isfinite(predicted).all() or not np.isfinite(target).all():
        raise ValueError(
            "Non-finite prediction or target; refusing to hide invalid values in averages."
        )
    columns = {}
    for i, name in enumerate(names):
        is_xyz = name.endswith(("tcp.x", "tcp.y", "tcp.z"))
        error = (predicted[:, i] - target[:, i]) * (1000.0 if is_xyz else 1.0)
        columns[name] = dict(
            unit="mm" if is_xyz else "unitless",
            mae=float(abs(error).mean()),
            rmse=float(np.sqrt(np.mean(error**2))),
            bias=float(error.mean()),
        )
    arms = {}
    for side in ("left", "right"):
        xyz = [names.index(f"{side}_tcp.{axis}") for axis in ("x", "y", "z")]
        r6 = [names.index(f"{side}_tcp.r{i}") for i in range(1, 7)]
        grip = names.index(f"{side}_gripper.pos")
        rp, vp = rotation_matrices(predicted[:, r6])
        rt, vt = rotation_matrices(target[:, r6])
        valid = vp & vt
        cosine = (np.einsum("nij,nij->n", rp[valid], rt[valid]) - 1.0) / 2.0
        angles = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
        arms[side] = {
            "position_error_mm": scalar_stats(
                np.linalg.norm(predicted[:, xyz] - target[:, xyz], axis=1) * 1000.0
            ),
            "rotation_error_deg": scalar_stats(angles),
            "invalid_predicted_rotation_count": int(np.count_nonzero(~vp)),
            "invalid_target_rotation_count": int(np.count_nonzero(~vt)),
            "gripper_mae": float(np.mean(abs(predicted[:, grip] - target[:, grip]))),
            "gripper_state_accuracy_at_0.5": float(
                np.mean((predicted[:, grip] >= 0.5) == (target[:, grip] >= 0.5))
            ),
            "gripper_prediction_outside_0_1_count": int(
                np.count_nonzero(
                    (predicted[:, grip] < 0.0) | (predicted[:, grip] > 1.0)
                )
            ),
        }
    return {"count": len(predicted), "arms": arms, "columns": columns}


def resolve_checkpoint(
    training_dir: Path, explicit: Path | None, wait_seconds: float
) -> Path:
    path = explicit or training_dir / "checkpoints/last/pretrained_model"
    if path.name != "pretrained_model" and (path / "pretrained_model").is_dir():
        path = path / "pretrained_model"
    required = (
        "config.json",
        "model.safetensors",
        "train_config.json",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
    )
    deadline = time.monotonic() + wait_seconds
    while not all((path / name).is_file() for name in required):
        if time.monotonic() >= deadline:
            raise FileNotFoundError(
                f"No complete saved ACT checkpoint at {path}. The live training model is not a checkpoint."
            )
        print(f"Waiting for a saved checkpoint: {path}", flush=True)
        time.sleep(min(30.0, max(0.0, deadline - time.monotonic())))
    return path.resolve()


def save_plots(report, rows, destination):
    """Export physical-unit error summaries and a representative EE comparison."""
    try:
        import matplotlib
    except ModuleNotFoundError:
        print(
            "Metrics saved; matplotlib is unavailable in this environment, so plot export was skipped."
        )
        return

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"left": "#2563eb", "right": "#e76f24"}
    horizons = report["metrics_by_future_frame_offset"]
    offsets = sorted(map(int, horizons))
    seconds = np.array(offsets) / report["fps"]
    first = horizons["0"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    labels = [
        f"{side[0].upper()} {axis.upper()}"
        for side in ("left", "right")
        for axis in "xyz"
    ]
    mae = [
        first["columns"][f"{side}_tcp.{axis}"]["mae"]
        for side in ("left", "right")
        for axis in "xyz"
    ]
    axes[0, 0].bar(
        labels, mae, color=[colors[side] for side in ("left", "right") for _ in "xyz"]
    )
    axes[0, 0].set(title="First-step position MAE by coordinate", ylabel="mm")
    for side in ("left", "right"):
        axes[0, 1].plot(
            seconds,
            [
                horizons[str(h)]["arms"][side]["rotation_error_deg"]["mean"]
                for h in offsets
            ],
            "o-",
            label=f"{side} mean",
            color=colors[side],
        )
        axes[0, 1].plot(
            seconds,
            [
                horizons[str(h)]["arms"][side]["rotation_error_deg"]["p95"]
                for h in offsets
            ],
            "--",
            label=f"{side} P95",
            color=colors[side],
            alpha=0.65,
        )
        axes[1, 1].plot(
            seconds,
            [horizons[str(h)]["arms"][side]["gripper_mae"] for h in offsets],
            "o-",
            label=side,
            color=colors[side],
        )
    axes[0, 1].set(
        title="Orientation prediction error",
        ylabel="degrees",
        xlabel="Future offset (s)",
    )
    axes[0, 1].legend(fontsize=8)
    episodes = sorted(report["per_episode_first_step"], key=int)
    x = np.arange(len(episodes))
    for shift, side in ((-0.18, "left"), (0.18, "right")):
        values = [
            report["per_episode_first_step"][ep]["arms"][side]["position_error_mm"][
                "mean"
            ]
            for ep in episodes
        ]
        axes[1, 0].bar(x + shift, values, width=0.36, label=side, color=colors[side])
    axes[1, 0].set(
        xticks=x,
        xticklabels=episodes,
        xlabel="Episode",
        ylabel="mm",
        title="First-step mean Euclidean position error",
    )
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].set(
        title="Gripper prediction MAE",
        xlabel="Future offset (s)",
        ylabel="Normalized position (0-1)",
    )
    axes[1, 1].legend(fontsize=8)
    for ax in axes.flat:
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle(
        f"ACT {Path(report['checkpoint']).parent.name} | training data, open-loop | {report['sampled_observations']} inputs"
    )
    fig.savefig(destination / "error_summary.png", dpi=180)
    plt.close(fig)

    # Show predictions vs recorded action labels AND the contemporaneous input
    # EE pose. Input-to-output displacement is not a prediction error.
    episode = min(int(row["episode"]) for row in rows)
    subset = sorted(
        [
            row
            for row in rows
            if int(row["episode"]) == episode and int(row["horizon"]) == 0
        ],
        key=lambda row: int(row["frame"]),
    )
    clock = np.array([int(row["frame"]) for row in subset]) / report["fps"]
    fig, axes = plt.subplots(
        5, 2, figsize=(14, 13), sharex=True, constrained_layout=True
    )
    for col, side in enumerate(("left", "right")):
        for row_index, key in enumerate(("tcp.x", "tcp.y", "tcp.z", "gripper.pos")):
            name = f"{side}_{key}"
            scale = 1000.0 if key.startswith("tcp.") else 1.0
            ax = axes[row_index, col]
            for source, label, style, color in (
                ("target", "Recorded action target", "-", "#151515"),
                ("predicted", "Model first-step prediction", "o-", colors[side]),
                ("input_current", "Input measured EE/gripper", "--", "#8b8b8b"),
            ):
                ax.plot(
                    clock,
                    [float(row[f"{source}.{name}"]) * scale for row in subset],
                    style,
                    color=color,
                    markersize=3,
                    linewidth=1.25,
                    label=label,
                )
            ax.set_ylabel(f"{key}: {'mm' if scale == 1000.0 else '0-1'}")
            ax.grid(alpha=0.2)
            for gap in report["recording_gaps"].get(
                str(episode), report["recording_gaps"].get(episode, [])
            ):
                ax.axvline(
                    gap / report["fps"], color="#b91c1c", linestyle=":", alpha=0.6
                )
        rkeys = [f"{side}_tcp.r{i}" for i in range(1, 7)]
        rp, vp = rotation_matrices(
            [[float(row[f"predicted.{name}"]) for name in rkeys] for row in subset]
        )
        rt, vt = rotation_matrices(
            [[float(row[f"target.{name}"]) for name in rkeys] for row in subset]
        )
        angles = np.degrees(
            np.arccos(np.clip((np.einsum("nij,nij->n", rp, rt) - 1.0) / 2.0, -1.0, 1.0))
        )
        angles[~(vp & vt)] = np.nan
        axes[4, col].plot(clock, angles, "o-", color=colors[side], markersize=3)
        axes[4, col].set(
            ylabel="Rotation error: degrees", xlabel="Recorded nominal time (s)"
        )
        axes[4, col].grid(alpha=0.2)
        axes[0, col].set_title(f"{side.capitalize()} arm")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        f"Episode {episode}: sampled fresh first-step predictions (not a robot rollout)\n"
        "EE xyz / gripper values; r1-r6 are rotation encoding, not joints"
    )
    fig.savefig(destination / f"episode_{episode:02d}_comparison.png", dpi=180)
    plt.close(fig)


def evaluate(args):
    import draccus
    import torch
    import yaml
    from torch.utils.data import DataLoader, Subset

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.datasets.factory import make_dataset
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    torch.set_num_threads(args.cpu_threads)
    requested = draccus.decode(
        TrainPipelineConfig, yaml.safe_load(args.config.read_text())
    )
    checkpoint = resolve_checkpoint(
        Path(requested.output_dir), args.checkpoint, args.wait_seconds
    )
    # Use the checkpoint's own dataset configuration and saved normalization,
    # not a potentially edited training YAML or freshly fitted statistics.
    with (
        (checkpoint / "train_config.json").open() as handle,
        draccus.config_type("json"),
    ):
        training = draccus.load(TrainPipelineConfig, handle)
    policy_config = draccus.decode(
        PreTrainedConfig, json.loads((checkpoint / "config.json").read_text())
    )
    if not isinstance(policy_config, ACTConfig):
        raise ValueError("This evaluator requires an ACT checkpoint.")
    policy_config.device = args.device
    policy_config.pretrained_backbone_weights = (
        None  # Full backbone is in the checkpoint.
    )
    training.policy = policy_config
    training.dataset.image_transforms.enable = False
    dataset = make_dataset(training)
    names = list(dataset.meta.features["action"]["names"])
    expected = [
        f"{side}_{key}"
        for side in ("left", "right")
        for key in (
            "tcp.x",
            "tcp.y",
            "tcp.z",
            *(f"tcp.r{i}" for i in range(1, 7)),
            "gripper.pos",
        )
    ]
    if names != expected:
        raise ValueError(
            f"This evaluator expects the B601 20-dimensional Cartesian schema, got {names}."
        )
    state_names = list(dataset.meta.features["observation.state"]["names"])
    current_columns = [state_names.index(name) for name in names]
    table = dataset.hf_dataset.select_columns(
        ["episode_index", "frame_index", "observation.state"]
    )
    episodes = np.asarray(table["episode_index"], dtype=int)
    frame_indices = np.asarray(table["frame_index"], dtype=int)
    states = np.asarray([np.asarray(row["observation.state"]) for row in table])
    joint_columns = [
        state_names.index(f"{side}_joint_{j}.pos")
        for side in ("left", "right")
        for j in range(1, 7)
    ]
    gaps, selected = {}, []
    for ep in np.unique(episodes):
        indices = np.flatnonzero(episodes == ep)
        selected.extend(
            indices[
                np.unique(
                    np.linspace(
                        0,
                        len(indices) - 1,
                        min(args.samples_per_episode, len(indices)),
                        dtype=int,
                    )
                )
            ]
        )
        jumps = (
            np.max(abs(np.diff(states[indices][:, joint_columns], axis=0)), axis=1)
            > 10.0
        )
        gaps[int(ep)] = frame_indices[indices[1:][jumps]].tolist()

    policy = ACTPolicy.from_pretrained(
        checkpoint, config=policy_config, local_files_only=True, strict=True
    ).eval()
    pre, post = make_pre_post_processors(
        policy_config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": args.device}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    loader = DataLoader(
        Subset(dataset, selected),
        batch_size=args.batch_size,
        num_workers=0,
        shuffle=False,
    )
    predictions, targets, padding, currents, ep_rows, frame_rows = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    with torch.inference_mode():
        for batch in loader:
            # Do not pass ground-truth actions to the model / VAE encoder.
            observations = {key: batch[key] for key in policy_config.input_features}
            processed = pre(observations)
            output = post(policy.predict_action_chunk(processed)).cpu().numpy()
            predictions.append(output)
            targets.append(batch["action"].numpy())
            padding.append(batch["action_is_pad"].numpy())
            currents.append(batch["observation.state"].numpy()[:, current_columns])
            ep_rows.append(batch["episode_index"].numpy().reshape(-1))
            frame_rows.append(batch["frame_index"].numpy().reshape(-1))
            print(
                f"Evaluated {sum(len(x) for x in predictions)}/{len(selected)} recorded observations",
                flush=True,
            )

    predicted, target, pad = map(np.concatenate, (predictions, targets, padding))
    current, ep_rows, frame_rows = map(np.concatenate, (currents, ep_rows, frame_rows))
    horizons = sorted(
        set(
            [
                0,
                min(9, predicted.shape[1] - 1),
                min(29, predicted.shape[1] - 1),
                predicted.shape[1] - 1,
            ]
        )
    )
    metrics, csv_rows = {}, []
    for horizon in horizons:
        valid = ~pad[:, horizon].astype(bool)
        crosses_gap = np.array(
            [
                any(frame < gap <= frame + horizon for gap in gaps[int(ep)])
                for ep, frame in zip(ep_rows, frame_rows, strict=True)
            ]
        )
        valid &= ~crosses_gap
        if not valid.any():
            continue
        key = str(horizon)
        metrics[key] = cartesian_metrics(
            predicted[valid, horizon], target[valid, horizon], names
        )
        metrics[key]["excluded_padding_or_recording_gap"] = int(
            np.count_nonzero(~valid)
        )
        for index in np.flatnonzero(valid):
            row = {
                "episode": int(ep_rows[index]),
                "frame": int(frame_rows[index]),
                "horizon": horizon,
            }
            for column, name in enumerate(names):
                row[f"predicted.{name}"] = float(predicted[index, horizon, column])
                row[f"target.{name}"] = float(target[index, horizon, column])
                row[f"input_current.{name}"] = float(current[index, column])
            csv_rows.append(row)
    result = {
        "checkpoint": str(checkpoint),
        "checkpoint_step": json.loads(
            (checkpoint.parent / "training_state/training_step.json").read_text()
        )["step"]
        if (checkpoint.parent / "training_state/training_step.json").is_file()
        else None,
        "dataset_root": str(training.dataset.root),
        "evaluation_type": "open-loop predictions on training data; not held-out validation or robot execution",
        "input_features": {
            k: {"type": v.type.value, "shape": list(v.shape)}
            for k, v in policy_config.input_features.items()
        },
        "input_state_names": state_names,
        "output_action_names": names,
        "fps": dataset.fps,
        "sampled_observations": len(selected),
        "recording_gaps": gaps,
        "metrics_by_future_frame_offset": metrics,
        "current_pose_baseline_vs_target_first_step": cartesian_metrics(
            current, target[:, 0], names
        ),
        "predicted_first_step_vs_current_input_commanded_displacement_not_prediction_error": cartesian_metrics(
            predicted[:, 0], current, names
        ),
        "per_episode_first_step": {
            str(int(ep)): cartesian_metrics(
                predicted[ep_rows == ep, 0], target[ep_rows == ep, 0], names
            )
            for ep in np.unique(ep_rows)
        },
        "joint_output_errors": None,
        "joint_output_errors_note": "The policy outputs EE poses, not joint angles. IK is nonunique; no joint-angle errors are inferred.",
    }
    destination = (
        args.output
        or Path(requested.output_dir) / "evaluation" / checkpoint.parent.name
    )
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    with (destination / "predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    save_plots(result, csv_rows, destination)
    for horizon, metric in metrics.items():
        print(
            f"Horizon +{horizon} frames ({int(horizon) / dataset.fps:.2f}s), {metric['count']} samples:"
        )
        for side, arm in metric["arms"].items():
            print(
                f"  {side}: position MAE(norm)={arm['position_error_mm']['mean']:.2f}mm, "
                f"rotation mean={arm['rotation_error_deg']['mean']}deg, gripper MAE={arm['gripper_mae']:.4f}"
            )
    print(
        f"Report: {destination / 'report.json'}\nPredictions: {destination / 'predictions.csv'}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=Path(__file__).resolve().parents[1] / "lerobot/configs/train_act.yaml",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--wait-seconds", type=float, default=0.0)
    parser.add_argument("--samples-per-episode", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (
        min(args.samples_per_episode, args.batch_size, args.cpu_threads) <= 0
        or args.wait_seconds < 0
    ):
        parser.error(
            "Sample, batch and thread counts must be positive; wait seconds must be nonnegative."
        )
    evaluate(args)


if __name__ == "__main__":
    main()
