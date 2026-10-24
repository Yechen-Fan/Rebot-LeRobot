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

import numbers
import os
import socket
import subprocess
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from lerobot.processor import RobotAction, RobotObservation

from .constants import ACTION, ACTION_PREFIX, OBS_PREFIX, OBS_STR


def init_rerun(
    session_name: str = "lerobot_control_loop", ip: str | None = None, port: int | None = None,
    *, spawn_viewer: bool = True,
) -> None:
    """
    Initializes the Rerun SDK for visualizing the control loop.

    Args:
        session_name: Name of the Rerun session.
        ip: Optional IP for connecting to a Rerun server.
        port: Optional port for connecting to a Rerun server.
        spawn_viewer: Whether to launch a local viewer when no remote server is specified.
    """
    batch_size = os.getenv("RERUN_FLUSH_NUM_BYTES", "8000")
    os.environ["RERUN_FLUSH_NUM_BYTES"] = batch_size
    rr.init(session_name)
    memory_limit = os.getenv("LEROBOT_RERUN_MEMORY_LIMIT", "10%")
    if ip and port:
        rr.connect_grpc(url=f"rerun+http://{ip}:{port}/proxy")
    elif spawn_viewer:
        rr.spawn(memory_limit=memory_limit)


@contextmanager
def rerun_session(session_name: str, *, enabled: bool, ip: str | None = None, port: int | None = None):
    """Close the local viewer owned by this session on every recording exit path."""
    if not enabled:
        yield
        return

    viewer = None
    try:
        if ip and port:
            init_rerun(session_name, ip=ip, port=port)
        else:
            import rerun_cli

            init_rerun(session_name, spawn_viewer=False)
            # Use a separate port so an older/unrelated viewer is never reused
            # and then accidentally closed with this recording.
            with socket.socket() as reserved:
                reserved.bind(("127.0.0.1", 0))
                viewer_port = reserved.getsockname()[1]
            executable = os.environ.get("RERUN_CLI_PATH") or str(
                Path(rerun_cli.__file__).with_name("rerun.exe" if os.name == "nt" else "rerun")
            )
            viewer = subprocess.Popen(
                [executable, f"--port={viewer_port}",
                 f"--memory-limit={os.getenv('LEROBOT_RERUN_MEMORY_LIMIT', '10%')}",
                 "--server-memory-limit=0B", "--expect-data-soon"],
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                env={**os.environ, "RERUN_APP_ONLY": "true"},
            )
            rr.connect_grpc(url=f"rerun+http://127.0.0.1:{viewer_port}/proxy")
        yield
    finally:
        try:
            rr.rerun_shutdown()
        finally:
            if viewer is not None:
                if viewer.poll() is None:
                    viewer.terminate()
                try:
                    viewer.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    viewer.kill()
                    viewer.wait(timeout=2)


def configure_rerun_camera_views(camera_names: tuple[str, ...]) -> None:
    """Replace cached/automatic layouts with views for this session's connected cameras."""
    views = []
    for name in camera_names:
        path = name if name.startswith(OBS_PREFIX) else f"{OBS_STR}.{name}"
        views.append(rrb.Spatial2DView(name=name, origin=path, contents=[path]))
    layout = rrb.Grid(contents=views, grid_columns=min(3, len(views))) if views else rrb.Grid()
    rr.send_blueprint(
        rrb.Blueprint(
            layout,
            rrb.BlueprintPanel(expanded=False),
            rrb.SelectionPanel(expanded=False),
            rrb.TimePanel(expanded=False),
            auto_layout=False,
            auto_views=False,
        ),
        make_active=True,
        make_default=True,
    )


def _is_scalar(x):
    return isinstance(x, (float | numbers.Real | np.integer | np.floating)) or (
        isinstance(x, np.ndarray) and x.ndim == 0
    )


def log_rerun_data(
    observation: RobotObservation | None = None,
    action: RobotAction | None = None,
    compress_images: bool = False,
    camera_names: tuple[str, ...] | None = None,
) -> None:
    """
    Logs observation and action data to Rerun for real-time visualization.

    This function iterates through the provided observation and action dictionaries and sends their contents
    to the Rerun viewer. It handles different data types appropriately:
    - Scalars values (floats, ints) are logged as `rr.Scalars`.
    - 3D NumPy arrays that resemble images (e.g., with 1, 3, or 4 channels first) are transposed
      from CHW to HWC format, (optionally) compressed to JPEG and logged as `rr.Image` or `rr.EncodedImage`.
    - 1D NumPy arrays are logged as a series of individual scalars, with each element indexed.
    - Other multi-dimensional arrays are flattened and logged as individual scalars.

    Keys are automatically namespaced with "observation." or "action." if not already present.

    Args:
        observation: An optional dictionary containing observation data to log.
        action: An optional dictionary containing action data to log.
        compress_images: Whether to compress images before logging to save bandwidth & memory in exchange for cpu and quality.
        camera_names: When provided, display only these camera images, without state/action plots.
    """
    camera_paths = (
        None
        if camera_names is None
        else {name if name.startswith(OBS_PREFIX) else f"{OBS_STR}.{name}" for name in camera_names}
    )
    if observation:
        for k, v in observation.items():
            if v is None:
                continue
            key = k if str(k).startswith(OBS_PREFIX) else f"{OBS_STR}.{k}"
            if camera_paths is not None and (
                key not in camera_paths
                or not isinstance(v, np.ndarray)
                or not v.size
                or not (
                    v.ndim == 2 or (v.ndim == 3 and (v.shape[0] in (1, 3, 4) or v.shape[-1] in (1, 3, 4)))
                )
            ):
                continue

            if _is_scalar(v):
                rr.log(key, rr.Scalars(float(v)))
            elif isinstance(v, np.ndarray):
                arr = v
                # Convert CHW -> HWC when needed
                if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
                    arr = np.transpose(arr, (1, 2, 0))
                if arr.ndim == 1:
                    for i, vi in enumerate(arr):
                        rr.log(f"{key}_{i}", rr.Scalars(float(vi)))
                else:
                    img_entity = rr.Image(arr).compress() if compress_images else rr.Image(arr)
                    rr.log(key, entity=img_entity)

    if action and camera_paths is None:
        for k, v in action.items():
            if v is None:
                continue
            key = k if str(k).startswith(ACTION_PREFIX) else f"{ACTION}.{k}"

            if _is_scalar(v):
                rr.log(key, rr.Scalars(float(v)))
            elif isinstance(v, np.ndarray):
                if v.ndim == 1:
                    for i, vi in enumerate(v):
                        rr.log(f"{key}_{i}", rr.Scalars(float(vi)))
                else:
                    # Fall back to flattening higher-dimensional arrays
                    flat = v.flatten()
                    for i, vi in enumerate(flat):
                        rr.log(f"{key}_{i}", rr.Scalars(float(vi)))
