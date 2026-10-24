#!/usr/bin/env python3
"""MeshCat visualizer for optional simulation examples.

The visualizer uses Python pinocchio + meshcat to render the URDF. Kinematics
and trajectory calculations in the sim examples still use rebotarm_control_dm's
C++ bindings.
"""
from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_PYTHON = REPO_ROOT / "python"
if SOURCE_PYTHON.exists() and str(SOURCE_PYTHON) not in sys.path:
    sys.path.insert(0, str(SOURCE_PYTHON))

try:
    import meshcat
    import meshcat.geometry as mcg
    import pinocchio as pin
    from pinocchio.visualize import MeshcatVisualizer
except (ModuleNotFoundError, ImportError) as exc:  # pragma: no cover - optional GUI dependency
    missing = getattr(exc, "name", None)
    what = f": {missing}" if missing else ""
    raise SystemExit(
        f"example/python/sim cannot import optional visualization dependency{what}\n"
        f"{type(exc).__name__}: {exc}\n\n"
        "Install simulation visualization dependencies in the active environment:\n"
        "  pip install meshcat\n"
        '  conda install -c conda-forge "pinocchio>=3.2,<4"\n'
        "If your shell has sourced ROS, clear both ROS Python and library paths when running examples:\n"
        "  env -u PYTHONPATH -u LD_LIBRARY_PATH python example/python/sim/fk_sim.py\n"
        "Note: the RT package itself does not depend on Python pinocchio; "
        "only MeshCat visualization examples need it."
    ) from exc

from rebotarm_control_dm.kinematics import _URDF, compute_fk, load_robot_model


def _mesh_resolved_urdf() -> str:
    src = Path(_URDF)
    text = src.read_text(encoding="utf-8")
    text = text.replace(
        "package://reBot-DevArm_description_fixend/",
        f"file://{src.parents[1]}/",
    )
    tmp = tempfile.NamedTemporaryFile("w", suffix=".urdf", delete=False, encoding="utf-8")
    with tmp:
        tmp.write(text)
    return tmp.name


class Visualizer:
    def __init__(self, open_browser: bool = True) -> None:
        urdf_path = _mesh_resolved_urdf()

        self._rt_model = load_robot_model()
        self._model = pin.buildModelFromUrdf(urdf_path)
        self._data = self._model.createData()
        self._visual_model = pin.buildGeomFromUrdf(
            self._model, urdf_path, pin.GeometryType.VISUAL
        )
        self._visual_data = self._visual_model.createData()
        self._meshcat_viz = meshcat.Visualizer(zmq_url=None)
        self._viz = MeshcatVisualizer(
            self._model,
            collision_model=None,
            visual_model=self._visual_model,
            data=self._data,
            visual_data=self._visual_data,
        )
        self._viz.initViewer(self._meshcat_viz, loadModel=False)
        self._viz.loadViewerModel()

        if open_browser:
            print(f"MeshCat URL: {self._meshcat_viz.url()}")

    @property
    def meshcat(self):
        return self._meshcat_viz

    @property
    def nq(self) -> int:
        return self._rt_model.nq

    @property
    def model(self):
        return self._rt_model

    def update(self, q) -> None:
        q = np.asarray(q, dtype=float)
        if q.shape != (self.nq,):
            raise ValueError(f"q must have shape ({self.nq},), got {q.shape}")
        self._viz.display(q)

    def neutral(self) -> None:
        self.update(self._rt_model.neutral())

    def draw_path(self, points_xyz: list, node_name: str, color: int = 0x00AAFF) -> None:
        if len(points_xyz) < 2:
            return
        pts = np.array(points_xyz, dtype=np.float32).T
        line = mcg.Line(
            mcg.PointsGeometry(pts),
            mcg.LineBasicMaterial(color=color, linewidth=2),
        )
        self._meshcat_viz[node_name].set_object(line)

    def draw_ref_path(self, points_xyz: list) -> None:
        self.draw_path(points_xyz, "traj_path/ref", color=0x888888)

    def draw_actual_path(self, points_xyz: list, color: int = 0x00CC44) -> None:
        self.draw_path(points_xyz, "traj_path/actual", color=color)

    def clear_paths(self) -> None:
        for name in ("traj_path/ref", "traj_path/actual"):
            try:
                del self._meshcat_viz[name]
            except Exception:
                pass

    def clear_trajectory_line(self, name: str = "ee_trajectory") -> None:
        try:
            del self._meshcat_viz[name]
        except Exception:
            pass

    def plot_trajectory_line(self, joint_traj: list, color: int = 0xFF3300, name: str = "ee_trajectory") -> None:
        positions = []
        for pt in joint_traj:
            q = np.asarray(pt.q) if hasattr(pt, "q") else np.asarray(pt)
            _, _, transform = compute_fk(self._rt_model, q)
            positions.append(transform[:3, 3])
        if len(positions) < 2:
            return
        self.clear_trajectory_line(name)
        self._meshcat_viz[name].set_object(
            mcg.Line(
                mcg.PointsGeometry(np.asarray(positions, dtype=np.float32).T),
                mcg.LineBasicMaterial(color=color, linewidth=2),
            )
        )

    def play_trajectory(self, name: str, dt: float, q_list: list, path: list | None = None) -> None:
        print(f"[viz] play trajectory: {name}, points={len(q_list)}, dt={dt:.3f}s", flush=True)
        if path:
            self.draw_ref_path(path)
        visited = []
        for i, q in enumerate(q_list):
            self.update(np.asarray(q))
            if path and i < len(path):
                visited.append(path[i])
                self.draw_actual_path(visited)
            time.sleep(dt)
        print(f"[viz] trajectory {name!r} finished", flush=True)
