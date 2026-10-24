"""TCP calibration math for a fixed tool mounted on the flange.

The main workflow is the classic 4-point method:

    R_i @ u + t_i = O

where ``(R_i, t_i)`` are measured flange poses, ``u`` is the TCP position in
the flange frame, and ``O`` is the fixed touched point in the world frame.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np


def _as_pose(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=float)
    if pose.shape != (4, 4):
        raise ValueError(f"expected a 4x4 pose matrix, got shape {pose.shape}")
    return pose


def _normalize(vec: np.ndarray, *, name: str) -> np.ndarray:
    vec = np.asarray(vec, dtype=float).reshape(3)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-12:
        raise ValueError(f"{name} is too small to normalize")
    return vec / norm


def solve_tcp_position(flange_poses: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Solve TCP translation ``u`` and fixed world point ``O`` from flange poses.

    Args:
        flange_poses: At least four 4x4 flange poses whose TCP touches the same
            fixed point.

    Returns:
        ``(u, O, res_max_mm, res_mean_mm)``.
    """
    if len(flange_poses) < 4:
        raise ValueError("at least four flange poses are required")

    a_rows = []
    b_rows = []
    for pose in flange_poses:
        pose = _as_pose(pose)
        rot = pose[:3, :3]
        trans = pose[:3, 3]
        a_rows.append(np.hstack([rot, -np.eye(3)]))
        b_rows.append(-trans)

    a = np.vstack(a_rows)
    b = np.concatenate(b_rows)
    solution, *_ = np.linalg.lstsq(a, b, rcond=None)
    u = solution[:3]
    world_point = solution[3:6]

    residuals_m = []
    for pose in flange_poses:
        pose = _as_pose(pose)
        tip = pose[:3, :3] @ u + pose[:3, 3]
        residuals_m.append(float(np.linalg.norm(tip - world_point)))

    residuals_mm = np.asarray(residuals_m, dtype=float) * 1000.0
    return u, world_point, float(np.max(residuals_mm)), float(np.mean(residuals_mm))


def tool_axis_from_pose(flange_pose_k: np.ndarray, u: np.ndarray, world_point: np.ndarray) -> np.ndarray:
    """Return a tool axis expressed in the flange frame from one direction pose.

    The operator first touches the reference point ``O`` with the TCP, then
    moves the TCP along the desired tool axis and records ``flange_pose_k``.
    The world displacement is rotated back to the flange frame.
    """
    pose = _as_pose(flange_pose_k)
    u = np.asarray(u, dtype=float).reshape(3)
    world_point = np.asarray(world_point, dtype=float).reshape(3)

    tip_k = pose[:3, :3] @ u + pose[:3, 3]
    world_delta = tip_k - world_point
    axis = pose[:3, :3].T @ world_delta
    return _normalize(axis, name="tool axis")


def solve_tcp_full(
    touch_poses: Sequence[np.ndarray],
    z_pose: np.ndarray,
    x_pose: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """Solve full ``T_flange_tool`` from touch poses and +Z/+X direction poses."""
    u, world_point, res_max_mm, res_mean_mm = solve_tcp_position(touch_poses)
    z_axis = tool_axis_from_pose(z_pose, u, world_point)
    x_raw = tool_axis_from_pose(x_pose, u, world_point)

    x_axis = x_raw - float(np.dot(x_raw, z_axis)) * z_axis
    x_axis = _normalize(x_axis, name="tool +X axis after orthogonalization")
    y_axis = _normalize(np.cross(z_axis, x_axis), name="tool +Y axis")
    # Recompute X to remove numerical cross-product drift and keep det(R)=+1.
    x_axis = _normalize(np.cross(y_axis, z_axis), name="tool +X axis")

    transform = np.eye(4, dtype=float)
    transform[:3, :3] = np.column_stack([x_axis, y_axis, z_axis])
    transform[:3, 3] = u

    residuals = {
        "max": res_max_mm,
        "mean": res_mean_mm,
        "x_dot_z": float(np.dot(x_raw, z_axis)),
    }
    return transform, residuals


def matrix_to_xyz_rpy_deg(transform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert a 4x4 transform to xyz meters and XYZ fixed-axis RPY degrees."""
    transform = _as_pose(transform)
    rot = transform[:3, :3]
    sy = float(np.hypot(rot[0, 0], rot[1, 0]))
    singular = sy < 1e-12
    if not singular:
        roll = np.arctan2(rot[2, 1], rot[2, 2])
        pitch = np.arctan2(-rot[2, 0], sy)
        yaw = np.arctan2(rot[1, 0], rot[0, 0])
    else:
        roll = np.arctan2(-rot[1, 2], rot[1, 1])
        pitch = np.arctan2(-rot[2, 0], sy)
        yaw = 0.0
    return transform[:3, 3].copy(), np.degrees(np.array([roll, pitch, yaw], dtype=float))


def xyz_rpy_deg_to_matrix(xyz: Sequence[float], rpy_deg: Sequence[float]) -> np.ndarray:
    """Create a 4x4 transform from xyz meters and XYZ fixed-axis RPY degrees."""
    xyz = np.asarray(xyz, dtype=float).reshape(3)
    roll, pitch, yaw = np.radians(np.asarray(rpy_deg, dtype=float).reshape(3))

    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=float)
    rot_y = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=float)
    rot_z = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=float)

    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rot_z @ rot_y @ rot_x
    transform[:3, 3] = xyz
    return transform


def flange_pose(model, q: Sequence[float], frame: str = "link6") -> np.ndarray:
    """Return the flange pose from the native RobotModel FK."""
    _, _, transform = model.fk(np.asarray(q, dtype=float), frame)
    return np.asarray(transform, dtype=float)


def urdf_joint_origin_values(
    urdf_path: str | Path,
    *,
    joint_name: str = "end_joint",
) -> tuple[np.ndarray, np.ndarray]:
    """Read a URDF fixed joint origin as ``(xyz_m, rpy_rad)``."""
    urdf_path = Path(urdf_path)
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    joint = root.find(f"./joint[@name='{joint_name}']")
    if joint is None:
        raise ValueError(f"joint {joint_name!r} not found in {urdf_path}")

    origin = joint.find("origin")
    xyz = [0.0, 0.0, 0.0]
    rpy = [0.0, 0.0, 0.0]
    if origin is not None:
        if "xyz" in origin.attrib:
            xyz = [float(v) for v in origin.attrib["xyz"].split()]
        if "rpy" in origin.attrib:
            rpy = [float(v) for v in origin.attrib["rpy"].split()]
    if len(xyz) != 3 or len(rpy) != 3:
        raise ValueError(f"joint {joint_name!r} origin must have 3 xyz and 3 rpy values")
    return np.asarray(xyz, dtype=float), np.asarray(rpy, dtype=float)


def urdf_joint_origin_matrix(urdf_path: str | Path, *, joint_name: str = "end_joint") -> np.ndarray:
    """Read a URDF fixed joint origin as a 4x4 transform."""
    xyz, rpy = urdf_joint_origin_values(urdf_path, joint_name=joint_name)
    return xyz_rpy_deg_to_matrix(xyz, np.degrees(rpy))


def _format_origin_values(values: np.ndarray) -> str:
    return " ".join(f"{float(v):.9g}" for v in values)


def _replace_origin_attr(origin_tag: str, attr: str, value: str) -> str:
    attr_pattern = re.compile(rf"(\b{re.escape(attr)}\s*=\s*)([\"'])(.*?)(\2)", re.DOTALL)
    if attr_pattern.search(origin_tag):
        return attr_pattern.sub(lambda match: f"{match.group(1)}{match.group(2)}{value}{match.group(4)}", origin_tag, count=1)
    return origin_tag.replace("/>", f' {attr}="{value}" />', 1)


def _replace_joint_origin_text(
    text: str,
    *,
    joint_name: str,
    xyz_value: str,
    rpy_value: str | None,
) -> str | None:
    quoted_name = re.escape(joint_name)
    joint_pattern = re.compile(
        rf"<joint\b(?=[^>]*\bname\s*=\s*([\"']){quoted_name}\1)[\s\S]*?</joint>",
        re.MULTILINE,
    )
    joint_match = joint_pattern.search(text)
    if joint_match is None:
        return None

    joint_text = joint_match.group(0)
    origin_match = re.search(r"<origin\b[^>]*/>", joint_text, flags=re.MULTILINE)
    if origin_match is None:
        return None

    origin_text = origin_match.group(0)
    new_origin = _replace_origin_attr(origin_text, "xyz", xyz_value)
    if rpy_value is not None:
        new_origin = _replace_origin_attr(new_origin, "rpy", rpy_value)

    new_joint = joint_text[: origin_match.start()] + new_origin + joint_text[origin_match.end() :]
    return text[: joint_match.start()] + new_joint + text[joint_match.end() :]


def apply_tool_to_urdf(
    urdf_in: str | Path,
    transform: np.ndarray,
    urdf_out: str | Path,
    *,
    joint_name: str = "end_joint",
    preserve_rpy: bool = False,
) -> Path:
    """Write a URDF copy with ``joint_name`` origin replaced by ``transform``.

    The input URDF is not modified. URDF ``origin`` uses xyz in meters and rpy
    in radians. When ``preserve_rpy`` is true, only ``xyz`` is updated and the
    original URDF ``rpy`` text is left untouched.
    """
    transform = _as_pose(transform)
    xyz_m = transform[:3, 3].copy()
    rpy_value = None
    if not preserve_rpy:
        _, rpy_deg = matrix_to_xyz_rpy_deg(transform)
        rpy_value = _format_origin_values(np.radians(rpy_deg))

    urdf_in = Path(urdf_in)
    urdf_out = Path(urdf_out)
    xyz_value = _format_origin_values(xyz_m)

    text = urdf_in.read_text(encoding="utf-8")
    updated = _replace_joint_origin_text(text, joint_name=joint_name, xyz_value=xyz_value, rpy_value=rpy_value)
    if updated is not None:
        urdf_out.parent.mkdir(parents=True, exist_ok=True)
        urdf_out.write_text(updated, encoding="utf-8")
        return urdf_out

    tree = ET.parse(urdf_in)
    root = tree.getroot()
    joint = root.find(f"./joint[@name='{joint_name}']")
    if joint is None:
        raise ValueError(f"joint {joint_name!r} not found in {urdf_in}")

    origin = joint.find("origin")
    if origin is None:
        origin = ET.SubElement(joint, "origin")
    origin.set("xyz", xyz_value)
    if rpy_value is not None:
        origin.set("rpy", rpy_value)

    urdf_out.parent.mkdir(parents=True, exist_ok=True)
    tree.write(urdf_out, encoding="utf-8", xml_declaration=True)
    return urdf_out
