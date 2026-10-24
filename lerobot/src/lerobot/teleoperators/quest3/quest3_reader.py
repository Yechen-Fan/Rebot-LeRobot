from __future__ import annotations

from typing import Any

import numpy as np

from .oculus_reader.reader import OculusReader


def _matrix_to_quaternion_xyzw(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        qw = 0.25 * scale
        qx = (matrix[2, 1] - matrix[1, 2]) / scale
        qy = (matrix[0, 2] - matrix[2, 0]) / scale
        qz = (matrix[1, 0] - matrix[0, 1]) / scale
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = 2.0 * np.sqrt(max(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2], 0.0))
        qw = (matrix[2, 1] - matrix[1, 2]) / scale
        qx = 0.25 * scale
        qy = (matrix[0, 1] + matrix[1, 0]) / scale
        qz = (matrix[0, 2] + matrix[2, 0]) / scale
    elif matrix[1, 1] > matrix[2, 2]:
        scale = 2.0 * np.sqrt(max(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2], 0.0))
        qw = (matrix[0, 2] - matrix[2, 0]) / scale
        qx = (matrix[0, 1] + matrix[1, 0]) / scale
        qy = 0.25 * scale
        qz = (matrix[1, 2] + matrix[2, 1]) / scale
    else:
        scale = 2.0 * np.sqrt(max(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1], 0.0))
        qw = (matrix[1, 0] - matrix[0, 1]) / scale
        qx = (matrix[0, 2] + matrix[2, 0]) / scale
        qy = (matrix[1, 2] + matrix[2, 1]) / scale
        qz = 0.25 * scale

    quaternion = np.array([qx, qy, qz, qw], dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return (quaternion / norm).astype(np.float32)


class Quest3ReaderAdapter:
    """Expose OculusReader data through the controller API used by the TCP teleoperator."""

    def __init__(self, ip_address: str | None = None, port: int = 5555):
        self._reader = OculusReader(ip_address=ip_address, port=port)

    def _snapshot(self) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        transforms, buttons = self._reader.get_transformations_and_buttons()
        return dict(transforms or {}), dict(buttons or {})

    def _pose(self, hand: str) -> np.ndarray:
        transforms, _ = self._snapshot()
        transform = np.asarray(transforms.get(hand, np.zeros((4, 4))), dtype=np.float64)
        if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
            return np.zeros(7, dtype=np.float32)
        quaternion = _matrix_to_quaternion_xyzw(transform[:3, :3])
        return np.concatenate((transform[:3, 3].astype(np.float32), quaternion))

    @staticmethod
    def _scalar(buttons: dict[str, Any], key: str) -> float:
        value = buttons.get(key, 0.0)
        if isinstance(value, (tuple, list, np.ndarray)):
            value = value[0] if len(value) else 0.0
        return float(value)

    def get_left_controller_pose(self) -> np.ndarray:
        return self._pose("l")

    def get_right_controller_pose(self) -> np.ndarray:
        return self._pose("r")

    def get_left_grip(self) -> float:
        _, buttons = self._snapshot()
        return self._scalar(buttons, "leftGrip")

    def get_right_grip(self) -> float:
        _, buttons = self._snapshot()
        return self._scalar(buttons, "rightGrip")

    def get_left_trigger(self) -> float:
        _, buttons = self._snapshot()
        return self._scalar(buttons, "leftTrig")

    def get_right_trigger(self) -> float:
        _, buttons = self._snapshot()
        return self._scalar(buttons, "rightTrig")

    def get_A_button(self) -> bool:
        _, buttons = self._snapshot()
        return bool(buttons.get("A", False))

    def get_B_button(self) -> bool:
        _, buttons = self._snapshot()
        return bool(buttons.get("B", False))

    def get_X_button(self) -> bool:
        _, buttons = self._snapshot()
        return bool(buttons.get("X", False))

    def get_Y_button(self) -> bool:
        _, buttons = self._snapshot()
        return bool(buttons.get("Y", False))

    def close(self) -> None:
        self._reader.stop()
