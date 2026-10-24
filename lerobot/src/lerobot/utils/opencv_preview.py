from __future__ import annotations

import cv2
import numpy as np

from lerobot.processor import RobotObservation


class B601OpenCVPreview:
    """Render B601 camera and FlexiTac observations in one OpenCV window."""

    WINDOW_NAME = "B601 Sensors"
    WINDOW_WIDTH = 1440
    CAMERA_TILE_WIDTH = 480
    CAMERA_TILE_HEIGHT = 360
    TACTILE_TILE_WIDTH = 360
    TACTILE_TILE_HEIGHT = 180
    LABEL_HEIGHT = 24
    TACTILE_DIVIDER_WIDTH = 2
    TACTILE_FILTER_ALPHA = 0.2

    _CAMERAS = (
        ("left_wrist", "Left Wrist"),
        ("global", "Global"),
        ("right_wrist", "Right Wrist"),
    )
    _TACTILE = (
        ("left_outer", "Left Outer"),
        ("left_inner", "Left Inner"),
        ("right_inner", "Right Inner"),
        ("right_outer", "Right Outer"),
    )

    def __init__(self) -> None:
        self._enabled = True
        self._previous_tactile: dict[str, np.ndarray] = {}
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW_NAME, self.WINDOW_WIDTH, self.canvas_height)

    @property
    def canvas_height(self) -> int:
        return self.CAMERA_TILE_HEIGHT + self.TACTILE_TILE_HEIGHT

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def update(self, observation: RobotObservation) -> bool:
        """Display an observation and return false when the user closes the preview."""
        if not self._enabled:
            return False
        if not self._window_is_visible():
            self.close()
            return False

        cv2.imshow(self.WINDOW_NAME, self.build_canvas(observation))
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or not self._window_is_visible():
            self.close()
            return False
        return True

    def build_canvas(self, observation: RobotObservation) -> np.ndarray:
        """Build the fixed 3-camera plus 4-tactile visualization canvas."""
        canvas = np.zeros((self.canvas_height, self.WINDOW_WIDTH, 3), dtype=np.uint8)

        for index, (key, label) in enumerate(self._CAMERAS):
            tile = self._camera_tile(observation.get(key), label)
            x_start = index * self.CAMERA_TILE_WIDTH
            canvas[: self.CAMERA_TILE_HEIGHT, x_start : x_start + self.CAMERA_TILE_WIDTH] = tile

        tactile_y = self.CAMERA_TILE_HEIGHT
        for index, (name, label) in enumerate(self._TACTILE):
            tile = self._tactile_tile(observation.get(f"tactile.{name}"), name, label)
            x_start = index * self.TACTILE_TILE_WIDTH
            canvas[tactile_y:, x_start : x_start + self.TACTILE_TILE_WIDTH] = tile
            if index:
                divider_x = x_start - self.TACTILE_DIVIDER_WIDTH // 2
                canvas[tactile_y:, divider_x : divider_x + self.TACTILE_DIVIDER_WIDTH] = 0

        return canvas

    def close(self) -> None:
        if not self._enabled:
            return
        self._enabled = False
        try:
            cv2.destroyWindow(self.WINDOW_NAME)
        except cv2.error:
            pass

    def _window_is_visible(self) -> bool:
        try:
            return cv2.getWindowProperty(self.WINDOW_NAME, cv2.WND_PROP_VISIBLE) >= 1
        except cv2.error:
            return False

    def _camera_tile(self, image: object, label: str) -> np.ndarray:
        tile = self._empty_tile(self.CAMERA_TILE_WIDTH, self.CAMERA_TILE_HEIGHT, label)
        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[-1] != 3:
            return tile
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        tile[:] = self._resize_to_tile(
            bgr,
            self.CAMERA_TILE_WIDTH,
            self.CAMERA_TILE_HEIGHT,
            interpolation=cv2.INTER_AREA,
        )
        self._draw_label(tile, label)
        return tile

    def _tactile_tile(self, values: object, name: str, label: str) -> np.ndarray:
        tile = self._empty_tile(self.TACTILE_TILE_WIDTH, self.TACTILE_TILE_HEIGHT, label)
        if not isinstance(values, np.ndarray) or values.ndim != 2:
            return tile

        normalized = np.asarray(values, dtype=np.float32) / 255.0
        previous = self._previous_tactile.get(name)
        filtered = (
            normalized
            if previous is None
            else self.TACTILE_FILTER_ALPHA * normalized
            + (1.0 - self.TACTILE_FILTER_ALPHA) * previous
        )
        # Release should clear immediately when the measured contact is zero.
        filtered = np.where(normalized == 0, 0.0, filtered)
        self._previous_tactile[name] = filtered
        viridis_input = np.clip(filtered * 255.0, 0, 255).astype(np.uint8)
        colored = cv2.applyColorMap(viridis_input, cv2.COLORMAP_VIRIDIS)
        tile[:] = self._resize_to_tile(
            colored,
            self.TACTILE_TILE_WIDTH,
            self.TACTILE_TILE_HEIGHT,
            interpolation=cv2.INTER_NEAREST,
        )
        self._draw_label(tile, label)
        return tile

    def _empty_tile(self, width: int, height: int, label: str) -> np.ndarray:
        tile = np.full((height, width, 3), 24, dtype=np.uint8)
        self._draw_label(tile, label)
        return tile

    def _draw_label(self, tile: np.ndarray, label: str) -> None:
        tile[: self.LABEL_HEIGHT] = (42, 42, 42)
        cv2.putText(
            tile,
            label,
            (8, 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )

    @staticmethod
    def _resize_to_tile(image: np.ndarray, width: int, height: int, *, interpolation: int) -> np.ndarray:
        return cv2.resize(image, (width, height), interpolation=interpolation)
