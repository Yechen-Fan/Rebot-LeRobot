from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass

import numpy as np
import serial

from lerobot.utils.errors import RecoverableObservationError
from lerobot.utils.tactile import TACTILE_PREPROCESSING, contact_map

TACTILE_NAMES = ("left_inner", "left_outer", "right_inner", "right_outer")
SINGLE_TACTILE_NAMES = ("inner", "outer")
MAGIC = b"\xaa\x55"
ROWS = 16
COLS = 32
FRAME_SIZE = ROWS * COLS
OFFICIAL_THRESHOLD = 15.0
OFFICIAL_NOISE_SCALE = 60.0
OFFICIAL_MAX_CONTACT = 100.0
logger = logging.getLogger(__name__)


class FlexiTacError(RecoverableObservationError):
    pass


@dataclass(frozen=True)
class FlexiTacSnapshot:
    raw: np.ndarray
    timestamp: float
    sequence: int


class FlexiTacSensor:
    """Threaded reader for the FlexiTac 0xAA55 + 512-byte protocol."""

    def __init__(
        self,
        port: str,
        *,
        baudrate: int = 2_000_000,
        timeout_s: float = 0.05,
        serial_factory: Callable[..., object] = serial.Serial,
    ) -> None:
        self.port = port
        self.baudrate = baudrate
        self.timeout_s = timeout_s
        self._serial_factory = serial_factory
        self._serial = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._ready = threading.Condition()
        self._latest: FlexiTacSnapshot | None = None
        self._error: BaseException | None = None
        self._baseline: np.ndarray | None = None

    @property
    def is_connected(self) -> bool:
        return self._serial is not None and self._thread is not None and self._thread.is_alive()

    @property
    def baseline(self) -> np.ndarray:
        if self._baseline is None:
            raise FlexiTacError(f"FlexiTac {self.port} has not been calibrated")
        return self._baseline.copy()

    def connect(self) -> None:
        if self.is_connected:
            raise FlexiTacError(f"FlexiTac {self.port} is already connected")
        self._serial = self._serial_factory(self.port, self.baudrate, timeout=self.timeout_s)
        if hasattr(self._serial, "reset_input_buffer"):
            self._serial.reset_input_buffer()
        self._stop.clear()
        with self._ready:
            self._latest = None
            self._error = None
        self._thread = threading.Thread(target=self._read_loop, name=f"flexitac:{self.port}", daemon=True)
        self._thread.start()

    def calibrate(
        self,
        frames: int = 30,
        timeout_s: float = 5.0,
        max_span: float | None = 30.0,
        mean_min: float = 0.0,
        mean_max: float = 200.0,
    ) -> np.ndarray:
        samples = []
        with self._ready:
            last_sequence = self._latest.sequence if self._latest is not None else -1
        deadline = time.monotonic() + timeout_s
        while len(samples) < frames:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise FlexiTacError(
                    f"Timed out calibrating {self.port}: received {len(samples)}/{frames} frames"
                )
            snapshot = self.wait_for_frame(after_sequence=last_sequence, timeout_s=remaining)
            samples.append(snapshot.raw)
            last_sequence = snapshot.sequence
        samples_array = np.stack(samples)
        observed_span = float(np.ptp(samples_array.astype(np.int16), axis=0).max())
        observed_mean = float(samples_array.mean())
        logger.debug(
            "FlexiTac %s calibration: mean=%.2f max_span=%.0f frames=%d",
            self.port,
            observed_mean,
            observed_span,
            frames,
        )
        if not mean_min <= observed_mean <= mean_max:
            raise FlexiTacError(
                f"FlexiTac {self.port} calibration mean {observed_mean:.2f} is outside the expected "
                f"unloaded range [{mean_min:.2f}, {mean_max:.2f}]; remove contact or adjust the "
                "configured unloaded range"
            )
        if max_span is not None and observed_span > max_span:
            raise FlexiTacError(
                f"FlexiTac {self.port} changed by {observed_span:.0f} counts during calibration; "
                "remove contact and retry"
            )
        self._baseline = np.median(samples_array, axis=0).astype(np.float32)
        return self.baseline

    def wait_for_frame(self, *, after_sequence: int = -1, timeout_s: float | None = None) -> FlexiTacSnapshot:
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        with self._ready:
            if self._stop.is_set():
                raise FlexiTacError(f"FlexiTac {self.port} was disconnected while waiting for a frame")
            while self._latest is None or self._latest.sequence <= after_sequence:
                if self._stop.is_set():
                    raise FlexiTacError(f"FlexiTac {self.port} was disconnected while waiting for a frame")
                if self._error is not None:
                    raise FlexiTacError(f"FlexiTac {self.port} reader failed") from self._error
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise FlexiTacError(f"Timed out waiting for a frame from {self.port}")
                self._ready.wait(remaining)
            return FlexiTacSnapshot(self._latest.raw.copy(), self._latest.timestamp, self._latest.sequence)

    def snapshot(self, *, max_age_s: float) -> FlexiTacSnapshot:
        with self._ready:
            if self._error is not None:
                raise FlexiTacError(f"FlexiTac {self.port} reader failed") from self._error
            if self._latest is None:
                raise FlexiTacError(f"FlexiTac {self.port} has not produced a frame")
            age = time.monotonic() - self._latest.timestamp
            if age > max_age_s:
                raise FlexiTacError(f"FlexiTac {self.port} frame is stale ({age:.3f}s > {max_age_s:.3f}s)")
            return FlexiTacSnapshot(self._latest.raw.copy(), self._latest.timestamp, self._latest.sequence)

    def official_contact_map(self, raw: np.ndarray) -> np.ndarray:
        """Match the contact-map processing and color scale in fast_32_16.py."""
        return contact_map(raw, self.baseline)

    def disconnect(self) -> None:
        self._stop.set()
        with self._ready:
            self._ready.notify_all()
        serial_device = self._serial
        if serial_device is not None:
            with suppress(Exception):
                serial_device.close()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.timeout_s * 4))
        self._thread = None
        self._serial = None

    def _read_loop(self) -> None:
        buffer = bytearray()
        sequence = 0
        try:
            while not self._stop.is_set():
                chunk = self._serial.read(max(1, min(4096, self._serial.in_waiting)))
                if not chunk:
                    continue
                buffer.extend(chunk)
                while True:
                    marker = buffer.find(MAGIC)
                    if marker < 0:
                        if len(buffer) > 1:
                            del buffer[:-1]
                        break
                    if marker:
                        del buffer[:marker]
                    if len(buffer) < len(MAGIC) + FRAME_SIZE:
                        break
                    payload = bytes(buffer[2 : 2 + FRAME_SIZE])
                    del buffer[: 2 + FRAME_SIZE]
                    raw = np.frombuffer(payload, dtype=np.uint8).reshape(ROWS, COLS).copy()
                    with self._ready:
                        self._latest = FlexiTacSnapshot(raw, time.monotonic(), sequence)
                        sequence += 1
                        self._ready.notify_all()
        except BaseException as exc:
            if not self._stop.is_set():
                with self._ready:
                    self._error = exc
                    self._ready.notify_all()


class ReleaseZero:
    """One correction per observed release; stable pressure alone never arms it."""

    RELEASE_TIMEOUT_S = 30.0
    REMAINING_PEAK_RATIO = 0.7
    STABLE_SPAN = 10.0

    def __init__(self) -> None:
        self.armed = False
        self.peaks: dict[str, float] = {}
        self.release_time: float | None = None
        self.window: deque = deque(maxlen=256)
        self.last_sequences: tuple | None = None
        self.recontact_reference: dict[str, np.ndarray] | None = None

    def update(self, snapshots, baselines, gripper: float | None) -> dict[str, np.ndarray] | None:
        sequences = tuple(s.sequence for s in snapshots.values())
        if sequences == self.last_sequences:
            return None
        # Both pads must have supplied new frames; cached frames are not evidence of stability.
        if self.last_sequences is not None and any(
            a == b for a, b in zip(sequences, self.last_sequences, strict=True)
        ):
            return None
        self.last_sequences = sequences
        if gripper is None or not np.isfinite(gripper) or not 0.0 <= gripper <= 1.0:
            self.armed = False
            self.peaks.clear()
            self.release_time = None
            self.window.clear()
            self.recontact_reference = None
            return None
        now = min(s.timestamp for s in snapshots.values())
        raw = {n: s.raw.astype(np.float32) for n, s in snapshots.items()}
        contact = {n: np.maximum(v - baselines[n] - OFFICIAL_THRESHOLD, 0) for n, v in raw.items()}
        energy = {n: float(v.sum()) for n, v in contact.items()}
        if gripper >= 0.15:
            if not self.armed or self.release_time is not None:
                self.peaks.clear()
            self.armed = True
            self.release_time = None
            self.window.clear()
            self.recontact_reference = None
            for n, value in energy.items():
                self.peaks[n] = max(self.peaks.get(n, 0.0), value)
            return None
        if not self.armed or gripper > 0.05:
            self.window.clear()
            return None
        if self.release_time is None:
            self.release_time = now
        if now - self.release_time > self.RELEASE_TIMEOUT_S:
            self.armed = False
            self.peaks.clear()
            self.window.clear()
            self.recontact_reference = None
            return None
        # A measurable contact episode followed by a drop on every loaded pad is required.
        if max(self.peaks.values(), default=0.0) < 30.0 or any(
            value > self.REMAINING_PEAK_RATIO * self.peaks.get(n, 0.0) for n, value in energy.items()
        ):
            self.window.clear()
            return None
        if self.recontact_reference is not None:
            # A new touch pauses correction until that pressure falls again.
            # Persistent contact must not become a stable baseline on retry.
            if any(np.max(contact[n] - self.recontact_reference[n]) > self.STABLE_SPAN for n in raw):
                return None
            self.recontact_reference = None
        if self.window:
            first_time, first_gripper, first_raw = self.window[0]
            if abs(gripper - first_gripper) > 0.01 or any(
                np.max(abs(raw[n] - first_raw[n])) > self.STABLE_SPAN for n in raw
            ):
                first_contact = {
                    n: np.maximum(first_raw[n] - baselines[n] - OFFICIAL_THRESHOLD, 0) for n in raw
                }
                rising = any(np.max(contact[n] - first_contact[n]) > OFFICIAL_THRESHOLD for n in raw)
                self.window.clear()
                if rising:
                    self.recontact_reference = first_contact
                    return None
        self.window.append((now, gripper, raw))
        if len(self.window) < 12 or now - self.window[0][0] < 0.75:
            return None
        samples = {n: np.stack([entry[2][n] for entry in self.window]) for n in raw}
        if any(np.ptp(v, axis=0).max() > self.STABLE_SPAN for v in samples.values()):
            self.window.clear()
            return None
        result = {n: np.median(v, axis=0).astype(np.float32) for n, v in samples.items()}
        self.armed = False
        self.peaks.clear()
        self.window.clear()
        self.recontact_reference = None
        return result


class OpenZero:
    """Maintain one pad's baseline during sustained open, assumed unloaded rest.

    Pressure rises restart stability checks. A blocked pad can recover after
    a longer stable open interval, even if its old reference is never reached.
    Static contact cannot be distinguished from drift under the unloaded-rest
    assumption, including contact that remains stable after a detected rise.
    """

    OPEN_WAIT_S = 3.0
    STABLE_TIME_S = 1.5
    RECONTACT_STABLE_TIME_S = 3.0
    RISE_WINDOW_S = 0.5

    def __init__(self) -> None:
        self.last_sequence: int | None = None
        self.open_since: float | None = None
        self.history: deque = deque(maxlen=256)
        self.window: deque = deque(maxlen=1024)
        self.recontact_reference: np.ndarray | None = None

    def update(self, snapshot: FlexiTacSnapshot, baseline: np.ndarray, gripper: float | None):
        if snapshot.sequence == self.last_sequence:
            return None
        self.last_sequence = snapshot.sequence
        if gripper is None or not np.isfinite(gripper) or not 0 <= gripper <= 0.05:
            self.open_since = None
            self.history.clear()
            self.window.clear()
            self.recontact_reference = None
            return None

        now = snapshot.timestamp
        raw = snapshot.raw.astype(np.float32)
        contact = np.maximum(raw - baseline - OFFICIAL_THRESHOLD, 0)
        if self.open_since is None:
            self.open_since = now
        blocked = self.recontact_reference is not None
        if blocked and np.max(contact - self.recontact_reference) <= ReleaseZero.STABLE_SPAN:
            self.recontact_reference = None
            self.history.clear()
            self.window.clear()
            blocked = False

        while self.history and now - self.history[0][0] > self.RISE_WINDOW_S:
            self.history.popleft()
        if self.history:
            reference = np.minimum.reduce([entry[1] for entry in self.history])
            if np.max(contact - reference) > OFFICIAL_THRESHOLD:
                self.recontact_reference = reference
                self.history.clear()
                self.window.clear()
                return None
        self.history.append((now, contact))

        if not np.any(contact):
            self.window.clear()
            return None
        self.window.append((now, gripper, raw))
        stable_time = self.RECONTACT_STABLE_TIME_S if blocked else self.STABLE_TIME_S
        # Keep the latest stability interval, plus enough fresh frames at low
        # rates. A single outlier must age out instead of restarting the entire
        # interval and repeatedly missing a later, genuinely stable window.
        while len(self.window) > 24 and now - self.window[1][0] >= stable_time:
            self.window.popleft()
        if (
            now - self.open_since < self.OPEN_WAIT_S
            or len(self.window) < 24
            or now - self.window[0][0] < stable_time
        ):
            return None
        samples = np.stack([entry[2] for entry in self.window])
        grippers = [entry[1] for entry in self.window]
        if np.ptp(grippers) > 0.01 or np.ptp(samples, axis=0).max() > ReleaseZero.STABLE_SPAN:
            return None
        corrected = np.median(samples, axis=0).astype(np.float32)
        self.window.clear()
        self.history.clear()
        self.recontact_reference = None
        return corrected


class FlexiTacManager:
    def __init__(
        self,
        ports: dict[str, str],
        *,
        baudrate: int = 2_000_000,
        timeout_s: float = 0.05,
        max_frame_age_s: float = 0.2,
        auto_zero: bool = True,
    ) -> None:
        if not ports or any(not name or not port for name, port in ports.items()):
            raise ValueError("FlexiTac ports must be a non-empty mapping of sensor names to ports")
        if len(set(ports.values())) != len(ports):
            raise ValueError("Each FlexiTac sensor must use a distinct serial port")
        self.max_frame_age_s = max_frame_age_s
        self.auto_zero = auto_zero
        self._release_states: dict[str, ReleaseZero] = {}
        self._open_states: dict[str, OpenZero] = {}
        self._startup_baselines: dict[str, np.ndarray] = {}
        self._baseline_lock = threading.RLock()
        self.sensors = {
            name: FlexiTacSensor(port, baudrate=baudrate, timeout_s=timeout_s)
            for name, port in ports.items()
        }

    @property
    def is_connected(self) -> bool:
        return all(sensor.is_connected for sensor in self.sensors.values())

    def connect(
        self,
        *,
        calibration_frames: int,
        calibration_timeout_s: float,
        calibration_max_span: float,
        calibration_mean_min: float,
        calibration_mean_max: float,
    ) -> None:
        try:
            with ThreadPoolExecutor(max_workers=len(self.sensors)) as executor:
                futures = [executor.submit(sensor.connect) for sensor in self.sensors.values()]
                for future in futures:
                    future.result()
            self.recalibrate(
                calibration_frames=calibration_frames,
                calibration_timeout_s=calibration_timeout_s,
                calibration_max_span=calibration_max_span,
                calibration_mean_min=calibration_mean_min,
                calibration_mean_max=calibration_mean_max,
            )
        except Exception:
            self.disconnect()
            raise

    def get_contact_maps(self, grippers: dict[str, float] | None = None) -> dict[str, np.ndarray]:
        with self._baseline_lock:
            return self._get_contact_maps(grippers or {})

    def _get_contact_maps(self, grippers: dict[str, float]) -> dict[str, np.ndarray]:
        snapshots = {n: s.snapshot(max_age_s=self.max_frame_age_s) for n, s in self.sensors.items()}
        if self.auto_zero:
            for side in {n.split("_")[0] if "_" in n else "single" for n in self.sensors}:
                group = {n: s for n, s in snapshots.items()
                         if (n.split("_")[0] if "_" in n else "single") == side}
                state = self._release_states.setdefault(side, ReleaseZero())
                baseline = {n: self.sensors[n].baseline for n in group}
                corrected = state.update(group, baseline, grippers.get(side))
                if corrected is not None:
                    for n, value in corrected.items():
                        self.sensors[n]._baseline = value
                        self._open_states.pop(n, None)
                # Open-rest maintenance also covers an always-open gripper and
                # drift that reappears after the one-shot release correction.
                open_corrected = False
                for n, snapshot in group.items():
                    open_state = self._open_states.setdefault(n, OpenZero())
                    value = open_state.update(snapshot, self.sensors[n].baseline, grippers.get(side))
                    if value is not None:
                        self.sensors[n]._baseline = value
                        open_corrected = True
                if open_corrected:
                    # Closed-phase peaks/window refer to the previous baselines.
                    self._release_states[side] = ReleaseZero()
        contact_maps = {}
        for name, sensor in self.sensors.items():
            snapshot = snapshots[name]
            contact_maps[f"tactile.{name}"] = sensor.official_contact_map(snapshot.raw)
            contact_maps[f"tactile_baseline.{name}"] = np.zeros((ROWS, COLS), dtype=np.float32)
            contact_maps[f"tactile_raw.{name}"] = snapshot.raw.copy()
            contact_maps[f"tactile_raw_baseline.{name}"] = sensor.baseline
        return contact_maps

    def recalibrate(
        self,
        *,
        calibration_frames: int,
        calibration_timeout_s: float,
        calibration_max_span: float,
        calibration_mean_min: float,
        calibration_mean_max: float,
    ) -> None:
        """Replace all baselines together, restoring all of them if any sensor fails."""
        with self._baseline_lock:
            previous = {
                name: None if sensor._baseline is None else sensor.baseline
                for name, sensor in self.sensors.items()
            }
            try:
                with ThreadPoolExecutor(max_workers=len(self.sensors)) as executor:
                    futures = [executor.submit(
                        sensor.calibrate, calibration_frames, calibration_timeout_s,
                        calibration_max_span, calibration_mean_min, calibration_mean_max,
                    ) for sensor in self.sensors.values()]
                    for future in futures:
                        future.result()
            except BaseException:
                # The executor has joined every worker before rollback.
                for name, sensor in self.sensors.items():
                    sensor._baseline = previous[name]
                raise
            self._startup_baselines = {n: s.baseline for n, s in self.sensors.items()}
            self._release_states.clear()
            self._open_states.clear()

    def recover(self, *, timeout_s: float = 2.0) -> None:
        """Reconnect all readers while preserving the startup calibration baselines."""
        self._release_states.clear()
        self._open_states.clear()
        self.disconnect()
        try:
            with ThreadPoolExecutor(max_workers=len(self.sensors)) as executor:
                futures = [executor.submit(sensor.connect) for sensor in self.sensors.values()]
                for future in futures:
                    future.result()
            with ThreadPoolExecutor(max_workers=len(self.sensors)) as executor:
                futures = [
                    executor.submit(sensor.wait_for_frame, timeout_s=timeout_s)
                    for sensor in self.sensors.values()
                ]
                for future in futures:
                    future.result()
        except Exception:
            self.disconnect()
            raise

    def calibration_metadata(self) -> dict[str, dict]:
        with self._baseline_lock:
            return self._calibration_metadata()

    def _calibration_metadata(self) -> dict[str, dict]:
        return {
            name: {
                "port": sensor.port,
                "baseline": np.zeros((ROWS, COLS), dtype=np.float32).tolist(),
                "raw_baseline": self._startup_baselines.get(name, sensor.baseline).tolist(),
                "shape": [ROWS, COLS],
                "protocol": "flexitac-aa55-u8-v1",
                "preprocessing": TACTILE_PREPROCESSING,
            }
            for name, sensor in self.sensors.items()
        }

    def disconnect(self) -> None:
        with ThreadPoolExecutor(max_workers=len(self.sensors)) as executor:
            futures = [executor.submit(sensor.disconnect) for sensor in self.sensors.values()]
            for future in futures:
                future.result()
