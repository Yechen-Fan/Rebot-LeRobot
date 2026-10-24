from __future__ import annotations

import fcntl
import logging
import math
import os
import struct
import time
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from .config_switch_joycon import SwitchJoyconConfig

logger = logging.getLogger(__name__)

NINTENDO_VENDOR_ID = 0x057E
JOYCON_L_PRODUCT_ID = 0x2006
JOYCON_R_PRODUCT_ID = 0x2007
# Linux input.h: _IOW('E', 0xa0, int). This changes this fd's event clock only.
EVIOCSCLOCKID = 0x400445A0


def _load_evdev():
    try:
        from evdev import AbsInfo, InputDevice, ecodes, list_devices
    except ImportError as exc:
        raise ImportError(
            "Switch Joy-Con evdev backend requires evdev. Install it with `python -m pip install evdev`."
        ) from exc
    return AbsInfo, InputDevice, ecodes, list_devices


@dataclass
class EvdevDeviceSnapshot:
    path: str
    name: str
    phys: str
    vendor: int
    product: int


class EvdevJoycon:
    """Linux evdev Joy-Con reader with the button methods used by SwitchJoycon."""

    def __init__(self, config: SwitchJoyconConfig):
        self.config = config
        self.side = config.device
        self._abs_info_cls, self._input_device_cls, self._ecodes, self._list_devices = _load_evdev()
        self.devices = self._open_devices()
        self.control_devices = [dev for dev in self.devices if "(imu)" not in (dev.name or "").lower()]
        self.imu_devices = [dev for dev in self.devices if "(imu)" in (dev.name or "").lower()]
        if not self.control_devices:
            self.control_devices = self.devices
        self._key_state: dict[int, int] = {}
        self._control_abs_state: dict[int, int] = {}
        self._control_abs_info: dict[int, object] = {}
        self._imu_abs_state: dict[int, int] = {}
        self._imu_abs_info: dict[int, object] = {}
        self.orientation_rad = np.zeros(3, dtype=np.float32)

        for dev in self.devices:
            dev.grab_context = None
            self._set_nonblocking(dev)
            if dev in self.imu_devices:
                fcntl.ioctl(dev.fd, EVIOCSCLOCKID, struct.pack("i", time.CLOCK_MONOTONIC))
            caps = dev.capabilities(absinfo=True)
            for code, info in caps.get(self._ecodes.EV_ABS, []):
                if dev in self.imu_devices:
                    self._imu_abs_info[code] = info
                    self._imu_abs_state.setdefault(code, int(getattr(info, "value", 0)))
                else:
                    self._control_abs_info[code] = info
                    self._control_abs_state.setdefault(code, int(getattr(info, "value", 0)))

        self._button_codes = self._build_button_codes()
        if self.side == "left":
            self._stick_x_code = self._code("ABS_X")
            self._stick_y_code = self._code("ABS_Y")
        else:
            self._stick_x_code = self._code(self.config.evdev_stick_x_axis)
            self._stick_y_code = self._code(self.config.evdev_stick_y_axis)
        self._orientation_codes = [self._code(name) for name in self.config.evdev_orientation_axes]
        self._imu_rotation = Rotation.identity()
        self._gyro_bias_rad_s = np.zeros(3)
        self._imu_timestamp_us: int | None = None
        self._imu_report_time_us: int | None = None
        self._imu_report_samples: list[tuple[int, np.ndarray]] = []
        self._imu_discard_until_sync = False
        self._calibration_samples: list[tuple[int, np.ndarray]] | None = None
        self.angular_velocity_rad_s = np.zeros(3)
        if self.config.control_orientation and not all(
            code in self._imu_abs_info and self._imu_abs_info[code].resolution > 0
            for code in self._orientation_codes
        ):
            self.disconnect()
            raise OSError("Joy-Con rotation requires a readable IMU device with gyroscope axis resolutions.")
        self.update()

    @staticmethod
    def _set_nonblocking(dev) -> None:
        flags = fcntl.fcntl(dev.fd, fcntl.F_GETFL)
        fcntl.fcntl(dev.fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

    @staticmethod
    def snapshots() -> list[EvdevDeviceSnapshot]:
        _, input_device_cls, _, list_devices = _load_evdev()
        out = []
        for path in list_devices():
            try:
                dev = input_device_cls(path)
            except OSError:
                continue
            out.append(
                EvdevDeviceSnapshot(
                    path=path,
                    name=dev.name,
                    phys=dev.phys,
                    vendor=int(getattr(dev.info, "vendor", 0)),
                    product=int(getattr(dev.info, "product", 0)),
                )
            )
        return out

    def _open_devices(self):
        paths = self._candidate_paths()
        if not paths:
            raise OSError(
                f"No {self.side} Joy-Con evdev device found. Check /dev/input/event* permissions and Bluetooth pairing."
            )
        devices = []
        for path in paths:
            devices.append(self._input_device_cls(path))
        logger.info("Using Joy-Con evdev devices: %s", ", ".join(paths))
        return devices

    def _candidate_paths(self) -> list[str]:
        if self.config.evdev_device_path:
            return [self.config.evdev_device_path]

        expected_product = JOYCON_R_PRODUCT_ID if self.side == "right" else JOYCON_L_PRODUCT_ID
        exact = []
        fuzzy = []
        for path in self._list_devices():
            try:
                dev = self._input_device_cls(path)
            except OSError:
                continue
            vendor = int(getattr(dev.info, "vendor", 0))
            product = int(getattr(dev.info, "product", 0))
            name = (dev.name or "").lower()
            if vendor == NINTENDO_VENDOR_ID and product == expected_product:
                exact.append(path)
                continue
            if "joy-con" in name or "joycon" in name:
                wants_right = self.side == "right" and ("right" in name or "(r)" in name)
                wants_left = self.side == "left" and ("left" in name or "(l)" in name)
                if wants_right or wants_left:
                    fuzzy.append(path)
        return exact or fuzzy

    def _code(self, name: str | int) -> int:
        if isinstance(name, int):
            return name
        if isinstance(name, str) and name.isdigit():
            return int(name)
        try:
            return int(self._ecodes.ecodes[name])
        except KeyError as exc:
            raise ValueError(f"Unknown evdev code {name!r}.") from exc

    def _build_button_codes(self) -> dict[str, int]:
        if self.side == "left":
            return {
                "zr": self._code("BTN_TL2"),
                "zl": self._code("BTN_TL2"),
                "a": self._code("BTN_DPAD_RIGHT"),
                "b": self._code("BTN_DPAD_DOWN"),
                "x": self._code("BTN_DPAD_UP"),
                "y": self._code("BTN_DPAD_LEFT"),
                "plus": self._code("BTN_SELECT"),
                "minus": self._code("BTN_SELECT"),
                "home": self._code("BTN_SELECT"),
                "capture": self._code("BTN_Z"),
                "right_sr": self._code("BTN_TR"),
                "right_sl": self._code("BTN_TR2"),
                "left_sr": self._code("BTN_TR"),
                "left_sl": self._code("BTN_TR2"),
                "r_stick": self._code("BTN_THUMBL"),
                "l_stick": self._code("BTN_THUMBL"),
            }

        return {
            "zr": self._code(self.config.evdev_zr_button),
            "zl": self._code(self.config.evdev_zl_button),
            "a": self._code(self.config.evdev_a_button),
            "b": self._code(self.config.evdev_b_button),
            "x": self._code(self.config.evdev_x_button),
            "y": self._code(self.config.evdev_y_button),
            "plus": self._code(self.config.evdev_plus_button),
            "minus": self._code(self.config.evdev_minus_button),
            "home": self._code("BTN_MODE"),
            "capture": self._code("BTN_SELECT"),
            "right_sr": self._code(self.config.evdev_sr_button),
            "right_sl": self._code(self.config.evdev_sl_button),
            "left_sr": self._code(self.config.evdev_sr_button),
            "left_sl": self._code(self.config.evdev_sl_button),
            "r_stick": self._code(self.config.evdev_stick_button),
            "l_stick": self._code("BTN_THUMBL"),
        }

    def update(self) -> None:
        for dev in self.devices:
            try:
                events = list(dev.read())
            except BlockingIOError:
                continue
            except OSError:
                continue
            for event in events:
                if dev in self.imu_devices and self._imu_discard_until_sync:
                    if event.type == self._ecodes.EV_SYN and event.code == self._ecodes.SYN_REPORT:
                        # EV_ABS only emits changed axes: recover their current
                        # values after an evdev queue overflow before integrating.
                        for code in self._orientation_codes:
                            self._imu_abs_state[code] = dev.absinfo(code).value
                        self._imu_discard_until_sync = False
                    continue
                if event.type == self._ecodes.EV_KEY:
                    self._key_state[event.code] = int(event.value)
                elif event.type == self._ecodes.EV_ABS:
                    if dev in self.imu_devices:
                        self._imu_abs_state[event.code] = int(event.value)
                    else:
                        self._control_abs_state[event.code] = int(event.value)
                elif dev in self.imu_devices and event.type == self._ecodes.EV_SYN:
                    if event.code == self._ecodes.SYN_DROPPED:
                        self._imu_report_samples.clear()
                        self._imu_report_time_us = None
                        self._imu_timestamp_us = None
                        self._imu_discard_until_sync = True
                    elif event.code == self._ecodes.SYN_REPORT:
                        self._queue_imu_sample(event.sec * 1_000_000 + event.usec)

    def _gyro_rates(self) -> np.ndarray | None:
        rates = []
        for code in self._orientation_codes:
            info = self._imu_abs_info.get(code)
            if info is None or info.resolution <= 0:
                return
            # hid-nintendo reports angular velocity; resolution is counts/(degree/s).
            rates.append(math.radians(self._imu_abs_state[code] / info.resolution))
        return np.asarray(rates)

    def _queue_imu_sample(self, event_time_us: int) -> None:
        rates = self._gyro_rates()
        if rates is None:
            return
        # hid-nintendo emits three SYN_REPORT samples per hardware report,
        # a few microseconds apart. A gap marks a new report if we opened the
        # fd or resumed reading in the middle of one.
        if self._imu_report_samples and event_time_us - self._imu_report_samples[-1][0] > 1000:
            self._finish_imu_report()
        self._imu_report_samples.append((event_time_us, rates))
        if len(self._imu_report_samples) == 3:
            self._finish_imu_report()

    def _finish_imu_report(self) -> None:
        samples = self._imu_report_samples
        self._imu_report_samples = []
        report_time = samples[0][0]
        previous = self._imu_report_time_us
        self._imu_report_time_us = report_time
        interval = None if previous is None else report_time - previous
        if interval is not None and 0 <= interval < 3:
            # A burst can give consecutive reports the same microsecond. It
            # supplies no elapsed time or fresh calibration interval.
            self.angular_velocity_rad_s = samples[-1][1] - self._gyro_bias_rad_s
            return
        if len(samples) != 3 or interval is None or not 0 < interval <= 100_000:
            # Rebase on the host clock, without integrating across a missing or
            # incomplete report.
            self._imu_timestamp_us = None
            self._update_orientation(report_time, samples[-1][1])
            return
        # Event times describe report arrival, not the three sampling instants.
        # Spread the samples over that elapsed interval rather than assigning
        # all rotation to the first sample or using the polling-loop duration.
        for i, (_, rates) in enumerate(samples, 1):
            self._update_orientation(previous + round(interval * i / 3), rates)

    def _update_orientation(self, timestamp_us: int, rates: np.ndarray | None = None) -> None:
        if rates is None:
            rates = self._gyro_rates()
        if rates is None:
            return
        if self._calibration_samples is not None:
            self._calibration_samples.append((timestamp_us, rates))
        self.angular_velocity_rad_s = rates - self._gyro_bias_rad_s
        previous = self._imu_timestamp_us
        self._imu_timestamp_us = timestamp_us
        if previous is None or self._calibration_samples is not None:
            return
        dt = ((timestamp_us - previous) & 0xFFFFFFFF) * 1e-6
        # Ignore gaps/reconnects; never integrate a stale velocity across them.
        if not 0 < dt <= 0.1:
            return
        self._imu_rotation = self._imu_rotation * Rotation.from_rotvec(self.angular_velocity_rad_s * dt)
        self.orientation_rad = self._imu_rotation.as_rotvec().astype(np.float32)

    def _button(self, name: str) -> int:
        self.update()
        return int(self._key_state.get(self._button_codes[name], 0) > 0)

    def _axis_normalized(self, code: int, source: str = "control") -> float:
        if source == "imu":
            info = self._imu_abs_info.get(code)
            value = float(self._imu_abs_state.get(code, 0))
        else:
            info = self._control_abs_info.get(code)
            value = float(self._control_abs_state.get(code, 0))
        if info is None:
            return 0.0
        min_value = float(getattr(info, "min", 0))
        max_value = float(getattr(info, "max", 255))
        if max_value <= min_value:
            return 0.0
        center = (max_value + min_value) * 0.5
        half = (max_value - min_value) * 0.5
        return max(-1.0, min(1.0, (value - center) / half))

    def _stick_value(self, code: int, invert: bool = False) -> int:
        self.update()
        norm = self._axis_normalized(code, source="control")
        if invert:
            norm = -norm
        return int(round(2048 + norm * 2048))

    def get_stick_right_horizontal(self):
        return self._stick_value(self._stick_x_code, self.config.evdev_invert_stick_x)

    def get_stick_right_vertical(self):
        return self._stick_value(self._stick_y_code, self.config.evdev_invert_stick_y)

    def get_stick_left_horizontal(self):
        return self.get_stick_right_horizontal()

    def get_stick_left_vertical(self):
        return self.get_stick_right_vertical()

    def get_button_zr(self): return self._button("zr")
    def get_button_zl(self): return self._button("zl")
    def get_button_a(self): return self._button("a")
    def get_button_b(self): return self._button("b")
    def get_button_x(self): return self._button("x")
    def get_button_y(self): return self._button("y")
    def get_button_plus(self): return self._button("plus")
    def get_button_minus(self): return self._button("minus")
    def get_button_home(self): return self._button("home")
    def get_button_capture(self): return self._button("capture")
    def get_button_right_sr(self): return self._button("right_sr")
    def get_button_right_sl(self): return self._button("right_sl")
    def get_button_left_sr(self): return self._button("left_sr")
    def get_button_left_sl(self): return self._button("left_sl")
    def get_button_r_stick(self): return self._button("r_stick")
    def get_button_l_stick(self): return self._button("l_stick")

    def reset_joycon(self) -> None:
        self.update()  # Discard queued samples before collecting a fresh zero bias.
        self._imu_report_samples.clear()
        self._imu_report_time_us = None
        self._imu_timestamp_us = None
        self._calibration_samples = []
        try:
            deadline = time.perf_counter() + 8.0
            warned = False
            had_window = False
            checked_count = 0
            deviation = magnitude = 0.0
            while time.perf_counter() < deadline:
                self.update()
                history = self._calibration_samples
                if len(history) != checked_count and len(history) >= 30:
                    checked_count = len(history)
                    latest = history[-1][0]
                    window = [(stamp, rate) for stamp, rate in history
                              if ((latest - stamp) & 0xFFFFFFFF) <= 750_000]
                    stamps = [stamp for stamp, _ in window]
                    if (len(window) >= 30 and ((latest - stamps[0]) & 0xFFFFFFFF) >= 700_000
                            and all(0 < ((b - a) & 0xFFFFFFFF) <= 100_000
                                    for a, b in zip(stamps, stamps[1:]))):
                        had_window = True
                        samples = np.asarray([rate for _, rate in window])
                        bias = samples.mean(axis=0)
                        deviation = float(np.max(samples.std(axis=0)))
                        magnitude = float(np.linalg.norm(bias))
                        if deviation <= math.radians(2) and magnitude <= math.radians(5):
                            self._gyro_bias_rad_s = bias
                            break
                        if not warned:
                            logger.warning("Joy-Con IMU is waiting for a stationary calibration window; "
                                           "put the controller down and keep it still (timeout: 8 seconds).")
                            warned = True
                time.sleep(0.005)
            else:
                if not had_window:
                    raise OSError("Joy-Con IMU calibration received too few continuous samples in 8 seconds; "
                                  "check the IMU connection.")
                raise ValueError(
                    "Joy-Con IMU calibration timed out after 8 seconds; keep the controller stationary "
                    f"and retry (gyro std={math.degrees(deviation):.2f} deg/s, "
                    f"mean magnitude={math.degrees(magnitude):.2f} deg/s)."
                )
        finally:
            self._calibration_samples = None
        self._imu_rotation = Rotation.identity()
        self._imu_timestamp_us = None
        self.orientation_rad = np.zeros(3, dtype=np.float32)

    def disconnect(self) -> None:
        for dev in self.devices:
            dev.close()
