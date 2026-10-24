#!/usr/bin/env python3
"""单片 FlexiTac 通讯及按压响应检查。

运行（项目根目录）：
    conda run --no-capture-output -n rebot_lerobot_env python scripts/test_flexitac.py
    python scripts/test_flexitac.py --list-ports
    python scripts/test_flexitac.py --port /dev/ttyUSB0
    python scripts/test_flexitac.py --demo

项目内默认读取 lerobot/configs/collect_switch_bi_flexitac.yaml，可用 --config 指定其他文件。
可单独复制本文件运行；无项目 YAML 时使用内置的相同默认参数，不依赖 lerobot。
安装依赖：python -m pip install numpy pyserial opencv-python pyyaml
需要 numpy、pyyaml、pyserial 和带 GUI 支持的 OpenCV（opencv-python）。
启动时保持无接触，按配置采集基线；B 重新记录，Q/ESC 退出。
窗口仅显示采集驱动处理后的触觉图；数据过期时清空图像。
"""

from __future__ import annotations

import argparse
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import yaml
import serial
from serial.tools import list_ports


# 独立实现与当前项目 FlexiTac 驱动保持相同协议、校准和接触图算法。
MAGIC = b"\xaa\x55"
ROWS = 16
COLS = 32
FRAME_SIZE = ROWS * COLS
OFFICIAL_THRESHOLD = 15.0
OFFICIAL_NOISE_SCALE = 60.0
OFFICIAL_MAX_CONTACT = 100.0
logger = logging.getLogger(__name__)


class FlexiTacError(RuntimeError):
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
        last_sequence = -1
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
        contact = np.clip(
            raw.astype(np.float32) - self.baseline - OFFICIAL_THRESHOLD,
            0.0,
            OFFICIAL_MAX_CONTACT,
        )
        maximum = float(contact.max())
        normalized = (
            contact / OFFICIAL_NOISE_SCALE
            if maximum < OFFICIAL_THRESHOLD
            else contact / (maximum + 1e-6)
        )
        return np.clip(normalized * 255.0, 0.0, 255.0).astype(np.uint8)

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
                chunk = self._serial.read(4096)
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



def render(contact):
    heat = cv2.applyColorMap(contact, cv2.COLORMAP_TURBO)
    return cv2.resize(heat, (640, 320), interpolation=cv2.INTER_NEAREST)


def window_closed(window):
    try:
        return cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error as exc:
        # Qt 在最后一个窗口被关闭后会销毁 guiReceiver，此时查询会抛异常。
        if exc.code == -27:
            return True
        raise


# 与采集配置一致的内置默认值；独立复制此文件时无需项目源码或 YAML。
DEFAULT_SETTINGS = {'left_inner_port': '', 'left_outer_port': '', 'right_inner_port': '', 'right_outer_port': '', 'baudrate': 2000000, 'serial_timeout_s': 0.05, 'calibration_frames': 30, 'calibration_timeout_s': 5.0, 'calibration_max_span': 30.0, 'calibration_mean_min': 0.0, 'calibration_mean_max': 200.0, 'max_frame_age_s': 0.2}


def load_settings(path=None):
    settings = DEFAULT_SETTINGS.copy()
    if path is not None:
        with path.open() as stream:
            config = yaml.safe_load(stream)
        if not isinstance(config, dict) or not isinstance(config.get("robot"), dict):
            raise ValueError("YAML 必须包含 robot 配置")
        settings.update({k.removeprefix("flexitac_"): v for k, v in config["robot"].items()
                         if k.startswith("flexitac_")})
    for key in ("baudrate", "serial_timeout_s", "calibration_frames", "calibration_timeout_s", "max_frame_age_s"):
        if settings[key] <= 0:
            raise ValueError(f"flexitac_{key} 必须为正")
    if not isinstance(settings["calibration_frames"], int):
        raise ValueError("calibration_frames 必须为整数")
    if not 0 <= settings["calibration_mean_min"] <= settings["calibration_mean_max"] <= 255:
        raise ValueError("基线均值范围必须满足 0 <= min <= max <= 255")
    if settings["calibration_max_span"] is not None and settings["calibration_max_span"] < 0:
        raise ValueError("calibration_max_span 不能为负")
    return settings


class DemoSerial:
    """模拟相同串口协议，经过本文件的解析、校准与预处理。"""
    def __init__(self, *args, **kwargs):
        self.closed = False
        self.started = time.monotonic()

    def read(self, size):
        time.sleep(.02)
        if self.closed:
            return b""
        raw = np.full((16, 32), 20, np.uint8)
        elapsed = time.monotonic() - self.started
        if elapsed > 1.5:
            col = int(elapsed * 5) % 28
            raw[5:9, col:col + 4] = 150
        return b"\xaa\x55" + raw.tobytes()

    def close(self):
        self.closed = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    project_yaml = Path(__file__).resolve().parents[1] / "lerobot/configs/collect_switch_bi_flexitac.yaml"
    parser.add_argument("--config", type=Path, default=project_yaml if project_yaml.is_file() else None,
                        help="可选采集 YAML；项目内自动读取，独立运行使用内置默认值")
    parser.add_argument("--sensor", choices=("left_inner", "left_outer", "right_inner", "right_outer"), default="left_inner", help="使用 YAML 中对应传感器的串口")
    parser.add_argument("--port", help="覆盖 YAML 串口，例如 /dev/ttyUSB0")
    parser.add_argument("--list-ports", action="store_true")
    parser.add_argument("--demo", action="store_true", help="模拟数据，仅用于验证窗口")
    parser.add_argument("--headless", action="store_true", help="不创建窗口，仅检查收帧及校准")
    parser.add_argument("--duration", type=float, default=0, help="运行秒数，0 为持续运行")
    args = parser.parse_args()
    if args.duration < 0:
        parser.error("运行时长不能为负")
    ports = list(list_ports.comports())
    if args.list_ports:
        print("\n".join(f"{p.device}: {p.description} [{p.hwid}]" for p in ports) or "未发现串口设备")
        return 0
    try:
        cfg = load_settings(args.config)
    except (OSError, ValueError, TypeError, yaml.YAMLError) as exc:
        parser.error(str(exc))
    if not args.demo and not args.port:
        configured = cfg[f"{args.sensor}_port"]
        if configured and "REPLACE_" not in configured:
            args.port = configured
        else:
            candidates = [p.device for p in ports if p.vid is not None or p.device.startswith(("/dev/ttyUSB", "/dev/ttyACM"))]
            if len(candidates) != 1:
                parser.error("YAML 串口未配置且无法唯一识别 USB 串口，请通过 --port 指定")
            args.port = candidates[0]
    options = {"serial_factory": DemoSerial} if args.demo else {}
    sensor = FlexiTacSensor(args.port or "DEMO", baudrate=cfg["baudrate"], timeout_s=cfg["serial_timeout_s"], **options)
    executor = ThreadPoolExecutor(max_workers=1)
    future = None
    window = "FlexiTac single sensor test"
    raw = np.zeros((16, 32), np.uint8)
    contact = np.zeros_like(raw)
    baseline = None
    frames, sequence = 0, -1
    start = last_rx = time.monotonic()

    def calibrate():
        return executor.submit(sensor.calibrate,
            frames=cfg["calibration_frames"], timeout_s=cfg["calibration_timeout_s"],
            max_span=cfg["calibration_max_span"], mean_min=cfg["calibration_mean_min"],
            mean_max=cfg["calibration_mean_max"])

    try:
        sensor.connect()
        print(f"配置：{args.config or '内置默认值（与采集配置一致）'}\nFlexiTac 参数：{cfg}", flush=True)
        print(f"{'模拟数据 DEMO' if args.demo else args.port}：请保持无接触，采集 {cfg['calibration_frames']} 帧基线。", flush=True)
        future = calibrate()
        if not args.headless:
            cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
        while not args.duration or time.monotonic() - start < args.duration:
            now = time.monotonic()
            if future is not None and future.done():
                try:
                    baseline = future.result()
                    print("基线已通过采集驱动校验，请逐点按压。", flush=True)
                except FlexiTacError as exc:
                    print(f"基线校验失败：{exc}；松开传感器，按 B 重试。", flush=True)
                    if args.headless:
                        raise
                future = None
            try:
                snap = sensor.snapshot(max_age_s=cfg["max_frame_age_s"])
                if snap.sequence != sequence:
                    raw, sequence = snap.raw, snap.sequence
                    frames += 1
                    last_rx = snap.timestamp
                    if baseline is not None:
                        contact = sensor.official_contact_map(raw)
            except FlexiTacError as exc:
                contact[:] = 0
                if exc.__cause__ is not None:
                    raise
            if args.headless and now - last_rx > cfg["calibration_timeout_s"]:
                raise RuntimeError("持续未收到有效帧，请检查串口、接线和波特率")
            if not args.headless:
                cv2.imshow(window, render(contact))
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")) or window_closed(window):
                    return 0
                if key == ord("b") and future is None:
                    baseline = None
                    contact[:] = 0
                    future = calibrate()
            time.sleep(.005)
        print(f"结束：观察到 {frames} 帧。", flush=True)
        return 0 if baseline is not None and frames and time.monotonic() - last_rx <= cfg["max_frame_age_s"] else 1
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(f"测试失败：{exc}；原因：{exc.__cause__ or exc}", file=sys.stderr)
        return 1
    finally:
        sensor.disconnect()
        executor.shutdown(wait=True)
        if not args.headless:
            with suppress(cv2.error):
                cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
