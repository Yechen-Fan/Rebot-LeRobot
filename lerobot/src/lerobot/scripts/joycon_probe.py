import argparse
import time
from pprint import pprint

from lerobot.teleoperators.switch_joycon.config_switch_joycon import SwitchJoyconConfig
from lerobot.teleoperators.switch_joycon.evdev_joycon import EvdevJoycon
from lerobot.teleoperators.switch_joycon.switch_joycon import SwitchJoycon


def print_evdev_enumeration() -> None:
    print("evdev devices:")
    for snapshot in EvdevJoycon.snapshots():
        pprint(snapshot.__dict__)
    try:
        from evdev import InputDevice, ecodes, list_devices
    except ImportError:
        return
    for path in list_devices():
        try:
            dev = InputDevice(path)
        except OSError:
            continue
        is_joycon = int(getattr(dev.info, "vendor", 0)) == 0x057E or "joy-con" in (dev.name or "").lower()
        if not is_joycon:
            continue
        print(f"\n{path} {dev.name!r} capabilities:")
        caps = dev.capabilities(absinfo=True)
        key_names_by_code = ecodes.bytype.get(ecodes.EV_KEY, {})
        abs_names_by_code = ecodes.bytype.get(ecodes.EV_ABS, {})
        key_names = [
            key_names_by_code.get(code, str(code))
            for code in caps.get(ecodes.EV_KEY, [])
        ]
        abs_names = [
            abs_names_by_code.get(code, str(code))
            for code, _info in caps.get(ecodes.EV_ABS, [])
        ]
        print("  KEY:", key_names)
        print("  ABS:", abs_names)


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe Switch Joy-Con visibility and live input.")
    parser.add_argument("--device", choices=["right", "left"], default="right")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--rate", type=float, default=10.0)
    args = parser.parse_args()

    print_evdev_enumeration()

    teleop = SwitchJoycon(
        SwitchJoyconConfig(
            device=args.device,
            input_backend="evdev",
            imu_calibrate_on_connect=False,
            control_orientation=False,
        )
    )
    try:
        teleop.connect()
        deadline = time.perf_counter() + args.seconds
        while time.perf_counter() < deadline:
            teleop.get_action()
            joycon = teleop._joycon_device()
            stick_x, stick_y = teleop._stick_xy()
            print(
                {
                    "zr": bool(joycon.get_button_zr()),
                    "a": bool(joycon.get_button_a()),
                    "b": bool(joycon.get_button_b()),
                    "x": bool(joycon.get_button_x()),
                    "plus": bool(joycon.get_button_plus()),
                    "sr": bool(joycon.get_button_right_sr() if args.device == "right" else joycon.get_button_left_sr()),
                    "sl": bool(joycon.get_button_right_sl() if args.device == "right" else joycon.get_button_left_sl()),
                    "stick": (round(stick_x, 3), round(stick_y, 3)),
                    "orientation_rad": [round(float(v), 3) for v in teleop.joycon.orientation_rad],
                }
            )
            time.sleep(1.0 / args.rate)
    finally:
        if teleop.is_connected:
            teleop.disconnect()


if __name__ == "__main__":
    main()
