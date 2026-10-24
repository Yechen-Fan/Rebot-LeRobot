from argparse import Namespace
import unittest
from unittest.mock import patch

from motorbridge.cli import scan


class RobstrideScanTests(unittest.TestCase):
    def test_robstride_scan_uses_one_controller_at_a_time(self) -> None:
        events: list[tuple[str, int]] = []
        live_controllers = 0
        max_live_controllers = 0

        class FakeMotor:
            def __init__(self, mid: int, fid: int) -> None:
                self.mid = mid
                self.fid = fid

            def robstride_ping_host_id(self, fid: int, timeout_ms: int) -> tuple[int, int]:
                if self.mid == 1 and fid == 0xFD:
                    return self.mid, 0xFE
                raise RuntimeError("no reply")

            def robstride_get_param_f32_host_id(
                self, param_id: int, fid: int, timeout_ms: int
            ) -> float:
                raise RuntimeError("no reply")

            def close(self) -> None:
                pass

        class FakeController:
            def __init__(self, channel: str) -> None:
                nonlocal live_controllers, max_live_controllers
                live_controllers += 1
                max_live_controllers = max(max_live_controllers, live_controllers)
                self.bound = False

            def add_robstride_motor(self, mid: int, fid: int, model: str) -> FakeMotor:
                self.bound = True
                events.append(("add", fid))
                return FakeMotor(mid, fid)

            def close_bus(self) -> None:
                assert self.bound
                events.append(("close_bus", 0))

            def close(self) -> None:
                nonlocal live_controllers
                live_controllers -= 1

        args = Namespace(
            channel="can0",
            model="rs-00",
            feedback_ids="0xFD,0xFF",
            timeout_ms=80,
            param_id="0x7019",
            param_timeout_ms=120,
        )

        with patch.object(scan, "Controller", FakeController):
            found = scan._scan_robstride(args, 1, 1)

        self.assertEqual(
            found,
            [
                (
                    1,
                    "vendor=robstride via=ping feedback_id=0xFD device_id=1 responder_id=254",
                )
            ],
        )
        self.assertEqual(max_live_controllers, 1)
        self.assertEqual(live_controllers, 0)
        self.assertEqual(events, [("add", 0xFD), ("close_bus", 0)])
