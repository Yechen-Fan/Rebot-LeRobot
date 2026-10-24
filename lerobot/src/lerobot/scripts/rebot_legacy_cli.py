"""Deprecated reBot command aliases that forward to the standard LeRobot CLIs."""

import sys


def _warn(old_command: str, new_command: str) -> None:
    print(
        f"{old_command} is deprecated; use {new_command} with the same configuration.",
        file=sys.stderr,
    )


def record_quest3() -> None:
    _warn("lerobot-record-quest3", "lerobot-record")
    from lerobot.scripts.lerobot_record import main

    main()


def teleoperate_quest3() -> None:
    _warn("lerobot-teleoperate-quest3", "lerobot-teleoperate")
    from lerobot.scripts.lerobot_teleoperate import main

    main()


def teleoperate_switch() -> None:
    _warn("lerobot-teleoperate-switch", "lerobot-teleoperate")
    from lerobot.scripts.lerobot_teleoperate import main

    main()
