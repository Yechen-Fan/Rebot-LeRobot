"""Small helpers shared by hardware Python examples."""
from __future__ import annotations

import argparse
import atexit
import tempfile
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = _ROOT / "python" / "rebotarm_control_dm" / "config"


def add_port_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--port",
        default=None,
        help="Override the YAML channel, for example /dev/ttyACM0 or /dev/ttyACM1.",
    )


def config_with_port(config: str | None, port: str | None, *, gripper: bool = False) -> str | None:
    if not port:
        return config

    src = Path(config) if config else _CONFIG_DIR / ("gripper.yaml" if gripper else "arm.yaml")
    text = src.read_text(encoding="utf-8")
    lines = []
    replaced = False

    for line in text.splitlines():
        stripped = line.lstrip()
        indent = line[: len(line) - len(stripped)]
        if not replaced and stripped.startswith("channel:"):
            lines.append(f"{indent}channel: {port}")
            replaced = True
        else:
            lines.append(line)

    if not replaced:
        raise ValueError(f"{src} does not contain a channel: entry")

    tmp = tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        suffix=".yaml",
        prefix="rebotarm_rt_example_",
        delete=False,
    )
    with tmp:
        tmp.write("\n".join(lines))
        tmp.write("\n")
    atexit.register(lambda path=tmp.name: Path(path).unlink(missing_ok=True))
    return tmp.name
