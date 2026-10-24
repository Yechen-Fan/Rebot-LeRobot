#!/usr/bin/env python3
"""Compatibility entry point for example/python/tool_calibration.py."""
from pathlib import Path
import runpy
import sys


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent / "python"
    sys.path.insert(0, str(script_dir))
    runpy.run_path(str(script_dir / "tool_calibration.py"), run_name="__main__")
