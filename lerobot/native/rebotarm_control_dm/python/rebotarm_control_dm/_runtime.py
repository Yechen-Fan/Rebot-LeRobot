import ctypes
import sys
from pathlib import Path


_LIBSTDCPP_HANDLE = None


def ensure_compatible_libstdcpp() -> None:
    """Load the active Python environment's C++ runtime before native modules."""
    global _LIBSTDCPP_HANDLE
    if _LIBSTDCPP_HANDLE is not None or not sys.platform.startswith("linux"):
        return

    libstdcpp = Path(sys.prefix) / "lib" / "libstdc++.so.6"
    if libstdcpp.exists():
        _LIBSTDCPP_HANDLE = ctypes.CDLL(str(libstdcpp), mode=ctypes.RTLD_GLOBAL)
