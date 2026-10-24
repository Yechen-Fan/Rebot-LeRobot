import os
import platform
import shutil
import sys
from pathlib import Path

from setuptools import Distribution, find_packages, setup
from setuptools.command.build_py import build_py as _build_py


def _package_version() -> str:
    ns: dict[str, str] = {}
    version_file = Path(__file__).resolve().parent / "src" / "motorbridge" / "_version.py"
    exec(version_file.read_text(encoding="utf-8"), ns)
    return ns["VERSION"]


def _platform_lib_name() -> str:
    if sys.platform.startswith("win"):
        return "motor_abi.dll"
    if sys.platform == "darwin":
        return "libmotor_abi.dylib"
    return "libmotor_abi.so"


def _platform_gateway_name() -> str:
    if sys.platform.startswith("win"):
        return "ws_gateway.exe"
    return "ws_gateway"


def _dm_device_platform_relpath() -> Path | None:
    machine = platform.machine().lower()
    if sys.platform.startswith("linux"):
        if machine in {"x86_64", "amd64"}:
            return Path("linux/x86_64/libdm_device.so")
        if machine in {"aarch64", "arm64"}:
            return Path("linux/arm64/libdm_device.so")
    if sys.platform == "darwin":
        if machine in {"arm64", "aarch64"}:
            return Path("macos/arm64/libdm_device.dylib")
        if machine in {"x86_64", "amd64"}:
            return Path("macos/x86_64/libdm_device.dylib")
    if sys.platform.startswith("win") and machine in {"x86_64", "amd64"}:
        return Path("windows/msvc/dm_device.dll")
    return None


def _candidate_dm_device_paths() -> list[Path]:
    here = Path(__file__).resolve()
    repo_root = here.parents[2]
    rel = _dm_device_platform_relpath()
    candidates: list[Path] = []

    env = os.getenv("MOTOR_DM_DEVICE_LIB")
    if env:
        candidates.append(Path(env).expanduser())

    if rel is None:
        return candidates

    candidates.append(repo_root / "third_party" / "dm_device" / "v1.1.0" / rel)
    candidates.append(repo_root / "dm-device-sdk" / "C&C++" / "lib" / "v1.1.0" / rel)
    candidates.append(repo_root.parent / "dm-device-sdk" / "C&C++" / "lib" / "v1.1.0" / rel)
    candidates.append(here.parent / "src" / "motorbridge" / "lib" / "dm_device" / rel.name)
    return candidates


def _bundle_dm_device_runtime() -> bool:
    raw = os.getenv("MOTOR_DM_DEVICE_BUNDLE", "0").strip().lower()
    return raw not in {"0", "false", "off", "no"}


def _find_dm_device_path() -> Path | None:
    if not _bundle_dm_device_runtime():
        return None
    for path in _candidate_dm_device_paths():
        if path.exists():
            return path
    return None


def _candidate_gateway_paths() -> list[Path]:
    here = Path(__file__).resolve()
    repo_root = here.parents[2]
    bin_name = _platform_gateway_name()
    candidates: list[Path] = []

    env = os.getenv("MOTORBRIDGE_WS_GATEWAY_BIN")
    if env:
        candidates.append(Path(env).expanduser())

    candidates.append(repo_root / "target" / "release" / bin_name)
    candidates.append(here.parent / "src" / "motorbridge" / "bin" / bin_name)
    return candidates


def _resolve_gateway_path() -> Path:
    for path in _candidate_gateway_paths():
        if path.exists():
            return path
    tried = "\n".join(f"- {path}" for path in _candidate_gateway_paths())
    raise RuntimeError(
        "Cannot locate ws_gateway binary for wheel build.\n"
        f"Tried:\n{tried}\n"
        "Build gateway first (`cargo build -p ws_gateway --release`) or set MOTORBRIDGE_WS_GATEWAY_BIN."
    )


def _candidate_abi_paths() -> list[Path]:
    here = Path(__file__).resolve()
    repo_root = here.parents[2]
    lib_name = _platform_lib_name()
    candidates: list[Path] = []

    env = os.getenv("MOTORBRIDGE_LIB")
    if env:
        candidates.append(Path(env).expanduser())

    candidates.append(repo_root / "target" / "release" / lib_name)
    candidates.append(here.parent / "src" / "motorbridge" / "lib" / lib_name)
    return candidates


def _resolve_abi_path() -> Path:
    for p in _candidate_abi_paths():
        if p.exists():
            return p
    tried = "\n".join(f"- {p}" for p in _candidate_abi_paths())
    raise RuntimeError(
        "Cannot locate motor_abi shared library for wheel build.\n"
        f"Tried:\n{tried}\n"
        "Build ABI first (`cargo build -p motor_abi --release`) or set MOTORBRIDGE_LIB."
    )


class BuildPyWithAbi(_build_py):
    def run(self):
        super().run()
        abi_src = _resolve_abi_path()
        dst_dir = Path(self.build_lib) / "motorbridge" / "lib"
        if dst_dir.exists():
            shutil.rmtree(dst_dir)
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(abi_src, dst_dir / abi_src.name)

        dm_src = _find_dm_device_path()
        if dm_src is not None:
            dm_dst_dir = dst_dir / "dm_device"
            dm_dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(dm_src, dm_dst_dir / dm_src.name)

        gateway_src = _resolve_gateway_path()
        gateway_dir = Path(self.build_lib) / "motorbridge" / "bin"
        if gateway_dir.exists():
            shutil.rmtree(gateway_dir)
        gateway_dir.mkdir(parents=True, exist_ok=True)
        gateway_dst = gateway_dir / gateway_src.name
        shutil.copy2(gateway_src, gateway_dst)
        try:
            gateway_dst.chmod(0o755)
        except OSError:
            # Windows may not honor POSIX mode bits; keep best effort.
            pass


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


setup(
    name="motorbridge",
    version=_package_version(),
    description="Python SDK for motorbridge Rust ABI",
    long_description="Python SDK for the motorbridge Rust CAN motor-control ABI.",
    long_description_content_type="text/plain",
    author="motorbridge contributors",
    license="MIT",
    python_requires=">=3.10",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    package_data={"motorbridge": ["lib/*", "lib/dm_device/*", "bin/*"]},
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "motorbridge=motorbridge.cli:main",
            "motorbridge-cli=motorbridge.cli:main",
            "motorbridge-gateway=motorbridge.gateway:main",
            "motorbridge-install-dm-device=motorbridge.dm_device_runtime:main",
        ]
    },
    distclass=BinaryDistribution,
    cmdclass={"build_py": BuildPyWithAbi},
    zip_safe=False,
)
