"""Repository and resource paths used by examples and thin Python wrappers."""
from __future__ import annotations

from pathlib import Path


def package_root() -> Path:
    return Path(__file__).resolve().parent


def repo_root() -> Path:
    """Return the source repository root when running from a checkout.

    In an installed wheel there may be no repository root next to the package;
    callers should use the explicit resource paths below rather than assuming
    this path exists.
    """
    source_root = package_root().parents[1]
    if (source_root / "pyproject.toml").exists() and (source_root / "urdf").exists():
        return source_root

    cwd = Path.cwd()
    if (cwd / "pyproject.toml").exists() and (cwd / "urdf").exists():
        return cwd

    return source_root


def default_urdf_path() -> Path:
    """Return the default reBot-DevArm URDF path.

    The package-local directory is the canonical asset location so wheels and
    source checkouts use the same files. The project-level ``urdf/`` path is
    retained as a compatibility link for native examples and older callers.
    """
    rel = Path("reBot-DevArm_fixend_description") / "urdf" / "reBot-DevArm_fixend.urdf"
    candidates = [
        package_root() / "urdf" / rel,
        repo_root() / "urdf" / rel,
        package_root().parent / "urdf" / rel,
        package_root().parents[2] / "urdf" / rel,
        Path.cwd() / "urdf" / rel,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def default_calibration_dir() -> Path:
    root = repo_root()
    if (root / "pyproject.toml").exists():
        return root / "calibration"

    cwd = Path.cwd().resolve()
    for parent in [cwd, *cwd.parents]:
        candidates = [
            parent / "rebotarm_control_dm" / "calibration",
            parent / "rebot_lerobot" / "rebotarm_control_dm" / "calibration",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate

    return Path.cwd() / "calibration"


def resolve_urdf_path(urdf_path: str | Path | None = None) -> Path:
    """Resolve a URDF argument.

    ``None`` returns the original project URDF. Explicit existing paths are
    honored as-is. If the given path does not exist, the SDK-level
    ``calibration/`` directory is searched by filename. For example, both
    ``tool_calibration.urdf`` and ``some/package/tool_calibration.urdf`` resolve
    to ``calibration/tool_calibration.urdf`` when that file exists.
    """
    if urdf_path is None:
        return default_urdf_path()

    path = Path(urdf_path).expanduser()
    if path.is_absolute():
        return path

    calibration_path = default_calibration_dir() / path.name
    if calibration_path.exists():
        return calibration_path

    if path.exists():
        return path

    return path


__all__ = [
    "package_root",
    "repo_root",
    "default_urdf_path",
    "default_calibration_dir",
    "resolve_urdf_path",
]
