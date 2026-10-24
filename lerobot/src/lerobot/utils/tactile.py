from __future__ import annotations

import numpy as np

TACTILE_PREPROCESSING = "fast-32-16-release-zero-u8-v2"


def contact_map(raw: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """Shared live/training conversion; accepts (..., rows, columns)."""
    contact = np.clip(np.asarray(raw, dtype=np.float32) - baseline - 15.0, 0.0, 100.0)
    maximum = contact.max(axis=(-2, -1), keepdims=True)
    normalized = np.where(maximum < 15.0, contact / 60.0, contact / (maximum + 1e-6))
    return np.clip(normalized * 255.0, 0.0, 255.0).astype(np.uint8)
