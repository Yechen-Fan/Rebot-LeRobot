"""Gravity-compensated free-drive helper for manual calibration."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from rebotarm_control_dm.dynamics import compute_generalized_gravity


class FreeDrive:
    """Run a Python callback loop that makes the arm compliant under gravity.

    The loop sends MIT commands with ``kp=0`` and configurable damping ``kd``.
    Gravity feed-forward torque comes from the native dynamics model.
    """

    def __init__(
        self,
        arm,
        model,
        *,
        rate: float = 200.0,
        kd: float | Sequence[float] = 2.0,
        gravity_scale: float = 1.0,
        model_joints: int | None = None,
        request_feedback: bool = True,
    ) -> None:
        self.arm = arm
        self.model = model
        self.rate = float(rate)
        self.kd = kd
        self.gravity_scale = float(gravity_scale)
        self.model_joints = int(model_joints if model_joints is not None else model.nq)
        self.request_feedback = bool(request_feedback)
        self._running = False

    def start(self) -> None:
        if self._running:
            return
        self.arm.mode_mit()
        self.arm.start_control_loop(self._control, rate=self.rate)
        self._running = True

    def stop(self) -> None:
        if not self._running:
            return
        self.arm.stop_control_loop()
        self._running = False

    def capture(self) -> np.ndarray:
        return np.asarray(self.arm.get_positions(request=True), dtype=float)

    def _control(self, arm, dt: float) -> None:
        q_all = np.asarray(arm.get_positions(), dtype=float)
        n = arm.num_joints
        if q_all.size != n:
            q_all = np.resize(q_all, n)

        q_model = q_all[: self.model_joints]
        tau_model = np.asarray(compute_generalized_gravity(self.model, q_model), dtype=float)

        pos = q_all
        vel = np.zeros(n, dtype=float)
        kp = np.zeros(n, dtype=float)
        kd = self._kd_vector(n)
        tau = np.zeros(n, dtype=float)
        tau[: self.model_joints] = self.gravity_scale * tau_model[: self.model_joints]

        arm.mit(
            pos=pos.tolist(),
            vel=vel.tolist(),
            kp=kp.tolist(),
            kd=kd.tolist(),
            tau=tau.tolist(),
            request_feedback=self.request_feedback,
        )

    def _kd_vector(self, n: int) -> np.ndarray:
        kd = np.asarray(self.kd, dtype=float)
        if kd.ndim == 0:
            return np.full(n, float(kd), dtype=float)
        if kd.size != n:
            raise ValueError(f"kd must be scalar or length {n}, got length {kd.size}")
        return kd.reshape(n)

    def __enter__(self) -> "FreeDrive":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()
