from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict

import numpy as np
from scipy.linalg import solve_continuous_are

from idp_rl.config import PhysicsConfig
from idp_rl.dynamics import DoublePendulumCartDynamics


def linearize_dynamics(config: PhysicsConfig, epsilon: float = 1e-5) -> tuple[np.ndarray, np.ndarray]:
    dynamics = DoublePendulumCartDynamics(config)
    origin = np.zeros(6, dtype=np.float64)
    a_matrix = np.zeros((6, 6), dtype=np.float64)
    b_matrix = np.zeros((6, 1), dtype=np.float64)

    for index in range(6):
        perturb = np.zeros(6, dtype=np.float64)
        perturb[index] = epsilon
        a_matrix[:, index] = (
            dynamics.derivative(origin + perturb, 0.0) - dynamics.derivative(origin - perturb, 0.0)
        ) / (2.0 * epsilon)

    b_matrix[:, 0] = (
        dynamics.derivative(origin, epsilon) - dynamics.derivative(origin, -epsilon)
    ) / (2.0 * epsilon)
    return a_matrix, b_matrix


def lqr_gain(
    config: PhysicsConfig,
    q_diag: np.ndarray | list[float] | None = None,
    r: float = 0.1,
) -> np.ndarray:
    if "torch" in sys.modules:
        return _lqr_gain_subprocess(config, q_diag, r)
    return _lqr_gain_local(config, q_diag, r)


def _lqr_gain_local(
    config: PhysicsConfig,
    q_diag: np.ndarray | list[float] | None = None,
    r: float = 0.1,
) -> np.ndarray:
    q_diag = np.asarray(q_diag or [1.0, 1.0, 100.0, 10.0, 100.0, 10.0], dtype=np.float64)
    a_matrix, b_matrix = linearize_dynamics(config)
    q_matrix = np.diag(q_diag)
    r_matrix = np.array([[r]], dtype=np.float64)
    p_matrix = solve_continuous_are(a_matrix, b_matrix, q_matrix, r_matrix)
    return np.linalg.solve(r_matrix, b_matrix.T @ p_matrix)


def _lqr_gain_subprocess(
    config: PhysicsConfig,
    q_diag: np.ndarray | list[float] | None = None,
    r: float = 0.1,
) -> np.ndarray:
    payload = {
        "config": asdict(config),
        "q_diag": list(q_diag) if q_diag is not None else None,
        "r": r,
    }
    script = (
        "import json, numpy as np; "
        "from idp_rl.config import PhysicsConfig; "
        "from idp_rl.lqr import _lqr_gain_local; "
        "payload=json.loads(__import__('sys').stdin.read()); "
        "gain=_lqr_gain_local(PhysicsConfig(**payload['config']), payload['q_diag'], payload['r']); "
        "print(json.dumps(gain.tolist()))"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=True,
    )
    return np.asarray(json.loads(result.stdout), dtype=np.float64)


def lqr_action(state: np.ndarray, gain: np.ndarray, max_force: float) -> float:
    force = float((-gain @ np.asarray(state, dtype=np.float64).reshape(6, 1)).item())
    return float(np.clip(force, -max_force, max_force))
