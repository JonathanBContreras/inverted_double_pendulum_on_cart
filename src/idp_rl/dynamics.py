from __future__ import annotations

import numpy as np

from idp_rl.config import PhysicsConfig


class DoublePendulumCartDynamics:
    """Derived dynamics for two absolute-angle links mounted on a cart."""

    def __init__(self, config: PhysicsConfig):
        self.config = config

    def accelerations(self, state: np.ndarray, force: float) -> np.ndarray:
        x, x_dot, theta1, theta1_dot, theta2, theta2_dot = np.asarray(state, dtype=np.float64)
        cfg = self.config
        m0 = cfg.cart_mass
        m1 = cfg.link1_mass
        m2 = cfg.link2_mass
        l1 = cfg.link1_length
        l2 = cfg.link2_length
        g = cfg.gravity

        force = float(np.clip(force, -cfg.max_force, cfg.max_force))
        delta = theta1 - theta2

        mass_matrix = np.array(
            [
                [m0 + m1 + m2, (m1 + m2) * l1 * np.cos(theta1), m2 * l2 * np.cos(theta2)],
                [(m1 + m2) * l1 * np.cos(theta1), (m1 + m2) * l1**2, m2 * l1 * l2 * np.cos(delta)],
                [m2 * l2 * np.cos(theta2), m2 * l1 * l2 * np.cos(delta), m2 * l2**2],
            ],
            dtype=np.float64,
        )

        rhs = np.array(
            [
                force
                + (m1 + m2) * l1 * np.sin(theta1) * theta1_dot**2
                + m2 * l2 * np.sin(theta2) * theta2_dot**2,
                (m1 + m2) * g * l1 * np.sin(theta1)
                - m2 * l1 * l2 * np.sin(delta) * theta2_dot**2,
                m2 * g * l2 * np.sin(theta2)
                + m2 * l1 * l2 * np.sin(delta) * theta1_dot**2,
            ],
            dtype=np.float64,
        )

        return np.linalg.solve(mass_matrix, rhs)

    def derivative(self, state: np.ndarray, force: float) -> np.ndarray:
        state = np.asarray(state, dtype=np.float64)
        x_ddot, theta1_ddot, theta2_ddot = self.accelerations(state, force)
        return np.array(
            [state[1], x_ddot, state[3], theta1_ddot, state[5], theta2_ddot],
            dtype=np.float64,
        )

    def rk4_step(self, state: np.ndarray, force: float) -> np.ndarray:
        dt = self.config.dt
        state = np.asarray(state, dtype=np.float64)
        k1 = self.derivative(state, force)
        k2 = self.derivative(state + 0.5 * dt * k1, force)
        k3 = self.derivative(state + 0.5 * dt * k2, force)
        k4 = self.derivative(state + dt * k3, force)
        next_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        next_state[2] = wrap_angle(next_state[2])
        next_state[4] = wrap_angle(next_state[4])
        return next_state


def wrap_angle(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi
