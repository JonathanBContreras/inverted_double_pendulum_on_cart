from __future__ import annotations

from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ModuleNotFoundError:  # pragma: no cover - tests skip when dependency is absent
    gym = None
    spaces = None

from idp_rl.config import EnvConfig, PhysicsConfig, ProjectConfig
from idp_rl.dynamics import DoublePendulumCartDynamics


if gym is None:
    _BaseEnv = object
else:
    _BaseEnv = gym.Env


class InvertedDoublePendulumEnv(_BaseEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        config: ProjectConfig | None = None,
        *,
        env_config: EnvConfig | None = None,
        physics_config: PhysicsConfig | None = None,
        render_mode: str | None = None,
    ):
        if gym is None or spaces is None:
            raise RuntimeError("Gymnasium is required to construct InvertedDoublePendulumEnv.")
        super().__init__()
        project_config = config or ProjectConfig()
        self.env_config = env_config or project_config.env
        self.physics_config = physics_config or project_config.physics
        self.dynamics = DoublePendulumCartDynamics(self.physics_config)
        self.render_mode = render_mode

        high = np.array(
            [
                self.physics_config.track_limit,
                np.finfo(np.float32).max,
                1.0,
                1.0,
                np.finfo(np.float32).max,
                1.0,
                1.0,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([-self.physics_config.max_force], dtype=np.float32),
            high=np.array([self.physics_config.max_force], dtype=np.float32),
            dtype=np.float32,
        )
        self.state = np.zeros(6, dtype=np.float64)
        self.steps = 0
        self.hold_steps = 0
        self.max_hold_steps = 0
        self.handoff_ready_steps = 0
        self.max_handoff_ready_steps = 0
        self._figure = None
        self._axis = None

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        options = options or {}
        if "state" in options:
            self.state = np.asarray(options["state"], dtype=np.float64).copy()
        else:
            pcfg = self.physics_config
            reset_mode = options.get("reset_mode", self.env_config.reset_mode)
            angle_range = float(options.get("angle_range", pcfg.initial_angle_range))
            angular_velocity_range = float(
                options.get("angular_velocity_range", pcfg.initial_angular_velocity_range)
            )
            cart_position_range = float(options.get("cart_position_range", pcfg.initial_cart_position_range))
            cart_velocity_range = float(options.get("cart_velocity_range", pcfg.initial_cart_velocity_range))
            if reset_mode == "upright":
                theta_center = 0.0
            elif reset_mode == "hanging":
                theta_center = np.pi
            elif reset_mode == "custom":
                theta_center = float(options["angle_center"])
            else:
                raise ValueError(f"Unsupported reset_mode: {reset_mode}")
            self.state = np.array(
                [
                    self.np_random.uniform(-cart_position_range, cart_position_range),
                    self.np_random.uniform(-cart_velocity_range, cart_velocity_range),
                    theta_center + self.np_random.uniform(-angle_range, angle_range),
                    self.np_random.uniform(-angular_velocity_range, angular_velocity_range),
                    theta_center + self.np_random.uniform(-angle_range, angle_range),
                    self.np_random.uniform(-angular_velocity_range, angular_velocity_range),
                ],
                dtype=np.float64,
            )
            self.state[2] = ((self.state[2] + np.pi) % (2.0 * np.pi)) - np.pi
            self.state[4] = ((self.state[4] + np.pi) % (2.0 * np.pi)) - np.pi
        self.steps = 0
        self.hold_steps = 0
        self.max_hold_steps = 0
        self.handoff_ready_steps = 0
        self.max_handoff_ready_steps = 0
        self._update_hold_steps()
        return self._observation(), self._info(0.0)

    def step(self, action):
        force = float(np.asarray(action, dtype=np.float64).reshape(-1)[0])
        force = float(np.clip(force, -self.physics_config.max_force, self.physics_config.max_force))
        self.state = self.dynamics.rk4_step(self.state, force)
        self.steps += 1
        self._update_hold_steps()

        terminated = self._failed()
        truncated = self.steps >= self.env_config.max_episode_steps
        reward = self._reward(force, terminated)
        return self._observation(), reward, terminated, truncated, self._info(force)

    def _observation(self) -> np.ndarray:
        x, x_dot, theta1, theta1_dot, theta2, theta2_dot = self.state
        return np.array(
            [
                x,
                x_dot,
                np.sin(theta1),
                np.cos(theta1),
                theta1_dot,
                np.sin(theta2),
                np.cos(theta2),
                theta2_dot,
            ],
            dtype=np.float32,
        )

    def _failed(self) -> bool:
        x, _, theta1, _, theta2, _ = self.state
        cart_failed = abs(x) > self.physics_config.track_limit
        angle_failed = (
            self.env_config.terminate_on_angle
            and (
                abs(theta1) > self.env_config.angle_termination_radians
                or abs(theta2) > self.env_config.angle_termination_radians
            )
        )
        return bool(cart_failed or angle_failed)

    def _reward(self, force: float, failed: bool) -> float:
        x, x_dot, theta1, theta1_dot, theta2, theta2_dot = self.state
        upright_alignment = 0.5 * (np.cos(theta1) + np.cos(theta2) + 2.0)
        upright_gate = float(self._instant_upright())
        boundary_fraction = abs(x) / self.physics_config.track_limit
        boundary_penalty = self.env_config.boundary_reward_weight * boundary_fraction**4
        penalties = (
            self.env_config.position_reward_weight * x**2
            + self.env_config.velocity_reward_weight * (x_dot**2 + theta1_dot**2 + theta2_dot**2)
            + self.env_config.action_reward_weight * (force / self.physics_config.max_force) ** 2
            + boundary_penalty
        )
        calm_bonus = upright_gate * (1.0 / (1.0 + theta1_dot**2 + theta2_dot**2))
        hold_bonus = self.env_config.hold_reward * self.hold_steps
        reward = self.env_config.upright_reward * upright_alignment + calm_bonus + hold_bonus - penalties
        if self.is_success():
            reward += self.env_config.success_bonus
        if self.is_handoff_ready():
            reward += self.env_config.handoff_bonus
        elif self.in_handoff_angle_band():
            speed_error = max(0.0, self.handoff_speed_error())
            reward -= self.env_config.handoff_velocity_penalty_weight * speed_error**2
        if failed:
            reward -= self.env_config.failure_penalty
        return float(reward)

    def _info(self, force: float) -> dict[str, Any]:
        return {
            "state": self.state.copy(),
            "force": force,
            "upright_error": self.upright_error(),
            "hold_steps": self.hold_steps,
            "max_hold_steps": self.max_hold_steps,
            "handoff_ready_steps": self.handoff_ready_steps,
            "max_handoff_ready_steps": self.max_handoff_ready_steps,
            "is_handoff_ready": self.is_handoff_ready(),
            "handoff_speed_error": self.handoff_speed_error(),
            "handoff_angle_error": self.handoff_angle_error(),
            "cart_limit_margin": self.physics_config.track_limit - abs(float(self.state[0])),
            "action_magnitude": abs(float(force)),
            "is_success": self.is_success(),
        }

    def _instant_upright(self) -> bool:
        x, _, theta1, theta1_dot, theta2, theta2_dot = self.state
        return bool(
            abs(x) <= self.physics_config.track_limit
            and abs(theta1) <= self.env_config.success_angle_radians
            and abs(theta2) <= self.env_config.success_angle_radians
            and abs(theta1_dot) <= self.env_config.success_velocity_radians_per_second
            and abs(theta2_dot) <= self.env_config.success_velocity_radians_per_second
        )

    def _update_hold_steps(self) -> None:
        if self._instant_upright():
            self.hold_steps += 1
        else:
            self.hold_steps = 0
        self.max_hold_steps = max(self.max_hold_steps, self.hold_steps)
        if self.is_handoff_ready():
            self.handoff_ready_steps += 1
        else:
            self.handoff_ready_steps = 0
        self.max_handoff_ready_steps = max(self.max_handoff_ready_steps, self.handoff_ready_steps)

    def is_success(self) -> bool:
        return self.hold_steps >= self.env_config.success_hold_steps

    def upright_error(self) -> float:
        _, _, theta1, _, theta2, _ = self.state
        return float(max(abs(theta1), abs(theta2)))

    def handoff_angle_error(self) -> float:
        _, _, theta1, _, theta2, _ = self.state
        return float(max(abs(theta1), abs(theta2)) - self.env_config.handoff_angle_radians)

    def handoff_speed_error(self) -> float:
        _, _, _, theta1_dot, _, theta2_dot = self.state
        return float(max(abs(theta1_dot), abs(theta2_dot)) - self.env_config.handoff_velocity_radians_per_second)

    def in_handoff_angle_band(self) -> bool:
        _, _, theta1, _, theta2, _ = self.state
        return bool(max(abs(theta1), abs(theta2)) <= self.env_config.handoff_angle_band_radians)

    def is_handoff_ready(self) -> bool:
        x, _, theta1, theta1_dot, theta2, theta2_dot = self.state
        return bool(
            max(abs(theta1), abs(theta2)) <= self.env_config.handoff_angle_radians
            and max(abs(theta1_dot), abs(theta2_dot)) <= self.env_config.handoff_velocity_radians_per_second
            and self.physics_config.track_limit - abs(x) >= self.env_config.handoff_cart_margin
        )

    def render(self):
        import matplotlib.pyplot as plt

        if self._figure is None or self._axis is None:
            self._figure, self._axis = plt.subplots(figsize=(7, 4))
        ax = self._axis
        ax.clear()
        x, _, theta1, _, theta2, _ = self.state
        l1 = self.physics_config.link1_length
        l2 = self.physics_config.link2_length
        p0 = np.array([x, 0.0])
        p1 = p0 + np.array([l1 * np.sin(theta1), l1 * np.cos(theta1)])
        p2 = p1 + np.array([l2 * np.sin(theta2), l2 * np.cos(theta2)])

        ax.plot([-self.physics_config.track_limit, self.physics_config.track_limit], [0, 0], "k-", linewidth=1)
        ax.add_patch(plt.Rectangle((x - 0.15, -0.08), 0.3, 0.16, fill=False, linewidth=2))
        ax.plot([p0[0], p1[0], p2[0]], [p0[1], p1[1], p2[1]], "o-", linewidth=3)
        ax.set_xlim(-self.physics_config.track_limit - 0.5, self.physics_config.track_limit + 0.5)
        ax.set_ylim(-0.4, l1 + l2 + 0.3)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"step={self.steps}")
        self._figure.canvas.draw()

        if self.render_mode == "rgb_array":
            width, height = self._figure.canvas.get_width_height()
            rgba = np.frombuffer(self._figure.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)
            return rgba[:, :, :3].copy()
        plt.pause(self.physics_config.dt)
        return None

    def close(self):
        if self._figure is not None:
            import matplotlib.pyplot as plt

            plt.close(self._figure)
        self._figure = None
        self._axis = None
