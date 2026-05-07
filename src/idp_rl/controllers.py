from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch

from idp_rl.config import ProjectConfig
from idp_rl.dynamics import DoublePendulumCartDynamics
from idp_rl.hybrid_config import HybridConfig, MpcConfig
from idp_rl.lqr import lqr_action, lqr_gain
from idp_rl.ppo import ActorCritic


class Controller(Protocol):
    phase: str

    def reset(self) -> None:
        ...

    def act(self, observation: np.ndarray, info: dict[str, Any], env: Any) -> np.ndarray:
        ...


def state_to_observation(state: np.ndarray) -> np.ndarray:
    x, x_dot, theta1, theta1_dot, theta2, theta2_dot = np.asarray(state, dtype=np.float64)
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


def load_policy_model(config: ProjectConfig, checkpoint: str | Path, device: torch.device) -> ActorCritic:
    model = ActorCritic(8, 1, config.training.hidden_sizes).to(device)
    payload = torch.load(checkpoint, map_location=device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


@dataclass
class PolicyController:
    model: ActorCritic
    device: torch.device
    phase: str = "policy"

    def reset(self) -> None:
        pass

    def act(self, observation: np.ndarray, info: dict[str, Any], env: Any) -> np.ndarray:
        obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.model.actor_mean(obs_tensor).cpu().numpy()[0]
        return np.asarray(action, dtype=np.float32)


class LqrController:
    phase = "stabilize"

    def __init__(self, config: ProjectConfig, q_diag: list[float] | None = None, r: float = 0.1):
        self.max_force = config.physics.max_force
        self.gain = lqr_gain(config.physics, q_diag=q_diag, r=r)

    def reset(self) -> None:
        pass

    def act(self, observation: np.ndarray, info: dict[str, Any], env: Any) -> np.ndarray:
        return np.array([lqr_action(info["state"], self.gain, self.max_force)], dtype=np.float32)


class LqrViabilityGate:
    def __init__(
        self,
        config: ProjectConfig,
        gain: np.ndarray,
        *,
        steps: int,
        angle_limit: float,
        velocity_limit: float,
        cart_margin: float,
    ):
        self.dynamics = DoublePendulumCartDynamics(config.physics)
        self.gain = gain
        self.max_force = config.physics.max_force
        self.track_limit = config.physics.track_limit
        self.steps = steps
        self.angle_limit = angle_limit
        self.velocity_limit = velocity_limit
        self.cart_margin = cart_margin

    def is_viable(self, state: np.ndarray) -> bool:
        simulated = np.asarray(state, dtype=np.float64).copy()
        for _ in range(max(1, self.steps)):
            if not self._inside_limits(simulated):
                return False
            force = lqr_action(simulated, self.gain, self.max_force)
            simulated = self.dynamics.rk4_step(simulated, force)
        return self._inside_limits(simulated)

    def _inside_limits(self, state: np.ndarray) -> bool:
        x, _, theta1, theta1_dot, theta2, theta2_dot = np.asarray(state, dtype=np.float64)
        return bool(
            max(abs(theta1), abs(theta2)) <= self.angle_limit
            and max(abs(theta1_dot), abs(theta2_dot)) <= self.velocity_limit
            and self.track_limit - abs(x) >= self.cart_margin
        )


class MpcCaptureController:
    phase = "capture"

    def __init__(self, config: ProjectConfig, mpc_config: MpcConfig, viability_gate: LqrViabilityGate | None = None):
        self.physics = config.physics
        self.dynamics = DoublePendulumCartDynamics(config.physics)
        self.config = mpc_config
        self.viability_gate = viability_gate
        self.force_grid = np.clip(
            np.asarray(mpc_config.force_grid, dtype=np.float64),
            -config.physics.max_force,
            config.physics.max_force,
        )
        self._planned_actions: list[float] = []
        self._last_force = 0.0

    def reset(self) -> None:
        self._planned_actions.clear()
        self._last_force = 0.0

    def act(self, observation: np.ndarray, info: dict[str, Any], env: Any) -> np.ndarray:
        if not self._planned_actions:
            self._planned_actions = self.plan(info["state"], self._last_force)
            if self.config.replan_interval > 0:
                self._planned_actions = self._planned_actions[: self.config.replan_interval]
        force = float(self._planned_actions.pop(0))
        self._last_force = force
        return np.array([force], dtype=np.float32)

    def plan(self, state: np.ndarray, last_force: float = 0.0) -> list[float]:
        beams: list[tuple[float, np.ndarray, tuple[float, ...], float]] = [
            (0.0, np.asarray(state, dtype=np.float64).copy(), (), float(last_force))
        ]
        for depth in range(max(1, self.config.horizon)):
            candidates: list[tuple[float, np.ndarray, tuple[float, ...], float]] = []
            discount = self.config.discount**depth
            for cost, beam_state, actions, previous_force in beams:
                for force in self.force_grid:
                    next_state = self.dynamics.rk4_step(beam_state, float(force))
                    action_tuple = actions + (float(force),)
                    step_cost = self.cost(next_state, float(force), previous_force)
                    candidates.append((cost + discount * step_cost, next_state, action_tuple, float(force)))
            score = self.terminal_score if depth == max(1, self.config.horizon) - 1 else self.search_score
            candidates.sort(key=score)
            beams = candidates[: max(1, self.config.beam_width)]
        best = min(beams, key=self.terminal_score)
        return list(best[2]) or [0.0]

    def cost_after_constant_force(self, state: np.ndarray, force: float) -> float:
        simulated = np.asarray(state, dtype=np.float64).copy()
        total = 0.0
        previous_force = self._last_force
        for depth in range(max(1, self.config.horizon)):
            simulated = self.dynamics.rk4_step(simulated, force)
            total += (self.config.discount**depth) * self.cost(simulated, force, previous_force)
            previous_force = force
        return total + self.config.terminal_weight * self.state_cost(simulated) - self.basin_bonus(simulated)

    def cost_after_sequence(self, state: np.ndarray, forces: list[float]) -> float:
        simulated = np.asarray(state, dtype=np.float64).copy()
        total = 0.0
        previous_force = self._last_force
        for depth, force in enumerate(forces):
            simulated = self.dynamics.rk4_step(simulated, float(force))
            total += (self.config.discount**depth) * self.cost(simulated, float(force), previous_force)
            previous_force = float(force)
        return total + self.config.terminal_weight * self.state_cost(simulated) - self.basin_bonus(simulated)

    def search_score(self, item: tuple[float, np.ndarray, tuple[float, ...], float]) -> float:
        return item[0] + self.config.terminal_weight * self.state_cost(item[1])

    def terminal_score(self, item: tuple[float, np.ndarray, tuple[float, ...], float]) -> float:
        return self.search_score(item) - self.basin_bonus(item[1])

    def basin_bonus(self, state: np.ndarray) -> float:
        if self.config.terminal_basin_bonus <= 0.0 or self.viability_gate is None:
            return 0.0
        return self.config.terminal_basin_bonus if self.viability_gate.is_viable(state) else 0.0

    def cost(self, state: np.ndarray, force: float, previous_force: float) -> float:
        return (
            self.state_cost(state)
            + self.config.action_weight * (force / self.physics.max_force) ** 2
            + self.config.action_change_weight * ((force - previous_force) / self.physics.max_force) ** 2
        )

    def state_cost(self, state: np.ndarray) -> float:
        x, x_dot, theta1, theta1_dot, theta2, theta2_dot = np.asarray(state, dtype=np.float64)
        track_fraction = abs(x) / self.physics.track_limit
        boundary_cost = max(0.0, track_fraction - 0.75) ** 2
        return float(
            self.config.angle_weight * (theta1**2 + theta2**2)
            + self.config.angular_velocity_weight * (theta1_dot**2 + theta2_dot**2)
            + self.config.cart_position_weight * x**2
            + self.config.cart_velocity_weight * x_dot**2
            + self.config.boundary_weight * boundary_cost
        )


class HybridController:
    def __init__(
        self,
        swingup: PolicyController,
        capture: MpcCaptureController,
        stabilizer: LqrController,
        config: HybridConfig,
        viability_gate: LqrViabilityGate | None = None,
    ):
        self.swingup = swingup
        self.capture = capture
        self.stabilizer = stabilizer
        self.config = config
        self.viability_gate = viability_gate
        self.phase = "swingup"
        self._cooldown_remaining = 0
        self.capture_count = 0
        self.stabilize_count = 0
        self.fallback_count = 0
        self.handoff_count = 0
        self.capture_steps = 0
        self.stabilizer_steps = 0
        self.swingup_steps = 0
        self.safety_steps = 0

    def reset(self) -> None:
        self.phase = "swingup"
        self.capture_count = 0
        self.stabilize_count = 0
        self.fallback_count = 0
        self.handoff_count = 0
        self.capture_steps = 0
        self.stabilizer_steps = 0
        self.swingup_steps = 0
        self.safety_steps = 0
        self._cooldown_remaining = 0
        self.swingup.reset()
        self.capture.reset()
        self.stabilizer.reset()

    def act(self, observation: np.ndarray, info: dict[str, Any], env: Any) -> np.ndarray:
        self._update_phase(info["state"], env.physics_config.track_limit)
        if self.phase == "stabilize":
            self.stabilizer_steps += 1
            return self._apply_safety_filter(self.stabilizer.act(observation, info, env), info, env)
        if self.phase == "capture":
            self.capture_steps += 1
            return self._apply_safety_filter(self.capture.act(observation, info, env), info, env)
        self.swingup_steps += 1
        return self._apply_safety_filter(self.swingup.act(observation, info, env), info, env)

    def _apply_safety_filter(self, action: np.ndarray, info: dict[str, Any], env: Any) -> np.ndarray:
        if not self.config.safety_filter_enabled:
            return action
        x, x_dot, *_ = np.asarray(info["state"], dtype=np.float64)
        margin = env.physics_config.track_limit - abs(float(x))
        moving_outward = x * x_dot > 0.0
        if margin > self.config.safety_margin and not (
            margin < 2.0 * self.config.safety_margin and moving_outward
        ):
            return action
        force = -self.config.safety_position_gain * float(x) - self.config.safety_velocity_gain * float(x_dot)
        force = float(np.clip(force, -env.physics_config.max_force, env.physics_config.max_force))
        self.safety_steps += 1
        return np.array([force], dtype=np.float32)

    def _update_phase(self, state: np.ndarray, track_limit: float) -> None:
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
        if self.phase == "stabilize":
            if self._should_exit_stabilize(state, track_limit):
                self.phase = "capture" if self._in_capture_region(state, track_limit) else "swingup"
                self.fallback_count += 1
                self._cooldown_remaining = self.config.stabilizer_cooldown_steps
                self.capture.reset()
            return

        if self._in_stabilize_region(state, track_limit):
            self.phase = "stabilize"
            self.stabilize_count += 1
            self.handoff_count += 1
            self.capture.reset()
            return

        if self.phase == "capture":
            if self._should_exit_capture(state, track_limit):
                self.phase = "swingup"
                self.fallback_count += 1
                self.capture.reset()
            return

        if self._in_capture_region(state, track_limit):
            self.phase = "capture"
            self.capture_count += 1
            self.capture.reset()

    def _in_capture_region(self, state: np.ndarray, track_limit: float) -> bool:
        x, _, theta1, theta1_dot, theta2, theta2_dot = state
        return bool(
            max(abs(theta1), abs(theta2)) <= self.config.capture_enter_angle_radians
            and max(abs(theta1_dot), abs(theta2_dot)) <= self.config.capture_max_angular_velocity
            and track_limit - abs(x) >= self.config.capture_enter_cart_margin
        )

    def _should_exit_capture(self, state: np.ndarray, track_limit: float) -> bool:
        x, _, theta1, _, theta2, _ = state
        return bool(
            max(abs(theta1), abs(theta2)) >= self.config.capture_exit_angle_radians
            or track_limit - abs(x) <= self.config.capture_exit_cart_margin
        )

    def _in_stabilize_region(self, state: np.ndarray, track_limit: float) -> bool:
        if self._cooldown_remaining > 0:
            return False
        x, _, theta1, theta1_dot, theta2, theta2_dot = state
        inside_thresholds = bool(
            max(abs(theta1), abs(theta2)) <= self.config.stabilize_enter_angle_radians
            and max(abs(theta1_dot), abs(theta2_dot)) <= self.config.stabilize_enter_velocity_radians_per_second
            and track_limit - abs(x) >= self.config.stabilize_enter_cart_margin
        )
        if not inside_thresholds:
            return False
        return self.viability_gate.is_viable(state) if self.viability_gate is not None else True

    def _should_exit_stabilize(self, state: np.ndarray, track_limit: float) -> bool:
        x, _, theta1, theta1_dot, theta2, theta2_dot = state
        return bool(
            max(abs(theta1), abs(theta2)) >= self.config.stabilize_exit_angle_radians
            or max(abs(theta1_dot), abs(theta2_dot)) >= self.config.stabilize_exit_velocity_radians_per_second
            or track_limit - abs(x) <= self.config.capture_exit_cart_margin
        )


class PortfolioHybridController:
    def __init__(
        self,
        project_config: ProjectConfig,
        candidates: list[HybridController],
        candidate_names: list[str],
        max_selection_steps: int | None = None,
    ):
        if not candidates:
            raise ValueError("PortfolioHybridController requires at least one candidate.")
        self.project_config = project_config
        self.candidates = candidates
        self.candidate_names = candidate_names
        self.max_selection_steps = max_selection_steps or project_config.env.max_episode_steps
        self.active_index: int | None = None
        self.active: HybridController | None = None
        self.selected_name = ""
        self.phase = "portfolio"
        self.portfolio_evaluations = 0
        self._sync_counters()

    def reset(self) -> None:
        self.active_index = None
        self.active = None
        self.selected_name = ""
        self.phase = "portfolio"
        self.portfolio_evaluations = 0
        for candidate in self.candidates:
            candidate.reset()
        self._sync_counters()

    def act(self, observation: np.ndarray, info: dict[str, Any], env: Any) -> np.ndarray:
        if self.active is None:
            self._select_candidate(info["state"])
        assert self.active is not None
        action = self.active.act(observation, info, env)
        self.phase = self.active.phase
        self._sync_counters()
        return action

    def _select_candidate(self, initial_state: np.ndarray) -> None:
        scores = [self._score_candidate(candidate, initial_state) for candidate in self.candidates]
        self.active_index = max(range(len(scores)), key=lambda index: scores[index])
        self.active = self.candidates[self.active_index]
        self.selected_name = self.candidate_names[self.active_index]
        self.active.reset()
        self.portfolio_evaluations += len(self.candidates)

    def _score_candidate(self, candidate: HybridController, initial_state: np.ndarray) -> tuple[float, float, float]:
        from idp_rl.env import InvertedDoublePendulumEnv

        env = InvertedDoublePendulumEnv(self.project_config)
        obs, info = env.reset(options={"state": np.asarray(initial_state, dtype=np.float64)})
        candidate.reset()
        done = False
        length = 0
        max_hold = 0.0
        while not done and length < self.max_selection_steps:
            action = candidate.act(obs, info, env)
            obs, _, terminated, truncated, info = env.step(action)
            length += 1
            max_hold = max(max_hold, float(info["max_hold_steps"]))
            done = terminated or truncated
        env.close()
        success = float(max_hold >= self.project_config.env.success_hold_steps)
        return success, max_hold, -float(length)

    def _sync_counters(self) -> None:
        active = self.active
        for field in (
            "handoff_count",
            "capture_count",
            "stabilize_count",
            "fallback_count",
            "swingup_steps",
            "capture_steps",
            "stabilizer_steps",
            "safety_steps",
        ):
            setattr(self, field, 0 if active is None else getattr(active, field, 0))
