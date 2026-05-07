from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from idp_rl.config import load_config
from idp_rl.controllers import load_policy_model
from idp_rl.ppo import ActorCritic
from idp_rl.rollout import evaluate_controller
from idp_rl.runtime import device_report, resolve_device


@dataclass
class HandoffConfig:
    enter_angle_radians: float = 0.25
    enter_velocity_radians_per_second: float = 3.5
    enter_cart_margin: float = 0.35
    exit_angle_radians: float = 0.55
    exit_cart_margin: float = 0.2


class TwoStageAgent:
    def __init__(
        self,
        swingup_model: ActorCritic,
        stabilizer_model: ActorCritic,
        handoff_config: HandoffConfig | None = None,
        device: torch.device | None = None,
    ):
        self.swingup_model = swingup_model
        self.stabilizer_model = stabilizer_model
        self.handoff_config = handoff_config or HandoffConfig()
        self.device = device or next(swingup_model.parameters()).device
        self.phase = "swingup"
        self.handoff_count = 0
        self.stabilizer_steps = 0

    def reset(self) -> None:
        self.phase = "swingup"
        self.handoff_count = 0
        self.stabilizer_steps = 0

    def select_model(self, state: np.ndarray, track_limit: float) -> ActorCritic:
        if self.phase == "swingup" and should_enter_stabilizer(state, track_limit, self.handoff_config):
            self.phase = "stabilizer"
            self.handoff_count += 1
        elif self.phase == "stabilizer" and should_exit_stabilizer(state, track_limit, self.handoff_config):
            self.phase = "swingup"
        if self.phase == "stabilizer":
            self.stabilizer_steps += 1
            return self.stabilizer_model
        return self.swingup_model

    def act(self, observation: np.ndarray, info: dict[str, Any], env: Any) -> np.ndarray:
        model = self.select_model(info["state"], env.physics_config.track_limit)
        obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return model.actor_mean(obs_tensor).cpu().numpy()[0].astype(np.float32)


def should_enter_stabilizer(state: np.ndarray, track_limit: float, config: HandoffConfig | None = None) -> bool:
    config = config or HandoffConfig()
    x, _, theta1, theta1_dot, theta2, theta2_dot = state
    return bool(
        max(abs(theta1), abs(theta2)) <= config.enter_angle_radians
        and max(abs(theta1_dot), abs(theta2_dot)) <= config.enter_velocity_radians_per_second
        and track_limit - abs(x) >= config.enter_cart_margin
    )


def should_exit_stabilizer(state: np.ndarray, track_limit: float, config: HandoffConfig | None = None) -> bool:
    config = config or HandoffConfig()
    x, _, theta1, _, theta2, _ = state
    return bool(
        max(abs(theta1), abs(theta2)) >= config.exit_angle_radians
        or track_limit - abs(x) <= config.exit_cart_margin
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a two-stage swing-up plus stabilizer agent.")
    parser.add_argument("--swingup-config", required=True)
    parser.add_argument("--swingup-checkpoint", required=True)
    parser.add_argument("--stabilizer-config", required=True)
    parser.add_argument("--stabilizer-checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-animation", default=None)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--enter-angle-radians", type=float, default=HandoffConfig.enter_angle_radians)
    parser.add_argument(
        "--enter-velocity-radians-per-second",
        type=float,
        default=HandoffConfig.enter_velocity_radians_per_second,
    )
    parser.add_argument("--enter-cart-margin", type=float, default=HandoffConfig.enter_cart_margin)
    parser.add_argument("--exit-angle-radians", type=float, default=HandoffConfig.exit_angle_radians)
    parser.add_argument("--exit-cart-margin", type=float, default=HandoffConfig.exit_cart_margin)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate_two_stage(
        swingup_config_path=Path(args.swingup_config),
        swingup_checkpoint=Path(args.swingup_checkpoint),
        stabilizer_config_path=Path(args.stabilizer_config),
        stabilizer_checkpoint=Path(args.stabilizer_checkpoint),
        episodes=args.episodes,
        device=args.device,
        save_animation=args.save_animation,
        render=args.render,
        handoff_config=HandoffConfig(
            enter_angle_radians=args.enter_angle_radians,
            enter_velocity_radians_per_second=args.enter_velocity_radians_per_second,
            enter_cart_margin=args.enter_cart_margin,
            exit_angle_radians=args.exit_angle_radians,
            exit_cart_margin=args.exit_cart_margin,
        ),
    )
    print(
        " ".join(
            [
                f"mean_return={metrics['mean_return']:.3f}",
                f"success_rate={metrics['success_rate']:.3f}",
                f"mean_max_hold_steps={metrics['mean_max_hold_steps']:.1f}",
                f"max_hold_steps={metrics['max_hold_steps']:.1f}",
                f"mean_handoff_ready_steps={metrics['mean_handoff_ready_steps']:.1f}",
                f"handoff_count={metrics['handoff_count']}",
                f"stabilizer_active_steps={metrics['stabilizer_active_steps']}",
                f"mean_episode_length={metrics['mean_episode_length']:.1f}",
            ]
        )
    )


def load_policy(config, checkpoint: Path, device: torch.device) -> ActorCritic:
    return load_policy_model(config, checkpoint, device)


def evaluate_two_stage(
    swingup_config_path: Path,
    swingup_checkpoint: Path,
    stabilizer_config_path: Path,
    stabilizer_checkpoint: Path,
    episodes: int,
    device: str,
    save_animation: str | None = None,
    render: bool = False,
    handoff_config: HandoffConfig | None = None,
) -> dict[str, float]:
    resolved_device = resolve_device(device)
    print(device_report(resolved_device))
    swingup_config = load_config(swingup_config_path)
    stabilizer_config = load_config(stabilizer_config_path)
    agent = TwoStageAgent(
        load_policy(swingup_config, swingup_checkpoint, resolved_device),
        load_policy(stabilizer_config, stabilizer_checkpoint, resolved_device),
        handoff_config,
        resolved_device,
    )
    metrics = evaluate_controller(
        swingup_config,
        agent,
        episodes,
        render=render,
        save_animation=save_animation,
        seed_offset=0,
    )
    metrics["stabilizer_active_steps"] = metrics["stabilizer_steps"]
    return metrics


if __name__ == "__main__":
    main()
