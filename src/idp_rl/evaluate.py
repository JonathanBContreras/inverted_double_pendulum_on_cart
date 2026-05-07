from __future__ import annotations

import argparse
from pathlib import Path

from idp_rl.config import load_config
from idp_rl.controllers import PolicyController, load_policy_model
from idp_rl.rollout import evaluate_controller
from idp_rl.runtime import device_report, resolve_checkpoint, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained inverted double pendulum policy.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--save-animation", default=None)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    checkpoint = resolve_checkpoint(args.checkpoint, args.run_dir)
    metrics = evaluate(config, checkpoint, args.episodes, args.render, args.device, args.save_animation)
    print(
        " ".join(
            [
                f"mean_return={metrics['mean_return']:.3f}",
                f"success_rate={metrics['success_rate']:.3f}",
                f"mean_max_hold_steps={metrics['mean_max_hold_steps']:.1f}",
                f"mean_handoff_ready_steps={metrics['mean_handoff_ready_steps']:.1f}",
                f"mean_episode_length={metrics['mean_episode_length']:.1f}",
                f"returns={metrics['returns']}",
            ]
        )
    )


def evaluate(
    config,
    checkpoint: Path,
    episodes: int,
    render: bool,
    device: str,
    save_animation: str | None = None,
) -> dict:
    resolved_device = resolve_device(device)
    print(device_report(resolved_device))
    model = load_policy_model(config, checkpoint, resolved_device)
    controller = PolicyController(model, resolved_device)
    return evaluate_controller(
        config,
        controller,
        episodes,
        render=render,
        save_animation=save_animation,
        seed_offset=0,
    )


if __name__ == "__main__":
    main()
