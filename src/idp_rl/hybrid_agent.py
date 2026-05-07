from __future__ import annotations

import argparse
from pathlib import Path

from idp_rl.config import load_config
from idp_rl.controllers import (
    HybridController,
    LqrController,
    LqrViabilityGate,
    MpcCaptureController,
    PolicyController,
    PortfolioHybridController,
    load_policy_model,
)
from idp_rl.hybrid_config import HybridConfig, load_hybrid_config
from idp_rl.rollout import evaluate_controller
from idp_rl.runtime import device_report, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RL swing-up with MPC capture and LQR stabilization.")
    parser.add_argument("--swingup-config", required=True)
    parser.add_argument("--swingup-checkpoint", default=None)
    parser.add_argument("--swingup-checkpoints", nargs="+", default=None)
    parser.add_argument("--config", default="configs/hybrid_reliable.yaml")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--report-episodes", action="store_true")
    parser.add_argument("--metrics-csv", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-animation", default=None)
    parser.add_argument("--render", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate_hybrid(
        swingup_config_path=Path(args.swingup_config),
        swingup_checkpoint=Path(args.swingup_checkpoint) if args.swingup_checkpoint else None,
        swingup_checkpoints=[Path(item) for item in args.swingup_checkpoints] if args.swingup_checkpoints else None,
        hybrid_config_path=Path(args.config),
        episodes=args.episodes,
        device=args.device,
        seed_offset=args.seed_offset,
        save_animation=args.save_animation,
        render=args.render,
        metrics_csv=args.metrics_csv,
    )
    print(format_hybrid_metrics(metrics))
    if args.report_episodes:
        print_episode_report(metrics)


def evaluate_hybrid(
    swingup_config_path: Path,
    swingup_checkpoint: Path | None,
    hybrid_config_path: Path,
    episodes: int,
    device: str,
    swingup_checkpoints: list[Path] | None = None,
    seed_offset: int = 0,
    save_animation: str | None = None,
    render: bool = False,
    metrics_csv: str | None = None,
) -> dict:
    resolved_device = resolve_device(device)
    print(device_report(resolved_device))
    project_config = load_config(swingup_config_path)
    hybrid_config = load_hybrid_config(hybrid_config_path)
    checkpoints = swingup_checkpoints or ([swingup_checkpoint] if swingup_checkpoint is not None else [])
    if not checkpoints:
        raise ValueError("Provide --swingup-checkpoint or --swingup-checkpoints.")
    controller = (
        build_portfolio_controller(project_config, checkpoints, hybrid_config, resolved_device)
        if len(checkpoints) > 1
        else build_hybrid_controller(project_config, checkpoints[0], hybrid_config, resolved_device)
    )
    return evaluate_controller(
        project_config,
        controller,
        episodes,
        render=render,
        save_animation=save_animation,
        seed_offset=seed_offset,
        metrics_csv=metrics_csv,
    )


def build_hybrid_controller(
    project_config,
    swingup_checkpoint: Path,
    hybrid_config: HybridConfig,
    device,
) -> HybridController:
    swingup_model = load_policy_model(project_config, swingup_checkpoint, device)
    stabilizer = LqrController(project_config, q_diag=hybrid_config.lqr_q_diag, r=hybrid_config.lqr_r)
    viability_gate = None
    if hybrid_config.lqr_viability_enabled:
        viability_gate = LqrViabilityGate(
            project_config,
            stabilizer.gain,
            steps=hybrid_config.lqr_viability_steps,
            angle_limit=hybrid_config.lqr_viability_angle_radians,
            velocity_limit=hybrid_config.lqr_viability_velocity_radians_per_second,
            cart_margin=hybrid_config.lqr_viability_cart_margin,
        )
    capture = MpcCaptureController(project_config, hybrid_config.mpc, viability_gate)
    return HybridController(
        swingup=PolicyController(swingup_model, device, phase="swingup"),
        capture=capture,
        stabilizer=stabilizer,
        config=hybrid_config,
        viability_gate=viability_gate,
    )


def build_portfolio_controller(
    project_config,
    swingup_checkpoints: list[Path],
    hybrid_config: HybridConfig,
    device,
) -> PortfolioHybridController:
    candidates = [
        build_hybrid_controller(project_config, checkpoint, hybrid_config, device)
        for checkpoint in swingup_checkpoints
    ]
    return PortfolioHybridController(
        project_config,
        candidates,
        [str(checkpoint) for checkpoint in swingup_checkpoints],
    )


def format_hybrid_metrics(metrics: dict) -> str:
    phase_counts = metrics.get("phase_counts", {})
    return " ".join(
        [
            f"mean_return={metrics['mean_return']:.3f}",
            f"success_rate={metrics['success_rate']:.3f}",
            f"mean_max_hold_steps={metrics['mean_max_hold_steps']:.1f}",
            f"max_hold_steps={metrics['max_hold_steps']:.1f}",
            f"mean_handoff_ready_steps={metrics['mean_handoff_ready_steps']:.1f}",
            f"handoff_count={metrics['handoff_count']}",
            f"capture_count={metrics['capture_count']}",
            f"stabilize_count={metrics['stabilize_count']}",
            f"fallback_count={metrics['fallback_count']}",
            f"portfolio_evaluations={metrics.get('portfolio_evaluations', 0.0)}",
            f"phase_counts={phase_counts}",
            f"mean_episode_length={metrics['mean_episode_length']:.1f}",
        ]
    )


def print_episode_report(metrics: dict) -> None:
    for episode in metrics.get("episodes", []):
        print(
            "episode "
            f"index={episode['episode']} "
            f"seed={episode['seed']} "
            f"success={episode['success']} "
            f"max_hold_steps={episode['max_hold_steps']:.0f} "
            f"length={episode['length']} "
            f"reason={episode['failure_reason']} "
            f"capture_count={episode['capture_count']:.0f} "
            f"stabilize_count={episode['stabilize_count']:.0f} "
            f"fallback_count={episode['fallback_count']:.0f} "
            f"selected_controller={episode.get('selected_controller', '')} "
            f"phase_counts={episode['phase_counts']} "
            f"final_state={episode['final_state']}"
        )


if __name__ == "__main__":
    main()
