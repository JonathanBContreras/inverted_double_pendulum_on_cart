from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from idp_rl.config import load_config
from idp_rl.controllers import (
    HybridController,
    LqrController,
    LqrViabilityGate,
    MpcCaptureController,
    PolicyController,
    load_policy_model,
)
from idp_rl.hybrid_config import HybridConfig, MpcConfig, load_hybrid_config, save_hybrid_config
from idp_rl.rollout import evaluate_controller
from idp_rl.runtime import device_report, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deterministically tune hybrid controller parameters.")
    parser.add_argument("--swingup-config", required=True)
    parser.add_argument("--swingup-checkpoint", required=True)
    parser.add_argument("--base-config", default="configs/hybrid_reliable.yaml")
    parser.add_argument("--output-config", default="configs/hybrid_10of10.yaml")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-candidates", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = tune_hybrid(
        swingup_config_path=Path(args.swingup_config),
        swingup_checkpoint=Path(args.swingup_checkpoint),
        base_config_path=Path(args.base_config),
        output_config_path=Path(args.output_config),
        episodes=args.episodes,
        seed_offset=args.seed_offset,
        device=args.device,
        max_candidates=args.max_candidates,
    )
    print(
        " ".join(
            [
                f"best_success_rate={result['metrics']['success_rate']:.3f}",
                f"best_max_hold_steps={result['metrics']['max_hold_steps']:.1f}",
                f"best_mean_max_hold_steps={result['metrics']['mean_max_hold_steps']:.1f}",
                f"best_fallback_count={result['metrics']['fallback_count']:.0f}",
                f"output_config={result['output_config']}",
            ]
        )
    )


def tune_hybrid(
    swingup_config_path: Path,
    swingup_checkpoint: Path,
    base_config_path: Path,
    output_config_path: Path,
    episodes: int,
    seed_offset: int,
    device: str,
    max_candidates: int | None = None,
) -> dict:
    resolved_device = resolve_device(device)
    print(device_report(resolved_device))
    project_config = load_config(swingup_config_path)
    base_config = load_hybrid_config(base_config_path)
    model = load_policy_model(project_config, swingup_checkpoint, resolved_device)

    candidates = list(candidate_configs(base_config))
    if max_candidates is not None:
        candidates = candidates[:max_candidates]

    best_config = candidates[0]
    best_metrics = evaluate_candidate(project_config, model, best_config, episodes, seed_offset, resolved_device)
    print(format_candidate_result(0, best_metrics))
    if best_metrics["success_rate"] >= 1.0:
        save_hybrid_config(best_config, output_config_path)
        return {"config": best_config, "metrics": best_metrics, "output_config": str(output_config_path)}

    for index, candidate in enumerate(candidates[1:], start=1):
        metrics = evaluate_candidate(project_config, model, candidate, episodes, seed_offset, resolved_device)
        print(format_candidate_result(index, metrics))
        if candidate_score(metrics) > candidate_score(best_metrics):
            best_config = candidate
            best_metrics = metrics
        if metrics["success_rate"] >= 1.0:
            best_config = candidate
            best_metrics = metrics
            break

    save_hybrid_config(best_config, output_config_path)
    return {"config": best_config, "metrics": best_metrics, "output_config": str(output_config_path)}


def candidate_configs(base: HybridConfig):
    seen = {repr(base), repr(enable_viability(base))}
    yield base
    yield enable_viability(base)
    threshold_candidates = [
        {},
        {"stabilize_enter_angle_radians": 0.35, "stabilize_enter_velocity_radians_per_second": 5.0},
        {"stabilize_enter_angle_radians": 0.40, "stabilize_enter_velocity_radians_per_second": 5.0},
        {"stabilize_enter_angle_radians": 0.45, "stabilize_enter_velocity_radians_per_second": 6.0},
        {"capture_enter_angle_radians": 1.05, "capture_exit_angle_radians": 1.50},
        {"capture_enter_angle_radians": 1.15, "capture_exit_angle_radians": 1.60},
        {"stabilizer_cooldown_steps": 8},
        {"stabilizer_cooldown_steps": 16},
        {
            "stabilize_enter_angle_radians": 0.38,
            "stabilize_enter_velocity_radians_per_second": 5.5,
            "stabilizer_cooldown_steps": 8,
        },
        {
            "capture_enter_angle_radians": 1.10,
            "capture_exit_angle_radians": 1.55,
            "stabilize_enter_angle_radians": 0.38,
            "stabilize_enter_velocity_radians_per_second": 5.5,
        },
    ]
    mpc_candidates = [
        {},
        {"terminal_basin_bonus": 150.0},
        {"terminal_basin_bonus": 300.0},
        {"horizon": 10, "beam_width": 24, "terminal_basin_bonus": 250.0},
        {"horizon": 8, "beam_width": 32, "angular_velocity_weight": 14.0, "terminal_basin_bonus": 250.0},
        {
            "force_grid": [-20.0, -16.0, -12.0, -8.0, -4.0, 0.0, 4.0, 8.0, 12.0, 16.0, 20.0],
            "terminal_basin_bonus": 250.0,
        },
    ]
    for threshold_updates in threshold_candidates:
        threshold_config = replace(base, **threshold_updates)
        for mpc_updates in mpc_candidates:
            candidate = replace(threshold_config, mpc=replace(threshold_config.mpc, **mpc_updates))
            key = repr(candidate)
            if key in seen:
                continue
            seen.add(key)
            yield candidate
        viable_threshold_config = enable_viability(threshold_config)
        for mpc_updates in mpc_candidates[1:3]:
            candidate = replace(viable_threshold_config, mpc=replace(viable_threshold_config.mpc, **mpc_updates))
            key = repr(candidate)
            if key in seen:
                continue
            seen.add(key)
            yield candidate


def enable_viability(config: HybridConfig) -> HybridConfig:
    return replace(
        config,
        lqr_viability_enabled=True,
        lqr_viability_steps=max(config.lqr_viability_steps, 60),
        lqr_viability_angle_radians=max(config.lqr_viability_angle_radians, config.stabilize_exit_angle_radians),
        lqr_viability_velocity_radians_per_second=max(
            config.lqr_viability_velocity_radians_per_second,
            config.stabilize_exit_velocity_radians_per_second,
        ),
        lqr_viability_cart_margin=min(config.lqr_viability_cart_margin, config.capture_exit_cart_margin),
        stabilizer_cooldown_steps=max(config.stabilizer_cooldown_steps, 6),
        mpc=replace(config.mpc, terminal_basin_bonus=max(config.mpc.terminal_basin_bonus, 100.0)),
    )


def evaluate_candidate(
    project_config,
    model,
    hybrid_config: HybridConfig,
    episodes: int,
    seed_offset: int,
    device: torch.device,
) -> dict:
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
    controller = HybridController(
        swingup=PolicyController(model, device, phase="swingup"),
        capture=MpcCaptureController(project_config, hybrid_config.mpc, viability_gate),
        stabilizer=stabilizer,
        config=hybrid_config,
        viability_gate=viability_gate,
    )
    return evaluate_controller(project_config, controller, episodes, seed_offset=seed_offset)


def candidate_score(metrics: dict) -> tuple[float, float, float, float]:
    return (
        float(metrics["success_rate"]),
        float(metrics["max_hold_steps"]),
        float(metrics["mean_max_hold_steps"]),
        -float(metrics["fallback_count"]),
    )


def format_candidate_result(index: int, metrics: dict) -> str:
    return (
        f"candidate={index} "
        f"success_rate={metrics['success_rate']:.3f} "
        f"max_hold_steps={metrics['max_hold_steps']:.1f} "
        f"mean_max_hold_steps={metrics['mean_max_hold_steps']:.1f} "
        f"fallback_count={metrics['fallback_count']:.0f}"
    )


if __name__ == "__main__":
    main()
