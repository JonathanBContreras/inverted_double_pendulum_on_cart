from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from idp_rl.config import ProjectConfig
from idp_rl.controllers import Controller
from idp_rl.env import InvertedDoublePendulumEnv


COUNTER_FIELDS = (
    "handoff_count",
    "capture_count",
    "stabilize_count",
    "fallback_count",
    "swingup_steps",
    "capture_steps",
    "stabilizer_steps",
    "safety_steps",
    "portfolio_evaluations",
)


def evaluate_controller(
    config: ProjectConfig,
    controller: Controller,
    episodes: int,
    *,
    render: bool = False,
    save_animation: str | Path | None = None,
    seed_offset: int = 0,
    metrics_csv: str | Path | None = None,
) -> dict[str, Any]:
    env = InvertedDoublePendulumEnv(
        config,
        render_mode="rgb_array" if save_animation else ("human" if render else None),
    )
    try:
        return run_episodes(
            env,
            controller,
            episodes,
            config.training.seed + seed_offset,
            render,
            save_animation,
            metrics_csv,
        )
    finally:
        env.close()


def run_episodes(
    env: InvertedDoublePendulumEnv,
    controller: Controller,
    episodes: int,
    seed_start: int,
    render: bool = False,
    save_animation: str | Path | None = None,
    metrics_csv: str | Path | None = None,
) -> dict[str, Any]:
    returns: list[float] = []
    lengths: list[int] = []
    max_holds: list[float] = []
    max_handoff_ready_steps: list[float] = []
    successes: list[float] = []
    frames: list[np.ndarray] = []
    phase_counts: Counter[str] = Counter()
    counters: Counter[str] = Counter()
    episode_metrics: list[dict[str, Any]] = []

    for episode in range(episodes):
        seed = seed_start + episode
        obs, info = env.reset(seed=seed)
        controller.reset()
        done = False
        episode_return = 0.0
        episode_length = 0
        episode_phase_counts: Counter[str] = Counter()
        terminated = False
        truncated = False

        while not done:
            action = controller.act(obs, info, env)
            obs, reward, terminated, truncated, info = env.step(action)
            phase = str(getattr(controller, "phase", "policy"))
            phase_counts[phase] += 1
            episode_phase_counts[phase] += 1
            episode_return += reward
            episode_length += 1
            done = terminated or truncated

            if save_animation and episode == 0:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)
            elif render:
                env.render()

        returns.append(float(episode_return))
        lengths.append(episode_length)
        episode_max_hold = float(info["max_hold_steps"])
        episode_success = episode_max_hold >= env.env_config.success_hold_steps
        max_holds.append(episode_max_hold)
        max_handoff_ready_steps.append(float(info["max_handoff_ready_steps"]))
        successes.append(float(episode_success))
        for field in COUNTER_FIELDS:
            counters[field] += int(getattr(controller, field, 0))
        episode_metrics.append(
            {
                "episode": episode,
                "seed": seed,
                "return": float(episode_return),
                "length": episode_length,
                "success": bool(episode_success),
                "final_success": bool(info["is_success"]),
                "max_hold_steps": episode_max_hold,
                "max_handoff_ready_steps": float(info["max_handoff_ready_steps"]),
                "final_state": np.asarray(info["state"], dtype=np.float64).tolist(),
                "phase_counts": dict(episode_phase_counts),
                "failure_reason": failure_reason(env, info, episode_success, terminated, truncated),
                "selected_controller": str(getattr(controller, "selected_name", "")),
                **{field: float(getattr(controller, field, 0)) for field in COUNTER_FIELDS},
            }
        )

    if save_animation:
        save_gif(frames, Path(save_animation), fps=env.metadata["render_fps"])
    if metrics_csv:
        save_episode_metrics_csv(episode_metrics, Path(metrics_csv))

    metrics: dict[str, Any] = {
        "returns": returns,
        "mean_return": float(np.mean(returns)) if returns else 0.0,
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "mean_episode_length": float(np.mean(lengths)) if lengths else 0.0,
        "mean_max_hold_steps": float(np.mean(max_holds)) if max_holds else 0.0,
        "max_hold_steps": float(np.max(max_holds)) if max_holds else 0.0,
        "mean_handoff_ready_steps": float(np.mean(max_handoff_ready_steps)) if max_handoff_ready_steps else 0.0,
        "phase_counts": dict(phase_counts),
        "episodes": episode_metrics,
    }
    metrics.update({field: float(counters[field]) for field in COUNTER_FIELDS})
    for phase, count in phase_counts.items():
        metrics[f"phase_{phase}_steps"] = float(count)
    return metrics


def failure_reason(
    env: InvertedDoublePendulumEnv,
    info: dict[str, Any],
    episode_success: bool,
    terminated: bool,
    truncated: bool,
) -> str:
    if episode_success:
        return "success"
    state = np.asarray(info["state"], dtype=np.float64)
    if abs(float(state[0])) > env.physics_config.track_limit:
        return "cart_bounds"
    if (
        terminated
        and env.env_config.terminate_on_angle
        and max(abs(float(state[2])), abs(float(state[4]))) > env.env_config.angle_termination_radians
    ):
        return "angle_bounds"
    if truncated:
        return "max_steps"
    if terminated:
        return "terminated"
    return "unknown"


def save_episode_metrics_csv(episodes: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "episode",
        "seed",
        "success",
        "final_success",
        "max_hold_steps",
        "max_handoff_ready_steps",
        "length",
        "return",
        "failure_reason",
        "handoff_count",
        "capture_count",
        "stabilize_count",
        "fallback_count",
        "swingup_steps",
        "capture_steps",
        "stabilizer_steps",
        "safety_steps",
        "portfolio_evaluations",
        "selected_controller",
        "phase_counts",
        "final_state",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for episode in episodes:
            row = dict(episode)
            row["phase_counts"] = str(row["phase_counts"])
            row["final_state"] = str(row["final_state"])
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def save_gif(frames: list[np.ndarray], path: Path, fps: int) -> None:
    if not frames:
        raise RuntimeError("No animation frames were captured.")
    path.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.fromarray(frame) for frame in frames]
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=int(1000 / fps),
        loop=0,
    )
