from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - dependency is declared for runtime use
    yaml = None


@dataclass(frozen=True)
class PhysicsConfig:
    gravity: float = 9.80665
    cart_mass: float = 1.0
    link1_mass: float = 0.1
    link2_mass: float = 0.1
    link1_length: float = 0.5
    link2_length: float = 0.5
    track_limit: float = 2.4
    max_force: float = 20.0
    dt: float = 0.02
    initial_angle_range: float = 0.08
    initial_angular_velocity_range: float = 0.1
    initial_cart_position_range: float = 0.05
    initial_cart_velocity_range: float = 0.05


@dataclass(frozen=True)
class EnvConfig:
    reset_mode: str = "upright"
    max_episode_steps: int = 1000
    terminate_on_angle: bool = True
    angle_termination_radians: float = 0.7853981633974483
    success_angle_radians: float = 0.20943951023931953
    success_velocity_radians_per_second: float = 1.0
    success_hold_steps: int = 100
    position_reward_weight: float = 0.01
    velocity_reward_weight: float = 0.001
    action_reward_weight: float = 0.0005
    boundary_reward_weight: float = 0.2
    hold_reward: float = 0.05
    upright_reward: float = 2.0
    success_bonus: float = 5.0
    failure_penalty: float = 25.0
    handoff_angle_radians: float = 0.35
    handoff_velocity_radians_per_second: float = 3.5
    handoff_cart_margin: float = 0.35
    handoff_bonus: float = 0.0
    handoff_velocity_penalty_weight: float = 0.0
    handoff_angle_band_radians: float = 0.45


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 7
    total_steps: int = 200000
    rollout_steps: int = 2048
    num_envs: int = 1
    update_epochs: int = 10
    minibatch_size: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    entropy_coef: float = 0.0
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.03
    normalize_advantages: bool = True
    eval_every_updates: int = 5
    eval_episodes: int = 5
    success_threshold: float = 0.2
    learning_rate: float = 3e-4
    hidden_sizes: list[int] = field(default_factory=lambda: [128, 128])
    initial_log_std: float = 0.0
    reset_states_path: str | None = None
    save_every_updates: int = 10
    curriculum_stages: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class ProjectConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _build_dataclass(cls: type, values: dict[str, Any] | None):
    values = values or {}
    allowed = {field.name for field in cls.__dataclass_fields__.values()}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {', '.join(unknown)}")
    return cls(**values)


def load_config(path: str | Path | None = None) -> ProjectConfig:
    if path is None:
        return ProjectConfig()
    if yaml is None:
        raise RuntimeError("PyYAML is required to load YAML config files.")

    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    return ProjectConfig(
        env=_build_dataclass(EnvConfig, data.get("env")),
        physics=_build_dataclass(PhysicsConfig, data.get("physics")),
        training=_build_dataclass(TrainingConfig, data.get("training")),
    )


def save_config(config: ProjectConfig, path: str | Path) -> None:
    if yaml is None:
        raise RuntimeError("PyYAML is required to save YAML config files.")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False)
