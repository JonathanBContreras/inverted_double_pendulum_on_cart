from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - runtime dependency
    yaml = None


@dataclass(frozen=True)
class MpcConfig:
    horizon: int = 8
    beam_width: int = 24
    replan_interval: int = 2
    force_grid: list[float] = field(default_factory=lambda: [-20.0, -14.0, -8.0, -4.0, 0.0, 4.0, 8.0, 14.0, 20.0])
    angle_weight: float = 140.0
    angular_velocity_weight: float = 10.0
    cart_position_weight: float = 5.0
    cart_velocity_weight: float = 1.0
    action_weight: float = 0.02
    action_change_weight: float = 0.01
    boundary_weight: float = 80.0
    terminal_weight: float = 2.0
    terminal_basin_bonus: float = 0.0
    discount: float = 0.98


@dataclass(frozen=True)
class HybridConfig:
    capture_enter_angle_radians: float = 0.85
    capture_exit_angle_radians: float = 1.20
    capture_enter_cart_margin: float = 0.30
    capture_exit_cart_margin: float = 0.15
    capture_max_angular_velocity: float = 30.0
    stabilize_enter_angle_radians: float = 0.22
    stabilize_enter_velocity_radians_per_second: float = 2.5
    stabilize_exit_angle_radians: float = 0.50
    stabilize_exit_velocity_radians_per_second: float = 6.0
    stabilize_enter_cart_margin: float = 0.25
    lqr_viability_enabled: bool = False
    lqr_viability_steps: int = 80
    lqr_viability_angle_radians: float = 0.45
    lqr_viability_velocity_radians_per_second: float = 7.0
    lqr_viability_cart_margin: float = 0.15
    stabilizer_cooldown_steps: int = 0
    safety_filter_enabled: bool = False
    safety_margin: float = 0.30
    safety_position_gain: float = 24.0
    safety_velocity_gain: float = 10.0
    lqr_q_diag: list[float] = field(default_factory=lambda: [2.0, 1.0, 180.0, 18.0, 180.0, 18.0])
    lqr_r: float = 0.08
    mpc: MpcConfig = field(default_factory=MpcConfig)


def load_hybrid_config(path: str | Path | None) -> HybridConfig:
    if path is None:
        return HybridConfig()
    if yaml is None:
        raise RuntimeError("PyYAML is required to load hybrid YAML config files.")

    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return _build_hybrid_config(data)


def hybrid_config_to_dict(config: HybridConfig) -> dict[str, Any]:
    return {
        key: value
        for key, value in config.__dict__.items()
        if key != "mpc"
    } | {"mpc": dict(config.mpc.__dict__)}


def save_hybrid_config(config: HybridConfig, path: str | Path) -> None:
    if yaml is None:
        raise RuntimeError("PyYAML is required to save hybrid YAML config files.")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(hybrid_config_to_dict(config), handle, sort_keys=False)


def _build_hybrid_config(data: dict[str, Any]) -> HybridConfig:
    allowed = set(HybridConfig.__dataclass_fields__) - {"mpc"}
    unknown = sorted(set(data) - allowed - {"mpc"})
    if unknown:
        raise ValueError(f"Unknown HybridConfig keys: {', '.join(unknown)}")

    mpc_data = data.get("mpc") or {}
    mpc_allowed = set(MpcConfig.__dataclass_fields__)
    mpc_unknown = sorted(set(mpc_data) - mpc_allowed)
    if mpc_unknown:
        raise ValueError(f"Unknown MpcConfig keys: {', '.join(mpc_unknown)}")

    values = {key: value for key, value in data.items() if key != "mpc"}
    return HybridConfig(**values, mpc=MpcConfig(**mpc_data))
