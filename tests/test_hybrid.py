import importlib.util
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
import torch

from idp_rl.config import EnvConfig, PhysicsConfig, ProjectConfig, TrainingConfig, save_config
from idp_rl.controllers import (
    HybridController,
    LqrController,
    LqrViabilityGate,
    MpcCaptureController,
    PolicyController,
    state_to_observation,
)
from idp_rl.hybrid_config import HybridConfig, MpcConfig, load_hybrid_config
from idp_rl.ppo import ActorCritic

gymnasium_available = importlib.util.find_spec("gymnasium") is not None
scipy_available = importlib.util.find_spec("scipy") is not None


def test_mpc_action_is_finite_and_force_limited():
    config = ProjectConfig()
    mpc = MpcCaptureController(config, MpcConfig(horizon=2, beam_width=9))
    info = {"state": np.array([0.0, 0.0, 0.2, 5.0, -0.2, -5.0])}
    action = mpc.act(state_to_observation(info["state"]), info, _EnvStub(config.physics))
    assert action.shape == (1,)
    assert np.isfinite(action).all()
    assert abs(float(action[0])) <= config.physics.max_force


def test_mpc_plan_cost_is_no_worse_than_zero_force():
    config = ProjectConfig()
    mpc_config = MpcConfig(horizon=4, beam_width=81, force_grid=[-20.0, 0.0, 20.0])
    mpc = MpcCaptureController(config, mpc_config)
    state = np.array([0.0, 0.0, 0.2, 5.0, -0.2, -5.0])
    plan = mpc.plan(state)
    assert mpc.cost_after_sequence(state, plan) <= mpc.cost_after_constant_force(state, 0.0)


@pytest.mark.skipif(not scipy_available, reason="SciPy is required")
def test_lqr_viability_gate_accepts_easy_and_rejects_fast_states():
    config = ProjectConfig()
    stabilizer = LqrController(config)
    gate = LqrViabilityGate(
        config,
        stabilizer.gain,
        steps=5,
        angle_limit=0.6,
        velocity_limit=10.0,
        cart_margin=0.1,
    )
    assert gate.is_viable(np.zeros(6))
    assert not gate.is_viable(np.array([0.0, 0.0, 0.1, 20.0, -0.1, -20.0]))


@pytest.mark.skipif(not scipy_available, reason="SciPy is required")
def test_mpc_terminal_basin_bonus_prefers_viable_state():
    config = ProjectConfig()
    stabilizer = LqrController(config)
    gate = LqrViabilityGate(config, stabilizer.gain, steps=3, angle_limit=0.6, velocity_limit=10.0, cart_margin=0.1)
    mpc = MpcCaptureController(config, MpcConfig(terminal_basin_bonus=123.0), gate)
    assert mpc.basin_bonus(np.zeros(6)) == 123.0
    assert mpc.basin_bonus(np.array([0.0, 0.0, 0.1, 20.0, -0.1, -20.0])) == 0.0


@pytest.mark.skipif(not (gymnasium_available and scipy_available), reason="Gymnasium and SciPy are required")
def test_hybrid_phase_transitions_are_deterministic():
    from idp_rl.env import InvertedDoublePendulumEnv

    config = ProjectConfig()
    model = ActorCritic(8, 1, [8])
    controller = HybridController(
        swingup=PolicyController(model, torch.device("cpu"), phase="swingup"),
        capture=MpcCaptureController(config, MpcConfig(horizon=1, beam_width=3, force_grid=[0.0])),
        stabilizer=LqrController(config),
        config=HybridConfig(capture_enter_angle_radians=0.9, stabilize_enter_angle_radians=0.25),
    )
    env = InvertedDoublePendulumEnv(config)
    env.reset(options={"state": np.array([0.0, 0.0, 0.6, 8.0, -0.5, -7.0])})
    controller.act(state_to_observation(env.state), {"state": env.state.copy()}, env)
    assert controller.phase == "capture"
    env.reset(options={"state": np.array([0.0, 0.0, 0.1, 0.5, -0.1, -0.5])})
    controller.act(state_to_observation(env.state), {"state": env.state.copy()}, env)
    assert controller.phase == "stabilize"
    env.close()


@pytest.mark.skipif(not (gymnasium_available and scipy_available), reason="Gymnasium and SciPy are required")
def test_hybrid_respects_stabilizer_cooldown():
    from idp_rl.env import InvertedDoublePendulumEnv

    config = ProjectConfig()
    model = ActorCritic(8, 1, [8])
    controller = HybridController(
        swingup=PolicyController(model, torch.device("cpu"), phase="swingup"),
        capture=MpcCaptureController(config, MpcConfig(horizon=1, beam_width=3, force_grid=[0.0])),
        stabilizer=LqrController(config),
        config=HybridConfig(
            capture_enter_angle_radians=0.9,
            stabilize_enter_angle_radians=0.5,
            stabilize_enter_velocity_radians_per_second=10.0,
            stabilize_exit_angle_radians=0.2,
            stabilizer_cooldown_steps=2,
        ),
    )
    env = InvertedDoublePendulumEnv(config)
    controller.phase = "stabilize"
    env.reset(options={"state": np.array([0.0, 0.0, 0.3, 0.0, 0.1, 0.0])})
    controller.act(state_to_observation(env.state), {"state": env.state.copy()}, env)
    env.reset(options={"state": np.zeros(6)})
    controller.act(state_to_observation(env.state), {"state": env.state.copy()}, env)
    assert controller.phase != "stabilize"
    env.close()


@pytest.mark.skipif(not (gymnasium_available and scipy_available), reason="Gymnasium and SciPy are required")
def test_hybrid_evaluator_writes_metrics_and_gif():
    from idp_rl.hybrid_agent import evaluate_hybrid
    from idp_rl.train import save_checkpoint

    project_config = ProjectConfig(
        env=EnvConfig(max_episode_steps=5, reset_mode="hanging", terminate_on_angle=False),
        physics=PhysicsConfig(),
        training=TrainingConfig(hidden_sizes=[8], seed=3),
    )
    run_dir = Path("test_runs") / f"hybrid_smoke_{uuid4().hex}"
    run_dir.mkdir(parents=True)
    project_config_path = run_dir / "project.yaml"
    hybrid_config_path = run_dir / "hybrid.yaml"
    checkpoint_path = run_dir / "latest.pt"
    animation_path = run_dir / "hybrid.gif"
    save_config(project_config, project_config_path)
    hybrid_config_path.write_text(
        "\n".join(
            [
                "capture_enter_angle_radians: 3.2",
                "mpc:",
                "  horizon: 1",
                "  beam_width: 3",
                "  replan_interval: 1",
                "  force_grid: [-20.0, 0.0, 20.0]",
            ]
        ),
        encoding="utf-8",
    )
    model = ActorCritic(8, 1, [8])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    save_checkpoint(model, optimizer, checkpoint_path, project_config, 0)

    metrics = evaluate_hybrid(
        swingup_config_path=project_config_path,
        swingup_checkpoint=checkpoint_path,
        hybrid_config_path=hybrid_config_path,
        episodes=1,
        device="cpu",
        save_animation=str(animation_path),
    )
    assert "phase_counts" in metrics
    assert "capture_count" in metrics
    assert metrics["episodes"][0]["seed"] == project_config.training.seed
    assert "failure_reason" in metrics["episodes"][0]
    assert animation_path.exists()
    assert animation_path.stat().st_size > 0

    portfolio_metrics = evaluate_hybrid(
        swingup_config_path=project_config_path,
        swingup_checkpoint=None,
        swingup_checkpoints=[checkpoint_path, checkpoint_path],
        hybrid_config_path=hybrid_config_path,
        episodes=1,
        device="cpu",
    )
    assert portfolio_metrics["portfolio_evaluations"] == 2.0
    assert portfolio_metrics["episodes"][0]["selected_controller"]


@pytest.mark.skipif(not (gymnasium_available and scipy_available), reason="Gymnasium and SciPy are required")
def test_tuner_writes_valid_output_config():
    from idp_rl.train import save_checkpoint
    from idp_rl.tune_hybrid import tune_hybrid

    project_config = ProjectConfig(
        env=EnvConfig(max_episode_steps=3, reset_mode="hanging", terminate_on_angle=False),
        physics=PhysicsConfig(),
        training=TrainingConfig(hidden_sizes=[8], seed=4),
    )
    run_dir = Path("test_runs") / f"hybrid_tune_{uuid4().hex}"
    run_dir.mkdir(parents=True)
    project_config_path = run_dir / "project.yaml"
    base_config_path = run_dir / "base_hybrid.yaml"
    output_config_path = run_dir / "output_hybrid.yaml"
    checkpoint_path = run_dir / "latest.pt"
    save_config(project_config, project_config_path)
    base_config_path.write_text(
        "\n".join(
            [
                "capture_enter_angle_radians: 3.2",
                "mpc:",
                "  horizon: 1",
                "  beam_width: 3",
                "  force_grid: [-20.0, 0.0, 20.0]",
            ]
        ),
        encoding="utf-8",
    )
    model = ActorCritic(8, 1, [8])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    save_checkpoint(model, optimizer, checkpoint_path, project_config, 0)

    result = tune_hybrid(
        swingup_config_path=project_config_path,
        swingup_checkpoint=checkpoint_path,
        base_config_path=base_config_path,
        output_config_path=output_config_path,
        episodes=1,
        seed_offset=0,
        device="cpu",
        max_candidates=1,
    )
    assert output_config_path.exists()
    assert load_hybrid_config(output_config_path).mpc.horizon >= 1
    assert "success_rate" in result["metrics"]


class _EnvStub:
    def __init__(self, physics_config):
        self.physics_config = physics_config
