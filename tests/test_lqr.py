import importlib.util
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
import torch

from idp_rl.config import EnvConfig, PhysicsConfig, ProjectConfig, TrainingConfig
from idp_rl.env import InvertedDoublePendulumEnv
from idp_rl.lqr import lqr_action, lqr_gain

scipy_available = importlib.util.find_spec("scipy") is not None
gymnasium_available = importlib.util.find_spec("gymnasium") is not None


@pytest.mark.skipif(not scipy_available, reason="SciPy is not installed")
def test_lqr_gain_is_finite_and_shaped():
    gain = lqr_gain(PhysicsConfig())
    assert gain.shape == (1, 6)
    assert np.isfinite(gain).all()


@pytest.mark.skipif(not (scipy_available and gymnasium_available), reason="SciPy and Gymnasium are required")
def test_lqr_stabilizes_easy_near_upright_state():
    config = ProjectConfig(
        env=EnvConfig(reset_mode="upright", terminate_on_angle=True, angle_termination_radians=0.9, success_hold_steps=100),
        physics=PhysicsConfig(),
    )
    gain = lqr_gain(config.physics)
    env = InvertedDoublePendulumEnv(config)
    env.reset(
        seed=123,
        options={
            "reset_mode": "upright",
            "angle_range": 0.04,
            "angular_velocity_range": 0.2,
            "cart_position_range": 0.03,
            "cart_velocity_range": 0.1,
        },
    )
    done = False
    info = {"max_hold_steps": 0, "is_success": False}
    while not done:
        action = lqr_action(env.state, gain, config.physics.max_force)
        _, _, terminated, truncated, info = env.step([action])
        done = terminated or truncated
    assert info["is_success"]
    assert info["max_hold_steps"] >= 100
    env.close()


@pytest.mark.skipif(not scipy_available, reason="SciPy is not installed")
def test_pretraining_writes_loadable_checkpoint():
    from idp_rl.pretrain_stabilizer import pretrain

    run_dir = Path("test_runs") / f"pretrain_smoke_{uuid4().hex}"
    config = ProjectConfig(
        env=EnvConfig(reset_mode="upright"),
        physics=PhysicsConfig(),
        training=TrainingConfig(hidden_sizes=[16], learning_rate=1e-3, initial_log_std=-1.0),
    )
    pretrain(config, run_dir, samples=512, epochs=1, batch_size=128, device="cpu", seed=1)
    payload = torch.load(run_dir / "checkpoints" / "latest.pt", map_location="cpu")
    assert "model_state_dict" in payload
