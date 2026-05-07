import importlib.util
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
import torch

from idp_rl.agent import HandoffConfig, evaluate_two_stage, should_enter_stabilizer, should_exit_stabilizer
from idp_rl.config import load_config
from idp_rl.ppo import ActorCritic
from idp_rl.train import save_checkpoint

gymnasium_available = importlib.util.find_spec("gymnasium") is not None


def test_handoff_enters_inside_capture_region():
    state = np.array([0.0, 0.0, 0.2, 3.0, -0.1, -3.4])
    assert should_enter_stabilizer(state, track_limit=2.4, config=HandoffConfig())


def test_handoff_rejects_too_fast_capture_region():
    state = np.array([0.0, 0.0, 0.2, 3.0, -0.1, -4.0])
    assert not should_enter_stabilizer(state, track_limit=2.4, config=HandoffConfig())


def test_handoff_exits_outside_escape_region():
    state = np.array([0.0, 0.0, 0.6, 0.0, 0.1, 0.0])
    assert should_exit_stabilizer(state, track_limit=2.4, config=HandoffConfig())


@pytest.mark.skipif(not gymnasium_available, reason="Gymnasium is not installed")
def test_two_stage_evaluator_loads_checkpoints_and_saves_gif():
    run_dir = Path("test_runs") / f"agent_smoke_{uuid4().hex}"
    swingup_dir = run_dir / "swingup"
    stabilizer_dir = run_dir / "stabilizer"
    swingup_dir.mkdir(parents=True)
    stabilizer_dir.mkdir(parents=True)

    config = load_config("configs/swingup.yaml")
    stabilizer_config = load_config("configs/stabilizer.yaml")
    swingup_model = ActorCritic(8, 1, config.training.hidden_sizes)
    stabilizer_model = ActorCritic(8, 1, stabilizer_config.training.hidden_sizes)
    optimizer = torch.optim.Adam(swingup_model.parameters(), lr=1e-3)
    save_checkpoint(swingup_model, optimizer, swingup_dir / "latest.pt", config, 0)
    optimizer = torch.optim.Adam(stabilizer_model.parameters(), lr=1e-3)
    save_checkpoint(stabilizer_model, optimizer, stabilizer_dir / "latest.pt", stabilizer_config, 0)

    # Reuse existing YAML-backed configs so evaluate_two_stage exercises the public loading path.
    animation_path = run_dir / "agent.gif"
    metrics = evaluate_two_stage(
        swingup_config_path=Path("configs/swingup.yaml"),
        swingup_checkpoint=swingup_dir / "latest.pt",
        stabilizer_config_path=Path("configs/stabilizer.yaml"),
        stabilizer_checkpoint=stabilizer_dir / "latest.pt",
        episodes=1,
        device="cpu",
        save_animation=str(animation_path),
    )
    assert "handoff_count" in metrics
    assert "stabilizer_active_steps" in metrics
    assert animation_path.exists()
    assert animation_path.stat().st_size > 0
