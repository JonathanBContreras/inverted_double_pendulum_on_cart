import importlib.util
import csv
from pathlib import Path
from uuid import uuid4

import pytest

from idp_rl.config import EnvConfig, PhysicsConfig, ProjectConfig, TrainingConfig

torch_available = importlib.util.find_spec("torch") is not None
gymnasium_available = importlib.util.find_spec("gymnasium") is not None


@pytest.mark.skipif(not (torch_available and gymnasium_available), reason="PyTorch and Gymnasium are required")
def test_short_training_writes_checkpoint_and_rollout():
    from idp_rl.train import train

    run_dir = Path("test_runs") / f"training_smoke_{uuid4().hex}"
    config = ProjectConfig(
        env=EnvConfig(max_episode_steps=20),
        physics=PhysicsConfig(),
        training=TrainingConfig(
            total_steps=32,
            rollout_steps=16,
            update_epochs=1,
            minibatch_size=8,
            hidden_sizes=[16],
            save_every_updates=1,
        ),
    )
    train(config, run_dir, total_steps=32, rollout_steps=16, device="cpu")
    assert (run_dir / "checkpoints" / "latest.pt").exists()
    assert (run_dir / "checkpoints" / "best.pt").exists()
    assert list((run_dir / "rollouts").glob("rollout_*.npz"))


@pytest.mark.skipif(not (torch_available and gymnasium_available), reason="PyTorch and Gymnasium are required")
def test_evaluate_resolves_latest_and_saves_animation():
    from idp_rl.evaluate import evaluate
    from idp_rl.runtime import resolve_checkpoint
    from idp_rl.train import train

    run_dir = Path("test_runs") / f"evaluate_smoke_{uuid4().hex}"
    config = ProjectConfig(
        env=EnvConfig(max_episode_steps=5, reset_mode="hanging", terminate_on_angle=False),
        physics=PhysicsConfig(),
        training=TrainingConfig(
            total_steps=8,
            rollout_steps=4,
            num_envs=2,
            update_epochs=1,
            minibatch_size=4,
            hidden_sizes=[16],
            eval_every_updates=1,
            save_every_updates=1,
        ),
    )
    train(config, run_dir, total_steps=8, rollout_steps=4, device="cpu")
    checkpoint = resolve_checkpoint(None, run_dir)
    animation_path = run_dir / "animation.gif"
    metrics = evaluate(config, checkpoint, episodes=1, render=False, device="cpu", save_animation=str(animation_path))
    assert len(metrics["returns"]) == 1
    assert "success_rate" in metrics
    assert "mean_max_hold_steps" in metrics
    assert "mean_handoff_ready_steps" in metrics
    assert animation_path.exists()
    assert animation_path.stat().st_size > 0


@pytest.mark.skipif(not (torch_available and gymnasium_available), reason="PyTorch and Gymnasium are required")
def test_training_writes_best_two_stage_checkpoint_and_metrics():
    from idp_rl.train import save_checkpoint, train
    from idp_rl.ppo import ActorCritic

    run_dir = Path("test_runs") / f"two_stage_training_smoke_{uuid4().hex}"
    stabilizer_dir = run_dir / "stabilizer"
    stabilizer_dir.mkdir(parents=True)
    config = ProjectConfig(
        env=EnvConfig(max_episode_steps=5, reset_mode="hanging", terminate_on_angle=False),
        physics=PhysicsConfig(),
        training=TrainingConfig(
            total_steps=8,
            rollout_steps=4,
            num_envs=2,
            update_epochs=1,
            minibatch_size=4,
            hidden_sizes=[16],
            eval_every_updates=1,
            save_every_updates=1,
        ),
    )
    stabilizer_model = ActorCritic(8, 1, [16])
    optimizer = __import__("torch").optim.Adam(stabilizer_model.parameters(), lr=1e-3)
    save_checkpoint(stabilizer_model, optimizer, stabilizer_dir / "latest.pt", config, 0)
    train(
        config,
        run_dir,
        total_steps=8,
        rollout_steps=4,
        device="cpu",
        two_stage_stabilizer_config=None,
        two_stage_stabilizer_checkpoint=stabilizer_dir / "latest.pt",
        two_stage_eval_episodes=1,
        two_stage_eval_every_updates=1,
    )
    assert (run_dir / "checkpoints" / "best_two_stage.pt").exists()
    with (run_dir / "metrics.csv").open("r", encoding="utf-8") as handle:
        header = next(csv.reader(handle))
    assert "two_stage_success_rate" in header
    assert "two_stage_max_hold_steps" in header
