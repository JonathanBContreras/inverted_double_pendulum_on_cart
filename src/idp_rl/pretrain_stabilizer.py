from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch import nn

from idp_rl.config import load_config, save_config
from idp_rl.lqr import lqr_action, lqr_gain
from idp_rl.ppo import ActorCritic
from idp_rl.runtime import device_report, resolve_device
from idp_rl.train import save_checkpoint, write_training_notes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Supervised pretrain the stabilizer actor from an LQR teacher.")
    parser.add_argument("--config", default="configs/stabilizer_lqr.yaml")
    parser.add_argument("--run-dir", default="runs/stabilizer_lqr")
    parser.add_argument("--samples", type=int, default=200_000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    pretrain(
        config=config,
        run_dir=Path(args.run_dir),
        samples=args.samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed if args.seed is not None else config.training.seed,
    )


def pretrain(config, run_dir: Path, samples: int, epochs: int, batch_size: int, device: str, seed: int) -> None:
    resolved_device = resolve_device(device)
    print(device_report(resolved_device))
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    save_config(config, run_dir / "config.yaml")
    write_training_notes(config, run_dir / "training_notes.md")

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    states = sample_stabilizer_states(config, samples, rng)
    gain = lqr_gain(config.physics)
    targets = np.array(
        [lqr_action(state, gain, config.physics.max_force) for state in states],
        dtype=np.float32,
    ).reshape(-1, 1)
    observations = states_to_observations(states)

    model = ActorCritic(8, 1, config.training.hidden_sizes).to(resolved_device)
    with torch.no_grad():
        model.log_std.fill_(config.training.initial_log_std)
    optimizer = torch.optim.Adam(model.actor_mean.parameters(), lr=config.training.learning_rate)
    loss_fn = nn.MSELoss()

    obs_t = torch.as_tensor(observations, dtype=torch.float32, device=resolved_device)
    target_t = torch.as_tensor(targets, dtype=torch.float32, device=resolved_device)
    metrics_path = run_dir / "pretrain_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "loss"])
        writer.writeheader()

    indexes = np.arange(samples)
    latest_loss = 0.0
    for epoch in range(1, epochs + 1):
        rng.shuffle(indexes)
        for start in range(0, samples, batch_size):
            batch_indexes = indexes[start : start + batch_size]
            prediction = model.actor_mean(obs_t[batch_indexes])
            loss = loss_fn(prediction, target_t[batch_indexes])
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.actor_mean.parameters(), config.training.max_grad_norm)
            optimizer.step()
            latest_loss = float(loss.detach().cpu().item())
        with metrics_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["epoch", "loss"])
            writer.writerow({"epoch": epoch, "loss": latest_loss})

    # Give PPO fine-tuning a critic that is at least finite around the stabilizer basin.
    critic_optimizer = torch.optim.Adam(model.critic.parameters(), lr=config.training.learning_rate)
    with torch.no_grad():
        value_targets = -torch.mean((target_t - model.actor_mean(obs_t)) ** 2, dim=1)
    for _ in range(3):
        for start in range(0, samples, batch_size):
            batch = slice(start, min(start + batch_size, samples))
            value = model.get_value(obs_t[batch])
            loss = loss_fn(value, value_targets[batch])
            critic_optimizer.zero_grad()
            loss.backward()
            critic_optimizer.step()

    save_checkpoint(model, optimizer, checkpoint_dir / "latest.pt", config, 0)
    save_checkpoint(model, optimizer, checkpoint_dir / "best.pt", config, 0)


def sample_stabilizer_states(config, samples: int, rng: np.random.Generator) -> np.ndarray:
    # A mixture keeps the teacher accurate near upright while showing moderate capture velocities.
    buckets = [
        (0.04, 0.2, 0.03, 0.1, 0.35),
        (0.08, 0.6, 0.05, 0.2, 0.30),
        (0.15, 1.5, 0.10, 0.4, 0.25),
        (0.25, 3.5, 0.15, 0.6, 0.10),
    ]
    counts = [int(samples * bucket[-1]) for bucket in buckets]
    counts[-1] += samples - sum(counts)
    states = []
    for (angle_range, angular_velocity_range, cart_position_range, cart_velocity_range, _), count in zip(buckets, counts):
        block = np.column_stack(
            [
                rng.uniform(-cart_position_range, cart_position_range, count),
                rng.uniform(-cart_velocity_range, cart_velocity_range, count),
                rng.uniform(-angle_range, angle_range, count),
                rng.uniform(-angular_velocity_range, angular_velocity_range, count),
                rng.uniform(-angle_range, angle_range, count),
                rng.uniform(-angular_velocity_range, angular_velocity_range, count),
            ]
        )
        states.append(block)
    return np.concatenate(states, axis=0).astype(np.float32)


def states_to_observations(states: np.ndarray) -> np.ndarray:
    x = states[:, 0]
    x_dot = states[:, 1]
    theta1 = states[:, 2]
    theta1_dot = states[:, 3]
    theta2 = states[:, 4]
    theta2_dot = states[:, 5]
    return np.column_stack(
        [
            x,
            x_dot,
            np.sin(theta1),
            np.cos(theta1),
            theta1_dot,
            np.sin(theta2),
            np.cos(theta2),
            theta2_dot,
        ]
    ).astype(np.float32)


if __name__ == "__main__":
    main()
