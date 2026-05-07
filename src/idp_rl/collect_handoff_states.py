from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from idp_rl.config import load_config
from idp_rl.env import InvertedDoublePendulumEnv
from idp_rl.ppo import ActorCritic
from idp_rl.runtime import device_report, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect near-upright states produced by a swing-up policy.")
    parser.add_argument("--config", default="configs/swingup_curriculum.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--angle-threshold", type=float, default=0.45)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = resolve_device(args.device)
    print(device_report(device))
    states = collect_states(
        config=config,
        checkpoint=Path(args.checkpoint),
        episodes=args.episodes,
        angle_threshold=args.angle_threshold,
        device=device,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, states)
    print(f"saved_states={len(states)} output={output}")


def collect_states(config, checkpoint: Path, episodes: int, angle_threshold: float, device: torch.device) -> np.ndarray:
    env = InvertedDoublePendulumEnv(config)
    model = ActorCritic(8, 1, config.training.hidden_sizes).to(device)
    payload = torch.load(checkpoint, map_location=device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    states: list[np.ndarray] = []
    for episode in range(episodes):
        obs, info = env.reset(seed=config.training.seed + episode)
        done = False
        while not done:
            state = info["state"]
            if max(abs(state[2]), abs(state[4])) <= angle_threshold:
                states.append(state.copy())
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action = model.actor_mean(obs_tensor).cpu().numpy()[0]
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
    env.close()
    if not states:
        raise RuntimeError("No handoff states collected.")
    return np.asarray(states, dtype=np.float32)


if __name__ == "__main__":
    main()
