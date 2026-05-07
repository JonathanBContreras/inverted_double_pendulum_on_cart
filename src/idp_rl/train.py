from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from idp_rl.config import load_config, save_config
from idp_rl.controllers import PolicyController
from idp_rl.env import InvertedDoublePendulumEnv
from idp_rl.ppo import ActorCritic
from idp_rl.rollout import evaluate_controller
from idp_rl.runtime import device_report, resolve_device
from idp_rl.agent import HandoffConfig, TwoStageAgent, load_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on the inverted double pendulum environment.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--run-dir", default="runs/default")
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--max-train-seconds", type=float, default=None)
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--two-stage-stabilizer-config", default=None)
    parser.add_argument("--two-stage-stabilizer-checkpoint", default=None)
    parser.add_argument("--two-stage-eval-episodes", type=int, default=10)
    parser.add_argument("--two-stage-eval-every-updates", type=int, default=10)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    train(
        config,
        Path(args.run_dir),
        args.total_steps,
        args.rollout_steps,
        args.device,
        args.max_train_seconds,
        args.resume_checkpoint,
        args.two_stage_stabilizer_config,
        args.two_stage_stabilizer_checkpoint,
        args.two_stage_eval_episodes,
        args.two_stage_eval_every_updates,
    )


def train(
    config,
    run_dir: Path,
    total_steps: int | None = None,
    rollout_steps: int | None = None,
    device: str = "cpu",
    max_train_seconds: float | None = None,
    resume_checkpoint: str | Path | None = None,
    two_stage_stabilizer_config: str | Path | None = None,
    two_stage_stabilizer_checkpoint: str | Path | None = None,
    two_stage_eval_episodes: int = 10,
    two_stage_eval_every_updates: int = 10,
) -> None:
    resolved_device = resolve_device(device)
    print(device_report(resolved_device))
    tcfg = config.training
    total_steps = total_steps or tcfg.total_steps
    rollout_steps = rollout_steps or tcfg.rollout_steps
    num_envs = tcfg.num_envs
    start_time = time.monotonic()

    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = run_dir / "checkpoints"
    rollout_dir = run_dir / "rollouts"
    checkpoint_dir.mkdir(exist_ok=True)
    rollout_dir.mkdir(exist_ok=True)
    save_config(config, run_dir / "config.yaml")
    write_training_notes(config, run_dir / "training_notes.md")

    rng = np.random.default_rng(tcfg.seed)
    torch.manual_seed(tcfg.seed)
    reset_states = load_reset_states(tcfg.reset_states_path)
    envs = [InvertedDoublePendulumEnv(config) for _ in range(num_envs)]
    obs_list = []
    for index, env in enumerate(envs):
        obs_item, _ = reset_env(env, rng, tcfg, 0, reset_states, seed=tcfg.seed + index)
        obs_list.append(obs_item)
    obs = np.asarray(obs_list, dtype=np.float32)
    env = envs[0]
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    action_low = torch.as_tensor(env.action_space.low, dtype=torch.float32, device=resolved_device)
    action_high = torch.as_tensor(env.action_space.high, dtype=torch.float32, device=resolved_device)

    model = ActorCritic(obs_dim, action_dim, tcfg.hidden_sizes).to(resolved_device)
    with torch.no_grad():
        model.log_std.fill_(tcfg.initial_log_std)
    optimizer = torch.optim.Adam(model.parameters(), lr=tcfg.learning_rate, eps=1e-5)
    if resume_checkpoint is not None:
        payload = torch.load(resume_checkpoint, map_location=resolved_device)
        model.load_state_dict(payload["model_state_dict"])
        if "optimizer_state_dict" in payload:
            try:
                optimizer.load_state_dict(payload["optimizer_state_dict"])
            except ValueError:
                pass
            else:
                for group in optimizer.param_groups:
                    group["lr"] = tcfg.learning_rate
    two_stage_config = None
    two_stage_stabilizer_model = None
    if two_stage_stabilizer_checkpoint:
        two_stage_config = load_config(two_stage_stabilizer_config) if two_stage_stabilizer_config else config
        two_stage_stabilizer_model = load_policy(
            two_stage_config,
            Path(two_stage_stabilizer_checkpoint),
            resolved_device,
        )

    metrics_path = run_dir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "update",
                "global_step",
                "episode_return",
                "episode_length",
                "loss",
                "eval_success_rate",
                "eval_mean_return",
                "eval_mean_max_hold_steps",
                "eval_mean_handoff_ready_steps",
                "two_stage_success_rate",
                "two_stage_mean_max_hold_steps",
                "two_stage_max_hold_steps",
                "two_stage_mean_handoff_ready_steps",
                "two_stage_handoff_count",
                "two_stage_stabilizer_active_steps",
            ],
        )
        writer.writeheader()

    global_step = 0
    update = 0
    episode_returns = np.zeros(num_envs, dtype=np.float64)
    episode_lengths = np.zeros(num_envs, dtype=np.int32)
    completed_returns: list[float] = []
    completed_lengths: list[int] = []
    latest_loss = 0.0
    best_score = (-1.0, -1.0, -float("inf"))
    best_two_stage_score = (-1.0, -1.0, -1.0, -float("inf"))

    while global_step < total_steps:
        if max_train_seconds is not None and time.monotonic() - start_time >= max_train_seconds:
            break
        update += 1
        observations = np.zeros((rollout_steps, num_envs, obs_dim), dtype=np.float32)
        actions = np.zeros((rollout_steps, num_envs, action_dim), dtype=np.float32)
        log_probs = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        rewards = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        dones = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        values = np.zeros((rollout_steps, num_envs), dtype=np.float32)

        for step in range(rollout_steps):
            observations[step] = obs
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=resolved_device)
            with torch.no_grad():
                action_tensor, log_prob, _, value = model.get_action_and_value(obs_tensor)

            raw_actions = action_tensor.cpu().numpy()
            clipped_actions = torch.clamp(action_tensor, action_low, action_high).cpu().numpy()
            next_obs = np.zeros_like(obs)
            step_rewards = np.zeros(num_envs, dtype=np.float32)
            step_dones = np.zeros(num_envs, dtype=np.float32)

            for env_index, vector_env in enumerate(envs):
                obs_item, reward, terminated, truncated, _ = vector_env.step(clipped_actions[env_index])
                done = terminated or truncated
                step_rewards[env_index] = reward
                step_dones[env_index] = float(done)
                episode_returns[env_index] += reward
                episode_lengths[env_index] += 1

                if done:
                    completed_returns.append(float(episode_returns[env_index]))
                    completed_lengths.append(int(episode_lengths[env_index]))
                    reset_obs, _ = reset_env(vector_env, rng, tcfg, global_step, reset_states)
                    next_obs[env_index] = reset_obs
                    episode_returns[env_index] = 0.0
                    episode_lengths[env_index] = 0
                else:
                    next_obs[env_index] = obs_item

            actions[step] = raw_actions
            log_probs[step] = log_prob.cpu().numpy()
            rewards[step] = step_rewards
            dones[step] = step_dones
            values[step] = value.cpu().numpy()
            obs = next_obs
            global_step += num_envs

            if global_step >= total_steps or (
                max_train_seconds is not None and time.monotonic() - start_time >= max_train_seconds
            ):
                observations = observations[: step + 1]
                actions = actions[: step + 1]
                log_probs = log_probs[: step + 1]
                rewards = rewards[: step + 1]
                dones = dones[: step + 1]
                values = values[: step + 1]
                break

        with torch.no_grad():
            next_value = model.get_value(torch.as_tensor(obs, dtype=torch.float32, device=resolved_device)).cpu().numpy()
        advantages, returns = compute_gae(rewards, dones, values, next_value, tcfg.gamma, tcfg.gae_lambda)
        flat_observations = observations.reshape((-1, obs_dim))
        flat_actions = actions.reshape((-1, action_dim))
        flat_log_probs = log_probs.reshape(-1)
        flat_advantages = advantages.reshape(-1)
        flat_returns = returns.reshape(-1)
        latest_loss = update_policy(
            model,
            optimizer,
            flat_observations,
            flat_actions,
            flat_log_probs,
            flat_advantages,
            flat_returns,
            tcfg,
            str(resolved_device),
        )

        np.savez_compressed(
            rollout_dir / f"rollout_{update:05d}.npz",
            observations=observations,
            actions=actions,
            log_probs=log_probs,
            rewards=rewards,
            dones=dones,
            values=values,
            advantages=advantages,
            returns=returns,
        )

        mean_return = float(np.mean(completed_returns[-10:])) if completed_returns else float("nan")
        mean_length = float(np.mean(completed_lengths[-10:])) if completed_lengths else float("nan")
        eval_metrics = None
        two_stage_metrics = None
        if update % tcfg.eval_every_updates == 0 or global_step >= total_steps:
            eval_metrics = evaluate_policy(model, config, tcfg.eval_episodes, resolved_device)
            score = (
                eval_metrics["success_rate"],
                eval_metrics["mean_max_hold_steps"],
                eval_metrics["mean_return"],
            )
            if score > best_score:
                best_score = score
                save_checkpoint(model, optimizer, checkpoint_dir / "best.pt", config, global_step)
        if (
            two_stage_stabilizer_model is not None
            and two_stage_config is not None
            and (update % two_stage_eval_every_updates == 0 or global_step >= total_steps)
        ):
            two_stage_metrics = evaluate_two_stage_candidate(
                swingup_model=model,
                swingup_config=config,
                stabilizer_model=two_stage_stabilizer_model,
                episodes=two_stage_eval_episodes,
                device=resolved_device,
            )
            two_stage_score = (
                two_stage_metrics["success_rate"],
                two_stage_metrics["max_hold_steps"],
                two_stage_metrics["mean_handoff_ready_steps"],
                two_stage_metrics["mean_return"],
            )
            if two_stage_score > best_two_stage_score:
                best_two_stage_score = two_stage_score
                save_checkpoint(model, optimizer, checkpoint_dir / "best_two_stage.pt", config, global_step)
        with metrics_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "update",
                    "global_step",
                    "episode_return",
                    "episode_length",
                    "loss",
                    "eval_success_rate",
                    "eval_mean_return",
                    "eval_mean_max_hold_steps",
                    "eval_mean_handoff_ready_steps",
                    "two_stage_success_rate",
                    "two_stage_mean_max_hold_steps",
                    "two_stage_max_hold_steps",
                    "two_stage_mean_handoff_ready_steps",
                    "two_stage_handoff_count",
                    "two_stage_stabilizer_active_steps",
                ],
            )
            writer.writerow(
                {
                    "update": update,
                    "global_step": global_step,
                    "episode_return": mean_return,
                    "episode_length": mean_length,
                    "loss": latest_loss,
                    "eval_success_rate": "" if eval_metrics is None else eval_metrics["success_rate"],
                    "eval_mean_return": "" if eval_metrics is None else eval_metrics["mean_return"],
                    "eval_mean_max_hold_steps": "" if eval_metrics is None else eval_metrics["mean_max_hold_steps"],
                    "eval_mean_handoff_ready_steps": ""
                    if eval_metrics is None
                    else eval_metrics["mean_handoff_ready_steps"],
                    "two_stage_success_rate": ""
                    if two_stage_metrics is None
                    else two_stage_metrics["success_rate"],
                    "two_stage_mean_max_hold_steps": ""
                    if two_stage_metrics is None
                    else two_stage_metrics["mean_max_hold_steps"],
                    "two_stage_max_hold_steps": "" if two_stage_metrics is None else two_stage_metrics["max_hold_steps"],
                    "two_stage_mean_handoff_ready_steps": ""
                    if two_stage_metrics is None
                    else two_stage_metrics["mean_handoff_ready_steps"],
                    "two_stage_handoff_count": ""
                    if two_stage_metrics is None
                    else two_stage_metrics["handoff_count"],
                    "two_stage_stabilizer_active_steps": ""
                    if two_stage_metrics is None
                    else two_stage_metrics["stabilizer_active_steps"],
                }
            )

        if update % tcfg.save_every_updates == 0 or global_step >= total_steps:
            save_checkpoint(model, optimizer, checkpoint_dir / "latest.pt", config, global_step)
            save_checkpoint(model, optimizer, checkpoint_dir / f"step_{global_step}.pt", config, global_step)

    save_checkpoint(model, optimizer, checkpoint_dir / "latest.pt", config, global_step)
    if not (checkpoint_dir / "best.pt").exists():
        save_checkpoint(model, optimizer, checkpoint_dir / "best.pt", config, global_step)
    if two_stage_stabilizer_model is not None and not (checkpoint_dir / "best_two_stage.pt").exists():
        save_checkpoint(model, optimizer, checkpoint_dir / "best_two_stage.pt", config, global_step)
    for vector_env in envs:
        vector_env.close()


def compute_gae(
    rewards: np.ndarray,
    dones: np.ndarray,
    values: np.ndarray,
    next_value: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae = np.zeros(rewards.shape[1], dtype=np.float32)
    for t in reversed(range(len(rewards))):
        next_non_terminal = 1.0 - dones[t]
        next_values = next_value if t == len(rewards) - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_values * next_non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


def update_policy(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    observations: np.ndarray,
    actions: np.ndarray,
    old_log_probs: np.ndarray,
    advantages: np.ndarray,
    returns: np.ndarray,
    tcfg,
    device: str,
) -> float:
    obs_t = torch.as_tensor(observations, dtype=torch.float32, device=device)
    actions_t = torch.as_tensor(actions, dtype=torch.float32, device=device)
    old_log_probs_t = torch.as_tensor(old_log_probs, dtype=torch.float32, device=device)
    advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=device)
    returns_t = torch.as_tensor(returns, dtype=torch.float32, device=device)
    if tcfg.normalize_advantages:
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std(unbiased=False) + 1e-8)

    batch_size = len(observations)
    indexes = np.arange(batch_size)
    latest_loss = 0.0
    for _ in range(tcfg.update_epochs):
        np.random.shuffle(indexes)
        for start in range(0, batch_size, tcfg.minibatch_size):
            batch_indexes = indexes[start : start + tcfg.minibatch_size]
            _, new_log_prob, entropy, new_value = model.get_action_and_value(obs_t[batch_indexes], actions_t[batch_indexes])
            log_ratio = new_log_prob - old_log_probs_t[batch_indexes]
            ratio = log_ratio.exp()
            approx_kl = ((ratio - 1.0) - log_ratio).mean()
            policy_loss_1 = -advantages_t[batch_indexes] * ratio
            policy_loss_2 = -advantages_t[batch_indexes] * torch.clamp(ratio, 1.0 - tcfg.clip_coef, 1.0 + tcfg.clip_coef)
            policy_loss = torch.max(policy_loss_1, policy_loss_2).mean()
            value_loss = 0.5 * ((new_value - returns_t[batch_indexes]) ** 2).mean()
            entropy_loss = entropy.mean()
            loss = policy_loss + tcfg.value_coef * value_loss - tcfg.entropy_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), tcfg.max_grad_norm)
            optimizer.step()
            latest_loss = float(loss.detach().cpu().item())
        if tcfg.target_kl and float(approx_kl.detach().cpu().item()) > tcfg.target_kl:
            break
    return latest_loss


def curriculum_options(tcfg, global_step: int) -> dict | None:
    for stage in tcfg.curriculum_stages:
        if global_step < int(stage.get("until_step", 0)):
            return {
                "reset_mode": stage.get("reset_mode", "custom"),
                "angle_center": stage.get("angle_center", 0.0),
                "angle_range": stage.get("angle_range", 0.1),
                "angular_velocity_range": stage.get("angular_velocity_range", 0.1),
                "cart_position_range": stage.get("cart_position_range", 0.05),
                "cart_velocity_range": stage.get("cart_velocity_range", 0.05),
            }
    return None


def load_reset_states(path: str | None) -> np.ndarray | None:
    if not path:
        return None
    states = np.load(path)
    if states.ndim != 2 or states.shape[1] != 6:
        raise ValueError(f"Expected reset state array with shape (N, 6), got {states.shape}")
    return states.astype(np.float64)


def reset_env(env: InvertedDoublePendulumEnv, rng: np.random.Generator, tcfg, global_step: int, reset_states, seed=None):
    if reset_states is not None:
        index = int(rng.integers(0, len(reset_states)))
        return env.reset(seed=seed, options={"state": reset_states[index]})
    return env.reset(seed=seed if seed is not None else int(rng.integers(0, 2**31 - 1)), options=curriculum_options(tcfg, global_step))


def evaluate_policy(model: ActorCritic, config, episodes: int, device: torch.device) -> dict[str, float]:
    return evaluate_controller(
        config,
        PolicyController(model, device),
        episodes,
        seed_offset=10_000,
    )


def evaluate_two_stage_candidate(
    swingup_model: ActorCritic,
    swingup_config,
    stabilizer_model: ActorCritic,
    episodes: int,
    device: torch.device,
) -> dict[str, float]:
    metrics = evaluate_controller(
        swingup_config,
        TwoStageAgent(swingup_model, stabilizer_model, HandoffConfig(), device),
        episodes,
        seed_offset=20_000,
    )
    metrics["stabilizer_active_steps"] = metrics["stabilizer_steps"]
    return metrics


def write_training_notes(config, path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "# Training Notes",
                "",
                "PPO is used because it is stable for continuous control and easy to inspect.",
                f"The actor and critic are MLPs with hidden sizes `{config.training.hidden_sizes}` and tanh activations.",
                "The Gaussian policy outputs continuous cart force commands; the environment clips those commands to physical limits.",
                f"`gamma={config.training.gamma}` and `gae_lambda={config.training.gae_lambda}` balance long-horizon swing-up credit assignment with variance.",
                f"`entropy_coef={config.training.entropy_coef}` keeps exploration alive while the curriculum moves toward hanging starts.",
                f"`num_envs={config.training.num_envs}` gathers diverse rollouts per PPO update and improves GPU utilization.",
                "Curriculum stages start easier than full hanging swing-up, then progressively move the reset distribution toward hanging.",
            ]
        ),
        encoding="utf-8",
    )


def save_checkpoint(model: ActorCritic, optimizer: torch.optim.Optimizer, path: Path, config, global_step: int) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config.to_dict(),
            "global_step": global_step,
        },
        path,
    )
    with path.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump({"global_step": global_step}, handle, indent=2)


if __name__ == "__main__":
    main()
