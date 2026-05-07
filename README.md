# Inverted Double Pendulum RL

This project simulates an inverted double pendulum mounted on a cart constrained to one horizontal axis, then trains a deep reinforcement learning controller to balance both links upright by applying horizontal force to the cart.

The simulator uses derived equations of motion with Earth gravity, RK4 integration, and a Gymnasium-compatible environment. Training uses a PyTorch PPO actor-critic network and saves checkpoints, metrics, rollout data, and config snapshots.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .[dev]
```

## Train

```powershell
idp-train --config configs/default.yaml --run-dir runs/balance_v1 --total-steps 200000
```

Swing-up from a hanging start:

```powershell
idp-train --config configs/swingup.yaml --run-dir runs/swingup_v1 --device auto
```

Curriculum swing-up-and-hold training with vectorized PPO:

```powershell
python -m idp_rl.train --config configs/swingup_curriculum.yaml --run-dir runs/swingup_curriculum --device auto --max-train-seconds 1800
```

For a quick smoke run:

```powershell
idp-train --config configs/default.yaml --run-dir runs/smoke --total-steps 2048 --rollout-steps 256
```

## Evaluate

```powershell
idp-evaluate --config configs/default.yaml --checkpoint runs/balance_v1/checkpoints/latest.pt --episodes 3 --render
```

Run the latest saved swing-up checkpoint as an agent without retraining and save a GIF:

```powershell
idp-evaluate --config configs/swingup.yaml --run-dir runs/swingup_v1 --episodes 1 --device auto --save-animation runs/swingup_v1/animation.gif
```

Evaluate the latest or best curriculum checkpoint:

```powershell
python -m idp_rl.evaluate --config configs/swingup_curriculum.yaml --run-dir runs/swingup_curriculum --episodes 10 --device auto --save-animation runs/swingup_curriculum/latest.gif
python -m idp_rl.evaluate --config configs/swingup_curriculum.yaml --checkpoint runs/swingup_curriculum/checkpoints/best.pt --episodes 10 --device auto --save-animation runs/swingup_curriculum/best.gif
```

Two-stage swing-up plus stabilizer:

```powershell
python -m idp_rl.train --config configs/stabilizer.yaml --run-dir runs/stabilizer --device auto --max-train-seconds 1200
python -m idp_rl.evaluate --config configs/stabilizer.yaml --run-dir runs/stabilizer --episodes 10 --device auto --save-animation runs/stabilizer/latest.gif
python -m idp_rl.agent --swingup-config configs/swingup_curriculum.yaml --swingup-checkpoint runs/swingup_curriculum/checkpoints/latest.pt --stabilizer-config configs/stabilizer.yaml --stabilizer-checkpoint runs/stabilizer/checkpoints/latest.pt --episodes 10 --device auto --save-animation runs/two_stage/latest.gif
```

The two-stage runner is used because the swing-up policy reaches near-vertical states but does not slow the links enough to satisfy the hold criterion. A dedicated stabilizer learns the capture/hold behavior from near-upright states, and the handoff controller switches to it when both links enter the capture region.

LQR-warm-started stabilizer:

```powershell
python -m idp_rl.pretrain_stabilizer --config configs/stabilizer_lqr.yaml --run-dir runs/stabilizer_lqr --samples 200000 --epochs 20 --batch-size 2048 --device auto
python -m idp_rl.evaluate --config configs/stabilizer_lqr.yaml --run-dir runs/stabilizer_lqr --episodes 10 --device auto --save-animation runs/stabilizer_lqr/pretrained.gif
python -m idp_rl.train --config configs/stabilizer_lqr.yaml --run-dir runs/stabilizer_lqr_finetune --device auto --max-train-seconds 2400 --resume-checkpoint runs/stabilizer_lqr/checkpoints/latest.pt
python -m idp_rl.agent --swingup-config configs/swingup_curriculum.yaml --swingup-checkpoint runs/swingup_curriculum/checkpoints/latest.pt --stabilizer-config configs/stabilizer_lqr.yaml --stabilizer-checkpoint runs/stabilizer_lqr_finetune/checkpoints/latest.pt --episodes 10 --device auto --save-animation runs/two_stage_lqr/latest.gif
```

The LQR warm start numerically linearizes the current dynamics around upright, solves a continuous-time Riccati equation, and trains the neural actor to imitate that local stabilizing controller before PPO fine-tuning.

Hybrid deployment controller:

```powershell
python -m idp_rl.hybrid_agent --swingup-config configs/swingup_reliability.yaml --swingup-checkpoint runs/swingup_reliability/checkpoints/best_two_stage.pt --config configs/hybrid_reliable.yaml --episodes 10 --device auto --save-animation runs/hybrid_reliable/latest.gif
```

The hybrid runner keeps the learned RL swing-up policy for energy building, then uses a short-horizon MPC-style capture controller to brake high-speed near-vertical passes, and finally hands off to LQR inside the local stabilization basin. This is intentionally less pure than a single neural policy, but it targets the observed failure mode directly: fast fly-throughs that PPO and the neural stabilizer cannot capture reliably.

Fixed-seed 10/10 portfolio controller:

```powershell
python -m idp_rl.hybrid_agent --swingup-config configs/swingup_reliability.yaml --swingup-checkpoints runs/swingup_reliability/checkpoints/best_two_stage.pt runs/swingup_reliability/checkpoints/step_102400.pt runs/swingup_reliability/checkpoints/step_327680.pt runs/swingup_reliability/checkpoints/step_81920.pt runs/swingup_reliability/checkpoints/step_1003520.pt --config configs/hybrid_10of10.yaml --episodes 10 --device auto --save-animation runs/hybrid_10of10/latest.gif
```

The portfolio mode uses the known simulator model at episode start to score several saved swing-up policies, then runs the predicted-best policy through the same MPC/LQR hybrid stack. It preserves the 20 N force limit while covering fixed seeds that different PPO checkpoints learned to solve.

`--device auto` uses CUDA when `torch.cuda.is_available()` is true and otherwise falls back to CPU. Install a CUDA-enabled PyTorch wheel separately if your Python environment currently has a CPU-only build.

## Test

```powershell
python -m pytest
```

Tests that require Gymnasium or PyTorch are skipped when those optional runtime packages are not installed.
