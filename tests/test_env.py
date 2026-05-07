import importlib.util

import numpy as np
import pytest

from idp_rl.config import EnvConfig, ProjectConfig

gymnasium_available = importlib.util.find_spec("gymnasium") is not None


@pytest.mark.skipif(not gymnasium_available, reason="Gymnasium is not installed")
def test_seeded_reset_is_deterministic():
    from idp_rl.env import InvertedDoublePendulumEnv

    env = InvertedDoublePendulumEnv(ProjectConfig())
    obs1, info1 = env.reset(seed=123)
    obs2, info2 = env.reset(seed=123)
    np.testing.assert_allclose(obs1, obs2)
    np.testing.assert_allclose(info1["state"], info2["state"])
    env.close()


@pytest.mark.skipif(not gymnasium_available, reason="Gymnasium is not installed")
def test_step_shapes_and_termination():
    from idp_rl.env import InvertedDoublePendulumEnv

    env = InvertedDoublePendulumEnv(ProjectConfig())
    env.reset(options={"state": np.array([3.0, 0.0, 0.0, 0.0, 0.0, 0.0])})
    obs, reward, terminated, truncated, info = env.step([0.0])
    assert obs.shape == env.observation_space.shape
    assert np.isfinite(reward)
    assert terminated
    assert not truncated
    assert not info["is_success"]
    env.close()


@pytest.mark.skipif(not gymnasium_available, reason="Gymnasium is not installed")
def test_hanging_reset_is_deterministic_and_allowed():
    from idp_rl.env import InvertedDoublePendulumEnv

    config = ProjectConfig(env=EnvConfig(reset_mode="hanging", terminate_on_angle=False))
    env = InvertedDoublePendulumEnv(config)
    obs1, info1 = env.reset(seed=123)
    obs2, info2 = env.reset(seed=123)
    np.testing.assert_allclose(obs1, obs2)
    np.testing.assert_allclose(info1["state"], info2["state"])
    assert abs(abs(info1["state"][2]) - np.pi) < 0.1
    _, _, terminated, _, _ = env.step([0.0])
    assert not terminated
    env.close()


@pytest.mark.skipif(not gymnasium_available, reason="Gymnasium is not installed")
def test_curriculum_reset_options_control_angle_range():
    from idp_rl.env import InvertedDoublePendulumEnv

    env = InvertedDoublePendulumEnv(ProjectConfig())
    _, info = env.reset(
        seed=123,
        options={
            "reset_mode": "custom",
            "angle_center": 1.5,
            "angle_range": 0.01,
            "angular_velocity_range": 0.01,
        },
    )
    assert abs(info["state"][2] - 1.5) < 0.02
    assert abs(info["state"][4] - 1.5) < 0.02
    env.close()


@pytest.mark.skipif(not gymnasium_available, reason="Gymnasium is not installed")
def test_reset_options_control_cart_and_velocity_ranges():
    from idp_rl.env import InvertedDoublePendulumEnv

    env = InvertedDoublePendulumEnv(ProjectConfig())
    _, info = env.reset(
        seed=123,
        options={
            "reset_mode": "upright",
            "angle_range": 0.35,
            "angular_velocity_range": 8.0,
            "cart_position_range": 0.2,
            "cart_velocity_range": 1.0,
        },
    )
    assert abs(info["state"][0]) <= 0.2
    assert abs(info["state"][1]) <= 1.0
    assert abs(info["state"][3]) <= 8.0
    assert abs(info["state"][5]) <= 8.0
    env.close()


@pytest.mark.skipif(not gymnasium_available, reason="Gymnasium is not installed")
def test_success_requires_hold_window():
    from idp_rl.env import InvertedDoublePendulumEnv

    config = ProjectConfig(env=EnvConfig(success_hold_steps=3))
    env = InvertedDoublePendulumEnv(config)
    env.reset(options={"state": np.zeros(6)})
    assert not env.is_success()
    _, _, _, _, info = env.step([0.0])
    assert not info["is_success"]
    _, _, _, _, info = env.step([0.0])
    assert info["is_success"]
    assert info["hold_steps"] >= 3
    env.close()


@pytest.mark.skipif(not gymnasium_available, reason="Gymnasium is not installed")
def test_upright_reward_exceeds_hanging_reward():
    from idp_rl.env import InvertedDoublePendulumEnv

    config = ProjectConfig(env=EnvConfig(reset_mode="hanging", terminate_on_angle=False))
    env = InvertedDoublePendulumEnv(config)
    env.reset(options={"state": np.zeros(6)})
    upright_reward = env._reward(0.0, False)
    env.reset(options={"state": np.array([0.0, 0.0, np.pi, 0.0, np.pi, 0.0])})
    hanging_reward = env._reward(0.0, False)
    assert upright_reward > hanging_reward
    env.close()


@pytest.mark.skipif(not gymnasium_available, reason="Gymnasium is not installed")
def test_slow_upright_reward_exceeds_fast_upright_reward():
    from idp_rl.env import InvertedDoublePendulumEnv

    config = ProjectConfig(
        env=EnvConfig(
            terminate_on_angle=False,
            handoff_bonus=10.0,
            handoff_velocity_penalty_weight=0.1,
            handoff_angle_band_radians=0.55,
        )
    )
    env = InvertedDoublePendulumEnv(config)
    env.reset(options={"state": np.array([0.0, 0.0, 0.1, 0.2, -0.1, -0.2])})
    slow_reward = env._reward(0.0, False)
    env.reset(options={"state": np.array([0.0, 0.0, 0.1, 8.0, -0.1, -8.0])})
    fast_reward = env._reward(0.0, False)
    assert slow_reward > fast_reward
    env.close()


@pytest.mark.skipif(not gymnasium_available, reason="Gymnasium is not installed")
def test_handoff_ready_requires_angle_velocity_and_margin():
    from idp_rl.env import InvertedDoublePendulumEnv

    env = InvertedDoublePendulumEnv(ProjectConfig())
    env.reset(options={"state": np.array([0.0, 0.0, 0.2, 2.0, -0.2, -2.0])})
    assert env.is_handoff_ready()
    env.reset(options={"state": np.array([0.0, 0.0, 0.2, 4.0, -0.2, -2.0])})
    assert not env.is_handoff_ready()
    env.reset(options={"state": np.array([2.2, 0.0, 0.2, 2.0, -0.2, -2.0])})
    assert not env.is_handoff_ready()
    env.close()


@pytest.mark.skipif(not gymnasium_available, reason="Gymnasium is not installed")
def test_gymnasium_env_checker():
    from gymnasium.utils.env_checker import check_env

    from idp_rl.env import InvertedDoublePendulumEnv

    check_env(InvertedDoublePendulumEnv(ProjectConfig()), skip_render_check=True)
