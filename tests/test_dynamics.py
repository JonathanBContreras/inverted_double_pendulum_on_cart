import numpy as np

from idp_rl.config import PhysicsConfig
from idp_rl.dynamics import DoublePendulumCartDynamics


def test_accelerations_are_finite_and_shaped():
    dynamics = DoublePendulumCartDynamics(PhysicsConfig())
    state = np.array([0.0, 0.1, 0.05, 0.0, -0.04, 0.02])
    accelerations = dynamics.accelerations(state, 0.0)
    assert accelerations.shape == (3,)
    assert np.isfinite(accelerations).all()


def test_rk4_step_is_deterministic():
    dynamics = DoublePendulumCartDynamics(PhysicsConfig())
    state = np.array([0.0, 0.0, 0.03, 0.0, -0.02, 0.0])
    first = dynamics.rk4_step(state, 1.5)
    second = dynamics.rk4_step(state, 1.5)
    np.testing.assert_allclose(first, second)


def test_upright_zero_force_has_no_acceleration():
    dynamics = DoublePendulumCartDynamics(PhysicsConfig())
    state = np.zeros(6)
    derivative = dynamics.derivative(state, 0.0)
    np.testing.assert_allclose(derivative, np.zeros(6), atol=1e-12)
