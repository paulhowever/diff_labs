import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analytical import (analytical_solution, analytical_velocity,
                            damping_regime)


def numeric_second_derivative(t: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.gradient(np.gradient(x, t), t)


def test_initial_conditions_underdamped():
    omega0, delta = 2.0, 0.2
    x0, v0 = 1.0, 0.0
    t = np.linspace(0.0, 1e-6, 3)
    x = analytical_solution(t, omega0, delta, x0, v0)
    v = analytical_velocity(t, omega0, delta, x0, v0)
    assert math.isclose(x[0], x0, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(v[0], v0, rel_tol=1e-9, abs_tol=1e-9)


def test_satisfies_ode_underdamped():
    omega0, delta = 2.0, 0.2
    x0, v0 = 1.0, 0.0
    t = np.linspace(0.0, 15.0, 4000)
    x = analytical_solution(t, omega0, delta, x0, v0)
    v = analytical_velocity(t, omega0, delta, x0, v0)
    a = numeric_second_derivative(t, x)
    residual = a + 2.0 * delta * v + (omega0 ** 2) * x
    interior = residual[5:-5]
    assert np.max(np.abs(interior)) < 1e-2


def test_regimes():
    assert damping_regime(2.0, 0.2) == "underdamped"
    assert damping_regime(1.5, 1.5) == "critical"
    assert damping_regime(2.0, 3.0) == "overdamped"


def test_critical_solution():
    omega0, delta = 1.5, 1.5
    x0, v0 = 1.0, 0.0
    t = np.array([0.0, 0.5, 1.0, 2.0])
    x = analytical_solution(t, omega0, delta, x0, v0)
    expected = np.exp(-delta * t) * (x0 + (v0 + delta * x0) * t)
    np.testing.assert_allclose(x, expected, rtol=1e-12, atol=1e-12)


def test_overdamped_solution():
    omega0, delta = 1.0, 2.0
    x0, v0 = 1.0, 0.0
    t = np.linspace(0.0, 5.0, 5000)
    x = analytical_solution(t, omega0, delta, x0, v0)
    v = analytical_velocity(t, omega0, delta, x0, v0)
    a = numeric_second_derivative(t, x)
    residual = a + 2.0 * delta * v + (omega0 ** 2) * x
    assert np.max(np.abs(residual[10:-10])) < 1e-3
