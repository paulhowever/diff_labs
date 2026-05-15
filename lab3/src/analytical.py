import math

import numpy as np


def damping_regime(omega0: float, delta: float) -> str:
    if abs(delta - omega0) < 1e-12:
        return "critical"
    if delta < omega0:
        return "underdamped"
    return "overdamped"


def analytical_solution(t: np.ndarray, omega0: float, delta: float,
                        x0: float, v0: float) -> np.ndarray:
    regime = damping_regime(omega0, delta)
    if regime == "underdamped":
        omega = math.sqrt(omega0 * omega0 - delta * delta)
        a = x0
        b = (v0 + delta * x0) / omega
        return np.exp(-delta * t) * (a * np.cos(omega * t) + b * np.sin(omega * t))
    if regime == "critical":
        a = x0
        b = v0 + delta * x0
        return np.exp(-delta * t) * (a + b * t)
    gamma = math.sqrt(delta * delta - omega0 * omega0)
    r1 = -delta + gamma
    r2 = -delta - gamma
    c1 = (v0 - r2 * x0) / (r1 - r2)
    c2 = x0 - c1
    return c1 * np.exp(r1 * t) + c2 * np.exp(r2 * t)


def analytical_velocity(t: np.ndarray, omega0: float, delta: float,
                        x0: float, v0: float) -> np.ndarray:
    regime = damping_regime(omega0, delta)
    if regime == "underdamped":
        omega = math.sqrt(omega0 * omega0 - delta * delta)
        a = x0
        b = (v0 + delta * x0) / omega
        e = np.exp(-delta * t)
        c = np.cos(omega * t)
        s = np.sin(omega * t)
        return e * (-delta * (a * c + b * s) + (-a * omega * s + b * omega * c))
    if regime == "critical":
        a = x0
        b = v0 + delta * x0
        e = np.exp(-delta * t)
        return e * (b - delta * (a + b * t))
    gamma = math.sqrt(delta * delta - omega0 * omega0)
    r1 = -delta + gamma
    r2 = -delta - gamma
    c1 = (v0 - r2 * x0) / (r1 - r2)
    c2 = x0 - c1
    return c1 * r1 * np.exp(r1 * t) + c2 * r2 * np.exp(r2 * t)
