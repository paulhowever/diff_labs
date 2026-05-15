import numpy as np

from .analytical import analytical_solution


def generate_noisy_data(omega0: float, delta: float, x0: float, v0: float,
                        T: float, n_points: int = 80,
                        noise_std: float = 0.05, seed: int = 42):
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, T, n_points)
    x_true = analytical_solution(t, omega0, delta, x0, v0)
    y = x_true + rng.normal(0.0, noise_std, size=t.shape)
    return t, x_true, y
