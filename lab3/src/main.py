import math
from pathlib import Path

import numpy as np
import torch

from .analytical import analytical_solution, damping_regime
from .config import NOISE_STD, SEED, VARIANT
from .data import generate_noisy_data
from .plots import (plot_delta_trajectory, plot_forward, plot_loss,
                    plot_noisy, save_csv, save_history_csv)
from .train_forward import ForwardConfig, train_forward
from .train_inverse import InverseConfig, train_inverse


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"


def run_forward(device: str = "cpu"):
    cfg = ForwardConfig(
        omega0=VARIANT.omega0,
        delta=VARIANT.delta,
        x0=VARIANT.x0,
        v0=VARIANT.v0,
        T=VARIANT.T,
    )
    model, history = train_forward(cfg, seed=SEED, device=device)

    n_eval = 600
    t_eval = np.linspace(0.0, VARIANT.T, n_eval)
    with torch.no_grad():
        t_torch = torch.tensor(t_eval, dtype=torch.float32,
                               device=device).view(-1, 1)
        x_pinn = model(t_torch).cpu().numpy().ravel()
    x_true = analytical_solution(t_eval, VARIANT.omega0, VARIANT.delta,
                                 VARIANT.x0, VARIANT.v0)
    err = np.abs(x_pinn - x_true)
    metrics = {
        "max_abs_error": float(err.max()),
        "rmse": float(np.sqrt(np.mean((x_pinn - x_true) ** 2))),
        "mean_abs_error": float(err.mean()),
    }

    save_csv(DATA_DIR / "forward_solution.csv",
             ["t", "x_analytical", "x_pinn", "abs_error"],
             list(zip(t_eval.tolist(), x_true.tolist(),
                      x_pinn.tolist(), err.tolist())))
    save_history_csv(DATA_DIR / "forward_history.csv", history)
    save_csv(DATA_DIR / "forward_metrics.csv",
             ["metric", "value"],
             [[k, v] for k, v in metrics.items()])

    plot_forward(t_eval, x_true, x_pinn, FIG_DIR / "forward_solution.png",
                 title=fr"Вариант 6: $\omega_0={VARIANT.omega0}$, "
                       fr"$\delta={VARIANT.delta}$, T={VARIANT.T}")
    plot_loss(history, FIG_DIR / "forward_loss.png",
              keys=("loss", "loss_phys", "loss_ic_x", "loss_ic_v"))

    return metrics


def run_inverse(device: str = "cpu"):
    t_data, x_true, y_noisy = generate_noisy_data(
        omega0=VARIANT.omega0,
        delta=VARIANT.delta,
        x0=VARIANT.x0,
        v0=VARIANT.v0,
        T=VARIANT.T,
        n_points=80,
        noise_std=NOISE_STD,
        seed=SEED,
    )
    save_csv(DATA_DIR / "noisy_data.csv",
             ["t", "x_true", "y_noisy"],
             list(zip(t_data.tolist(), x_true.tolist(), y_noisy.tolist())))

    cfg = InverseConfig(
        omega0=VARIANT.omega0,
        x0=VARIANT.x0,
        v0=VARIANT.v0,
        T=VARIANT.T,
        delta_init=1.0,
    )
    model, delta_est, history = train_inverse(cfg, t_data, y_noisy,
                                              seed=SEED, device=device)

    n_eval = 600
    t_eval = np.linspace(0.0, VARIANT.T, n_eval)
    with torch.no_grad():
        t_torch = torch.tensor(t_eval, dtype=torch.float32,
                               device=device).view(-1, 1)
        x_pinn = model(t_torch).cpu().numpy().ravel()
    x_true_dense = analytical_solution(t_eval, VARIANT.omega0,
                                       VARIANT.delta, VARIANT.x0, VARIANT.v0)

    save_csv(DATA_DIR / "inverse_solution.csv",
             ["t", "x_true", "x_pinn"],
             list(zip(t_eval.tolist(), x_true_dense.tolist(),
                      x_pinn.tolist())))
    save_history_csv(DATA_DIR / "inverse_history.csv", history)

    abs_err = abs(delta_est - VARIANT.delta)
    rel_err = abs_err / VARIANT.delta if VARIANT.delta != 0 else float("nan")
    save_csv(DATA_DIR / "inverse_metrics.csv",
             ["metric", "value"],
             [
                 ["delta_true", VARIANT.delta],
                 ["delta_estimated", delta_est],
                 ["abs_error", abs_err],
                 ["rel_error", rel_err],
             ])

    plot_noisy(t_data, x_true, y_noisy, t_eval, x_pinn,
               FIG_DIR / "inverse_fit.png",
               title=fr"Идентификация $\delta$: оценка = {delta_est:.4f}, "
                     fr"истинное = {VARIANT.delta}")
    plot_delta_trajectory(history, VARIANT.delta,
                          FIG_DIR / "inverse_delta.png")
    plot_loss(history, FIG_DIR / "inverse_loss.png",
              keys=("loss", "loss_phys", "loss_data"))

    return {
        "delta_true": VARIANT.delta,
        "delta_estimated": delta_est,
        "abs_error": abs_err,
        "rel_error": rel_err,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    regime = damping_regime(VARIANT.omega0, VARIANT.delta)
    omega = math.sqrt(max(VARIANT.omega0 ** 2 - VARIANT.delta ** 2, 0.0))
    print(f"Variant {VARIANT.number}: omega0={VARIANT.omega0}, "
          f"delta={VARIANT.delta}, x0={VARIANT.x0}, v0={VARIANT.v0}, "
          f"T={VARIANT.T}, estimated={VARIANT.estimated}")
    print(f"Regime: {regime}; damped frequency omega = {omega:.6f}")
    print(f"Device: {device}")

    print("\n[1/2] Forward task")
    fwd = run_forward(device=device)
    for k, v in fwd.items():
        print(f"  {k}: {v:.6e}")

    print("\n[2/2] Inverse task")
    inv = run_inverse(device=device)
    for k, v in inv.items():
        print(f"  {k}: {v:.6f}")


if __name__ == "__main__":
    main()
