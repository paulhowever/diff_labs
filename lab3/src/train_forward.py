from dataclasses import dataclass
from typing import List

import torch

from .model import PINN, first_derivative, second_derivative


@dataclass
class ForwardConfig:
    omega0: float
    delta: float
    x0: float
    v0: float
    T: float
    n_collocation: int = 800
    epochs: int = 20000
    lr: float = 1e-3
    lambda_ic_x: float = 200.0
    lambda_ic_v: float = 200.0
    lambda_phys: float = 1.0
    hidden_layers: int = 5
    hidden_size: int = 64
    log_every: int = 500


def train_forward(cfg: ForwardConfig, seed: int = 42,
                  device: str = "cpu"):
    torch.manual_seed(seed)
    omega = max((cfg.omega0 ** 2 - cfg.delta ** 2), 0.0) ** 0.5
    n_fourier = 4
    max_freq = max(omega, 1.0) * 1.5
    model = PINN(cfg.hidden_layers, cfg.hidden_size, t_scale=cfg.T,
                 n_fourier=n_fourier,
                 fourier_max_freq=max_freq).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=4000,
                                                gamma=0.5)

    t_coll = torch.linspace(0.0, cfg.T, cfg.n_collocation, device=device)
    t_coll = t_coll.view(-1, 1).requires_grad_(True)
    t0 = torch.zeros(1, 1, device=device, requires_grad=True)

    history: List[dict] = []
    for epoch in range(cfg.epochs):
        optim.zero_grad()
        x_pred = model(t_coll)
        dx = first_derivative(x_pred, t_coll)
        d2x = second_derivative(x_pred, t_coll)
        residual = d2x + 2.0 * cfg.delta * dx + (cfg.omega0 ** 2) * x_pred
        loss_phys = torch.mean(residual ** 2)

        x0_pred = model(t0)
        dx0_pred = first_derivative(x0_pred, t0)
        loss_ic_x = (x0_pred - cfg.x0) ** 2
        loss_ic_v = (dx0_pred - cfg.v0) ** 2

        loss = (cfg.lambda_phys * loss_phys
                + cfg.lambda_ic_x * loss_ic_x.squeeze()
                + cfg.lambda_ic_v * loss_ic_v.squeeze())
        loss.backward()
        optim.step()
        scheduler.step()

        if epoch % cfg.log_every == 0 or epoch == cfg.epochs - 1:
            history.append({
                "epoch": epoch,
                "loss": float(loss.item()),
                "loss_phys": float(loss_phys.item()),
                "loss_ic_x": float(loss_ic_x.item()),
                "loss_ic_v": float(loss_ic_v.item()),
            })

    lbfgs = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=800,
                              history_size=60,
                              tolerance_grad=1e-10, tolerance_change=1e-14,
                              line_search_fn="strong_wolfe")

    def closure():
        lbfgs.zero_grad()
        x_pred = model(t_coll)
        dx = first_derivative(x_pred, t_coll)
        d2x = second_derivative(x_pred, t_coll)
        residual = d2x + 2.0 * cfg.delta * dx + (cfg.omega0 ** 2) * x_pred
        loss_phys = torch.mean(residual ** 2)
        x0_pred = model(t0)
        dx0_pred = first_derivative(x0_pred, t0)
        loss_ic_x = (x0_pred - cfg.x0) ** 2
        loss_ic_v = (dx0_pred - cfg.v0) ** 2
        loss = (cfg.lambda_phys * loss_phys
                + cfg.lambda_ic_x * loss_ic_x.squeeze()
                + cfg.lambda_ic_v * loss_ic_v.squeeze())
        loss.backward()
        return loss

    final_loss = lbfgs.step(closure)
    history.append({
        "epoch": cfg.epochs,
        "loss": float(final_loss.item()),
        "loss_phys": float(final_loss.item()),
        "loss_ic_x": float("nan"),
        "loss_ic_v": float("nan"),
    })

    return model, history
