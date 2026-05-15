from dataclasses import dataclass
from typing import List

import numpy as np
import torch

from .model import PINN, first_derivative, second_derivative


@dataclass
class InverseConfig:
    omega0: float
    x0: float
    v0: float
    T: float
    delta_init: float = 1.0
    n_collocation: int = 400
    epochs: int = 15000
    lr: float = 1e-3
    lambda_ic_x: float = 100.0
    lambda_ic_v: float = 100.0
    lambda_phys: float = 1.0
    lambda_data: float = 100.0
    hidden_layers: int = 4
    hidden_size: int = 48
    log_every: int = 500


def train_inverse(cfg: InverseConfig, t_data: np.ndarray, y_data: np.ndarray,
                  seed: int = 42, device: str = "cpu"):
    torch.manual_seed(seed)
    model = PINN(cfg.hidden_layers, cfg.hidden_size,
                 t_scale=cfg.T, n_fourier=4,
                 fourier_max_freq=max(cfg.omega0, 1.0) * 1.5).to(device)
    delta_param = torch.nn.Parameter(torch.tensor(cfg.delta_init,
                                                  dtype=torch.float32,
                                                  device=device))
    params = list(model.parameters()) + [delta_param]
    optim = torch.optim.Adam(params, lr=cfg.lr)

    t_coll = torch.linspace(0.0, cfg.T, cfg.n_collocation, device=device)
    t_coll = t_coll.view(-1, 1).requires_grad_(True)
    t0 = torch.zeros(1, 1, device=device, requires_grad=True)

    t_d = torch.tensor(t_data, dtype=torch.float32,
                       device=device).view(-1, 1)
    y_d = torch.tensor(y_data, dtype=torch.float32,
                       device=device).view(-1, 1)

    history: List[dict] = []
    for epoch in range(cfg.epochs):
        optim.zero_grad()
        x_pred = model(t_coll)
        dx = first_derivative(x_pred, t_coll)
        d2x = second_derivative(x_pred, t_coll)
        residual = d2x + 2.0 * delta_param * dx + (cfg.omega0 ** 2) * x_pred
        loss_phys = torch.mean(residual ** 2)

        x0_pred = model(t0)
        dx0_pred = first_derivative(x0_pred, t0)
        loss_ic_x = (x0_pred - cfg.x0) ** 2
        loss_ic_v = (dx0_pred - cfg.v0) ** 2

        x_at_data = model(t_d)
        loss_data = torch.mean((x_at_data - y_d) ** 2)

        loss = (cfg.lambda_phys * loss_phys
                + cfg.lambda_ic_x * loss_ic_x.squeeze()
                + cfg.lambda_ic_v * loss_ic_v.squeeze()
                + cfg.lambda_data * loss_data)
        loss.backward()
        optim.step()

        if epoch % cfg.log_every == 0 or epoch == cfg.epochs - 1:
            history.append({
                "epoch": epoch,
                "loss": float(loss.item()),
                "loss_phys": float(loss_phys.item()),
                "loss_data": float(loss_data.item()),
                "delta": float(delta_param.item()),
            })

    return model, float(delta_param.item()), history
