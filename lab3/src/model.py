import torch
import torch.nn as nn


class PINN(nn.Module):
    def __init__(self, hidden_layers: int = 4, hidden_size: int = 48,
                 t_scale: float = 1.0, n_fourier: int = 0,
                 fourier_max_freq: float = 1.0):
        super().__init__()
        self.t_scale = t_scale
        self.n_fourier = n_fourier
        if n_fourier > 0:
            freqs = torch.linspace(1.0, fourier_max_freq, n_fourier)
            self.register_buffer("freqs", freqs)
            in_dim = 1 + 2 * n_fourier
        else:
            in_dim = 1
        layers = [nn.Linear(in_dim, hidden_size), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.Tanh()]
        layers += [nn.Linear(hidden_size, 1)]
        self.net = nn.Sequential(*layers)
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def _features(self, t: torch.Tensor) -> torch.Tensor:
        t_norm = 2.0 * t / self.t_scale - 1.0
        if self.n_fourier == 0:
            return t_norm
        ang = t * self.freqs
        return torch.cat([t_norm, torch.sin(ang), torch.cos(ang)], dim=-1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(self._features(t))


class HardICPINN(nn.Module):
    def __init__(self, x0: float, v0: float, hidden_layers: int = 4,
                 hidden_size: int = 48, t_scale: float = 1.0):
        super().__init__()
        self.x0 = x0
        self.v0 = v0
        self.t_scale = t_scale
        self.inner = PINN(hidden_layers, hidden_size, t_scale=t_scale)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        n = self.inner(t)
        return self.x0 + self.v0 * t + t * t * n


def first_derivative(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(x, t, grad_outputs=torch.ones_like(x),
                               create_graph=True, retain_graph=True)[0]


def second_derivative(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    dx = first_derivative(x, t)
    return torch.autograd.grad(dx, t, grad_outputs=torch.ones_like(dx),
                               create_graph=True, retain_graph=True)[0]
