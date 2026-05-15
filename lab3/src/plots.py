import csv
from pathlib import Path
from typing import List, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in rows:
            w.writerow(row)


def save_history_csv(path: Path, history: List[dict]) -> None:
    if not history:
        return
    keys = list(history[0].keys())
    rows = [[h[k] for k in keys] for h in history]
    save_csv(path, keys, rows)


def plot_forward(t: np.ndarray, x_analytical: np.ndarray,
                 x_pinn: np.ndarray, out_path: Path,
                 title: str = "PINN vs аналитическое решение") -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    axes[0].plot(t, x_analytical, label="аналитическое", lw=2)
    axes[0].plot(t, x_pinn, "--", label="PINN", lw=2)
    axes[0].set_ylabel("x(t)")
    axes[0].set_title(title)
    axes[0].grid(True, alpha=0.4)
    axes[0].legend()

    axes[1].plot(t, x_pinn - x_analytical, color="crimson")
    axes[1].axhline(0.0, color="k", lw=0.5)
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("ошибка")
    axes[1].grid(True, alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_noisy(t: np.ndarray, x_true: np.ndarray, y_noisy: np.ndarray,
               t_pinn: np.ndarray, x_pinn: np.ndarray, out_path: Path,
               title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(t, x_true, label="истинное", lw=2)
    ax.scatter(t, y_noisy, s=15, color="gray", alpha=0.7,
               label="зашумленные данные")
    ax.plot(t_pinn, x_pinn, "--", color="crimson", lw=2, label="PINN")
    ax.set_xlabel("t")
    ax.set_ylabel("x(t)")
    ax.set_title(title)
    ax.grid(True, alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_delta_trajectory(history: List[dict], true_delta: float,
                          out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [h["epoch"] for h in history]
    deltas = [h["delta"] for h in history]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(epochs, deltas, label=r"$\delta$ (PINN)")
    ax.axhline(true_delta, color="k", ls="--",
               label=fr"истинное $\delta={true_delta}$")
    ax.set_xlabel("эпоха")
    ax.set_ylabel(r"$\delta$")
    ax.set_title("Сходимость оценки δ")
    ax.grid(True, alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_loss(history: List[dict], out_path: Path,
              keys: Sequence[str] = ("loss", "loss_phys")) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4))
    epochs = [h["epoch"] for h in history]
    for k in keys:
        if k in history[0]:
            ax.plot(epochs, [h[k] for h in history], label=k)
    ax.set_yscale("log")
    ax.set_xlabel("эпоха")
    ax.set_ylabel("loss")
    ax.set_title("Динамика функции потерь")
    ax.grid(True, alpha=0.4, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
