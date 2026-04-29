from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "lab1" / "data"
IMAGES_DIR = ROOT / "reports" / "lab1" / "images"
TABLES_DIR = ROOT / "reports" / "lab1" / "tables"


def read_csv(path: Path) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            normalized = {k.lstrip("\ufeff"): v for k, v in row.items()}
            rows.append(
                {
                    "method": normalized["method"],
                    "x": float(normalized["x"]),
                    "approx": float(normalized["approx"]),
                    "exact": float(normalized["exact"]),
                    "abs_error": float(normalized["abs_error"]),
                    "rel_error": float(normalized["rel_error"]),
                }
            )
    return rows


def write_error_table(results: dict[str, list[dict[str, float | str]]]) -> None:
    lines = [
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Метод & $\Delta_l$ & $\delta_l$ \\",
        r"\midrule",
    ]
    for method, rows in results.items():
        max_abs = max(r["abs_error"] for r in rows)
        max_rel = max(r["rel_error"] for r in rows)
        lines.append(f"{method} & {max_abs:.6e} & {max_rel:.6e} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (TABLES_DIR / "errors.tex").write_text("\n".join(lines), encoding="utf-8")


def write_points_table(results: dict[str, list[dict[str, float | str]]]) -> None:
    method_order = ["Euler", "Heun", "RungeKutta4"]
    anchor = results[method_order[0]]
    n = len(anchor)
    sample_indices = sorted({0, n // 4, n // 2, 3 * n // 4, n - 1})
    lines = [
        r"\begin{tabular}{cccccc}",
        r"\toprule",
        r"$x_k$ & $\varphi(x_k)$ & Euler & Heun & RK4 & $|e_{RK4}|$ \\",
        r"\midrule",
    ]
    for idx in sample_indices:
        x = anchor[idx]["x"]
        exact = anchor[idx]["exact"]
        euler = results["Euler"][idx]["approx"]
        heun = results["Heun"][idx]["approx"]
        rk4 = results["RungeKutta4"][idx]["approx"]
        err = abs(exact - rk4)
        lines.append(
            f"{x:.3f} & {exact:.6f} & {euler:.6f} & {heun:.6f} & {rk4:.6f} & {err:.3e} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (TABLES_DIR / "sample_points.tex").write_text("\n".join(lines), encoding="utf-8")


def plot_solutions(results: dict[str, list[dict[str, float | str]]]) -> None:
    x = [r["x"] for r in results["RungeKutta4"]]
    exact = [r["exact"] for r in results["RungeKutta4"]]
    plt.figure(figsize=(10, 6))
    plt.plot(x, exact, "k--", linewidth=2.0, label="Точное решение")
    for method, rows in results.items():
        y = [r["approx"] for r in rows]
        plt.plot(x, y, label=method)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("ЛР1: точное и численные решения")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "solutions.png", dpi=220)
    plt.close()


def plot_errors(results: dict[str, list[dict[str, float | str]]]) -> None:
    x = [r["x"] for r in results["RungeKutta4"]]
    plt.figure(figsize=(10, 6))
    for method, rows in results.items():
        err = [r["abs_error"] for r in rows]
        plt.plot(x, err, label=method)
    plt.yscale("log")
    plt.xlabel("x")
    plt.ylabel("|phi - phi_tilde|")
    plt.title("ЛР1: абсолютные ошибки (логарифмическая шкала)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "errors.png", dpi=220)
    plt.close()


def main() -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "Euler": read_csv(DATA_DIR / "euler.csv"),
        "Heun": read_csv(DATA_DIR / "heun.csv"),
        "RungeKutta4": read_csv(DATA_DIR / "rungekutta4.csv"),
    }
    write_error_table(results)
    write_points_table(results)
    plot_solutions(results)
    plot_errors(results)


if __name__ == "__main__":
    main()
