from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "lab2" / "data" / "rungekutta4.csv"
IMAGES_DIR = ROOT / "reports" / "lab2" / "images"
TABLES_DIR = ROOT / "reports" / "lab2" / "tables"


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


def write_error_table(rows: list[dict[str, float | str]]) -> None:
    max_abs = max(r["abs_error"] for r in rows)
    max_rel = max(r["rel_error"] for r in rows)
    lines = [
        r"\begin{tabular}{lc}",
        r"\toprule",
        r"Параметр & Значение \\",
        r"\midrule",
        f"$\\Delta$ & {max_abs:.6e} \\\\",
        f"$\\delta$ & {max_rel:.6e} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    (TABLES_DIR / "errors.tex").write_text("\n".join(lines), encoding="utf-8")


def write_points_table(rows: list[dict[str, float | str]]) -> None:
    n = len(rows)
    sample_indices = sorted({0, n // 4, n // 2, 3 * n // 4, n - 1})
    lines = [
        r"\begin{tabular}{ccccc}",
        r"\toprule",
        r"$x_k$ & $\varphi(x_k)$ & $\tilde\varphi(x_k)$ & $|e_k|$ & $\varepsilon_k$ \\",
        r"\midrule",
    ]
    for idx in sample_indices:
        r = rows[idx]
        lines.append(
            f"{r['x']:.3f} & {r['exact']:.6f} & {r['approx']:.6f} & {r['abs_error']:.3e} & {r['rel_error']:.3e} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (TABLES_DIR / "sample_points.tex").write_text("\n".join(lines), encoding="utf-8")


def plot_solutions(rows: list[dict[str, float | str]]) -> None:
    x = [r["x"] for r in rows]
    approx = [r["approx"] for r in rows]
    exact = [r["exact"] for r in rows]
    plt.figure(figsize=(10, 6))
    plt.plot(x, exact, "k--", linewidth=2, label="Точное решение")
    plt.plot(x, approx, label="RK4")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("ЛР2: точное и численное решения")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "solutions.png", dpi=220)
    plt.close()


def plot_error(rows: list[dict[str, float | str]]) -> None:
    x = [r["x"] for r in rows]
    err = [r["abs_error"] for r in rows]
    plt.figure(figsize=(10, 6))
    plt.plot(x, err, color="tab:red", label="|phi - phi_tilde|")
    plt.yscale("log")
    plt.xlabel("x")
    plt.ylabel("Ошибка")
    plt.title("ЛР2: абсолютная ошибка RK4")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "errors.png", dpi=220)
    plt.close()


def main() -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_csv(DATA_PATH)
    write_error_table(rows)
    write_points_table(rows)
    plot_solutions(rows)
    plot_error(rows)


if __name__ == "__main__":
    main()
