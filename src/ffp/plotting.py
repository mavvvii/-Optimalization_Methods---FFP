"""Wykresy generowane automatycznie z wynikow eksperymentu (matplotlib).

Dla kazdej instancji:
  * krzywe zbieznosci (best uratowany koszt vs pokolenie) – wszystkie warianty,
  * slupki koncowej jakosci (avg +/- std) z linia odniesienia heurystyki zachlannej,
  * boxplot rozkladu najlepszych wynikow z powtorzen.
Zbiorczo:
  * procentowa poprawa wariantow wzgledem baseline, usredniona po instancjach.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend bez GUI – zapis do plikow
import matplotlib.pyplot as plt

from .experiment import InstanceReport

_COLORS = {
    "baseline": "#555555",
    "cost": "#1f77b4",
    "degree": "#2ca02c",
    "dijkstra": "#d62728",
    "hybrid": "#9467bd",
}


def _color(strategy: str) -> str:
    return _COLORS.get(strategy, "#888888")


def plot_convergence(report: InstanceReport, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for strat, st in report.stats.items():
        xs = range(1, len(st.gen_best) + 1)
        ax.plot(xs, st.gen_best, label=strat, color=_color(strat), linewidth=2)
    ax.set_xlabel("Pokolenie")
    ax.set_ylabel("Najlepszy uratowany koszt (srednia z powtorzen)")
    ax.set_title(f"Zbieznosc GA – {report.instance.name}")
    ax.legend(title="Wariant")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / f"{report.instance.name}_convergence.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_final_bars(report: InstanceReport, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    strategies = list(report.stats.keys())
    avgs = [report.stats[s].avg for s in strategies]
    stds = [report.stats[s].std for s in strategies]
    colors = [_color(s) for s in strategies]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(strategies, avgs, yerr=stds, capsize=5, color=colors, alpha=0.85)

    # Linie odniesienia – heurystyki zachlanne (bez GA).
    for s in strategies:
        ref = report.greedy_reference.get(s)
        if ref is not None:
            xpos = strategies.index(s)
            ax.hlines(ref, xpos - 0.4, xpos + 0.4, colors="black", linestyles="dashed", linewidth=1)

    for bar, val in zip(bars, avgs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.0f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Sredni uratowany koszt (best z runu)")
    ax.set_title(f"Jakosc koncowa – {report.instance.name}\n(linia przerywana = heurystyka zachlanna bez GA)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = out_dir / f"{report.instance.name}_final_bars.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_boxplot(report: InstanceReport, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    strategies = list(report.stats.keys())
    data = [report.stats[s].final_bests for s in strategies]

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data, patch_artist=True)
    ax.set_xticks(range(1, len(strategies) + 1))
    ax.set_xticklabels(strategies)
    for patch, s in zip(bp["boxes"], strategies):
        patch.set_facecolor(_color(s))
        patch.set_alpha(0.6)
    ax.set_ylabel("Najlepszy uratowany koszt (per run)")
    ax.set_title(f"Rozklad wynikow z powtorzen – {report.instance.name}")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = out_dir / f"{report.instance.name}_boxplot.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_overall_improvement(reports: list[InstanceReport], out_dir: Path) -> Path | None:
    """Srednia procentowa poprawa kazdego usprawnienia wzgledem baseline."""
    out_dir.mkdir(parents=True, exist_ok=True)
    improvements: dict[str, list[float]] = {}
    for rep in reports:
        if "baseline" not in rep.stats:
            continue
        base = rep.stats["baseline"].avg
        if base == 0:
            continue
        for strat, st in rep.stats.items():
            if strat == "baseline":
                continue
            improvements.setdefault(strat, []).append(100.0 * (st.avg - base) / base)

    if not improvements:
        return None

    strategies = list(improvements.keys())
    means = [sum(v) / len(v) for v in improvements.values()]
    colors = [_color(s) for s in strategies]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(strategies, means, color=colors, alpha=0.85)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:+.1f}%", ha="center",
                va="bottom" if val >= 0 else "top", fontsize=9)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Srednia poprawa wzgledem baseline [%]")
    ax.set_title("Zbiorcza poprawa usprawnien wzgledem czystego GA\n(srednia po wszystkich instancjach)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = out_dir / "overall_improvement.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def generate_all_plots(reports: list[InstanceReport], out_root: Path) -> list[Path]:
    paths: list[Path] = []
    for rep in reports:
        inst_dir = out_root / rep.instance.name
        paths.append(plot_convergence(rep, inst_dir))
        paths.append(plot_final_bars(rep, inst_dir))
        paths.append(plot_boxplot(rep, inst_dir))
    overall = plot_overall_improvement(reports, out_root)
    if overall is not None:
        paths.append(overall)
    return paths
