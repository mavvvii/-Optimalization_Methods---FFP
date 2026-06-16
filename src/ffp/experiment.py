"""Orkiestracja eksperymentu FFP.

Dla kazdej instancji uruchamia czysty GA oraz warianty z usprawnieniami,
po `runs` niezaleznych powtorzen kazdy. Pilnuje, by WSZYSTKIE warianty mialy
identyczny budzet ewaluacji, liczy statystyki (best/worst/avg/std) i zapisuje
wyniki do plikow CSV.
"""

from __future__ import annotations

import csv
import statistics
from dataclasses import dataclass
from pathlib import Path

from .genetic import GAConfig, GAResult, GeneticAlgorithm
from .heuristics import ALL_STRATEGIES
from .instance import FFPInstance, load_instance
from .simulator import simulate


@dataclass
class StrategyStats:
    strategy: str
    best: float
    worst: float
    avg: float
    std: float
    budget: int
    evaluations_per_run: int
    # Usrednione po powtorzeniach krzywe zbieznosci.
    gen_best: list[float]
    gen_avg: list[float]
    gen_worst: list[float]
    final_bests: list[float]   # najlepszy wynik kazdego powtorzenia (do boxplotu)


def _average_curves(results: list[GAResult]) -> tuple[list[float], list[float], list[float]]:
    g = len(results[0].gen_best)
    avg_best = [statistics.mean(r.gen_best[i] for r in results) for i in range(g)]
    avg_avg = [statistics.mean(r.gen_avg[i] for r in results) for i in range(g)]
    avg_worst = [statistics.mean(r.gen_worst[i] for r in results) for i in range(g)]
    return avg_best, avg_avg, avg_worst


def run_strategy(instance: FFPInstance, config: GAConfig, strategy: str, runs: int, base_seed: int) -> StrategyStats:
    results: list[GAResult] = []
    for i in range(runs):
        ga = GeneticAlgorithm(instance, config, strategy, seed=base_seed + i)
        results.append(ga.run())

    finals = [r.best_cost for r in results]
    avg_best, avg_avg, avg_worst = _average_curves(results)
    evals = results[0].evaluations_used

    return StrategyStats(
        strategy=strategy,
        best=max(finals),
        worst=min(finals),
        avg=statistics.mean(finals),
        std=statistics.pstdev(finals) if len(finals) > 1 else 0.0,
        budget=config.budget,
        evaluations_per_run=evals,
        gen_best=avg_best,
        gen_avg=avg_avg,
        gen_worst=avg_worst,
        final_bests=finals,
    )


def _greedy_reference(instance: FFPInstance, config: GAConfig) -> dict[str, float]:
    """Wynik jednorazowej (zachlannej) heurystyki bez GA – punkt odniesienia."""
    from .heuristics import COST, DEGREE, DIJKSTRA, HYBRID, priority_order
    ref: dict[str, float] = {}
    for strat in (COST, DEGREE, DIJKSTRA, HYBRID):
        order = priority_order(instance, strat, config.hybrid_weights)
        ref[strat] = simulate(instance, order, config.num_firefighters).saved_cost
    return ref


@dataclass
class InstanceReport:
    instance: FFPInstance
    stats: dict[str, StrategyStats]
    greedy_reference: dict[str, float]


def run_instance(
    dataset_root: Path,
    n_obj: int,
    size: int,
    idx: int,
    config: GAConfig,
    strategies: list[str],
    runs: int,
    base_seed: int,
) -> InstanceReport:
    instance = load_instance(dataset_root, n_obj, size, idx)

    print(f"\n{'='*70}")
    print(f"  Instancja : {instance.name}")
    print(f"  Wezly     : {instance.num_nodes}   Krawedzie : {len(instance.edges)}   "
          f"Start pozaru : {list(instance.start_nodes)}")
    print(f"  Koszt calk.: {instance.total_cost:.1f}   Strazacy/ture : {config.num_firefighters}")
    print(f"  Budzet    : {config.budget} ewaluacji/run "
          f"(pop={config.pop_size} x gen={config.generations})   Powtorzen : {runs}")
    print(f"{'='*70}")

    stats: dict[str, StrategyStats] = {}
    for strat in strategies:
        s = run_strategy(instance, config, strat, runs, base_seed)
        stats[strat] = s
        print(f"  {strat:9s}: best={s.best:9.1f}  avg={s.avg:9.1f}  "
              f"worst={s.worst:9.1f}  std={s.std:7.2f}  evals/run={s.evaluations_per_run}")

    # --- KONTROLA BUDZETU: wszystkie warianty musza miec identyczny budzet ---
    budgets = {strat: s.evaluations_per_run for strat, s in stats.items()}
    unique = set(budgets.values())
    if len(unique) != 1:
        raise RuntimeError(
            f"Rozny budzet ewaluacji miedzy wariantami: {budgets}. "
            f"Wszystkie warianty musza zuzyc tyle samo ewaluacji."
        )
    if unique.pop() != config.budget:
        raise RuntimeError(
            f"Budzet rzeczywisty != deklarowany ({budgets} vs {config.budget})."
        )
    print(f"  [OK] Rowny budzet dla wszystkich wariantow: {config.budget} ewaluacji/run.")

    greedy = _greedy_reference(instance, config)
    return InstanceReport(instance=instance, stats=stats, greedy_reference=greedy)


# --------------------------------------------------------------------------- #
#  Zapis CSV
# --------------------------------------------------------------------------- #
def write_convergence_csv(report: InstanceReport, out_dir: Path) -> Path:
    """Krzywe zbieznosci (usrednione po powtorzeniach) – jeden plik na instancje."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{report.instance.name}_convergence.csv"
    strategies = list(report.stats.keys())
    generations = len(next(iter(report.stats.values())).gen_best)

    header = ["generation"]
    for s in strategies:
        header += [f"{s}_best", f"{s}_avg", f"{s}_worst"]

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for g in range(generations):
            row: list[object] = [g + 1]
            for s in strategies:
                st = report.stats[s]
                row += [f"{st.gen_best[g]:.4f}", f"{st.gen_avg[g]:.4f}", f"{st.gen_worst[g]:.4f}"]
            w.writerow(row)
    return path


def write_summary_csv(reports: list[InstanceReport], config: GAConfig, path: Path) -> Path:
    """Zbiorcze podsumowanie wszystkich instancji i wariantow."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "instance", "num_nodes", "strategy",
            "best", "avg", "worst", "std",
            "improvement_vs_baseline_%", "greedy_reference",
            "budget", "evaluations_per_run",
        ])
        for rep in reports:
            base_avg = rep.stats["baseline"].avg if "baseline" in rep.stats else None
            for strat, st in rep.stats.items():
                if base_avg is not None and base_avg != 0 and strat != "baseline":
                    impr = 100.0 * (st.avg - base_avg) / base_avg
                    impr_str = f"{impr:.2f}"
                else:
                    impr_str = "0.00" if strat == "baseline" else ""
                greedy = rep.greedy_reference.get(strat, "")
                greedy_str = f"{greedy:.2f}" if greedy != "" else ""
                w.writerow([
                    rep.instance.name, rep.instance.num_nodes, strat,
                    f"{st.best:.4f}", f"{st.avg:.4f}", f"{st.worst:.4f}", f"{st.std:.4f}",
                    impr_str, greedy_str,
                    st.budget, st.evaluations_per_run,
                ])
    return path
