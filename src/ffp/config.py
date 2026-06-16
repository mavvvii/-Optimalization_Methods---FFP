"""Wczytywanie konfiguracji eksperymentu z pliku TOML (tomllib – biblioteka standardowa)."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from .genetic import GAConfig
from .heuristics import ALL_STRATEGIES


@dataclass(frozen=True)
class InstanceSpec:
    size: int
    idx: int


@dataclass(frozen=True)
class ExperimentConfig:
    dataset_root: Path
    n_obj: int
    instances: list[InstanceSpec]
    strategies: list[str]
    runs: int
    base_seed: int
    results_dir: Path
    ga: GAConfig


def load_config(path: Path) -> ExperimentConfig:
    with path.open("rb") as f:
        data = tomllib.load(f)

    ds = data.get("dataset", {})
    exp = data.get("experiment", {})
    ga = data.get("ga", {})

    strategies = exp.get("strategies", list(ALL_STRATEGIES))
    unknown = [s for s in strategies if s not in ALL_STRATEGIES]
    if unknown:
        raise ValueError(f"Nieznane strategie w configu: {unknown}. Dozwolone: {list(ALL_STRATEGIES)}")

    instances = [InstanceSpec(size=int(i["size"]), idx=int(i.get("idx", 1)))
                 for i in data.get("instances", [])]
    if not instances:
        raise ValueError("Config nie zawiera zadnej instancji (sekcje [[instances]]).")

    weights = ga.get("hybrid_weights", [1.0, 1.0, 1.0])
    ga_config = GAConfig(
        pop_size=int(ga.get("pop_size", 100)),
        generations=int(ga.get("generations", 100)),
        crossover_prob=float(ga.get("crossover_prob", 0.9)),
        mutation_prob=float(ga.get("mutation_prob", 0.2)),
        tournament_size=int(ga.get("tournament_size", 5)),
        elite_size=int(ga.get("elite_size", 2)),
        crossover=str(ga.get("crossover", "ordered")),
        mutation=str(ga.get("mutation", "swap")),
        num_firefighters=int(ga.get("num_firefighters", 2)),
        seed_fraction=float(ga.get("seed_fraction", 0.25)),
        guided_mutation_prob=float(ga.get("guided_mutation_prob", 0.5)),
        hybrid_weights=(float(weights[0]), float(weights[1]), float(weights[2])),
    )

    return ExperimentConfig(
        dataset_root=Path(ds.get("root", "instances")),
        n_obj=int(ds.get("n_obj", 2)),
        instances=instances,
        strategies=strategies,
        runs=int(exp.get("runs", 10)),
        base_seed=int(exp.get("base_seed", 42)),
        results_dir=Path(exp.get("results_dir", "results")),
        ga=ga_config,
    )
