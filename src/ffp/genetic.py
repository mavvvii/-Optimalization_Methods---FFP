"""Algorytm genetyczny dla FFP.

Jedna klasa obslugujaca zarowno czysty GA (`baseline`), jak i warianty z
usprawnieniami (`cost`, `degree`, `dijkstra`, `hybrid`). Roznica miedzy
wariantami sprowadza sie do:
  * zasiania populacji poczatkowej heurystyka (seeding),
  * mutacji kierowanej heurystyka.

BUDZET: liczba ewaluacji funkcji celu = pop_size * generations dla KAZDEGO
wariantu (cala populacja jest oceniana w kazdym pokoleniu, bez cache'owania).
Heurystyki NIE zuzywaja dodatkowych ewaluacji – to gwarantuje rowny budzet.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from . import operators as ops
from .heuristics import BASELINE, priority_order
from .instance import FFPInstance
from .simulator import simulate


@dataclass(frozen=True)
class GAConfig:
    pop_size: int = 100
    generations: int = 100
    crossover_prob: float = 0.9
    mutation_prob: float = 0.2
    tournament_size: int = 5
    elite_size: int = 2
    crossover: str = "ordered"          # ordered | pmx | both
    mutation: str = "swap"              # swap | inversion | both
    num_firefighters: int = 2
    seed_fraction: float = 0.25         # jaka czesc populacji zasiac heurystyka
    guided_mutation_prob: float = 0.5   # p. mutacji kierowanej (tylko warianty z heur.)
    hybrid_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)  # cost, degree, dist

    @property
    def budget(self) -> int:
        return self.pop_size * self.generations


@dataclass
class GAResult:
    strategy: str
    best_cost: float                       # najlepszy uratowany koszt (cel)
    best_permutation: list[int]
    evaluations_used: int                  # rzeczywista liczba ewaluacji (kontrola budzetu)
    gen_best: list[float]                  # najlepsza ocena w kazdym pokoleniu
    gen_avg: list[float]                   # srednia ocena w kazdym pokoleniu
    gen_worst: list[float]                 # najgorsza ocena w kazdym pokoleniu


class GeneticAlgorithm:
    def __init__(self, instance: FFPInstance, config: GAConfig, strategy: str, seed: int) -> None:
        self._inst = instance
        self._cfg = config
        self._strategy = strategy
        self._rng = random.Random(seed)
        self._evaluations = 0

        # Porzadek heurystyczny + rangi (pozycja wezla w tym porzadku).
        if strategy == BASELINE:
            self._priority: list[int] | None = None
            self._priority_rank: dict[int, int] | None = None
        else:
            self._priority = priority_order(instance, strategy, config.hybrid_weights)
            self._priority_rank = {node: pos for pos, node in enumerate(self._priority)}

    # --- ocena = symulacja pozaru (jedyne miejsce zliczajace budzet) ---
    def _evaluate(self, perm: list[int]) -> float:
        self._evaluations += 1
        return simulate(self._inst, perm, self._cfg.num_firefighters).saved_cost

    def _random_perm(self) -> list[int]:
        return self._rng.sample(range(self._inst.num_nodes), k=self._inst.num_nodes)

    def _perturb(self, base: list[int], swaps: int) -> list[int]:
        out = base.copy()
        n = len(out)
        for _ in range(swaps):
            i, j = self._rng.sample(range(n), k=2)
            out[i], out[j] = out[j], out[i]
        return out

    def _initial_population(self) -> list[list[int]]:
        pop: list[list[int]] = []
        if self._priority is not None:
            n_seed = max(1, int(self._cfg.seed_fraction * self._cfg.pop_size))
            pop.append(self._priority.copy())                 # czysta heurystyka
            for _ in range(n_seed - 1):                       # lekko zaburzone kopie
                pop.append(self._perturb(self._priority, swaps=max(1, self._inst.num_nodes // 20)))
        while len(pop) < self._cfg.pop_size:
            pop.append(self._random_perm())
        return pop[: self._cfg.pop_size]

    def run(self) -> GAResult:
        cfg = self._cfg
        population = self._initial_population()

        best_cost = float("-inf")
        best_perm: list[int] = []
        gen_best: list[float] = []
        gen_avg: list[float] = []
        gen_worst: list[float] = []

        for _ in range(cfg.generations):
            scored: list[ops.Scored] = []
            for perm in population:
                fitness = self._evaluate(perm)
                scored.append((fitness, perm))

            fits = [f for f, _ in scored]
            gen_best.append(max(fits))
            gen_avg.append(sum(fits) / len(fits))
            gen_worst.append(min(fits))

            gen_top_fit, gen_top_perm = max(scored, key=lambda e: e[0])
            if gen_top_fit > best_cost:
                best_cost = gen_top_fit
                best_perm = gen_top_perm.copy()

            # --- nowa populacja: elita + potomstwo ---
            next_pop = ops.elitism(scored, cfg.elite_size)
            while len(next_pop) < cfg.pop_size:
                parent1 = ops.tournament(scored, cfg.tournament_size, self._rng)
                parent2 = ops.roulette(scored, self._rng)
                if self._rng.random() < cfg.crossover_prob:
                    child = ops.apply_crossover(cfg.crossover, parent1, parent2, self._rng)
                else:
                    child = parent1.copy()
                if self._rng.random() < cfg.mutation_prob:
                    child = ops.apply_mutation(
                        cfg.mutation, child, self._rng,
                        self._priority_rank, cfg.guided_mutation_prob,
                    )
                next_pop.append(child)
            population = next_pop

        return GAResult(
            strategy=self._strategy,
            best_cost=best_cost,
            best_permutation=best_perm,
            evaluations_used=self._evaluations,
            gen_best=gen_best,
            gen_avg=gen_avg,
            gen_worst=gen_worst,
        )
