"""Symulator rozprzestrzeniania pozaru = funkcja celu FFP.

Reprezentacja rozwiazania: PERMUTACJA wezlow = kolejnosc priorytetow ochrony.
Dekoder symuluje proces w czasie dyskretnym (konwencja Hartnella / Michalaka):

    1. Strazacy chronia (na stale) pierwsze `num_firefighters` wezlow z listy
       priorytetow, ktore nie pala sie i nie sa juz chronione.
    2. Pozar rozprzestrzenia sie na wszystkich niechronionych sasiadow plonacych
       wezlow.
    3. Powtarzamy, az pozar nie moze sie dalej rozszerzyc.

Cel (maksymalizowany): laczny KOSZT uratowanych wezlow (im wiecej, tym lepiej).
"""

from __future__ import annotations

from dataclasses import dataclass

from .instance import FFPInstance


@dataclass(frozen=True)
class SimulationResult:
    saved_cost: float
    saved_nodes: int
    burned_nodes: int
    protected_order: tuple[int, ...]
    burned_set: frozenset[int]


def simulate(instance: FFPInstance, priority: list[int], num_firefighters: int) -> SimulationResult:
    """Uruchamia symulacje pozaru dla zadanej kolejnosci priorytetow ochrony."""
    adjacency = instance.adjacency
    costs = instance.costs
    n = instance.num_nodes

    burning = [False] * n
    protected = [False] * n
    for s in instance.start_nodes:
        burning[s] = True

    fire_front = list(instance.start_nodes)
    protected_seq: list[int] = []
    ptr = 0

    while fire_front:
        placed = 0
        while placed < num_firefighters and ptr < len(priority):
            node = priority[ptr]
            ptr += 1
            if not burning[node] and not protected[node]:
                protected[node] = True
                protected_seq.append(node)
                placed += 1

        newly_burning: list[int] = []
        for u in fire_front:
            for v in adjacency[u]:
                if not burning[v] and not protected[v]:
                    burning[v] = True
                    newly_burning.append(v)

        if not newly_burning:
            break
        fire_front = newly_burning

    saved_cost = 0.0
    saved_nodes = 0
    for i in range(n):
        if not burning[i]:
            saved_cost += costs[i]
            saved_nodes += 1

    return SimulationResult(
        saved_cost=saved_cost,
        saved_nodes=saved_nodes,
        burned_nodes=n - saved_nodes,
        protected_order=tuple(protected_seq),
        burned_set=frozenset(i for i in range(n) if burning[i]),
    )
