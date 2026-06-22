"""Operatory algorytmu genetycznego dzialajace na permutacjach.

  * Krzyzowanie: OX (Order Crossover) oraz PMX (Partially Matched Crossover).
  * Mutacja: swap, inversion oraz mutacja KIEROWANA heurystyka (guided).
  * Selekcja: turniejowa, ruletkowa, elityzm.

UWAGA: FFP MAKSYMALIZUJE funkcje celu (uratowany koszt), dlatego selekcja
wybiera osobniki o NAJWYZSZEJ ocenie (a ruletka wazy proporcjonalnie do oceny).
"""

from __future__ import annotations

import random

Scored = tuple[float, list[int]]


def order_crossover(p1: list[int], p2: list[int], rng: random.Random) -> list[int]:
    """OX – zachowuje fragment jednego rodzica, reszte uzupelnia z drugiego."""
    n = len(p1)
    if n < 3:
        return p1.copy()
    a, b = sorted(rng.sample(range(1, n), k=2))
    child = [-1] * n
    child[a:b] = p1[a:b]
    used = set(child[a:b])
    fill = b % n
    for value in p2[b:] + p2[:b]:
        if value in used:
            continue
        while child[fill] != -1:
            fill = (fill + 1) % n
        child[fill] = value
        fill = (fill + 1) % n
    return child


def pmx_crossover(p1: list[int], p2: list[int], rng: random.Random) -> list[int]:
    """PMX – krzyzowanie z czesciowym dopasowaniem (mapowaniem)."""
    n = len(p1)
    if n < 3:
        return p1.copy()
    a, b = sorted(rng.sample(range(1, n), k=2))
    child = [-1] * n
    child[a:b] = p1[a:b]
    seg1, seg2 = p1[a:b], p2[a:b]
    for i in list(range(0, a)) + list(range(b, n)):
        cand = p2[i]
        while cand in seg1:
            cand = seg2[seg1.index(cand)]
        child[i] = cand
    return child


def apply_crossover(mode: str, p1: list[int], p2: list[int], rng: random.Random) -> list[int]:
    if mode == "ordered":
        return order_crossover(p1, p2, rng)
    if mode == "pmx":
        return pmx_crossover(p1, p2, rng)
    return order_crossover(p1, p2, rng) if rng.random() < 0.5 else pmx_crossover(p1, p2, rng)


def swap_mutation(perm: list[int], i: int, j: int) -> list[int]:
    out = perm.copy()
    out[i], out[j] = out[j], out[i]
    return out


def inversion_mutation(perm: list[int], i: int, j: int) -> list[int]:
    out = perm.copy()
    out[i:j + 1] = reversed(out[i:j + 1])
    return out


def guided_mutation(perm: list[int], priority_rank: dict[int, int], rng: random.Random) -> list[int]:
    """Mutacja KIEROWANA heurystyka.

    Wybiera wezel o wysokim priorytecie (wg heurystyki), ktory aktualnie lezy
    pozno w permutacji, i przesuwa go blizej poczatku (wczesniejsza ochrona).
    To budzetowo neutralne 'wstrzykniecie' wiedzy o problemie.
    """
    n = len(perm)
    if n < 3:
        return perm.copy()
    best_node, best_gap, best_pos = None, -1, 0
    sample = rng.sample(range(n), k=min(8, n))
    for pos in sample:
        node = perm[pos]
        gap = pos - priority_rank.get(node, pos)
        if gap > best_gap:
            best_node, best_gap, best_pos = node, gap, pos
    if best_node is None or best_gap <= 0:
        i, j = sorted(rng.sample(range(n), k=2))
        return swap_mutation(perm, i, j)
    out = perm.copy()
    out.pop(best_pos)
    target = priority_rank.get(best_node, 0)
    target = max(0, min(target, len(out)))
    out.insert(target, best_node)
    return out


def apply_mutation(
    mode: str,
    perm: list[int],
    rng: random.Random,
    priority_rank: dict[int, int] | None,
    guided_prob: float,
) -> list[int]:
    n = len(perm)
    if n < 2:
        return perm.copy()
    if priority_rank is not None and rng.random() < guided_prob:
        return guided_mutation(perm, priority_rank, rng)
    i, j = sorted(rng.sample(range(n), k=2))
    if mode == "swap":
        return swap_mutation(perm, i, j)
    if mode == "inversion":
        return inversion_mutation(perm, i, j)
    return swap_mutation(perm, i, j) if rng.random() < 0.5 else inversion_mutation(perm, i, j)


def tournament(scored: list[Scored], size: int, rng: random.Random) -> list[int]:
    pool = rng.sample(scored, k=min(size, len(scored)))
    pool.sort(key=lambda e: e[0], reverse=True)
    return pool[0][1].copy()


def roulette(scored: list[Scored], rng: random.Random) -> list[int]:
    fitness = [s for s, _ in scored]
    lo = min(fitness)
    weights = [(f - lo) + 1e-6 for f in fitness]
    idx = rng.choices(range(len(scored)), weights=weights, k=1)[0]
    return scored[idx][1].copy()


def elitism(scored: list[Scored], elite_size: int) -> list[list[int]]:
    ordered = sorted(scored, key=lambda e: e[0], reverse=True)
    return [perm.copy() for _, perm in ordered[:elite_size]]
