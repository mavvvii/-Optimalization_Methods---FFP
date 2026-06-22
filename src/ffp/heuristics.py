"""Heurystyki problemowe = TRZY usprawnienia GA (+ hybryda).

Kazda heurystyka zwraca PORZADEK PRIORYTETOW (permutacje wezlow), w jakim
warto chronic wezly. Porzadek ten:
  * sluzy do ZASIANIA czesci populacji poczatkowej (seeding),
  * steruje MUTACJA KIEROWANA (promuje wazne wezly blizej poczatku listy).

Usprawnienia (zgodnie z zalozeniami projektu):
  1. cost     – chron najpierw wezly o NAJWYZSZYM KOSZCIE.
  2. degree   – chron najpierw wezly o NAJWIEKSZEJ LICZBIE POLACZEN (huby).
  3. dijkstra – chron najpierw wezly NAJBLIZSZE pozarowi (najkrotsza droga ognia),
                aby jak najszybciej odciac mu dostep do reszty grafu.
  4. hybrid   – (pomysl wlasny) wazona kombinacja powyzszych: cenny + dobrze
                polaczony + blisko ognia.
"""

from __future__ import annotations

from .instance import FFPInstance

BASELINE = "baseline"
COST = "cost"
DEGREE = "degree"
DIJKSTRA = "dijkstra"
HYBRID = "hybrid"

IMPROVEMENTS = (COST, DEGREE, DIJKSTRA, HYBRID)
ALL_STRATEGIES = (BASELINE,) + IMPROVEMENTS


def _normalize(values: list[float]) -> list[float]:
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def priority_order(instance: FFPInstance, strategy: str, hybrid_weights: tuple[float, float, float]) -> list[int]:
    """Buduje porzadek priorytetow ochrony dla danej heurystyki."""
    n = instance.num_nodes
    nodes = list(range(n))

    if strategy == COST:
        return sorted(nodes, key=lambda i: -instance.costs[i])

    if strategy == DEGREE:
        return sorted(nodes, key=lambda i: (-instance.degree[i], -instance.costs[i]))

    if strategy == DIJKSTRA:
        return sorted(nodes, key=lambda i: (instance.distance_from_fire[i], -instance.degree[i]))

    if strategy == HYBRID:
        w_cost, w_deg, w_dist = hybrid_weights
        cost_n = _normalize([instance.costs[i] for i in nodes])
        deg_n = _normalize([float(instance.degree[i]) for i in nodes])
        close_n = _normalize([1.0 / (1.0 + instance.distance_from_fire[i]) for i in nodes])
        score = [w_cost * cost_n[i] + w_deg * deg_n[i] + w_dist * close_n[i] for i in nodes]
        return sorted(nodes, key=lambda i: -score[i])

    raise ValueError(f"Nieznana strategia heurystyki: {strategy!r}")
