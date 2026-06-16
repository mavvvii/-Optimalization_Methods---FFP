"""Wczytywanie instancji problemu strażackiego (Firefighter Problem, FFP).

Format danych (zbiory K. Michalaka, ed-ls_ffp_instances):

    <root>/2-obj/costs/0050_0002/0050_0002_01.txt
    <root>/2-obj/edges/0050_0.0500/edges_0050_0.0500_01.txt
    <root>/2-obj/starting_points/0050_0001/starting_points_0050_0001_01.txt

  * costs           – jedna linia na wezel; dla 2-obj / 3-obj jest kilka kolumn,
                      ale ZAWSZE bierzemy pierwsza (lewa) kolumne -> problem
                      jest jednokryterialny (zgodnie z zalozeniem projektu).
  * edges           – pary "u  v" (graf nieskierowany, indeksowany od 0).
  * starting_points – numery wezlow, w ktorych wybucha pozar.

Sciezki roznia sie zerowym dopelnieniem miedzy 2-obj a 3-obj, dlatego pliki
wyszukujemy globem zamiast budowac sztywna sciezke.
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

# Liczba zmiennoprzecinkowa lub calkowita (obsluga notacji naukowej).
_NUMBER = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _read_numeric_rows(path: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        found = _NUMBER.findall(line)
        if found:
            rows.append(found)
    return rows


@dataclass(frozen=True)
class FFPInstance:
    """Pojedyncza instancja FFP gotowa do symulacji."""

    name: str
    num_nodes: int
    costs: tuple[float, ...]                      # koszt (wartosc) kazdego wezla
    adjacency: tuple[frozenset[int], ...]         # lista sasiedztwa
    start_nodes: tuple[int, ...]                  # wezly, w ktorych zaczyna sie pozar
    edges: tuple[tuple[int, int], ...]            # surowa lista krawedzi (do podgladu)

    # Pola wyliczane (cache) – przydatne dla heurystyk.
    degree: tuple[int, ...] = field(default=())
    distance_from_fire: tuple[int, ...] = field(default=())

    @property
    def total_cost(self) -> float:
        return float(sum(self.costs))


def _bfs_distances(adjacency: list[set[int]], sources: list[int], n: int) -> list[int]:
    """Odleglosc (liczba krokow) od najblizszego zrodla pozaru do kazdego wezla.

    Na grafie nieskierowanym o jednostkowych wagach BFS daje dokladnie te sama
    odleglosc co algorytm Dijkstry – wykorzystujemy to w heurystyce 'dijkstra'.
    """
    INF = 10**9
    dist = [INF] * n
    q: deque[int] = deque()
    for s in sources:
        dist[s] = 0
        q.append(s)
    while q:
        u = q.popleft()
        for v in adjacency[u]:
            if dist[v] == INF:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def find_instance_files(root: Path, n_obj: int, size: int, idx: int) -> tuple[Path, Path, Path]:
    """Lokalizuje pliki costs / edges / starting_points dla zadanej instancji.

    Zwraca (costs_path, edges_path, starting_points_path).
    """
    base = root / f"{n_obj}-obj"
    if not base.is_dir():
        raise FileNotFoundError(
            f"Nie znaleziono katalogu '{base}'. Ustaw [dataset].root na folder "
            f"zawierajacy podkatalogi '2-obj/' i '3-obj/'."
        )

    # Zerowe dopelnienie rozni sie miedzy 2-obj (4 cyfry) a 3-obj (6 cyfr) –
    # probujemy obu wariantow przez glob.
    size_globs = [f"{size:04d}", f"{size:06d}", str(size)]
    idx_globs = [f"{idx:02d}", f"{idx:03d}", str(idx)]

    def _first_match(subdir: str, pattern_builder) -> Path:
        for sz in size_globs:
            for ix in idx_globs:
                for hit in sorted((base / subdir).glob(pattern_builder(sz, ix))):
                    return hit
        raise FileNotFoundError(
            f"Brak pliku w '{base / subdir}' dla size={size}, idx={idx}."
        )

    costs = _first_match("costs", lambda sz, ix: f"*/{sz}_*_{ix}.txt")
    edges = _first_match("edges", lambda sz, ix: f"*/edges_{sz}_*_{ix}.txt")
    starts = _first_match("starting_points", lambda sz, ix: f"*/starting_points_{sz}_*_{ix}.txt")
    return costs, edges, starts


def load_instance(root: Path, n_obj: int, size: int, idx: int) -> FFPInstance:
    """Wczytuje instancje FFP i wstepnie wylicza stopnie wezlow oraz odleglosci."""
    costs_path, edges_path, starts_path = find_instance_files(root, n_obj, size, idx)

    # --- koszty (bierzemy tylko pierwsza kolumne) ---
    cost_rows = _read_numeric_rows(costs_path)
    costs = [float(r[0]) for r in cost_rows]
    n = len(costs)

    # --- krawedzie ---
    edge_rows = _read_numeric_rows(edges_path)
    edges = [(int(float(a)), int(float(b))) for a, b in edge_rows]

    adjacency: list[set[int]] = [set() for _ in range(n)]
    for a, b in edges:
        if a == b:
            continue
        adjacency[a].add(b)
        adjacency[b].add(a)

    # --- punkty startowe ---
    start_tokens = _NUMBER.findall(starts_path.read_text(encoding="utf-8"))
    start_nodes = [int(float(t)) for t in start_tokens]
    if not start_nodes:
        raise ValueError(f"Plik {starts_path} nie zawiera punktu startowego.")

    degree = [len(adjacency[i]) for i in range(n)]
    distance = _bfs_distances(adjacency, start_nodes, n)

    name = f"{n_obj}obj_{size:04d}_{idx:02d}"
    return FFPInstance(
        name=name,
        num_nodes=n,
        costs=tuple(costs),
        adjacency=tuple(frozenset(s) for s in adjacency),
        start_nodes=tuple(start_nodes),
        edges=tuple(edges),
        degree=tuple(degree),
        distance_from_fire=tuple(distance),
    )
