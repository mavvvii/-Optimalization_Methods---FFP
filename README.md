# FFP-Genetic — algorytm genetyczny dla problemu strażackiego

Projekt akademicki: rozwiązanie **problemu strażackiego** (*Firefighter Problem*, FFP)
za pomocą **algorytmu genetycznego** (GA) oraz porównanie czystego GA z **trzema
usprawnieniami** opartymi na wiedzy o problemie (+ wariant hybrydowy).

Bazuje na sformułowaniu i zbiorach instancji K. Michalaka
(`ed-ls_ffp_instances`, <https://krzysztof-michalak.pl/research_graph.html>).

---

## 1. Problem (w skrócie)

Dany jest nieskierowany graf `G = (V, E)`. W węźle (lub węzłach) startowym wybucha
pożar. Proces przebiega w krokach czasu dyskretnego:

1. **Strażacy** chronią na stałe `num_firefighters` wybranych, niepłonących węzłów.
2. **Pożar** rozprzestrzenia się na wszystkich niechronionych sąsiadów płonących węzłów.
3. Powtarzamy, aż pożar nie może się dalej rozszerzyć.

**Cel (maksymalizowany):** łączny **koszt uratowanych węzłów**.

> **Założenie projektu — jednokryterialność.** Instancje 2-obj / 3-obj zawierają po
> 2 / 3 kolumny kosztów, ale **zawsze używamy tylko pierwszej (lewej) kolumny**.
> Problem NIE jest rozpatrywany wielokryterialnie.

**Reprezentacja rozwiązania:** permutacja węzłów = kolejność priorytetów ochrony.
Dekoder (symulator pożaru) zamienia permutację na wartość uratowanych węzłów.

---

## 2. Trzy usprawnienia (+ hybryda)

Każde usprawnienie wstrzykuje wiedzę o problemie do GA przez **zasianie populacji
początkowej** (seeding) i **mutację kierowaną** — bez zużywania dodatkowego budżetu.

| Wariant     | Pomysł |
|-------------|--------|
| `baseline`  | czysty GA, bez wiedzy o problemie (losowa inicjalizacja, klasyczne operatory) |
| `cost`      | chroń najpierw węzły o **najwyższym koszcie** |
| `degree`    | chroń najpierw węzły o **największej liczbie połączeń** (huby) |
| `dijkstra`  | chroń najpierw węzły **najbliższe pożarowi** (najkrótsza droga ognia → odetnij dostęp) |
| `hybrid`    | *(pomysł własny)* ważona kombinacja: cenny + dobrze połączony + blisko ognia |

> Na grafie nieskierowanym o jednostkowych wagach algorytm Dijkstry daje tę samą
> odległość co BFS — wykorzystujemy to w wariancie `dijkstra`.

---

## 3. Instalacja (Poetry)

```bash
poetry install
```

Wymagany Python ≥ 3.11 (konfiguracja czytana przez wbudowane `tomllib`).

---

## 4. Uruchomienie

Cały eksperyment definiuje plik **`config.toml`** (nie trzeba dotykać kodu):

```bash
poetry run python run.py
# lub przez zainstalowany skrypt:
poetry run ffp
# inny plik konfiguracyjny:
poetry run python run.py --config moj_config.toml
```

W repozytorium dołączono **mały podzbiór instancji** (`instances/`), więc projekt
działa od razu. Aby użyć **pełnego zbioru** Michalaka, rozpakuj `ed-ls_ffp_instances`
i wskaż jego ścieżkę w `config.toml`:

```toml
[dataset]
root = "/ścieżka/do/ed-ls_ffp_instances"
```

---

## 5. Wyniki (generowane automatycznie)

Każde uruchomienie tworzy katalog `results/run_RRRRMMDD_GGMMSS/` zawierający:

* `summary.csv` — zbiorcze podsumowanie (best/avg/worst/std, **% poprawy względem
  baseline**, wynik heurystyki zachłannej, budżet) dla każdej instancji i wariantu;
* dla każdej instancji: `<nazwa>_convergence.csv` — uśrednione krzywe zbieżności;
* wykresy PNG:
  * `<nazwa>_convergence.png` — zbieżność (best vs pokolenie) wszystkich wariantów,
  * `<nazwa>_final_bars.png` — jakość końcowa (avg ± std) + linia heurystyki zachłannej,
  * `<nazwa>_boxplot.png` — rozkład wyników z powtórzeń,
  * `overall_improvement.png` — średnia % poprawa usprawnień względem baseline.

---

## 6. Równy budżet (kontrola)

Budżet = `pop_size × generations` ewaluacji funkcji celu na jedno uruchomienie.
Obowiązuje **identycznie każdy wariant** — cała populacja jest oceniana w każdym
pokoleniu, a heurystyki nie zużywają dodatkowych ewaluacji. Kod **sprawdza** to po
każdej instancji i przerywa z błędem, jeśli budżety się rozjadą.

---

## 7. Struktura projektu

```
ffp-genetic/
├── config.toml              # wszystkie parametry eksperymentu
├── run.py                   # punkt wejścia (działa też bez `poetry install`)
├── pyproject.toml           # Poetry
├── instances/               # mały podzbiór instancji FFP (2-obj + 3-obj)
└── src/ffp/
    ├── instance.py          # wczytywanie instancji + BFS/Dijkstra od pożaru
    ├── simulator.py         # symulator pożaru = funkcja celu
    ├── heuristics.py        # 3 usprawnienia + hybryda (porządki priorytetów)
    ├── operators.py         # krzyżowanie, mutacja (w tym kierowana), selekcja
    ├── genetic.py           # algorytm genetyczny + kontrola budżetu
    ├── experiment.py        # orkiestracja, statystyki, zapis CSV
    ├── plotting.py          # wykresy
    ├── config.py            # wczytywanie config.toml
    └── cli.py               # spięcie całości
```
