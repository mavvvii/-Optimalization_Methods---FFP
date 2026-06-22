"""Punkt wejscia: czyta config.toml, uruchamia eksperyment, zapisuje CSV + wykresy."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from .config import load_config
from .experiment import (
    InstanceReport,
    run_instance,
    write_convergence_csv,
    write_summary_csv,
)
from .plotting import generate_all_plots


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Algorytm genetyczny dla problemu strazackiego (FFP) z usprawnieniami."
    )
    parser.add_argument(
        "--config", type=Path, default=Path("config.toml"),
        help="Sciezka do pliku konfiguracyjnego TOML (domyslnie ./config.toml).",
    )
    args = parser.parse_args(argv)

    if not args.config.is_file():
        raise SystemExit(f"Nie znaleziono pliku konfiguracyjnego: {args.config}")

    cfg = load_config(args.config)

    stamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    out_root = cfg.results_dir / stamp
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Wyniki beda zapisane w: {out_root.resolve()}")

    reports: list[InstanceReport] = []
    for spec in cfg.instances:
        report = run_instance(
            dataset_root=cfg.dataset_root,
            n_obj=cfg.n_obj,
            size=spec.size,
            idx=spec.idx,
            config=cfg.ga,
            strategies=cfg.strategies,
            runs=cfg.runs,
            base_seed=cfg.base_seed,
        )
        reports.append(report)
        write_convergence_csv(report, out_root / report.instance.name)

    summary_path = write_summary_csv(reports, cfg.ga, out_root / "summary.csv")
    plot_paths = generate_all_plots(reports, out_root)

    print(f"\n{'='*70}")
    print("  GOTOWE.")
    print(f"  Podsumowanie CSV : {summary_path}")
    print(f"  Wykresy ({len(plot_paths)}) zapisane w: {out_root}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
