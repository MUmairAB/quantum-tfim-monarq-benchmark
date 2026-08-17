"""Exact ground-state energy of the open-BC TFIM via QuSpin (Phase 1, step 1).

This is the ground-truth method that NNQS and VQE are checked against. Usage:

    python -m src.exact --config configs/<sweep>.yaml
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml

from src.hamiltonian import build_h_grid, build_quspin_basis, build_quspin_hamiltonian

REPO_ROOT = Path(__file__).resolve().parent.parent


def ground_state_energy(L: int, h: float, J: float = 1.0, basis=None) -> float:
    """Return the ground-state energy E0 for one (L, h, J) point via sparse Lanczos."""
    H = build_quspin_hamiltonian(L, h, J, basis=basis)
    E0 = H.eigsh(k=1, which="SA")[0][0]
    return float(E0)


def run_sweep(config: dict, results_dir: Path) -> None:
    """Run the full L x h sweep from a config dict and write one JSON file per point.

    The QuSpin basis only depends on L, so it's built once per L and reused
    across every h in the sweep rather than rebuilt from scratch each time —
    at L=24 that basis build alone costs ~20s per point, so this matters.

    Resumable: a point whose output file already exists is skipped, so an
    interrupted run (crash, sleep, Ctrl-C) only costs the interrupted point,
    not the whole sweep — just rerun the same command.
    """
    J = config.get("J", 1.0)
    h_grid = build_h_grid(**config.get("h_grid", {}))
    L_values = config["L_values"]

    for L in L_values:
        out_dir = results_dir / str(L)
        out_dir.mkdir(parents=True, exist_ok=True)
        remaining = [h for h in h_grid if not (out_dir / f"{h:.3f}.json").exists()]
        if len(remaining) < len(h_grid):
            print(f"L={L:<3} skipping {len(h_grid) - len(remaining)} already-done point(s)")
        if not remaining:
            continue
        basis = build_quspin_basis(L)
        for h in remaining:
            start = time.perf_counter()
            E0 = ground_state_energy(L, h, J, basis=basis)
            elapsed = time.perf_counter() - start
            out_path = out_dir / f"{h:.3f}.json"
            out_path.write_text(
                json.dumps(
                    {
                        "L": L,
                        "h": h,
                        "J": J,
                        "E0": E0,
                        "method": "exact",
                        "boundary": "open",
                        "wall_time_s": elapsed,
                    },
                    indent=2,
                )
            )
            print(f"L={L:<3} h={h:.3f}  E0={E0:.8f}  ({elapsed:.2f}s) -> {out_path}")


def main() -> None:
    """CLI entry point: load a sweep config and run it, writing results to disk."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--results-dir", type=Path, default=REPO_ROOT / "results" / "exact"
    )
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    run_sweep(config, args.results_dir)


if __name__ == "__main__":
    main()
