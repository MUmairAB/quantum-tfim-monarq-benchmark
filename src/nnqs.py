"""Neural-network quantum state (RBM) ground-state energy for the open-BC TFIM.

Checked against the exact-diagonalization results from src/exact.py. Usage:

    python -m src.nnqs --config configs/<sweep>.yaml
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml

from src.hamiltonian import build_h_grid, build_netket_hamiltonian

REPO_ROOT = Path(__file__).resolve().parent.parent


def train_nnqs(
    L: int,
    h: float,
    J: float = 1.0,
    alpha: int = 2,
    n_samples: int = 1024,
    n_iter: int = 300,
    learning_rate: float = 0.02,
    diag_shift: float = 0.01,
    seed: int | None = None,
    show_progress: bool = True,
) -> tuple[float, float]:
    """Train an RBM ground state via VMC and return (energy_mean, energy_error).

    energy_error is the Monte Carlo standard error on the mean, not a bound
    on distance to the true ground state.

    show_progress drives NetKet's per-iteration progress bar. It is on by
    default so an interactive sweep still shows one, but batch callers should
    turn it off: the bar redraws every iteration and writes roughly 50 kB per
    run, which becomes tens of megabytes of SLURM logs across a full ensemble.
    """
    import netket as nk

    hi, H = build_netket_hamiltonian(L, h, J)
    model = nk.models.RBM(alpha=alpha, param_dtype=complex)
    sampler = nk.sampler.MetropolisLocal(hi)
    vstate = nk.vqs.MCState(sampler, model, n_samples=n_samples, seed=seed)

    optimizer = nk.optimizer.Sgd(learning_rate=learning_rate)
    preconditioner = nk.optimizer.SR(diag_shift=diag_shift)
    gs = nk.driver.VMC(
        H, optimizer, variational_state=vstate, preconditioner=preconditioner
    )
    gs.run(n_iter=n_iter, out=None, show_progress=show_progress)

    stats = vstate.expect(H)
    return float(stats.mean.real), float(stats.error_of_mean)


def load_exact_E0(L: int, h: float, exact_dir: Path) -> float | None:
    """Look up the matching exact-diagonalization result, if one was computed."""
    path = exact_dir / str(L) / f"{h:.3f}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())["E0"]


def run_sweep(config: dict, results_dir: Path, exact_dir: Path) -> None:
    """Train an RBM at every (L, h) point in the config and save results
    alongside the matching exact-diagonalization comparison.

    h-values come from an explicit `h_values` list or a `h_grid` spec
    (build_h_grid kwargs). Resumable: points that already have an output
    file are skipped.
    """
    J = config.get("J", 1.0)
    L_values = config["L_values"]
    h_values = config["h_values"] if "h_values" in config else build_h_grid(**config.get("h_grid", {}))
    train_kwargs = config.get("train", {})

    for L in L_values:
        out_dir = results_dir / str(L)
        out_dir.mkdir(parents=True, exist_ok=True)
        for h in h_values:
            out_path = out_dir / f"{h:.3f}.json"
            if out_path.exists():
                print(f"L={L:<3} h={h:.3f}  already done, skipping -> {out_path}")
                continue
            start = time.perf_counter()
            E_mean, E_err = train_nnqs(L, h, J, **train_kwargs)
            elapsed = time.perf_counter() - start

            E_exact = load_exact_E0(L, h, exact_dir)
            rel_error = abs(E_mean - E_exact) / abs(E_exact) if E_exact is not None else None

            # At h=0 the RBM represents the ground state exactly, so the Monte
            # Carlo variance is identically zero and NetKet returns NaN. JSON has
            # no NaN literal (RFC 8259), and a bare NaN makes the file unreadable
            # to strict parsers, so store it as null instead.
            E_err_out = None if E_err != E_err else E_err

            record = {
                "L": L,
                "h": h,
                "J": J,
                "E0": E_mean,
                "E0_mc_error": E_err_out,
                "E0_exact": E_exact,
                "rel_error": rel_error,
                "method": "nnqs",
                "boundary": "open",
                "wall_time_s": elapsed,
                **train_kwargs,
            }
            # allow_nan=False turns any remaining non-finite value into an
            # error rather than an unparseable file.
            out_path.write_text(json.dumps(record, indent=2, allow_nan=False))

            rel_str = f"{rel_error:.4%}" if rel_error is not None else "n/a (no exact ref)"
            err_str = f"{E_err:.1e}" if E_err_out is not None else "n/a (exact, zero variance)"
            print(
                f"L={L:<3} h={h:.3f}  E0={E_mean:.6f}±{err_str}  "
                f"rel_err={rel_str}  ({elapsed:.1f}s) -> {out_path}"
            )


def main() -> None:
    """CLI entry point: load a sweep config and run it, writing results to disk."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--results-dir", type=Path, default=REPO_ROOT / "results" / "nnqs"
    )
    parser.add_argument(
        "--exact-dir", type=Path, default=REPO_ROOT / "results" / "exact"
    )
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    run_sweep(config, args.results_dir, args.exact_dir)


if __name__ == "__main__":
    main()
