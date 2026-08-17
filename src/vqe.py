"""VQE (Hamiltonian Variational Ansatz) ground-state energy for the open-BC TFIM.

Phase 1, step 3: a first working VQE, simulator only (PennyLane's default.qubit),
checked against the exact-diagonalization results from src/exact.py. Real MonarQ
hardware is Phase 3, not this module — confirm before pointing this at anything
but a local simulator device. Usage:

    python -m src.vqe --config configs/<sweep>.yaml
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml

from src.hamiltonian import build_h_grid, build_pennylane_hamiltonian
from src.nnqs import load_exact_E0

REPO_ROOT = Path(__file__).resolve().parent.parent


def hva_layer(gamma: float, beta: float, L: int) -> None:
    """One Hamiltonian Variational Ansatz layer: ZZ-entangle then X-rotate, mirroring the TFIM's own terms."""
    import pennylane as qml

    for i in range(L - 1):
        qml.IsingZZ(gamma, wires=[i, i + 1])
    for w in range(L):
        qml.RX(beta, wires=w)


def build_circuit(L: int, H, dev):
    """Build the HVA QNode: Hadamard layer (uniform superposition) then n_layers of hva_layer."""
    import pennylane as qml

    @qml.qnode(dev)
    def circuit(params):
        for w in range(L):
            qml.Hadamard(wires=w)
        for gamma, beta in params:
            hva_layer(gamma, beta, L)
        return qml.expval(H)

    return circuit


def train_vqe(
    L: int,
    h: float,
    J: float = 1.0,
    n_layers: int = 4,
    n_steps: int = 200,
    stepsize: float = 0.1,
    seed: int | None = None,
) -> tuple[float, int]:
    """Train the HVA on PennyLane's default.qubit simulator and return (energy, two_qubit_gate_count).

    two_qubit_gate_count is (L-1) * n_layers — the IsingZZ gates. This is the
    quantity that actually gates hardware feasibility on MonarQ (Phase 3), so
    it's returned here even though it doesn't matter for simulator runs.
    """
    import numpy as np
    import pennylane as qml
    from pennylane import numpy as pnp

    dev = qml.device("default.qubit", wires=L)
    H = build_pennylane_hamiltonian(L, h, J)
    circuit = build_circuit(L, H, dev)

    rng = np.random.default_rng(seed)
    params = pnp.array(rng.uniform(0, 2 * np.pi, size=(n_layers, 2)), requires_grad=True)

    opt = qml.AdamOptimizer(stepsize=stepsize)
    for _ in range(n_steps):
        params, _ = opt.step_and_cost(circuit, params)

    final_energy = float(circuit(params))
    two_qubit_gate_count = (L - 1) * n_layers
    return final_energy, two_qubit_gate_count


def run_sweep(config: dict, results_dir: Path, exact_dir: Path) -> None:
    """Train the HVA VQE at every (L, h) point in the config and save results + exact comparison.

    h-values come from an explicit `h_values` list (small validation sweeps)
    or a `h_grid` spec (build_h_grid kwargs, for matching the full exact sweep).

    Resumable: a point whose output file already exists is skipped, so an
    interrupted run (crash, sleep, Ctrl-C) only costs the interrupted point,
    not the whole sweep — just rerun the same command. This matters most for
    this module: the L=20 leg alone takes on the order of 10 hours.
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
            E, two_qubit_gates = train_vqe(L, h, J, **train_kwargs)
            elapsed = time.perf_counter() - start

            E_exact = load_exact_E0(L, h, exact_dir)
            rel_error = abs(E - E_exact) / abs(E_exact) if E_exact is not None else None

            record = {
                "L": L,
                "h": h,
                "J": J,
                "E0": E,
                "E0_exact": E_exact,
                "rel_error": rel_error,
                "method": "vqe_sim",
                "device": "default.qubit",
                "boundary": "open",
                "two_qubit_gate_count": two_qubit_gates,
                "wall_time_s": elapsed,
                **train_kwargs,
            }
            out_path.write_text(json.dumps(record, indent=2))

            rel_str = f"{rel_error:.4%}" if rel_error is not None else "n/a (no exact ref)"
            print(
                f"L={L:<3} h={h:.3f}  E0={E:.6f}  rel_err={rel_str}  "
                f"2q_gates={two_qubit_gates}  ({elapsed:.1f}s) -> {out_path}"
            )


def main() -> None:
    """CLI entry point: load a sweep config and run it, writing results to disk."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--results-dir", type=Path, default=REPO_ROOT / "results" / "vqe_sim"
    )
    parser.add_argument(
        "--exact-dir", type=Path, default=REPO_ROOT / "results" / "exact"
    )
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    run_sweep(config, args.results_dir, args.exact_dir)


if __name__ == "__main__":
    main()
