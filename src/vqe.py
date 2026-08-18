"""VQE (Hamiltonian Variational Ansatz) ground-state energy for the open-BC TFIM.

Simulator only — real MonarQ hardware runs are a separate step, confirm
before pointing this at anything but a local simulator device. Usage:

    python -m src.vqe --config configs/<sweep>.yaml

Training runs through JAX (jit-compiled, optax Adam) on lightning.qubit with
diff_method="adjoint", rather than default.qubit's default backprop.
default.qubit+backprop keeps a full statevector per layer, so memory scales
with circuit depth and OOMs at L=20 with enough layers; lightning.qubit's
adjoint is natively C++ and keeps memory ~O(2^L) regardless of depth, at
backprop-comparable speed. The device name is config-driven (`device` key in
run_sweep) in case a GPU adjoint device is worth trying later.
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
    """Build the HVA QNode: a Hadamard layer, then n_layers of hva_layer."""
    import pennylane as qml

    @qml.qnode(dev, interface="jax", diff_method="adjoint")
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
    device: str = "lightning.qubit",
) -> tuple[float, int]:
    """Train the HVA and return (energy, two_qubit_gate_count).

    two_qubit_gate_count = (L-1) * n_layers, the IsingZZ gate count — the
    quantity that gates hardware feasibility on MonarQ.
    """
    import jax

    # x64 isn't JAX's default; enable it explicitly to match the rest of
    # the repo's float64 precision (QuSpin, NetKet).
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import numpy as np
    import optax
    import pennylane as qml

    dev = qml.device(device, wires=L)
    H = build_pennylane_hamiltonian(L, h, J)
    circuit = build_circuit(L, H, dev)

    rng = np.random.default_rng(seed)
    params = jnp.array(rng.uniform(0, 2 * np.pi, size=(n_layers, 2)))

    # b1/b2/eps match qml.AdamOptimizer's defaults rather than optax's own
    # (beta2=0.99 vs optax's 0.999) — the mismatch changes which local
    # minimum training converges to.
    opt = optax.adam(stepsize, b1=0.9, b2=0.99, eps=1e-8)
    opt_state = opt.init(params)

    @jax.jit
    def step(params, opt_state):
        loss, grads = jax.value_and_grad(circuit)(params)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state

    for _ in range(n_steps):
        params, opt_state = step(params, opt_state)

    final_energy = float(circuit(params))
    two_qubit_gate_count = (L - 1) * n_layers
    return final_energy, two_qubit_gate_count


def run_sweep(config: dict, results_dir: Path, exact_dir: Path) -> None:
    """Train the HVA VQE at every (L, h, n_layers) point in the config and
    save results alongside the matching exact-diagonalization comparison.

    `train.n_layers` can be a single value or a list, to scan circuit depth
    at fixed L/h; output filenames only get a `_layersN` suffix when scanning.
    Resumable: points that already have an output file are skipped.
    """
    import jax

    jax.config.update("jax_enable_x64", True)

    J = config.get("J", 1.0)
    L_values = config["L_values"]
    h_values = config["h_values"] if "h_values" in config else build_h_grid(**config.get("h_grid", {}))
    method = config.get("method", "vqe_sim")
    device = config.get("device", "lightning.qubit")
    train_kwargs = config.get("train", {})

    n_layers_values = train_kwargs.pop("n_layers")
    if not isinstance(n_layers_values, list):
        n_layers_values = [n_layers_values]
    scanning_layers = len(n_layers_values) > 1

    print(f"JAX backend: {jax.default_backend()}  devices: {jax.devices()}")

    for L in L_values:
        out_dir = results_dir / str(L)
        out_dir.mkdir(parents=True, exist_ok=True)
        for h in h_values:
            for n_layers in n_layers_values:
                suffix = f"{h:.3f}_layers{n_layers}.json" if scanning_layers else f"{h:.3f}.json"
                out_path = out_dir / suffix
                if out_path.exists():
                    print(f"L={L:<3} h={h:.3f} layers={n_layers:<3}  already done, skipping -> {out_path}")
                    continue

                run_kwargs = {**train_kwargs, "n_layers": n_layers, "device": device}
                start = time.perf_counter()
                E, two_qubit_gates = train_vqe(L, h, J, **run_kwargs)
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
                    "method": method,
                    "jax_backend": jax.default_backend(),
                    "boundary": "open",
                    "two_qubit_gate_count": two_qubit_gates,
                    "wall_time_s": elapsed,
                    **run_kwargs,
                }
                out_path.write_text(json.dumps(record, indent=2))

                rel_str = f"{rel_error:.4%}" if rel_error is not None else "n/a (no exact ref)"
                print(
                    f"L={L:<3} h={h:.3f} layers={n_layers:<3}  E0={E:.6f}  rel_err={rel_str}  "
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
