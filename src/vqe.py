"""VQE (Hamiltonian Variational Ansatz) ground-state energy for the open-BC TFIM.

Phase 1, step 3: a first working VQE, simulator only, checked against the
exact-diagonalization results from src/exact.py. Real MonarQ hardware is
Phase 3, not this module — confirm before pointing this at anything but a
local simulator device.

Training runs through JAX (jit-compiled, optax Adam) rather than PennyLane's
own autograd-based optimizer, on lightning.qubit (PennyLane's C++ backend)
with diff_method="adjoint" rather than default.qubit's default backprop.
This combination was arrived at after two things that didn't work for the
L=20 deeper-layer scan this module was built to run:

  - default.qubit + backprop: backprop keeps the full statevector after
    every gate for the backward pass, so memory scales with n_layers. At
    L=20/16 layers this needed ~10GB in one allocation and OOM'd a Compute
    Canada GPU's 10GB MIG slice.
  - default.qubit + adjoint (which fixes the memory scaling): correct, but
    6-44x slower than backprop in local testing, growing worse with L —
    default.qubit's adjoint isn't well jit-compiled under JAX and appears
    to fall back to a non-XLA-compiled path per step.

lightning.qubit's adjoint is natively C++, not a JAX-interfaced Python
fallback: it matched backprop's speed exactly in testing (down to matching
default.qubit+backprop's wall-clock at L=6/12/16) while keeping memory
~O(2^L) regardless of depth, same as any adjoint method. It's CPU-only, not
GPU — but since it already matches the speed we'd measured on Compute
Canada's GPU while sidestepping the memory ceiling entirely, and needs no
new dependency (pennylane-lightning ships as part of a plain `pennylane`
install), it's the simulator this module actually uses now. The device name
is still config-driven (see run_sweep's `device` key) in case a GPU-native
adjoint device (e.g. lightning.gpu) is worth revisiting later. Usage:

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
    """Build the HVA QNode: Hadamard layer (uniform superposition) then n_layers of hva_layer.

    interface="jax" so the QNode's parameters, execution, and gradients all
    run through JAX. diff_method="adjoint" keeps memory ~O(2^L) regardless
    of circuit depth — see the module docstring for why this needs to be
    paired with lightning.qubit specifically, not just any device.
    """
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
    """Train the HVA (JAX, jit-compiled, adjoint diff) and return (energy, two_qubit_gate_count).

    two_qubit_gate_count is (L-1) * n_layers — the IsingZZ gates. This is the
    quantity that actually gates hardware feasibility on MonarQ (Phase 3), so
    it's returned here even though it doesn't matter for simulator runs.
    """
    import jax

    # JAX defaults to float32/complex64, silently truncating precision below
    # what the rest of the repo uses (QuSpin, NetKet, and the old autograd
    # VQE are all float64) — enable x64 explicitly so switching backends
    # doesn't also switch precision.
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import numpy as np
    import optax
    import pennylane as qml

    dev = qml.device(device, wires=L)
    H = build_pennylane_hamiltonian(L, h, J)
    circuit = build_circuit(L, H, dev)

    # Params are initialized with plain numpy (not jax.random) so a given
    # seed produces the exact same starting point as the old autograd-based
    # implementation — isolates "more layers" as the only thing changing
    # relative to earlier vqe_sim results.
    rng = np.random.default_rng(seed)
    params = jnp.array(rng.uniform(0, 2 * np.pi, size=(n_layers, 2)))

    # b1/b2/eps match qml.AdamOptimizer's defaults (beta1=0.9, beta2=0.99,
    # eps=1e-8) exactly — optax.adam's own default beta2 is 0.999, and that
    # mismatch alone was enough to send this non-convex landscape to a
    # different local minimum at L=12 during testing. Matching them keeps
    # "more layers" the only real variable relative to the old vqe_sim results.
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
    """Train the HVA VQE at every (L, h, n_layers) point in the config and save results + exact comparison.

    h-values come from an explicit `h_values` list (small validation sweeps)
    or a `h_grid` spec (build_h_grid kwargs, for matching the full exact sweep).
    `train.n_layers` can be a single value (as in the original sim sweep) or
    a list, to scan circuit depth at fixed L/h — used for the L=20 GPU rerun
    that checks whether more layers resolves the accuracy degradation found
    at 10 layers. Output filenames only carry a `_layersN` suffix when a scan
    is actually configured (len > 1), so the original single-depth sweep's
    file naming (and its resumability against already-existing results) is
    unaffected.

    Resumable: a point whose output file already exists is skipped, so an
    interrupted run (crash, sleep, Ctrl-C) only costs the interrupted point,
    not the whole sweep — just rerun the same command. This matters most for
    this module: the L=20 leg alone takes on the order of 10 hours per depth.
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

    # jax.default_backend() reflects JAX's own array ops (the optimizer/glue
    # code), not what ran the circuit simulation itself — that's whatever
    # `device` (below, in each result record) says. lightning.qubit always
    # simulates in C++ on CPU regardless of what JAX itself defaults to.
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
