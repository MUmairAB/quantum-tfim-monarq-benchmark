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

`evaluate_on_device` is the MonarQ hardware entry point. Training itself
never runs on hardware — parameter-shift gradients would need thousands of
real circuit evaluations per VQE run, which isn't practical against a
shot-limited, queued device. Instead, params come from a normal `train_vqe`
simulation run, and `evaluate_on_device` executes that fixed, already-trained
circuit once (with shots) on `monarq.sim` or the real `monarq.default`
backend, to measure how much hardware noise degrades an already-good
solution.
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


def build_circuit(L: int, H, dev, interface: str | None = "jax", diff_method: str | None = "adjoint", shots=None):
    """Build the HVA QNode: a Hadamard layer, then n_layers of hva_layer.

    Defaults match the simulator training path (`train_vqe`). The hardware
    path (`evaluate_on_device`) overrides all three: no interface/diff_method
    needed since it only evaluates a fixed circuit, and `shots` is required
    on any real or noisy-sampled device.
    """
    import pennylane as qml

    @qml.qnode(dev, interface=interface, diff_method=diff_method, shots=shots)
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
    return_params: bool = False,
):
    """Train the HVA and return (energy, two_qubit_gate_count).

    two_qubit_gate_count = (L-1) * n_layers, the IsingZZ gate count — the
    quantity that gates hardware feasibility on MonarQ. With
    return_params=True, also returns the trained params (needed to hand a
    fixed, already-optimized circuit to evaluate_on_device for a hardware
    run) as a third element.
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
    if return_params:
        return final_energy, two_qubit_gate_count, params
    return final_energy, two_qubit_gate_count


def evaluate_on_device(
    L: int,
    h: float,
    params,
    J: float = 1.0,
    device: str = "monarq.sim",
    shots: int = 1000,
    client=None,
) -> float:
    """Run an already-trained HVA circuit once on `device` and return the measured energy.

    `params` should come from `train_vqe` (a plain array of shape
    (n_layers, 2)) — this function does no training, just one circuit
    execution per Hamiltonian term with the given shot count. `client` is a
    `MonarqClient` instance, required for the real `monarq.default` backend
    and unused for `monarq.sim`. Credentials are the caller's responsibility
    (see `local_only/access_monarq.py`) — this module stays credential-free
    since it's the part of the repo that's pushed to GitHub.

    Measures each Hamiltonian term separately and sums the weighted results
    in plain Python, rather than a single qml.expval(H) call: confirmed by
    direct test that monarq.sim's multi-term Hamiltonian expval returns
    near-zero nonsense (a state with known expval -1.0 came back as -0.005),
    while single-term expval values are correct. Per-term measurement works
    around it and costs one extra circuit execution per term either way.

    params gets converted to plain floats before use — passing the JAX
    arrays train_vqe returns straight through measurably degraded results
    on monarq.sim relative to the same circuit with plain-float angles,
    consistent with the plugin's API layer not handling JAX's array type
    correctly when serializing gate parameters.
    """
    import numpy as np
    import pennylane as qml

    params = np.asarray(params, dtype=float)

    device_kwargs = {"wires": L}
    if client is not None:
        device_kwargs["client"] = client
    dev = qml.device(device, **device_kwargs)

    H = build_pennylane_hamiltonian(L, h, J)
    coeffs, ops = H.terms()
    total = 0.0
    for coeff, op in zip(coeffs, ops):
        circuit = build_circuit(L, op, dev, interface=None, diff_method=None, shots=shots)
        total += coeff * circuit(params)
    return float(total)


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
