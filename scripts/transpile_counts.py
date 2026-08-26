"""Count what the HVA circuits actually become once transpiled for MonarQ.

The result files record a `two_qubit_gate_count` of (L-1)*n_layers, which is the
number of IsingZZ gates as written in the ansatz. That is not what the hardware
runs. MonarQ's native two-qubit gate is CZ, and this script measures how many of
them each circuit really costs, along with the X90 count and the circuit depth
that MonarQ's published budget is quoted against.

Everything runs locally. The decomposition steps used here perform no
calibration lookup and open no client, so nothing contacts the device.

One trap worth knowing about: PreProcessor.get_processor wraps the whole
transpilation in a try/except that returns the tape UNCHANGED on failure, so a
call that silently did nothing is indistinguishable from one that worked. This
script applies the steps directly, so a failure raises instead.

Usage:  python scripts/transpile_counts.py
"""
from __future__ import annotations

import sys
import warnings
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

warnings.filterwarnings("ignore")

import pennylane as qml  # noqa: E402
from pennylane.tape import QuantumScript  # noqa: E402
from pennylane_calculquebec.processing.config import NoPlaceNoRouteConfig  # noqa: E402
from pennylane_calculquebec.processing.interfaces import PreProcStep  # noqa: E402

from src.vqe import train_vqe  # noqa: E402

# The configurations that were actually executed, with the seed each one used.
CONFIGS = [
    (2, 1.0, 1, 0),
    (2, 1.0, 2, 0),
    (2, 1.0, 4, 0),
    (6, 1.0, 1, 11),
    (6, 1.0, 2, 0),
    (6, 1.0, 4, 0),
]


def depth_of(ops, L: int) -> int:
    """Circuit depth: pack operations into layers by which wires they occupy."""
    last = {w: 0 for w in range(L)}
    for op in ops:
        wires = list(op.wires)
        layer = max(last[w] for w in wires) + 1
        for w in wires:
            last[w] = layer
    return max(last.values()) if last else 0


def transpile(tape):
    """Run the native-decomposition steps, letting any failure raise."""
    steps = [s for s in NoPlaceNoRouteConfig().steps if isinstance(s, PreProcStep)]
    with qml.QueuingManager.stop_recording():
        for step in steps:
            tape = step.execute(tape)
    return tape


def census(L: int, h: float, n_layers: int, seed: int) -> dict:
    """Train the circuit, transpile it, and count the native gates it becomes.

    Parameters are recovered by retraining at the recorded seed and
    hyperparameters, since the result files store the seed rather than the
    trained angles.
    """
    _, _, params = train_vqe(
        L, h, n_layers=n_layers, n_steps=600, stepsize=0.1, seed=seed,
        return_params=True,
    )

    ops = [qml.Hadamard(w) for w in range(L)]
    for gamma, beta in params:
        for i in range(L - 1):
            ops.append(qml.IsingZZ(float(gamma), wires=[i, i + 1]))
        for w in range(L):
            ops.append(qml.RX(float(beta), wires=w))
    tape = QuantumScript(ops, [qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))])

    out = transpile(tape)
    counts = Counter(op.name for op in out.operations)
    cz_ops = [op for op in out.operations if op.name == "CZ"]
    logical_zz = (L - 1) * n_layers

    return {
        "L": L,
        "n_layers": n_layers,
        "logical_zz": logical_zz,
        "native_cz": len(cz_ops),
        "cz_per_zz": len(cz_ops) / logical_zz,
        "x90": counts.get("X90", 0),
        # Z rotations are virtual on this hardware, applied by phase tracking
        # rather than a pulse, so they are counted separately from X90.
        "other_1q": sum(v for k, v in counts.items() if k not in ("CZ", "X90")),
        "cz_depth": depth_of(cz_ops, L),
        "total_depth": depth_of(out.operations, L),
    }


def main() -> None:
    header = (f"{'config':<18}{'logZZ':>7}{'CZ':>5}{'CZ/ZZ':>7}{'X90':>6}"
              f"{'other1q':>9}{'CZdepth':>9}{'depth':>7}")
    print(header)
    print("-" * len(header))
    for L, h, n_layers, seed in CONFIGS:
        r = census(L, h, n_layers, seed)
        label = f"L={r['L']}, {r['n_layers']} layer" + ("s" if r["n_layers"] > 1 else "")
        print(f"{label:<18}{r['logical_zz']:>7}{r['native_cz']:>5}"
              f"{r['cz_per_zz']:>7.2f}{r['x90']:>6}{r['other_1q']:>9}"
              f"{r['cz_depth']:>9}{r['total_depth']:>7}")


if __name__ == "__main__":
    main()
