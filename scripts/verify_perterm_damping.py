"""Recompute the per-term damping of the L=4 and L=6 hardware runs.

Those runs stored the measured expectation value of each Hamiltonian term but
not the noiseless reference to divide it by, so the damping factors cannot be
read straight off the files. The reference is recoverable: the records keep the
seed and every hyperparameter, so retraining reproduces the same circuit and the
term values can be evaluated analytically. The script checks that the retrained
energy matches the stored E0_sim_noiseless before trusting the reconstruction.

What it shows: one physical qubit on this chip damps its X term far harder than
the others, and which logical index that lands on moves with L. At L=4 it is
X(0), at L=6 it is X(1) — in both cases the wire mapped onto physical qubit 9.

Two of the L=6 repeats lost their per-term breakdown to a filename collision and
carry a note saying so; they are skipped and reported.

Usage:  python scripts/verify_perterm_damping.py
"""
from __future__ import annotations

import glob
import json
import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pennylane as qml  # noqa: E402

from src.vqe import hva_layer, train_vqe  # noqa: E402

# (L, seed) as recorded in the ladder result files.
RUNS = [(4, 3), (6, 11)]
H = 1.0
N_LAYERS = 1


def noiseless_x_terms(L: int, h: float, n_layers: int, seed: int):
    """Retrain at the recorded seed and evaluate each X term analytically."""
    energy, _, params = train_vqe(
        L, h, n_layers=n_layers, n_steps=600, stepsize=0.1, seed=seed,
        return_params=True,
    )
    dev = qml.device("default.qubit", wires=L)

    def prepare():
        for w in range(L):
            qml.Hadamard(w)
        for gamma, beta in params:
            hva_layer(float(gamma), float(beta), L)

    refs = {}
    for i in range(L):
        node = qml.QNode(
            lambda i=i: (prepare(), qml.expval(qml.PauliX(i)))[1], dev
        )
        refs[f"X({i})"] = float(node())
    return energy, refs


def main() -> None:
    for L, seed in RUNS:
        pattern = f"results/vqe_hardware/{L}/{H:.3f}_layers{N_LAYERS}_rep*_default_*.json"
        files = sorted(glob.glob(str(REPO_ROOT / pattern)))
        if not files:
            print(f"L={L}: no records found")
            continue

        stored = json.loads(Path(files[0]).read_text())["E0_sim_noiseless"]
        energy, refs = noiseless_x_terms(L, H, N_LAYERS, seed)
        if abs(energy - stored) > 1e-6:
            print(f"L={L}: retrained energy {energy:.9f} does not match the stored "
                  f"{stored:.9f}; reconstruction rejected")
            continue

        damping: dict[str, list[float]] = {}
        skipped = []
        for path in files:
            record = json.loads(Path(path).read_text())
            if "terms" not in record:
                skipped.append((Path(path).name, record.get("note", "")))
                continue
            for term in record["terms"]:
                name = term["term"]
                if name.startswith("X") and abs(refs[name]) > 1e-9:
                    damping.setdefault(name, []).append(term["expval"] / refs[name])

        print(f"L={L}, seed={seed}: reconstruction reproduces the stored noiseless "
              f"energy to {abs(energy - stored):.1e}")
        for name in sorted(damping, key=lambda s: int(s[2])):
            values = np.asarray(damping[name])
            sem = values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
            flag = "  <- anomalous" if values.mean() < 0.2 else ""
            print(f"    {name}: {values.mean():+.3f} +/- {sem:.3f}  "
                  f"(n={len(values)}){flag}")
        for name, note in skipped:
            print(f"    skipped {name}: {note[:60]}")
        print()


if __name__ == "__main__":
    main()
