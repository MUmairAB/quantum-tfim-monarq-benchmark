"""Builds the open-BC TFIM Hamiltonian: H = -J * sum_i Z_i Z_{i+1} - h * sum_i X_i.

Every method (exact, NNQS, VQE) builds its Hamiltonian through this module,
so the three-way comparison stays on the same model.
"""
from __future__ import annotations

import numpy as np


def build_quspin_basis(L: int):
    """QuSpin spin-1/2 basis for a chain of length L. Depends only on L, so
    reuse across an h-sweep instead of rebuilding per point."""
    from quspin.basis import spin_basis_1d

    return spin_basis_1d(L, pauli=True)


def build_quspin_hamiltonian(
    L: int,
    h: float,
    J: float = 1.0,
    basis=None,
    check_herm: bool = False,
    check_symm: bool = False,
    check_pcon: bool = False,
):
    """Open-BC TFIM as a QuSpin operator, ready for `eigsh`.

    Hermiticity/symmetry checks are off by default — they don't change the
    ground-state energy but roughly triple the cost at large L. Pass
    check_herm/check_symm/check_pcon=True to re-enable for debugging.
    """
    from quspin.operators import hamiltonian

    if basis is None:
        basis = build_quspin_basis(L)
    zz = [[-J, i, i + 1] for i in range(L - 1)]
    x = [[-h, i] for i in range(L)]
    return hamiltonian(
        [["zz", zz], ["x", x]],
        [],
        basis=basis,
        dtype=np.float64,
        check_herm=check_herm,
        check_symm=check_symm,
        check_pcon=check_pcon,
    )


def build_netket_hamiltonian(L: int, h: float, J: float = 1.0):
    """Open-BC TFIM as a NetKet operator, for NNQS training. Returns
    (hilbert_space, hamiltonian_operator)."""
    import netket as nk

    g = nk.graph.Chain(L, pbc=False)
    hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)
    H = sum(
        -J * (nk.operator.spin.sigmaz(hi, i) @ nk.operator.spin.sigmaz(hi, j))
        for i, j in g.edges()
    )
    H += sum(-h * nk.operator.spin.sigmax(hi, i) for i in g.nodes())
    return hi, H


def build_pennylane_hamiltonian(L: int, h: float, J: float = 1.0):
    """Open-BC TFIM as a PennyLane Hamiltonian, for the VQE circuit (simulator
    and MonarQ hardware both use this — only the device differs)."""
    import pennylane as qml

    coeffs, ops = [], []
    for i in range(L - 1):
        coeffs.append(-J)
        ops.append(qml.PauliZ(i) @ qml.PauliZ(i + 1))
    for i in range(L):
        coeffs.append(-h)
        ops.append(qml.PauliX(i))
    return qml.Hamiltonian(coeffs, ops)


def build_h_grid(
    h_min: float = 0.0,
    h_max: float = 2.0,
    coarse_step: float = 0.2,
    fine_lo: float = 0.7,
    fine_hi: float = 1.3,
    fine_step: float = 0.05,
    ndigits: int = 6,
) -> list[float]:
    """Sweep grid over h: coarse step away from the critical point, fine step
    near it (h in [fine_lo, fine_hi])."""
    coarse = np.arange(h_min, h_max + coarse_step / 2, coarse_step)
    fine = np.arange(fine_lo, fine_hi + fine_step / 2, fine_step)
    coarse_outside = coarse[(coarse < fine_lo) | (coarse > fine_hi)]
    grid = np.concatenate([coarse_outside, fine])
    grid = np.round(grid, ndigits)
    return sorted(set(grid.tolist()))
