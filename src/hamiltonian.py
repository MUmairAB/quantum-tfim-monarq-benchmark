"""Shared TFIM convention: H = -J * sum_i Z_i Z_{i+1} - h * sum_i X_i, open BC.

See README.md for the locked convention. Every method (exact, NNQS, VQE) must
build its Hamiltonian through this module so the three-way comparison can't
silently drift onto three different models.
"""
from __future__ import annotations

import numpy as np


def build_quspin_basis(L: int):
    """Build the QuSpin spin-1/2 basis for a chain of length L (no symmetry reduction).

    This only depends on L, not on h or J, so callers sweeping over h for a
    fixed L should build it once and reuse it across the whole sweep.
    """
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
    """Build the open-BC TFIM Hamiltonian as a QuSpin operator for QuSpin's eigsh.

    Pass in a `basis` (from build_quspin_basis) to reuse it across an h-sweep
    instead of rebuilding it for every point — at L=24 that basis build alone
    costs ~20s, so reusing it matters for sweep wall-clock time.

    QuSpin's Hermiticity/symmetry checks are off by default here: they were
    verified to change nothing about the resulting ground-state energy but
    roughly triple the per-point cost at L=24 (confirmed against the roadmap's
    own reference value, L=6 h=1.0 -> E0=-7.29622981, both with checks on and off).
    Pass check_herm/check_symm/check_pcon=True to re-enable them if debugging.
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
    """Build the open-BC TFIM as a NetKet operator, for NNQS training (Phase 1/2).

    Returns (hilbert_space, hamiltonian_operator) since NetKet's variational
    state needs the Hilbert space object alongside the operator itself.
    """
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
    """Build the open-BC TFIM as a PennyLane Hamiltonian, for VQE (Phase 1/3).

    Used both for the simulator-only VQE in Phase 1 and, later, for real
    MonarQ hardware runs in Phase 3 — the Hamiltonian itself never changes
    between those two; only the device the circuit runs on does.
    """
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
    """Build the sweep grid: coarse step away from criticality, fine step near it.

    Matches README.md's locked sweep grid: coarse_step outside [fine_lo, fine_hi],
    fine_step inside it (since the physics changes fastest near the critical
    point h_c, that's where extra resolution actually earns its keep).
    """
    coarse = np.arange(h_min, h_max + coarse_step / 2, coarse_step)
    fine = np.arange(fine_lo, fine_hi + fine_step / 2, fine_step)
    coarse_outside = coarse[(coarse < fine_lo) | (coarse > fine_hi)]
    grid = np.concatenate([coarse_outside, fine])
    grid = np.round(grid, ndigits)
    return sorted(set(grid.tolist()))
