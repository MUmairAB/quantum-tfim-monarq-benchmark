# TFIM: NNQS vs. VQE vs. exact

Three-way comparison of neural-network quantum states (NNQS), variational quantum
eigensolver (VQE) on MonarQ, and exact diagonalization for the transverse-field
Ising model (TFIM). See [quantum_implementation_roadmap.md](quantum_implementation_roadmap.md)
for the full phased plan; this README holds the conventions that every method
(exact, NNQS, VQE) must agree on.

## Hamiltonian convention

```
H = -J * sum_i Z_i Z_{i+1}  -  h * sum_i X_i
```

- `J = 1` (fixed).
- Critical point at `h_c = J = 1`.
- **Boundary conditions: open** (no wraparound `Z_{L-1} Z_0` term). Chosen because
  it maps cleanly onto hardware connectivity with no extra SWAPs, and avoids the
  periodic case's fermion-parity-sector subtlety in the analytic solution.

Every method — QuSpin exact diagonalization, NetKet NNQS, and the PennyLane HVA
VQE (simulator and MonarQ hardware) — must use exactly this Hamiltonian and
boundary condition. Any deviation makes the "three-way comparison" silently
compare three different models.

## Sweep grid

- `h/J` range: 0 to 2.
  - Coarse step 0.2 away from criticality.
  - Fine step 0.05 within `h ∈ [0.7, 1.3]` (near `h_c`).
- **System sizes — simulation track:** `L ∈ {4, 6, 8, 10, 12, 16, 20, 24}`
  (NNQS + exact diagonalization; classical compute only, unaffected by hardware
  limits).
- **System sizes — hardware ladder:** `L ∈ {2, 4, 6}`, capped by the 6 qubits
  currently available on MonarQ (confirmed via program demo + benchmark pull:
  qubits 9–14, couplers 15–19 real, everything else zero). Re-check this
  right before submitting the Phase 3 batch — the *count* is settled, but which
  physical qubits carry the calibration can drift.

## Scope decisions (locked)

- Mixed-field extension: parked. Optional stretch goal only if the core
  comparison finishes early — not a build target.

## Repo layout

```
src/          # hamiltonian.py, nnqs.py, vqe.py, exact.py — reusable library, CLI-driven
notebooks/    # one notebook per phase/method: imports from src/, runs it, narrates in markdown
configs/      # one yaml per sweep (L, h, method as parameters — config-driven, not hardcoded)
results/{exact,nnqs,vqe_sim,vqe_hardware}/{L}/{h}.json
figures/
```
