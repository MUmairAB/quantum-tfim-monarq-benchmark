# TFIM: NNQS vs. VQE vs. exact diagonalization

A three-way comparison of methods for finding ground-state energies of the transverse-field Ising model (TFIM): exact diagonalization, neural-network quantum states (NNQS), and a variational quantum eigensolver (VQE) running on MonarQ, Calcul Québec's superconducting-qubit backend.

## 1. Model

```
H = -J * sum_i Z_i Z_{i+1}  -  h * sum_i X_i
```

- `J = 1` (fixed), critical point at `h_c = 1`.
- Open boundary conditions (no `Z_{L-1} Z_0` wraparound term) — this maps directly onto hardware connectivity without extra SWAP gates.

All three methods build this same Hamiltonian and boundary condition through `src/hamiltonian.py`, so the comparison stays apples-to-apples.

## 2. Sweep grid

- `h/J` from 0 to 2, step 0.2 away from criticality and 0.05 within `[0.7, 1.3]`.
- Simulation track (exact + NNQS, classical compute): `L ∈ {4, 6, 8, 10, 12, 16, 20, 24}`.
- Hardware track (VQE on MonarQ): `L ∈ {2, 4, 6}`, capped by MonarQ's currently available qubits.

## 3. Repo structure

```
src/          hamiltonian.py, exact.py, nnqs.py, vqe.py — reusable library, run via CLI
notebooks/    one notebook per method, imports from src/ and narrates the results
configs/      one YAML per sweep
results/      {exact,nnqs,vqe_sim,vqe_hardware}/{L}/{h}.json
figures/
```

## 4. Setup

```bash
python3.11 -m venv monarq-env
source monarq-env/bin/activate
pip install -r requirements.txt
```

`quspin` needs OpenMP at the system level (`brew install libomp` on macOS). On Compute Canada, use `setup_computecanada.sh` instead of a plain `pip install` — see the script for why.

## 5. Running a sweep

```bash
python -m src.exact --config configs/exact_simulation_sweep.yaml
python -m src.nnqs  --config configs/nnqs_simulation_sweep.yaml
python -m src.vqe   --config configs/vqe_simulation_sweep.yaml
```

Each run writes one JSON file per `(L, h)` point and skips points that already have a result, so an interrupted sweep can just be rerun.

## 6. Results

`notebooks/01_exact_diagonalization.ipynb` and `notebooks/02_nnqs.ipynb` walk through the simulation-track results and produce the figures in `figures/`. NNQS matches exact diagonalization to well under 1% across the full `L` range, including at the critical point.
