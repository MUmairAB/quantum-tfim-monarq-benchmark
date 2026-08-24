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
- Simulation track (exact + NNQS, classical compute): `L ∈ {4, 6, 8, 10, 12, 16, 20, 24}`. VQE covers a 7-point subset of `h` up to `L=20` (see `notebooks/03_vqe.ipynb`).
- Hardware track (VQE on real MonarQ hardware): `L=2` only, 1 circuit layer. `L ∈ {4, 6}` were characterized on MonarQ's noise-model simulator first, to pick a point where a signal would survive — see `notebooks/05_monarq_hardware.ipynb` for that characterization and for what actually sets the noise.

## 3. Repo structure

```
src/          hamiltonian.py, exact.py, nnqs.py, vqe.py — reusable library, run via CLI
              mitigation.py — verified readout mitigation for MonarQ
notebooks/    one notebook per method (01-03), plus cross-method analysis (04),
              MonarQ hardware characterization (05), and the VQE seed ensemble (06)
configs/      one YAML per sweep
results/      {exact,nnqs,vqe_sim}/{L}/{h}.json — simulation-track sweeps
              vqe_gpu/{L}/{h}_layers{n}.json — VQE circuit-depth scan
              vqe_seeds/{L}/{h}_layers{n}_seed{s}.json — the VQE grid repeated over
                8 random seeds, so each point has a spread rather than one run;
                summary.json aggregates each configuration
              monarq_sim/{L}/{h}_layers{n}.json — MonarQ noise-model characterization,
                first pass: one training run and one measurement per point
              monarq_sim_restarts/{L}/{h}_layers{n}.json — the same ladder retrained
                from 20 random starts and measured 10 times; supersedes monarq_sim
              monarq_mitigation/{L}/{h}_layers{n}_{raw,mitigated}.json — readout
                mitigation measured against its own unmitigated counts
              monarq_sim/{L}/{h}_layers{n}_rep{r}_{sim,sim_calibrated}.json — the L=2
                per-term repeats, 5 each, run against the generic noise constants
                (sim) and against live calibration (sim_calibrated)
              vqe_hardware/{L}/{h}.json — real MonarQ hardware results, first pass:
                one unrepeated measurement per point, no uncertainty
              vqe_hardware/{L}/{h}_layers{n}_rep{r}_{default,best}.json — per-term
                hardware repeats, 5 each, measuring every Hamiltonian term
                separately. Two qubit placements: whichever pair the plugin picks
                (default) and the pair that measures best (best) — these were the
                same pair when chosen, but see coupler*.json below
              vqe_hardware/2/coupler{c}_{h}_rep{r}_perterm.json — the same circuit
                on all five usable couplers, which is how the placement was checked
                against the advertised fidelity rather than assumed from it
              vqe_hardware/2/per_term_summary.json — the above aggregated to a mean
                and standard error per point
figures/
```

## 4. Setup

```bash
python3.11 -m venv monarq-env
source monarq-env/bin/activate
pip install -r requirements.txt
```

`quspin` needs OpenMP at the system level (`brew install libomp` on macOS). On a Compute Canada / Digital Research Alliance cluster, use `setup_computecanada.sh` instead of a plain `pip install` — see the script for why.

## 5. Running a sweep

```bash
python -m src.exact --config configs/exact_simulation_sweep.yaml
python -m src.nnqs  --config configs/nnqs_simulation_sweep.yaml
python -m src.vqe   --config configs/vqe_simulation_sweep.yaml
python -m src.vqe   --config configs/vqe_gpu_L20_layer_scan.yaml --results-dir results/vqe_gpu
```

Each run writes one JSON file per `(L, h)` point and skips points that already have a result, so an interrupted sweep can just be rerun. The last command is the `L=20` circuit-depth scan, not a GPU run despite the filename — see `src/vqe.py`'s module docstring.

MonarQ hardware runs (`src/vqe.py`'s `evaluate_on_device`) need real credentials and aren't a config-driven sweep — see `notebooks/05_monarq_hardware.ipynb` for the exact calls used.

## 6. Results

- `notebooks/01_exact_diagonalization.ipynb`, `02_nnqs.ipynb`, `03_vqe.ipynb` — one method each. NNQS matches exact to well under 1% across the full range, including at the critical point. VQE behaves differently in the two phases, and repeating each point over several seeds changes what that difference is — see `06_vqe_uncertainty.ipynb`.
- `notebooks/04_analysis.ipynb` — all methods on shared figures, including the real hardware comparison at `L=2`.
- `notebooks/05_monarq_hardware.ipynb` — the MonarQ noise characterization. Noise overwhelms this ansatz well before 10 circuit layers, and what sets the loss is two-qubit gates *per qubit* rather than total gate count, so circuit depth is the binding constraint and qubit count only a secondary one. Readout error mitigation recovers a real part of the error at one layer but turns harmful by four. Real hardware performs measurably worse than the noise-model simulator, which models gate noise only — no relaxation or dephasing.
- `notebooks/06_vqe_uncertainty.ipynb` — every VQE point repeated over 8 seeds. A single training run is one draw from a distribution of local minima, and the spread turns out to matter: near the critical point at `L=20` the optimizer either converges to ~0.4% or fails outright, so the chain length changes how often it fails rather than how accurate it is. Added depth does not change the converged accuracy there; it only helps in the ferromagnetic regime. Quote the medians and the count of failed seeds rather than a mean and standard deviation — where the seeds split into two groups the mean describes neither.
