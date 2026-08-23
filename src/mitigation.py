"""Readout-error mitigation on MonarQ, with a guard against the plugin's silent no-op.

The plugin ships `MatrixReadoutMitigation`, a calibration-matrix correction that
runs as a post-processing step on the measured counts. It catches every
exception it raises internally, logs it, and hands back the counts it was given
unchanged — so a correction that failed outright looks exactly like one that ran
and found nothing to fix. There are two easy ways to trip it, and this project
hit both:

1. Building `monarq.sim` without a `client` leaves the plugin's API adapter
   uninitialized, so the calibration lookup fails before any correction happens.
2. Even with a client, MonarQ reports all-zero calibration for qubits that are
   not currently live (only 9-14 carry real numbers). A circuit left on logical
   wires 0..L-1 builds a degenerate readout matrix, and the correction is
   skipped again.

`VerifiedReadoutMitigation` checks the calibration before correcting, records
what it did on every call, and turns a skipped correction into a loud error via
`raise_if_not_applied`. That check has to happen outside the plugin, which
swallows exceptions raised inside a processing step as well.

One thing to keep in mind when reading simulator numbers from this module:
`monarq.sim` injects readout noise using the same per-qubit calibration matrix
the mitigation step inverts, so on the simulator the correction is exact by
construction. What it recovers is the readout-error share of the total error —
an upper bound on what mitigation could recover on real hardware, not a
prediction of it.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, stdev

from pennylane_calculquebec.processing.config import MonarqDefaultConfig, ProcessingConfig
from pennylane_calculquebec.processing.steps.readout_error_mitigation import (
    MatrixReadoutMitigation,
    all_results,
    get_readout_fidelities,
)
from pennylane_calculquebec.utility.debug import compute_expval, get_measurement_wires

from src.hamiltonian import build_pennylane_hamiltonian
from src.vqe import evaluate_on_device

# The name monarq.sim and monarq.default use internally for the machine. It is
# not "monarq" — that is the backend name, and the calibration lookup fails with
# it.
MACHINE_NAME = "yamaska"


@dataclass
class MitigationRecord:
    """What the mitigation step did during one circuit execution."""

    wires: list[int]
    readout0: list[float]
    readout1: list[float]
    raw_counts: dict[str, float]
    mitigated_counts: dict[str, float]
    applied: bool
    counts_changed: bool = False
    problem: str | None = None

    def as_dict(self) -> dict:
        """Plain-dict form, for writing into a results file."""
        return {
            "wires": self.wires,
            "readout0": self.readout0,
            "readout1": self.readout1,
            "raw_counts": self.raw_counts,
            "mitigated_counts": self.mitigated_counts,
            "applied": self.applied,
            "counts_changed": self.counts_changed,
            "problem": self.problem,
        }


class VerifiedReadoutMitigation(MatrixReadoutMitigation):
    """Calibration-matrix readout mitigation that refuses to fail quietly.

    Drop-in replacement for the plugin's `MatrixReadoutMitigation` — append it
    to a `ProcessingConfig` the same way. It adds three things:

    - The calibration is checked before the correction runs: the API adapter has
      to be initialized, and every readout fidelity has to be a real number
      above zero. A qubit MonarQ is not currently calibrating reports zero, and
      the correction is meaningless there.
    - The plugin caches its calibration matrix on the *class*, keyed by nothing
      at all, and only reconsiders it once a day. Consecutive Hamiltonian terms
      measure different qubits, so that cache is cleared on every call instead
      of being silently reused at the wrong size.
    - Every call is recorded, so `raise_if_not_applied` can report a correction
      that never happened. Whether it happened is decided by object identity,
      not by whether the counts moved — see `execute` for why that distinction
      matters.
    """

    def __init__(self, machine_name: str = MACHINE_NAME):
        super().__init__(machine_name)
        self.records: list[MitigationRecord] = []

    def reset(self) -> None:
        """Drop the records from a previous measurement. Call before each fresh run."""
        self.records = []

    def execute(self, tape, results):
        """Correct one circuit's counts, and remember whether the correction actually happened."""
        # Clear the plugin's class-level cache: it was built for whichever qubits
        # the previous term measured, and reusing it here is either wrong or
        # raises on a shape mismatch (which the plugin would then swallow).
        MatrixReadoutMitigation._readout_matrix_normalized = None
        MatrixReadoutMitigation._readout_matrix_reduced = None
        MatrixReadoutMitigation._readout_matrix_reduced_inverted = None

        wires = sorted(int(w) for w in get_measurement_wires(tape))
        raw = _as_counts(results, len(wires))

        try:
            readout0, readout1 = get_readout_fidelities(self.machine_name, wires)
            readout0 = [float(v) for v in readout0]
            readout1 = [float(v) for v in readout1]
        except Exception as exc:
            self._record(
                wires, [], [], raw, raw, False, False,
                f"calibration lookup failed: {type(exc).__name__}: {exc}",
            )
            return results

        fidelities = readout0 + readout1
        if min(fidelities) <= 0.0:
            self._record(
                wires, readout0, readout1, raw, raw, False, False,
                f"qubits {wires} report zero readout fidelity — not currently calibrated",
            )
            return results
        if min(fidelities) >= 1.0:
            self._record(
                wires, readout0, readout1, raw, raw, False, False,
                f"qubits {wires} report perfect readout — there is nothing to correct",
            )
            return results

        mitigated = super().execute(tape, results)

        # Object identity is the exact test for whether the correction ran. Each
        # of the plugin's bail-out paths hands back the very object it was given;
        # the one path that actually corrects builds a fresh dict. Comparing the
        # counts instead looks tempting and is wrong: a correction that ran
        # perfectly can still leave them untouched, because a distribution
        # already sitting at the calibration matrix's fixed point moves by less
        # than half a count and rounds straight back. On a qubit with readout
        # fidelities 0.975/0.917 that is any result between 765 and 772 of 1000.
        applied = mitigated is not results
        mitigated_counts = _as_counts(mitigated, len(wires))
        changed = any(abs(mitigated_counts[label] - raw[label]) > 1e-9 for label in raw)
        problem = None if applied else "the plugin handed back its input untouched, so no correction ran"
        self._record(wires, readout0, readout1, raw, mitigated_counts, applied, changed, problem)
        return mitigated

    def _record(self, wires, readout0, readout1, raw, mitigated, applied,
                counts_changed=False, problem=None) -> None:
        """Append one MitigationRecord."""
        self.records.append(
            MitigationRecord(wires, readout0, readout1, raw, mitigated,
                             applied, counts_changed, problem)
        )

    def raise_if_not_applied(self) -> None:
        """Raise if any circuit went through without its counts actually being corrected.

        The plugin never raises on its own, so without this a run that reports
        "mitigated" energies may be reporting raw ones.
        """
        if not self.records:
            raise RuntimeError(
                "readout mitigation was configured but never ran — no circuit reached the "
                "post-processing step"
            )
        failed = [r for r in self.records if not r.applied]
        if failed:
            details = "; ".join(f"wires {r.wires}: {r.problem}" for r in failed)
            raise RuntimeError(
                f"readout mitigation did not apply on {len(failed)} of {len(self.records)} "
                f"circuits — {details}"
            )

    @property
    def measured_wires(self) -> list[list[int]]:
        """The physical qubits each circuit was measured on, in execution order."""
        return [r.wires for r in self.records]


def _as_counts(results, n_wires: int) -> dict[str, float]:
    """Normalise a counts dict to plain floats over every bitstring of n_wires."""
    return {label: float(value) for label, value in all_results(results, n_wires).items()}


def mitigated_config(machine_name: str = MACHINE_NAME, use_benchmark: bool = True) -> ProcessingConfig:
    """MonarQ's default transpilation pipeline with verified readout mitigation on the end.

    `use_benchmark` has to agree with whether the device is given a client:
    monarq.sim sets its own noise model from `client is not None`, and a config
    that disagrees makes placement skip its benchmark-driven steps, which
    surfaces much later as "Your circuit should contain only MonarQ native
    gates. Cannot simulate noise."

    The mitigation step is the last one, so grab it with `config.steps[-1]` if
    you want its records.
    """
    config = MonarqDefaultConfig(machine_name, use_benchmark=use_benchmark)
    config.steps.append(VerifiedReadoutMitigation(machine_name))
    return config


def measure_raw_and_mitigated(L, h, params, J=1.0, client=None, shots=1000, machine_name=MACHINE_NAME):
    """Measure a trained circuit on monarq.sim and return raw and mitigated energies from the same shots.

    Readout mitigation is pure post-processing of measured counts, so both
    energies come out of one set of circuit executions: the mitigation step
    keeps the counts it was handed alongside the ones it produced. Pairing them
    this way takes shot noise out of the comparison entirely — the two energies
    differ by the correction and nothing else.

    A `client` is required. Without one the plugin cannot read the calibration,
    and this raises rather than quietly reporting a raw number twice.

    Returns (E_raw, E_mitigated, records).
    """
    config = mitigated_config(machine_name, use_benchmark=client is not None)
    step = config.steps[-1]
    step.reset()

    E_mitigated = evaluate_on_device(
        L, h, params, J=J, device="monarq.sim", shots=shots,
        client=client, processing_config=config,
    )
    step.raise_if_not_applied()

    coeffs, _ = build_pennylane_hamiltonian(L, h, J).terms()
    if len(step.records) != len(coeffs):
        raise RuntimeError(
            f"expected one mitigation record per Hamiltonian term "
            f"({len(coeffs)}), got {len(step.records)}"
        )

    # Same expectation-value code the device itself uses, applied to the counts
    # as they were before the correction.
    E_raw = sum(
        float(coeff) * compute_expval(record.raw_counts)
        for coeff, record in zip(coeffs, step.records)
    )
    return float(E_raw), float(E_mitigated), list(step.records)


def repeat_raw_and_mitigated(L, h, params, n_repeats=10, J=1.0, client=None, shots=1000,
                             machine_name=MACHINE_NAME, verbose=True) -> dict:
    """Repeat `measure_raw_and_mitigated` and summarise the spread.

    A single 1000-shot measurement of this energy carries a few percent of
    statistical error, which is the same size as the effect being looked for, so
    a single pair of numbers cannot settle whether mitigation helps. This runs
    the measurement `n_repeats` times and reports mean and standard error for
    each arm, plus the paired difference — the per-repeat mitigated-minus-raw
    gap, which is the sharper statistic because the two arms share their shots.
    """
    raw_values, mitigated_values, wire_sets = [], [], []
    n_circuits = n_moved = 0
    for i in range(n_repeats):
        E_raw, E_mitigated, records = measure_raw_and_mitigated(
            L, h, params, J=J, client=client, shots=shots, machine_name=machine_name
        )
        raw_values.append(E_raw)
        mitigated_values.append(E_mitigated)
        wire_sets = [r.wires for r in records]
        n_circuits += len(records)
        n_moved += sum(1 for r in records if r.counts_changed)
        if verbose:
            print(f"  repeat {i + 1}/{n_repeats}: raw={E_raw:+.4f}  mitigated={E_mitigated:+.4f}")

    differences = [m - r for m, r in zip(mitigated_values, raw_values)]
    return {
        "n_repeats": n_repeats,
        "raw": _summarise(raw_values),
        "mitigated": _summarise(mitigated_values),
        "paired_difference": _summarise(differences),
        "measured_wires": wire_sets,
        # Reaching this line means every circuit passed raise_if_not_applied.
        # n_counts_moved is lower than n_circuits when a corrected distribution
        # rounds back to where it started, which is a real outcome, not a miss.
        "mitigation_verified": True,
        "n_circuits": n_circuits,
        "n_counts_moved": n_moved,
    }


def _summarise(values: list[float]) -> dict:
    """Mean, sample standard deviation and standard error of a list of measurements."""
    n = len(values)
    sd = stdev(values) if n > 1 else 0.0
    return {
        "values": [float(v) for v in values],
        "mean": float(mean(values)),
        "std": float(sd),
        "standard_error": float(sd / (n ** 0.5)) if n > 1 else 0.0,
    }
