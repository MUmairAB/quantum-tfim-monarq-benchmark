"""Repeat every VQE point across several random seeds, so each energy comes
with a real uncertainty instead of being a single unrepeated optimization.

The VQE results already in `results/vqe_sim/` are one run each at `seed=0`.
The HVA optimizer is not convex, so a different parameter initialization can
land in a different local minimum — and the spread between seeds turns out to
be comparable to some of the effects we want to report. This module reruns the
same grid over a set of seeds and writes one file per (L, h, n_layers, seed),
so downstream analysis has a spread to quote rather than a single number.

Quote the median and a count of seeds over 1%, not mean +/- std. Near the
critical point the seeds do not form one cluster: most converge to a few tenths
of a percent and a few fail outright at 6-8%. A mean sits in the gap between the
two groups and names a value no run produced. `--summarize` records median,
quartiles and `n_above_1pct` for exactly this reason.

Usage:

    # how many runs does this config expand to?
    python -m src.vqe_seeds --config configs/vqe_seed_ensemble.yaml --list

    # run everything serially (fine for the small-L blocks)
    python -m src.vqe_seeds --config configs/vqe_seed_ensemble.yaml

    # run one slice, for a SLURM job array
    python -m src.vqe_seeds --config configs/vqe_seed_ensemble.yaml --index $SLURM_ARRAY_TASK_ID

    # aggregate the finished runs into results/vqe_seeds/summary.json
    python -m src.vqe_seeds --config configs/vqe_seed_ensemble.yaml --summarize

Resumable in the same way as the other runners here: a point whose output
file already exists is skipped, so a requeued or interrupted array job only
costs the runs that were actually in flight.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import yaml

from src.nnqs import load_exact_E0
from src.vqe import train_vqe

REPO_ROOT = Path(__file__).resolve().parent.parent


def expand_specs(config: dict) -> list[dict]:
    """Turn a config's `blocks` into a flat, ordered list of single-run specs.

    Each block is a cartesian product of L x h x n_layers x seed. Ordering is
    deterministic (block order, then L, h, n_layers, seed) because SLURM array
    indices refer to positions in this list — reordering a config after
    submitting would point the indices at different runs.
    """
    J = config.get("J", 1.0)
    method = config.get("method", "vqe_seed_ensemble")
    device = config.get("device", "lightning.qubit")

    specs: list[dict] = []
    seen: set[tuple] = set()
    for block in config["blocks"]:
        seeds = block.get("seeds", config.get("seeds"))
        n_steps = block.get("n_steps", config.get("n_steps", 600))
        stepsize = block.get("stepsize", config.get("stepsize", 0.1))
        for L in block["L_values"]:
            for h in block["h_values"]:
                for n_layers in block["n_layers_values"]:
                    for seed in seeds:
                        # Blocks are allowed to overlap (the L=20 h-scan repeats
                        # points the L=20 depth study already covers). Keep the
                        # first occurrence only, so one run never gets two array
                        # indices racing to write the same file.
                        key = (L, float(h), n_layers, seed)
                        if key in seen:
                            continue
                        seen.add(key)
                        specs.append(
                            {
                                "L": L,
                                "h": float(h),
                                "J": J,
                                "n_layers": n_layers,
                                "n_steps": n_steps,
                                "stepsize": stepsize,
                                "seed": seed,
                                "device": device,
                                "method": method,
                                "block": block.get("name", ""),
                            }
                        )
    return specs


def estimated_cost(spec: dict) -> float:
    """Rough relative cost of one run, for ordering work longest-first.

    Statevector simulation is O(2^L) per gate and the gate count is linear in
    both depth and steps, so 2^L * n_layers * n_steps tracks the real runtime
    closely enough to schedule with. Only the ordering matters, not the units.
    """
    return (2 ** spec["L"]) * spec["n_layers"] * spec["n_steps"]


def output_path(spec: dict, results_dir: Path) -> Path:
    """Where a single run's JSON lands: results/vqe_seeds/{L}/{h}_layers{n}_seed{s}.json"""
    return results_dir / str(spec["L"]) / f"{spec['h']:.3f}_layers{spec['n_layers']}_seed{spec['seed']}.json"


def run_one(spec: dict, results_dir: Path, exact_dir: Path) -> bool:
    """Run a single VQE optimization and write its result file.

    Returns True if it actually ran, False if the output already existed and
    the point was skipped.
    """
    out_path = output_path(spec, results_dir)
    if out_path.exists():
        print(f"L={spec['L']:<3} h={spec['h']:.3f} layers={spec['n_layers']:<3} seed={spec['seed']}  "
              f"already done, skipping")
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    E, two_qubit_gates = train_vqe(
        spec["L"],
        spec["h"],
        spec["J"],
        n_layers=spec["n_layers"],
        n_steps=spec["n_steps"],
        stepsize=spec["stepsize"],
        seed=spec["seed"],
        device=spec["device"],
    )
    elapsed = time.perf_counter() - start

    E_exact = load_exact_E0(spec["L"], spec["h"], exact_dir)
    rel_error = abs(E - E_exact) / abs(E_exact) if E_exact is not None else None

    # Field names deliberately match results/vqe_sim/ so the same analysis
    # code reads both sets.
    record = {
        "L": spec["L"],
        "h": spec["h"],
        "J": spec["J"],
        "E0": E,
        "E0_exact": E_exact,
        "rel_error": rel_error,
        "method": spec["method"],
        "boundary": "open",
        "two_qubit_gate_count": two_qubit_gates,
        "wall_time_s": elapsed,
        "n_layers": spec["n_layers"],
        "n_steps": spec["n_steps"],
        "stepsize": spec["stepsize"],
        "seed": spec["seed"],
        "device": spec["device"],
    }
    # Write via a temp file and rename, so a job killed mid-write can't leave
    # a truncated file that the resume logic would then treat as complete.
    tmp_path = out_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(record, indent=2))
    tmp_path.rename(out_path)

    rel_str = f"{rel_error:.4%}" if rel_error is not None else "n/a (no exact ref)"
    print(
        f"L={spec['L']:<3} h={spec['h']:.3f} layers={spec['n_layers']:<3} seed={spec['seed']}  "
        f"E0={E:.6f}  rel_err={rel_str}  ({elapsed:.1f}s) -> {out_path}"
    )
    return True


def quantile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolated quantile of an already-sorted list."""
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def summarize(specs: list[dict], results_dir: Path) -> dict:
    """Collect the finished runs and reduce them to per-configuration statistics.

    Groups by (L, h, n_layers) and reports, over the seeds that finished:
    mean, sample standard deviation (ddof=1), and sem = std/sqrt(n), which is
    the number to compare effect sizes against.

    Median and quartiles are recorded alongside them because the seed
    distributions are not symmetric — most seeds land close together and an
    occasional one falls into a much worse local minimum. A single such run
    moves the mean and inflates the std a lot, so the median and IQR describe
    the typical run better, and the gap between mean and median is itself a
    useful signal that a configuration has a failure tail.

    `n_above_1pct` counts how many seeds exceeded 1% relative error. It is
    recorded because every other statistic here assumes the runs form one
    cluster with occasional outliers, and some configurations do not: at
    L=20 near criticality the optimizer either converges or fails outright,
    giving two separated groups. For a split like that the median falls in
    the empty gap between them and describes no actual run, and an
    outlier fence sits above the failed group and flags nothing. A plain
    count needs no assumption about the shape and stays honest either way.
    """
    groups: dict[tuple, list[dict]] = {}
    for spec in specs:
        path = output_path(spec, results_dir)
        if not path.exists():
            continue
        record = json.loads(path.read_text())
        groups.setdefault((spec["L"], spec["h"], spec["n_layers"]), []).append(record)

    summary = {}
    for (L, h, n_layers), records in sorted(groups.items()):
        records.sort(key=lambda r: r["seed"])
        above = sum(1 for r in records if r["rel_error"] is not None and r["rel_error"] > 0.01)
        entry = {"L": L, "h": h, "n_layers": n_layers, "n_seeds": len(records),
                 "n_above_1pct": above,
                 "seeds": [r["seed"] for r in records],
                 "E0_exact": records[0]["E0_exact"],
                 "n_steps": records[0]["n_steps"], "stepsize": records[0]["stepsize"]}
        for field in ("rel_error", "E0", "wall_time_s"):
            values = [r[field] for r in records if r[field] is not None]
            if not values:
                continue
            mean = sum(values) / len(values)
            std = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1)) if len(values) > 1 else 0.0
            ordered = sorted(values)
            entry[field] = {
                "mean": mean,
                "std": std,
                "sem": std / math.sqrt(len(values)) if len(values) > 1 else 0.0,
                "median": quantile(ordered, 0.5),
                "q1": quantile(ordered, 0.25),
                "q3": quantile(ordered, 0.75),
                "min": ordered[0],
                "max": ordered[-1],
                "n": len(values),
                "values": values,
            }
        summary[f"L{L}_h{h:.3f}_layers{n_layers}"] = entry
    return summary


def main() -> None:
    """CLI entry point: expand the config, then list, run, or summarize."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results" / "vqe_seeds")
    parser.add_argument("--exact-dir", type=Path, default=REPO_ROOT / "results" / "exact")
    parser.add_argument("--index", type=int, default=None,
                        help="run only the slice of runs for this array index (0-based)")
    parser.add_argument("--chunk-size", type=int, default=1,
                        help="how many runs one array index covers; lets short runs share a task")
    parser.add_argument("--blocks", type=str, default=None,
                        help="comma-separated block names to restrict to")
    parser.add_argument("--list", action="store_true", help="print the run count and array sizing, then exit")
    parser.add_argument("--task-order", action="store_true",
                        help="print task indices longest-job-first, so a parallel runner "
                             "starts the expensive runs early instead of trailing them")
    parser.add_argument("--summarize", action="store_true", help="write summary.json from finished runs and exit")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    specs = expand_specs(config)
    if args.blocks:
        wanted = {name.strip() for name in args.blocks.split(",")}
        specs = [s for s in specs if s["block"] in wanted]

    if args.list:
        n_tasks = math.ceil(len(specs) / args.chunk_size)
        print(f"{len(specs)} runs; with --chunk-size {args.chunk_size} that is "
              f"array 0-{n_tasks - 1} ({n_tasks} tasks)")
        pending = [s for s in specs if not output_path(s, args.results_dir).exists()]
        print(f"{len(pending)} still pending, {len(specs) - len(pending)} already on disk")
        by_block: dict[str, int] = {}
        for s in specs:
            by_block[s["block"]] = by_block.get(s["block"], 0) + 1
        for name, count in by_block.items():
            print(f"  block {name!r}: {count} runs")
        return

    if args.task_order:
        # Cost of a task is the cost of the runs it covers; skip tasks whose
        # runs are all already on disk so a resumed run does not re-queue them.
        n_tasks = math.ceil(len(specs) / args.chunk_size)
        tasks = []
        for i in range(n_tasks):
            chunk = specs[i * args.chunk_size:(i + 1) * args.chunk_size]
            if all(output_path(s, args.results_dir).exists() for s in chunk):
                continue
            tasks.append((sum(estimated_cost(s) for s in chunk), i))
        for _, i in sorted(tasks, reverse=True):
            print(i)
        return

    if args.summarize:
        summary = summarize(specs, args.results_dir)
        out_path = args.results_dir / "summary.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2))
        print(f"{len(summary)} configurations summarized -> {out_path}")
        return

    if args.index is not None:
        start = args.index * args.chunk_size
        specs = specs[start:start + args.chunk_size]
        if not specs:
            print(f"index {args.index} is past the end of the run list, nothing to do")
            return

    ran = 0
    for spec in specs:
        ran += run_one(spec, args.results_dir, args.exact_dir)
    print(f"done: {ran} runs executed, {len(specs) - ran} skipped")


if __name__ == "__main__":
    main()
