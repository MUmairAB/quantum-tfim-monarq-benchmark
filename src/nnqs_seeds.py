"""Repeat every NNQS point across several random seeds and sampling budgets, so
the RBM track carries the same kind of uncertainty as the VQE one.

Every file in `results/nnqs/` is a single run at `seed=0`, which is already an
asymmetry when the VQE track is quoted as a spread over eight. The sharper
reason is that the deviation from exact tracks the reported Monte Carlo error
closely across that grid, so the accuracy on record may be measuring the
sampling floor at `n_samples=1024` rather than the ansatz. One run per point
cannot tell those apart; repeats at several budgets can.

`--summarize` records two things for that question. `spread_over_mc_error` is
the standard deviation of E0 across seeds over the mean quoted error — near 1
when the error bar describes the run-to-run variation, well above 1 when it
understates it. `n_below_exact` counts seeds landing below the exact ground
state, which a variational expectation value cannot do, so a nonzero count
marks a point that is reporting noise rather than a bound.

Usage mirrors src/vqe_seeds.py exactly:

    # how many runs does this config expand to?
    python -m src.nnqs_seeds --config configs/nnqs_seed_ensemble.yaml --list

    # run everything serially
    python -m src.nnqs_seeds --config configs/nnqs_seed_ensemble.yaml

    # run one slice, for a SLURM job array
    python -m src.nnqs_seeds --config configs/nnqs_seed_ensemble.yaml --index $SLURM_ARRAY_TASK_ID

    # aggregate the finished runs into results/nnqs_seeds/summary.json
    python -m src.nnqs_seeds --config configs/nnqs_seed_ensemble.yaml --summarize

Resumable in the same way as the other runners here: a point whose output file
already exists is skipped, so a requeued or interrupted array job only costs
the runs that were actually in flight.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import yaml

from src.hamiltonian import build_h_grid
from src.nnqs import load_exact_E0, train_nnqs

REPO_ROOT = Path(__file__).resolve().parent.parent


def expand_specs(config: dict) -> list[dict]:
    """Turn a config's `blocks` into a flat, ordered list of single-run specs.

    Each block is a cartesian product of L x h x n_samples x seed. Ordering is
    deterministic (block order, then L, h, n_samples, seed) because SLURM array
    indices refer to positions in this list — reordering a config after
    submitting would point the indices at different runs.

    A block gives h-values either as an explicit `h_values` list or as an
    `h_grid` spec passed to build_h_grid, so a block can cover the full field
    grid without spelling out twenty-one numbers.
    """
    J = config.get("J", 1.0)
    method = config.get("method", "nnqs_seed_ensemble")

    specs: list[dict] = []
    seen: set[tuple] = set()
    for block in config["blocks"]:
        seeds = block.get("seeds", config.get("seeds"))
        train = dict(config.get("train", {}))
        train.update(block.get("train", {}))
        n_samples_values = block.get("n_samples_values", [train.get("n_samples", 1024)])
        if "h_values" in block:
            h_values = block["h_values"]
        else:
            h_values = build_h_grid(**block.get("h_grid", config.get("h_grid", {})))

        for L in block["L_values"]:
            for h in h_values:
                for n_samples in n_samples_values:
                    for seed in seeds:
                        # Blocks are allowed to overlap. Keep the first
                        # occurrence only, so one run never gets two array
                        # indices racing to write the same file.
                        key = (L, float(h), n_samples, seed)
                        if key in seen:
                            continue
                        seen.add(key)
                        specs.append(
                            {
                                "L": L,
                                "h": float(h),
                                "J": J,
                                "seed": seed,
                                "n_samples": n_samples,
                                "alpha": train.get("alpha", 2),
                                "n_iter": train.get("n_iter", 300),
                                "learning_rate": train.get("learning_rate", 0.02),
                                "diag_shift": train.get("diag_shift", 0.01),
                                "method": method,
                                "block": block.get("name", ""),
                            }
                        )
    return specs


def estimated_cost(spec: dict) -> float:
    """Rough relative cost of one run, for ordering work longest-first.

    VMC cost is dominated by drawing n_samples per iteration with a sweep of
    length L, and by the stochastic-reconfiguration solve over the RBM's
    2L^2 + 3L parameters. n_samples * n_iter * L^2 tracks both well enough to
    schedule with. Only the ordering matters, not the units.
    """
    return spec["n_samples"] * spec["n_iter"] * (spec["L"] ** 2)


def output_path(spec: dict, results_dir: Path) -> Path:
    """Where a single run's JSON lands: results/nnqs_seeds/{L}/{h}_ns{n}_seed{s}.json

    n_samples is in the filename because the sampling-budget ablation reruns
    the same (L, h, seed) at several budgets, and they must not overwrite each
    other.
    """
    return results_dir / str(spec["L"]) / f"{spec['h']:.3f}_ns{spec['n_samples']}_seed{spec['seed']}.json"


def run_one(spec: dict, results_dir: Path, exact_dir: Path) -> bool:
    """Train one RBM and write its result file.

    Returns True if it actually ran, False if the output already existed and
    the point was skipped.
    """
    out_path = output_path(spec, results_dir)
    if out_path.exists():
        print(f"L={spec['L']:<3} h={spec['h']:.3f} ns={spec['n_samples']:<6} seed={spec['seed']}  "
              f"already done, skipping")
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    E_mean, E_err = train_nnqs(
        spec["L"],
        spec["h"],
        spec["J"],
        alpha=spec["alpha"],
        n_samples=spec["n_samples"],
        n_iter=spec["n_iter"],
        learning_rate=spec["learning_rate"],
        diag_shift=spec["diag_shift"],
        seed=spec["seed"],
        # Batch runner: no per-iteration bar, see train_nnqs's docstring.
        show_progress=False,
    )
    elapsed = time.perf_counter() - start

    E_exact = load_exact_E0(spec["L"], spec["h"], exact_dir)
    rel_error = abs(E_mean - E_exact) / abs(E_exact) if E_exact is not None else None

    # At h=0 the RBM represents the ground state exactly, so the Monte Carlo
    # variance is identically zero and NetKet returns NaN. JSON has no NaN
    # literal, so store it as null rather than writing an unparseable file.
    E_err_out = None if E_err != E_err else E_err

    # Signed, not absolute: a negative value means the estimate fell below the
    # exact ground state, which a true variational expectation cannot do. That
    # is the diagnostic, so it must not be folded into a magnitude.
    signed_error = (E_mean - E_exact) if E_exact is not None else None

    # Field names deliberately match results/nnqs/ so the same analysis code
    # reads both sets.
    record = {
        "L": spec["L"],
        "h": spec["h"],
        "J": spec["J"],
        "E0": E_mean,
        "E0_mc_error": E_err_out,
        "E0_exact": E_exact,
        "rel_error": rel_error,
        "signed_error": signed_error,
        "method": spec["method"],
        "boundary": "open",
        "wall_time_s": elapsed,
        "alpha": spec["alpha"],
        "n_samples": spec["n_samples"],
        "n_iter": spec["n_iter"],
        "learning_rate": spec["learning_rate"],
        "diag_shift": spec["diag_shift"],
        "seed": spec["seed"],
    }
    # Write via a temp file and rename, so a job killed mid-write can't leave
    # a truncated file that the resume logic would then treat as complete.
    tmp_path = out_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(record, indent=2, allow_nan=False))
    tmp_path.rename(out_path)

    rel_str = f"{rel_error:.4%}" if rel_error is not None else "n/a (no exact ref)"
    err_str = f"{E_err:.1e}" if E_err_out is not None else "n/a (exact, zero variance)"
    print(
        f"L={spec['L']:<3} h={spec['h']:.3f} ns={spec['n_samples']:<6} seed={spec['seed']}  "
        f"E0={E_mean:.6f}±{err_str}  rel_err={rel_str}  ({elapsed:.1f}s) -> {out_path}"
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

    Groups by (L, h, n_samples) and reports mean, sample standard deviation
    (ddof=1), sem, median, quartiles and range over the seeds that finished,
    for rel_error, E0, E0_mc_error and wall_time_s.

    Two extra fields carry the questions this ensemble exists to answer:

    `spread_over_mc_error` is std(E0 across seeds) / mean(quoted MC error). The
    quoted error is a within-run sampling uncertainty; the seed spread is the
    between-run variation an independent repeat would see. If the quoted error
    is an honest description of the uncertainty, the ratio sits near 1. Well
    above 1 means repeats disagree by more than any single run admits.

    `n_below_exact` counts seeds whose energy fell below the exact ground
    state. A variational expectation value cannot, so a nonzero count means
    those points are reporting sampling noise rather than a variational bound,
    and their apparent accuracy should not be read as ansatz quality.
    """
    groups: dict[tuple, list[dict]] = {}
    for spec in specs:
        path = output_path(spec, results_dir)
        if not path.exists():
            continue
        record = json.loads(path.read_text())
        groups.setdefault((spec["L"], spec["h"], spec["n_samples"]), []).append(record)

    summary = {}
    for (L, h, n_samples), records in sorted(groups.items()):
        records.sort(key=lambda r: r["seed"])
        below = sum(1 for r in records
                    if r.get("signed_error") is not None and r["signed_error"] < 0)
        entry = {"L": L, "h": h, "n_samples": n_samples, "n_seeds": len(records),
                 "n_below_exact": below,
                 "seeds": [r["seed"] for r in records],
                 "E0_exact": records[0]["E0_exact"],
                 "alpha": records[0]["alpha"], "n_iter": records[0]["n_iter"]}
        for field in ("rel_error", "E0", "E0_mc_error", "signed_error", "wall_time_s"):
            values = [r[field] for r in records if r.get(field) is not None]
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

        # Guarded: at h=0 the variance is identically zero, so the quoted error
        # is null for every seed and the ratio is undefined rather than huge.
        mc_mean = entry.get("E0_mc_error", {}).get("mean")
        e0_std = entry.get("E0", {}).get("std")
        entry["spread_over_mc_error"] = (
            e0_std / mc_mean if mc_mean and e0_std is not None and mc_mean > 0 else None
        )
        summary[f"L{L}_h{h:.3f}_ns{n_samples}"] = entry
    return summary


def main() -> None:
    """CLI entry point: expand the config, then list, run, or summarize."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results" / "nnqs_seeds")
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
