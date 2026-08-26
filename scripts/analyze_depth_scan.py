"""Report the L=20 circuit-depth scan, and test what added depth actually repairs.

The depth-scan figure used to rest on three single-run points. This reads the
seed ensemble instead and asks the depth question as two separate ones, because
they can move independently:

  1. Does the TYPICAL run improve?   -> compare medians across depth.
  2. Does the OPTIMIZER FAIL LESS?   -> compare counts of tail seeds.

Depth could tighten the failure tail without shifting the median, or shift the
median without removing the tail. A single mean would blur the two together.

Central estimates are medians, not means. The seed distributions are
right-skewed: most seeds land close together and an occasional one falls into a
much worse local minimum, so one bad run moves the mean and inflates the
standard deviation a long way. Mean and std are printed alongside, but they
should not be read as a confidence interval — at n=8 the std largely reflects
whether a tail run happened to occur.

Usage:  python scripts/analyze_depth_scan.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.vqe_seeds import quantile

RESULTS = REPO_ROOT / "results" / "vqe_seeds" / "20"

H_VALUES = [0.4, 0.8, 0.95, 1.0, 1.05, 1.2]
DEPTHS = [10, 16, 20, 24, 30]
BASELINE_DEPTH = 10

# A run counts as a failure if it is anomalous by the standard Tukey fence AND
# large enough to matter by this project's existing 1% tolerance. The floor is
# the load-bearing half: without it the fence fires on fully converged
# configurations where every seed is essentially exact and nothing failed.
TUKEY_K = 1.5
ERROR_FLOOR = 0.01


def median_is_fictional(values: list[float]) -> bool:
    """True when the median falls in a gap between two groups of seeds.

    Near criticality at L=20 the optimizer either converges or fails outright,
    so the seeds separate into two groups with nothing between them. The median
    then lands in the empty space and reports a value no run ever produced.

    Checked exactly rather than by a shape heuristic: take the two middle runs
    that the median is averaged from, and ask whether the distance between them
    dominates the whole spread. If it does, the median is describing a gap.

    Kept identical to the version in analyze_seed_ensemble.py on purpose — the
    two analyses must agree on when a median is trustworthy.
    """
    v = sorted(values)
    n = len(v)
    if n < 4 or n % 2:
        return False
    middle_gap = v[n // 2] - v[n // 2 - 1]
    total = v[-1] - v[0]
    return total > 0 and middle_gap > 0.5 * total and middle_gap > 0.01


def load_group(h: float, n_layers: int) -> list[dict] | None:
    """All seed results for one (h, n_layers) configuration at L=20, or None."""
    paths = sorted(RESULTS.glob(f"{h:.3f}_layers{n_layers}_seed*.json"))
    if not paths:
        return None
    return [json.loads(p.read_text()) for p in paths]


def stats(records: list[dict]) -> dict:
    """Median, IQR, mean, std and tail-seed count for one configuration."""
    values = sorted(r["rel_error"] for r in records)
    n = len(values)
    q1, med, q3 = quantile(values, 0.25), quantile(values, 0.5), quantile(values, 0.75)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1) if n > 1 else 0.0
    fence = q3 + TUKEY_K * (q3 - q1)
    tail = [v for v in values if v > fence and v > ERROR_FLOOR]
    # Count of runs above the tolerance, independent of distribution shape.
    # The fence above assumes unimodal-with-outliers. If a configuration is
    # genuinely bimodal — half the seeds converging and half failing — then q3
    # sits inside the failed cluster, the fence lands above every point, and
    # the tail count reads zero on the least reliable configuration in the set.
    # The median misleads there too, reporting a value between the two modes
    # that no run ever produced. This count needs no shape assumption and
    # reuses the same 1% tolerance, so it adds no new arbitrary constant.
    failures = [v for v in values if v > ERROR_FLOOR]
    return {"n": n, "median": med, "q1": q1, "q3": q3, "mean": mean,
            "std": var ** 0.5, "min": values[0], "max": values[-1],
            "tail": tail, "failures": failures, "values": values}


def pct(x: float) -> str:
    return f"{x * 100:.2f}"


def main() -> None:
    """Print the depth-scan table, then the two depth claims per h-value."""
    grid: dict[tuple, dict] = {}
    for h in H_VALUES:
        for d in DEPTHS:
            records = load_group(h, d)
            if records:
                grid[(h, d)] = stats(records)

    if not grid:
        print(f"No results found in {RESULTS}. Nothing to report yet.")
        return

    print("=" * 92)
    print("L=20 circuit-depth scan — median relative error %, [seeds above 1% / n]")
    print("=" * 92)
    header = f"{'h':<7}" + "".join(f"{str(d) + ' layers':<17}" for d in DEPTHS)
    print(header)
    for h in H_VALUES:
        row = f"{h:<7}"
        for d in DEPTHS:
            s = grid.get((h, d))
            if s is None:
                row += f"{'—':<17}"
            else:
                # n_above_1% is shown as a fraction because it, not the median,
                # is what stays meaningful if the distribution is bimodal.
                row += f"{pct(s['median']) + ' [' + str(len(s['failures'])) + '/' + str(s['n']) + ']':<17}"
        print(row)

    incomplete = [(h, d) for h in H_VALUES for d in DEPTHS
                  if (h, d) in grid and grid[(h, d)]["n"] < 8]
    if incomplete:
        print("\nIncomplete configurations (fewer than 8 seeds so far):")
        for h, d in incomplete:
            print(f"  h={h}, {d} layers: n={grid[(h, d)]['n']}")

    print("\n" + "=" * 92)
    print("Mean +/- std for reference — NOT a confidence interval; at n=8 with right skew")
    print("the std mostly reflects whether a tail run happened to occur.")
    print("=" * 92)
    print(f"{'h':<7}" + "".join(f"{str(d) + ' layers':<17}" for d in DEPTHS))
    for h in H_VALUES:
        row = f"{h:<7}"
        for d in DEPTHS:
            s = grid.get((h, d))
            row += f"{'—':<17}" if s is None else f"{pct(s['mean']) + ' ± ' + pct(s['std']):<17}"
        print(row)

    print("\n" + "=" * 92)
    print(f"Does depth repair the failure? Each depth vs the {BASELINE_DEPTH}-layer baseline.")
    print("Two independent questions: does the typical run improve, and does the optimizer")
    print("fail less often. Tail counts are counts, not rates — at n=8 a single tail seed is")
    print("consistent with a true failure rate anywhere from a few percent to ~40%.")
    print("=" * 92)
    for h in H_VALUES:
        base = grid.get((h, BASELINE_DEPTH))
        print(f"\nh = {h}")
        if base is None:
            print(f"  no {BASELINE_DEPTH}-layer baseline on disk; cannot compare")
            continue
        print(f"  baseline {BASELINE_DEPTH:>2} layers: median {pct(base['median']):>6}%  "
              f"IQR {pct(base['q1'])}-{pct(base['q3'])}%  "
              f"above1% {len(base['failures'])}/{base['n']}  fence-tail {len(base['tail'])}")
        if median_is_fictional(base["values"]):
            print(f"           ^ the baseline median is FICTIONAL — the seeds split into two groups "
                  f"and no run produced anything near {pct(base['median'])}%. "
                  f"Compare the above-1% counts, not the medians, for this h.")
        for d in DEPTHS:
            if d == BASELINE_DEPTH or (h, d) not in grid:
                continue
            s = grid[(h, d)]
            ratio = base["median"] / s["median"] if s["median"] > 0 else float("inf")
            median_verdict = (f"median {ratio:.1f}x better" if ratio >= 1.5
                              else f"median {1 / ratio:.1f}x worse" if ratio <= 1 / 1.5
                              else "median ~unchanged")
            tail_verdict = (f"tail {len(base['tail'])} -> {len(s['tail'])}"
                            if base["tail"] or s["tail"] else "no tail either way")
            print(f"           {d:>2} layers: median {pct(s['median']):>6}%  "
                  f"IQR {pct(s['q1'])}-{pct(s['q3'])}%  "
                  f"above1% {len(s['failures'])}/{s['n']}   "
                  f"{median_verdict}; {tail_verdict}; "
                  f"above1% {len(base['failures'])}/{base['n']} -> {len(s['failures'])}/{s['n']}")

    print("\n" + "=" * 92)


if __name__ == "__main__":
    main()
