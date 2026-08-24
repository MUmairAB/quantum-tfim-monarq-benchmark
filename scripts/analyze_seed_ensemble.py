"""Read the seed ensemble and check which reported VQE effects survive the seed spread.

Every VQE number in the earlier results was a single optimization run, so an
apparent difference between two configurations could just be two draws from the
same distribution. This script compares each claimed effect against the spread
actually measured across seeds, using Welch's t statistic (unequal variances,
which is the right choice here since the spread varies a lot with h and depth).

An effect is only worth reporting if it is large compared with the standard
error of the difference. The printout gives that ratio for each claim.

Usage:  python scripts/analyze_seed_ensemble.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SUMMARY = REPO_ROOT / "results" / "vqe_seeds" / "summary.json"


def load() -> dict:
    """Load summary.json, keyed by (L, h, n_layers) for easy lookup."""
    raw = json.loads(SUMMARY.read_text())
    return {(e["L"], round(e["h"], 3), e["n_layers"]): e for e in raw.values()}


def welch(a: dict, b: dict) -> tuple[float, float, float]:
    """Compare two configurations' relative errors.

    Returns (difference in percentage points, standard error of that
    difference, t = difference / standard error). Both inputs are the
    `rel_error` sub-dictionaries written by src.vqe_seeds.summarize.
    """
    diff = (a["mean"] - b["mean"]) * 100
    se = math.sqrt((a["std"] * 100) ** 2 / a["n"] + (b["std"] * 100) ** 2 / b["n"])
    return diff, se, (diff / se if se > 0 else float("inf"))


def median_is_fictional(entry: dict) -> bool:
    """True when the median falls in a gap between two groups of seeds.

    Near criticality at L=20 the optimizer either converges or fails outright,
    so the seeds separate into two groups with nothing between them. The median
    then lands in the empty space and reports a value no run ever produced.

    Checked exactly rather than by a shape heuristic: take the two middle runs
    that the median is averaged from, and ask whether the distance between them
    dominates the whole spread. If it does, the median is describing a gap.
    """
    v = sorted(entry["rel_error"]["values"])
    n = len(v)
    if n < 4 or n % 2:
        return False
    middle_gap = v[n // 2] - v[n // 2 - 1]
    total = v[-1] - v[0]
    return total > 0 and middle_gap > 0.5 * total and middle_gap > 0.01


def fmt(entry: dict) -> str:
    """One configuration as 'mean ± std %, median, n seeds'.

    The median is shown alongside the mean because a single bad optimization
    run pulls the mean up noticeably; where the two disagree, the median is
    the better description of a typical run.
    """
    e = entry["rel_error"]
    txt = (f"{e['mean']*100:.3f} ± {e['std']*100:.3f} %  "
           f"(med {e['median']*100:.3f}, {entry['n_above_1pct']}/{e['n']} over 1%)")
    if median_is_fictional(entry):
        txt += "  <- SPLIT, median is not a real run"
    return txt


def claim(name: str, a: dict | None, b: dict | None, label_a: str, label_b: str) -> None:
    """Print one effect: both configurations, the difference, and how many
    standard errors that difference is."""
    print(f"\n{name}")
    if a is None or b is None:
        print("  missing data, cannot evaluate")
        return
    print(f"  {label_a:<28} {fmt(a)}")
    print(f"  {label_b:<28} {fmt(b)}")
    diff, se, t = welch(a["rel_error"], b["rel_error"])
    verdict = "SURVIVES" if abs(t) >= 3 else ("MARGINAL" if abs(t) >= 2 else "DOES NOT SURVIVE")
    print(f"  difference {diff:+.3f} pp,  se {se:.3f} pp,  t = {t:+.1f}  ->  {verdict}")


def main() -> None:
    """Walk through the five effects the analysis needs to rule in or out."""
    d = load()
    g = d.get

    print("=" * 78)
    print("VQE seed ensemble — do the reported effects exceed the seed-to-seed spread?")
    print("=" * 78)
    print("\nRule of thumb used below: |t| >= 3 survives, 2-3 marginal, < 2 does not.")

    claim("1. L=20 near-critical cliff at 10 layers (vs L=16)",
          g((20, 1.0, 10)), g((16, 1.0, 10)), "L=20, h=1.0, 10 layers", "L=16, h=1.0, 10 layers")

    claim("2. Depth repairs the L=20 near-critical failure (10 -> 16 layers)",
          g((20, 1.0, 10)), g((20, 1.0, 16)), "L=20, h=1.0, 10 layers", "L=20, h=1.0, 16 layers")

    claim("3. Any difference between 16 and 24 layers at L=20, h=1.0",
          g((20, 1.0, 16)), g((20, 1.0, 24)), "L=20, h=1.0, 16 layers", "L=20, h=1.0, 24 layers")

    claim("3b. Depth at h=0.4, L=20 (10 -> 16 layers)",
          g((20, 0.4, 10)), g((20, 0.4, 16)), "L=20, h=0.4, 10 layers", "L=20, h=0.4, 16 layers")

    print("\n\n4. h=0.4 gradual degradation with L (10 layers)")
    print(f"  {'L':<5}{'rel_error mean ± std (%)':<32}{'vs previous L':<24}")
    prev = None
    for L in (4, 6, 8, 10, 12, 16, 20):
        e = g((L, 0.4, 10))
        if e is None:
            print(f"  {L:<5}{'(missing)':<32}")
            continue
        line = f"  {L:<5}{fmt(e):<32}"
        if prev is not None:
            diff, se, t = welch(e["rel_error"], prev["rel_error"])
            line += f"{diff:+.3f} pp, t = {t:+.1f}"
        print(line)
        prev = e

    print("\n\n5. Non-monotonicity in L — is the mean monotonic once seeds are averaged?")
    print(f"  {'h':<7}{'sequence of means over L = 4,6,8,10,12,16,20 (%)':<58}{'monotonic?'}")
    for h in (0.4, 0.8, 0.95, 1.0, 1.05, 1.2, 1.6):
        means, drops = [], []
        for L in (4, 6, 8, 10, 12, 16, 20):
            e = g((L, h, 10))
            if e is None:
                continue
            if means and e["rel_error"]["mean"] < means[-1][1]:
                # A dip: check whether it is bigger than the noise on the pair.
                _, _, t = welch(e["rel_error"], means[-1][2])
                drops.append((L, t))
            means.append((L, e["rel_error"]["mean"], e["rel_error"]))
        seq = " -> ".join(f"{m*100:.2f}" for _, m, _ in means)
        if not drops:
            note = "yes, monotonic"
        else:
            worst = max(abs(t) for _, t in drops)
            note = ("no, but every dip is within noise" if worst < 2
                    else f"real dip at L={[L for L, t in drops if abs(t) == worst][0]} (t={worst:.1f})")
        print(f"  {h:<7}{seq:<58}{note}")

    print("\n\n6. Failure tail — seeds that landed in a much worse minimum")
    print("   A run counts as a failure if it is above Tukey's fence (q3 + 1.5*IQR) for its")
    print("   own configuration AND its error exceeds 1%. The fence alone is not enough:")
    print("   at L=4, h=1.6 every seed is under 0.2%, so nothing there has actually failed,")
    print("   but a fence test still flags the largest of them. 1% is the tolerance this")
    print("   project already works to, so it is the natural floor.")
    print(f"\n  {'configuration':<30}{'median':>9}{'IQR':>17}{'failed seeds':>26}")
    total = 0
    for (L, h, n), e in sorted(d.items()):
        r = e["rel_error"]
        iqr = r["q3"] - r["q1"]
        bad = [v for v in r["values"] if iqr > 0 and v > r["q3"] + 1.5 * iqr and v > 0.01]
        if not bad:
            continue
        total += len(bad)
        vals = ", ".join(f"{v*100:.2f}%" for v in sorted(bad))
        print(f"  L={L}, h={h}, {n} layers".ljust(32)
              + f"{r['median']*100:>7.2f}%"
              + f"{r['q1']*100:>8.2f}-{r['q3']*100:<8.2f}"
              + f"{vals:>24}")
    n_runs = sum(e["rel_error"]["n"] for e in d.values())
    print(f"\n  {total} failed runs out of {n_runs} ({total/n_runs*100:.1f}%), "
          f"in {sum(1 for _, e in d.items() if any(v > e['rel_error']['q3'] + 1.5*(e['rel_error']['q3']-e['rel_error']['q1']) and v > 0.01 for v in e['rel_error']['values']))} of {len(d)} configurations.")
    print("  With 8 seeds a single failure is consistent with a true rate anywhere from a")
    print("  few percent to ~40%, so these are counts, not a measured failure rate.")

    print("\n" + "=" * 78)


if __name__ == "__main__":
    main()
