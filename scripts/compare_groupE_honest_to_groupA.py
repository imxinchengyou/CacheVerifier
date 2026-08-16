"""Direction 1, Group E closing step: does each honestly-calibrated Group E
test-half point (threshold_calibration_ablation_groupE.py output) beat the
Group A static-threshold frontier at matched error rate?

Interpolates the Group A frontier (error_rate, hit_rate) linearly and
compares each Group E honest point's hit_rate against the interpolated
Group A hit_rate at that same error_rate -- the same "fair, matched-error-
rate" comparison PAPER.md describes for Group D/E's original grid-search
numbers. Win/tie/loss is decided by whether the point's own bootstrap CI
(already computed by threshold_calibration_ablation_groupE.py) clears the
interpolated Group A value, not just the point estimate -- consistent with
the CI-overlap convention already used elsewhere in this repo (e.g. the
bucketing ablation table in RESEARCH_PROPOSAL.md).

Usage:
    python scripts/compare_groupE_honest_to_groupA.py \
        --honest results/lmarena_groupE_honest_calibration.json \
        --group-a results/lmarena_groupA.json
"""

import argparse
import json
from pathlib import Path


def interpolate(frontier: list[tuple[float, float]], error_rate: float) -> float | None:
    """Piecewise-linear interpolation of hit_rate at a given error_rate.
    Returns None if error_rate falls outside the frontier's covered range."""
    frontier = sorted(frontier)
    if error_rate < frontier[0][0] or error_rate > frontier[-1][0]:
        return None
    for (e0, h0), (e1, h1) in zip(frontier, frontier[1:]):
        if e0 <= error_rate <= e1:
            if e1 == e0:
                return max(h0, h1)
            t = (error_rate - e0) / (e1 - e0)
            return h0 + t * (h1 - h0)
    return frontier[-1][1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--honest", required=True)
    parser.add_argument("--group-a", required=True)
    args = parser.parse_args()

    honest_points = json.loads(Path(args.honest).read_text(encoding="utf-8"))
    group_a = json.loads(Path(args.group_a).read_text(encoding="utf-8"))
    frontier = [(row["error_rate"], row["hit_rate"]) for row in group_a]

    win = tie = loss = out_of_range = 0
    best_net_lead = None
    print(f"{'tau_low':>8} {'error_rate':>11} {'hit_rate':>9} {'hit_rate_ci':>21} {'groupA@err':>11} {'verdict':>8}")
    for p in honest_points:
        er = p["error_rate"]
        hr = p["hit_rate"]
        hr_lo, hr_hi = p["hit_rate_ci"]
        a_hr = interpolate(frontier, er)
        if a_hr is None:
            out_of_range += 1
            print(f"{p['tau_low']:>8} {er:>11.4f} {hr:>9.4f} [{hr_lo:.4f},{hr_hi:.4f}] {'(out of range)':>11} {'skip':>8}")
            continue
        net_lead = hr - a_hr
        if best_net_lead is None or net_lead > best_net_lead:
            best_net_lead = net_lead
        if hr_lo > a_hr:
            verdict = "WIN"
            win += 1
        elif hr_hi < a_hr:
            verdict = "LOSS"
            loss += 1
        else:
            verdict = "TIE"
            tie += 1
        print(f"{p['tau_low']:>8} {er:>11.4f} {hr:>9.4f} [{hr_lo:.4f},{hr_hi:.4f}] {a_hr:>11.4f} {verdict:>8}")

    print()
    print(f"Win {win} / Tie {tie} / Loss {loss}  (out-of-range, skipped: {out_of_range})")
    if best_net_lead is not None:
        print(f"Best net lead (point estimate): {best_net_lead:+.4f} ({best_net_lead * 100:+.2f}pp)")


if __name__ == "__main__":
    main()
