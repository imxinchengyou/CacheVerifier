"""Cost-sensitive reanalysis of the paper's existing hit_rate/error_rate
results -- method "B" from the 2026-08-29 research-methods brainstorm
(see RESEARCH_PROPOSAL.md Sec 10, "研究现状总览").

Every result table in this paper (Groups A/C/D/E, Sections 5.1-5.16) reports
a hit_rate/error_rate Pareto frontier and lets the reader decide informally
whether a given point is "worth it." That framing implicitly treats every
false-accept and every miss as equally costly, which is never true in a real
deployment: a miss just costs one extra LLM call, a false-accept costs
customer trust/refunds/support escalation -- usually a much larger and
harder-to-quantify number. This script does NOT invent a real dollar figure
for either cost (that would be a fabricated business assumption dressed up
as a finding) -- it instead treats the cost RATIO r = C_false_accept /
C_miss as a free parameter and sweeps it, answering: at what cost ratios
does a real synchronous verifier (Group D/E) economically beat a static
threshold (Group A), and where exactly is the crossover?

This requires no new experiments -- it is a pure reanalysis of result JSONs
that already exist on disk, using each dataset's most credible numbers
(honest calibration for D/E, per PAPER.md Sec 5.4, not the original
grid-searched numbers) plus Group A's static-threshold frontier and Group C's
oracle ceiling as a reference.

Formula: expected cost per record, in units of C_miss, at cost ratio r:
    cost(r) = r * error_rate + (1 - hit_rate)
(error_rate is defined as false_accepts / N throughout this project, see
`cacheverifier/metrics/core.py`; hit_rate = (tp+fp)/N, so 1-hit_rate is the
miss rate -- both already reported in every existing result file, no new
measurement needed.)

For each r, the best operating point within a group is the one minimizing
cost(r); comparing best-of-A vs best-of-D vs best-of-E vs best-of-C at each r
gives an economically-grounded verdict, replacing "is +1.9pp hit rate worth
it" with "at what cost ratios is it worth it, and how far is that from a
plausible real one."

Usage:
    python scripts/cost_sensitive_reanalysis.py
    python scripts/cost_sensitive_reanalysis.py --output results/cost_sensitive_reanalysis.json
"""

import argparse
import json
from pathlib import Path

RATIO_GRID = [0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500]

DATASETS = {
    "lmarena": {
        "label": "LmArena",
        "groupA": "results/lmarena_groupA.json",
        "groupC": "results/lmarena_groupC.json",
        "groupD_honest": "results/threshold_calibration_lmarena_60k_merged.json",
        "groupE_honest": "results/lmarena_groupE_honest_calibration_merged.json",
    },
    "quora": {
        "label": "Quora",
        "groupA": "results/quora_groupA.json",
        "groupC": "results/quora_groupC.json",
        "groupD_honest": "results/threshold_calibration_quora_merged.json",
        "groupE_honest": "results/quora_groupE_honest_calibration_merged.json",
    },
    "search_queries": {
        "label": "SearchQueries (corrected)",
        "groupA": "results/search_queries_groupA.json",
        "groupC": "results/search_queries_groupC.json",
        "groupD_honest": "results/threshold_calibration_sq_corrected_merged.json",
        "groupE_honest": "results/search_queries_corrected_groupE_honest_calibration_merged.json",
    },
}
"""tau_low grids extended 2026-08-30 from {0.80..0.95} (6 pts) to {0.80..0.99}
(9 pts, matching Group A's full grid) via 6 new GPU-computed honest-calibration
runs at tau_low={0.97,0.98,0.99}, tau_high=0.999 (scripts/threshold_calibration_ablation.py
and _groupE.py, run on 47.85.85.178, merged with the original 6-point files). This
closes the grid-coverage gap that made the original high-r "A wins again" result
ambiguous (D/E previously had no operating point as conservative as A's tau=0.99)."""


def load_points(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8"))


def cost(point: dict, r: float) -> float:
    return r * point["error_rate"] + (1.0 - point["hit_rate"])


def best_in_group(points: list[dict], r: float) -> tuple[float, dict] | None:
    if not points:
        return None
    scored = [(cost(p, r), p) for p in points]
    scored.sort(key=lambda x: x[0])
    return scored[0]


def summarize_operating_point(p: dict) -> str:
    thr = p.get("threshold", p.get("calibrated_threshold"))
    tau_low = p.get("tau_low", p.get("tau_low"))
    bits = []
    if tau_low is not None:
        bits.append(f"tau_low={tau_low}")
    if thr is not None:
        bits.append(f"thr={thr:.3g}" if isinstance(thr, float) else f"thr={thr}")
    return ", ".join(bits) if bits else "?"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default="results/cost_sensitive_reanalysis.json")
    args = parser.parse_args()

    all_results = {}

    for key, cfg in DATASETS.items():
        pointsA = load_points(cfg["groupA"])
        pointsC = load_points(cfg["groupC"])
        pointsD = load_points(cfg["groupD_honest"])
        pointsE = load_points(cfg["groupE_honest"])

        print(f"\n{'=' * 70}")
        print(f"{cfg['label']}  (A: {len(pointsA)} pts, C: {len(pointsC)} pts, "
              f"D-honest: {len(pointsD)} pts, E-honest: {len(pointsE)} pts)")
        print(f"{'=' * 70}")
        header = f"{'r':>7} | {'A cost':>8} {'D cost':>8} {'E cost':>8} {'C cost':>8} | {'winner':>8} | {'D vs A':>9} {'E vs A':>9}"
        print(header)

        rows = []
        for r in RATIO_GRID:
            bestA = best_in_group(pointsA, r)
            bestC = best_in_group(pointsC, r)
            bestD = best_in_group(pointsD, r)
            bestE = best_in_group(pointsE, r)

            candidates = {"A": bestA, "C": bestC, "D": bestD, "E": bestE}
            # C (oracle) is a reference ceiling, not a real deployable competitor
            # to A/D/E -- exclude it from the "winner" among real policies.
            real_candidates = {k: v for k, v in candidates.items() if k in ("A", "D", "E") and v is not None}
            winner = min(real_candidates, key=lambda k: real_candidates[k][0])

            costA = bestA[0] if bestA else float("nan")
            costD = bestD[0] if bestD else float("nan")
            costE = bestE[0] if bestE else float("nan")
            costC = bestC[0] if bestC else float("nan")

            d_vs_a = (costA - costD) / costA * 100 if bestA and bestD and costA > 0 else float("nan")
            e_vs_a = (costA - costE) / costA * 100 if bestA and bestE and costA > 0 else float("nan")

            print(
                f"{r:>7.2f} | {costA:>8.4f} {costD:>8.4f} {costE:>8.4f} {costC:>8.4f} | "
                f"{winner:>8} | {d_vs_a:>+8.1f}% {e_vs_a:>+8.1f}%"
            )

            rows.append(
                {
                    "r": r,
                    "cost_A": costA,
                    "cost_D": costD,
                    "cost_E": costE,
                    "cost_C_oracle": costC,
                    "winner_among_ADE": winner,
                    "D_vs_A_pct_cost_reduction": d_vs_a,
                    "E_vs_A_pct_cost_reduction": e_vs_a,
                    "bestA_point": summarize_operating_point(bestA[1]) if bestA else None,
                    "bestD_point": summarize_operating_point(bestD[1]) if bestD else None,
                    "bestE_point": summarize_operating_point(bestE[1]) if bestE else None,
                }
            )

        # Find the full set of r-windows where D/E beats A -- best-of-group cost is
        # a min over several linear-in-r functions (piecewise-linear, concave), so
        # the difference between two such curves need NOT be monotonic or have a
        # single crossing; bisection would silently miss a "sweet spot" window
        # bounded on both sides. Scan a fine log-spaced grid instead and report
        # every contiguous r-range where the sign of (cost_verifier - cost_A) is
        # negative (verifier wins).
        fine_grid = [10 ** (x / 20) for x in range(-40, 61)]  # r in [1e-2, 1e3], ~100 points/decade
        windows = {}
        for label, points in {"D": pointsD, "E": pointsE}.items():
            if not points or not pointsA:
                windows[label] = []
                continue
            signs = [best_in_group(points, r)[0] - best_in_group(pointsA, r)[0] < 0 for r in fine_grid]
            ranges = []
            start = None
            for i, s in enumerate(signs):
                if s and start is None:
                    start = fine_grid[i]
                if not s and start is not None:
                    ranges.append((round(start, 3), round(fine_grid[i - 1], 3)))
                    start = None
            if start is not None:
                ranges.append((round(start, 3), round(fine_grid[-1], 3)))
            windows[label] = ranges

        print("\nr-windows where D/E beats A's best static threshold on expected cost (fine grid scan, r in [0.01, 1000]):")
        for label, ranges in windows.items():
            if not ranges:
                print(f"  {label}: never beats A anywhere in [0.01, 1000]")
            else:
                ranges_str = ", ".join(f"[{lo}, {hi}]" for lo, hi in ranges)
                print(f"  {label}: {ranges_str}")

        # Flag a grid-range caveat: if D/E's honest-calibration tau_low grid doesn't
        # reach as conservative an operating point as A's does, A can "win" at high r
        # purely because it has a more conservative option available, not because
        # verification stops helping -- report each group's tau_low/threshold range
        # so this is visible rather than silently baked into the high-r numbers.
        def tau_low_range(points: list[dict]) -> str:
            # Group A points key their operating variable "threshold" (it IS the
            # similarity cutoff for that group); Group C/D/E key it "tau_low".
            vals = [p.get("tau_low", p.get("threshold")) for p in points]
            vals = [v for v in vals if v is not None]
            return f"[{min(vals)}, {max(vals)}]" if vals else "n/a"

        print(f"\nGrid coverage (tau_low range) -- A: {tau_low_range(pointsA)}, "
              f"D-honest: {tau_low_range(pointsD)}, E-honest: {tau_low_range(pointsE)} "
              f"-- if D/E's range is narrower than A's, A 'winning' at very high r may "
              f"just reflect A having a more conservative grid point available, not a "
              f"real economic reversal.")

        all_results[key] = {
            "label": cfg["label"],
            "rows": rows,
            "winning_r_windows": windows,
            "tau_low_coverage": {
                "A": tau_low_range(pointsA),
                "D_honest": tau_low_range(pointsD),
                "E_honest": tau_low_range(pointsE),
            },
        }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
