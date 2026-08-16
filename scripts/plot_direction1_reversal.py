"""Minimal scatter for Reddit post 3 (social_media/reddit/post_3_...): the 6
honestly-calibrated test-half operating points (threshold_calibration_ablation.py
on SearchQueries_corrected) plotted against the interpolated Group A
static-threshold frontier (results/search_queries_groupA.json) they're being
compared to. No labels beyond axes -- the post's text carries the explanation,
this is just "look, every point clears the line."

Usage:
    python scripts/plot_direction1_reversal.py --out results/direction1_sq_honest_vs_groupA.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

# Group A static-threshold frontier, interpolation nodes (error_rate, hit_rate).
GROUP_A_PATH = Path("results/search_queries_groupA.json")

# threshold_calibration_ablation.py output on search_queries_corrected
# (honest calibration/test split, Youden's J threshold picked on the
# calibration half only, measured on the held-out test half).
HONEST_POINTS = [
    (0.1581, 0.6039),
    (0.1304, 0.5374),
    (0.1069, 0.4760),
    (0.0942, 0.4363),
    (0.0709, 0.3696),
    (0.0534, 0.3110),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    group_a = json.loads(GROUP_A_PATH.read_text(encoding="utf-8"))
    frontier = sorted([(row["error_rate"], row["hit_rate"]) for row in group_a])

    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    ax.plot(
        [p[0] for p in frontier], [p[1] for p in frontier],
        color="#999999", linestyle="--", linewidth=1.5, zorder=1,
    )

    ax.scatter(
        [p[0] for p in HONEST_POINTS], [p[1] for p in HONEST_POINTS],
        color="#2f6fed", s=70, zorder=2, edgecolors="white", linewidths=0.8,
    )

    ax.set_xlabel("Error rate")
    ax.set_ylabel("Hit rate")
    ax.set_title("SearchQueries, honest calibration", fontsize=10, color="#555555")
    ax.grid(True, alpha=0.25)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
