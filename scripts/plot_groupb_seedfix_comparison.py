"""Bar chart for the X/Reddit posts about the Group B seed-observation fix
(social_media/x/thread_v3_groupb_official_bug.md, social_media/reddit/
post_5_groupb_official_bug.md): hit rate before vs. after adding the two
synthetic bootstrap observations vCache's official EmbeddingMetadataObj
seeds and this paper's port originally lacked (see PAPER.md 5.2 end).

Numbers are the delta=0.01 row of the before/after table in PAPER.md 5.2
(results/{lmarena,quora,search_queries}_groupB.json vs
results/{lmarena,quora,search_queries}_groupB_seedfix.json), picked as the
headline delta since it has the largest, cleanest multiplier (LmArena
29.1x) and is the paper's tightest tested error budget.

Usage:
    python scripts/plot_groupb_seedfix_comparison.py --out results/groupb_seedfix_comparison.png
    python scripts/plot_groupb_seedfix_comparison.py --lang zh --out results/groupb_seedfix_comparison_zh.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# (dataset label, hit rate before %, hit rate after %, multiplier) at delta=0.01
DATA = [
    ("LmArena", 0.04, 1.21, "29.1x"),
    ("SearchQueries", 0.16, 0.80, "5.1x"),
    ("Quora", 0.02, 0.12, "5.4x"),
]

TEXT = {
    "en": {
        "before": "Before fix",
        "after": "After fix",
        "ylabel": "Group B hit rate (%)",
        "title": "Hit rate before/after the seed-observation fix (δ=0.01)",
    },
    "zh": {
        "before": "修复前",
        "after": "修复后",
        "ylabel": "Group B 命中率 (%)",
        "title": "种子观测点修复前后命中率对比（δ=0.01）",
        "font": "PingFang SC",
    },
}

BEFORE_COLOR = "#8a919c"   # muted gray -- baseline, not the headline; darkened from an earlier
                            # draft to clear the dataviz skill's >=3:1 surface-contrast check
AFTER_COLOR = "#2f6fed"    # accent blue -- matches plot_direction1_reversal.py's highlight color
                            # (gray/blue pair reads as "baseline vs. highlight," not a 2-category
                            # identity choice, so it's validated with direct value labels as the
                            # secondary encoding rather than relying on hue alone)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--lang", choices=["en", "zh"], default="en")
    args = parser.parse_args()

    t = TEXT[args.lang]
    if args.lang == "zh":
        plt.rcParams["font.sans-serif"] = [t["font"]]
        plt.rcParams["axes.unicode_minus"] = False

    labels = [d[0] for d in DATA]
    before = [d[1] for d in DATA]
    after = [d[2] for d in DATA]
    multipliers = [d[3] for d in DATA]

    x = np.arange(len(labels))
    width = 0.32

    fig, ax = plt.subplots(figsize=(6, 4.5))

    bars_before = ax.bar(x - width / 2, before, width, color=BEFORE_COLOR, label=t["before"], zorder=2)
    bars_after = ax.bar(x + width / 2, after, width, color=AFTER_COLOR, label=t["after"], zorder=2)

    # Direct value labels on every bar -- the gray/blue pair alone is below the
    # dataviz skill's CVD chroma floor (gray reads as "baseline," not a
    # separately-hued category), so labels are the required secondary encoding,
    # not just a nice-to-have.
    for xi, b in zip(x, before):
        ax.text(xi - width / 2, b + 0.06, f"{b:.2f}%", ha="center", va="bottom", fontsize=8.5, color="#555555")
    for xi, a in zip(x, after):
        ax.text(xi + width / 2, a + 0.06, f"{a:.2f}%", ha="center", va="bottom", fontsize=8.5, color="#1a3a8f")

    for xi, mult, a in zip(x, multipliers, after):
        ax.text(xi, a + 0.30, mult, ha="center", va="bottom", fontsize=11, color="#1a3a8f", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel(t["ylabel"])
    ax.set_title(t["title"], fontsize=10, color="#555555")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.grid(True, axis="y", alpha=0.25, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.set_ylim(0, max(after) * 1.65)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
