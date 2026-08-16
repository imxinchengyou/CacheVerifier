"""Grouped bar chart: off-the-shelf vs fine-tuned held-out verifier AUC
across the three datasets (PAPER.md Section 5.6's headline table). Built
for the X thread's second post, not part of the paper's own figure set.

Usage:
    python scripts/plot_auc_comparison.py --out results/auc_comparison.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

# (dataset label, off-the-shelf AUC, fine-tuned AUC) -- PAPER.md Section 5.6,
# SearchQueries using the corrected (post-erratum) numbers.
ROWS = [
    ("LmArena", 0.7212, 0.8789),
    ("SearchQueries", 0.5984, 0.7120),
    ("Quora", 0.6309, 0.7393),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    labels = [r[0] for r in ROWS]
    off_the_shelf = [r[1] for r in ROWS]
    fine_tuned = [r[2] for r in ROWS]

    x = range(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    b1 = ax.bar([i - width / 2 for i in x], off_the_shelf, width, label="Off-the-shelf verifier", color="#9aa5b1")
    b2 = ax.bar([i + width / 2 for i in x], fine_tuned, width, label="Fine-tuned on ~10-20k gray-zone labels", color="#2f6fed")

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.2f}", (bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=9)

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.text(len(labels) - 0.5, 0.505, "random", fontsize=8, color="gray")

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Held-out AUC")
    ax.set_ylim(0.4, 1.0)
    ax.set_title("Verifier discriminative power, before vs after fine-tuning")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Wrote plot to {out_path}")


if __name__ == "__main__":
    main()
