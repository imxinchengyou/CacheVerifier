"""Three figures for the Day-2 X thread (social_media/x/thread_v2_production_reality_check.md),
built from the numbers in PAPER.md Section 5.8/5.9 (Kaggle Twitter Customer
Support case study: AmazonHelp vs comcastcares). Not part of the paper's own
figure set -- purely for the social post.

Usage:
    python scripts/plot_twitter_case_study.py --out-dir results
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import date


def plot_timeline(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 2.6))

    rows = [
        ("AmazonHelp", date(2015, 6, 1), date(2017, 10, 1), "#2f6fed", "60k pairs · 2.4yr · stable"),
        ("comcastcares", date(2014, 7, 1), date(2017, 12, 1), "#e0575b", "30k pairs · 3.5yr · base rate drifted 1% → 5.2%"),
    ]

    for i, (label, start, end, color, note) in enumerate(rows):
        y = len(rows) - i
        ax.plot([start, end], [y, y], color=color, linewidth=10, solid_capstyle="butt", alpha=0.85)
        ax.text(start, y + 0.32, label, fontsize=11, fontweight="bold", color=color)
        ax.text(end, y - 0.38, note, fontsize=8.5, color="#555555", ha="right")

    ax.set_ylim(0.3, len(rows) + 0.9)
    ax.set_yticks([])
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for spine in ("top", "left", "right"):
        ax.spines[spine].set_visible(False)
    ax.set_title("Real production traffic windows: two brands, very different stability", fontsize=11)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_ablation_table(out_path: Path) -> None:
    columns = ["Ablation", "AmazonHelp (e-commerce)", "comcastcares (telecom)"]
    data = [
        ["Label noise\n(0% → 40% injected)", "Δ turns negative ~5-10%\n(smaller train set)", "Δ negative at EVERY level,\nincl. 0% clean: Δ=−0.186"],
        ["Cold start\n(more training data)", "Monotonically better,\nno saturation: Δ=+0.064→+0.097", "Gets WORSE with more data:\nΔ=−0.004→−0.569 (AUC 0.295)"],
        ["Drift\n(train/test time gap)", "Decays then plateaus\n(no further decline)", "Keeps decaying, never levels off\nΔ=−0.19 → −0.24"],
    ]

    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    ax.axis("off")
    tbl = ax.table(cellText=data, colLabels=columns, cellLoc="left", loc="center", colWidths=[0.22, 0.39, 0.39])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 2.6)

    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#dddddd")
        if row == 0:
            cell.set_facecolor("#2f2f2f")
            cell.set_text_props(color="white", fontweight="bold")
        elif col == 2:
            cell.set_facecolor("#fdecea")
        elif col == 1:
            cell.set_facecolor("#eaf1fd")
        else:
            cell.set_facecolor("#f7f7f7")

    ax.set_title("Same three deployment ablations, opposite outcomes", fontsize=12, pad=14)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_baseline_auc_drift(out_path: Path) -> None:
    segments = list(range(1, 9))
    comcastcares = [0.522, 0.39, 0.493, 0.408, 0.551, 0.569, 0.808, 0.877]

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(segments, comcastcares, marker="o", color="#e0575b", linewidth=2, label="comcastcares (unfine-tuned baseline AUC)")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.text(8.05, 0.505, "random", fontsize=8, color="gray")

    ax.set_xlabel("Time segment (chronological, 1 = earliest)")
    ax.set_ylabel("Baseline verifier AUC")
    ax.set_title("The base rate wasn't just shifted once — it was never stable")
    ax.set_xticks(segments)
    ax.set_ylim(0.25, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_timeline(out_dir / "twitter_case_study_timeline.png")
    plot_ablation_table(out_dir / "twitter_case_study_ablation_table.png")
    plot_baseline_auc_drift(out_dir / "twitter_case_study_baseline_auc_drift.png")


if __name__ == "__main__":
    main()
