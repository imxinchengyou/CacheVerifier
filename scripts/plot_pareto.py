"""Plot the hit-rate / error-rate Pareto frontier across one or more
`run_baselines.py` result files — the comparison proposal Section 5.3 wants
and what the Go/No-Go check (Section 6) is judged against.

Each point is one threshold (Group A) or target-error-rate/delta (Group B)
sweep value; the connected line is the non-dominated frontier
(`cacheverifier.metrics.pareto.pareto_frontier`) within that file's group.

Usage:
    python scripts/plot_pareto.py \\
        --results results/lmarena_groupA.json results/lmarena_groupB.json \\
        --out results/lmarena_pareto.png --title "SemCacheLMArena"
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt

from cacheverifier.metrics.pareto import ParetoPoint, pareto_frontier

GROUP_LABELS = {
    "A": "A: Static Threshold (GPTCache-style)",
    "B": "B: Adaptive Threshold (vCache-style)",
    "C": "C: Synchronous Krites",
    "D": "D: This proposal",
    "E": "E: Fine-tuned verifier",
}


def load_points(path: Path) -> tuple[str, list[ParetoPoint]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not rows:
        return "?", []
    group = rows[0]["group"]
    label_key = "threshold" if group == "A" else "target_error_rate"
    points = [
        ParetoPoint(label=str(row.get(label_key)), hit_rate=row["hit_rate"], error_rate=row["error_rate"])
        for row in rows
    ]
    return group, points


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", nargs="+", required=True, help="One or more results/*.json files")
    parser.add_argument("--out", required=True, help="Output image path (e.g. results/pareto.png)")
    parser.add_argument("--title", default="Hit rate vs error rate")
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, path_str in enumerate(args.results):
        group, points = load_points(Path(path_str))
        if not points:
            print(f"Skipping {path_str}: no rows")
            continue
        color = colors[i % len(colors)]
        label = GROUP_LABELS.get(group, f"Group {group}")

        ax.scatter([p.error_rate for p in points], [p.hit_rate for p in points], color=color, alpha=0.35, s=25)

        frontier = sorted(pareto_frontier(points), key=lambda p: p.error_rate)
        ax.plot(
            [p.error_rate for p in frontier],
            [p.hit_rate for p in frontier],
            color=color,
            marker="o",
            label=label,
            linewidth=2,
        )

    ax.set_xlabel("Error rate (fraction of all requests served an incorrect cached answer)")
    ax.set_ylabel("Hit rate")
    ax.set_title(args.title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Wrote plot to {out_path}")


if __name__ == "__main__":
    main()
