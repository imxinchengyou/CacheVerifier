"""Direction 15 v1: plot the risk-vs-deferral frontier for each dataset/verifier
from scripts/selective_deferral_experiment.py output, plus a kappa summary bar.

Usage:
    python scripts/plot_selective_deferral.py \
        results/selective_deferral_lmarena_groupD.json:LmArena-D \
        results/selective_deferral_lmarena_groupE.json:LmArena-E \
        results/selective_deferral_quora_groupD.json:Quora-D \
        results/selective_deferral_sq_groupD.json:SearchQ-D \
        results/selective_deferral_sq_groupE.json:SearchQ-E \
        --out results/selective_deferral_frontiers.png
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

COLORS = {"margin": "#1f77b4", "calibrated": "#ff7f0e",
          "random": "#7f7f7f", "oracle_informed": "#2ca02c"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="+", help="path.json:Label")
    ap.add_argument("--tau-low", default="0.8")
    ap.add_argument("--out", default="results/selective_deferral_frontiers.png")
    args = ap.parse_args()

    items = []
    for spec in args.specs:
        path, label = spec.rsplit(":", 1)
        items.append((label, json.loads(Path(path).read_text())))

    n = len(items)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 3.6), squeeze=False)
    for ax, (label, d) in zip(axes[0], items):
        sp = d["splits"].get(f"tau_low={args.tau_low}")
        if sp is None:
            ax.set_title(f"{label}\n(no tau_low={args.tau_low})")
            continue
        for sig, c in sp["curves"].items():
            ax.plot(c["realized_delta"], c["error_rate"], "-o", ms=3, lw=1.4,
                    color=COLORS[sig], label=sig)
        ax.axvline(0.30, color="k", ls=":", lw=0.8, alpha=0.5)
        km = sp["kappa"]["margin"]
        kc = sp["kappa"]["calibrated"]
        ax.set_title(f"{label}\nκ_margin={km:.2f}  κ_calib={kc:.2f}", fontsize=9)
        ax.set_xlabel("realized deferral rate δ (gray zone)")
        ax.set_ylabel("end-to-end error_rate")
        ax.grid(alpha=0.25)
    axes[0][0].legend(fontsize=7, loc="upper right")
    fig.suptitle(f"Selective deferral: risk vs deferral rate (τ_low={args.tau_low}, "
                 f"ε_oracle per each file)", fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
