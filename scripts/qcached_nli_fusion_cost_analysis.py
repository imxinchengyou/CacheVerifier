"""Direction 16 follow-up to qcached_nli_fusion_eval.py: that script found two
fusion mechanisms combining the zero-shot NLI entailment score with the
fine-tuned diff+adversarial checkpoint's score, each with a real but
differently-shaped natural-data cost -- weighted averaging (best total
adversarial FA 2.9% at w=0.2, natural AUC drops 0.87->0.71) and an AND-gate
veto (adversarial FA crushed to 0.7%/entity_swap 2.6%, but natural approve
rate crashes 95.6%->52.6%, with 44% of newly-rejected rows being genuine
correct answers). Both were reported as raw trade-off numbers with no
judgment on whether either is "worth it."

This script answers that question the same way Sec 5.17 (cost_sensitive_
reanalysis.py) already answers it for the paper's main Group A/C/D/E
comparison: treat the cost ratio r = C_false_accept / C_miss as a free
parameter, and at each r find whether ANY (w, threshold) fusion grid point
beats the current production baseline (pure fine-tuned, w=1.0, threshold=0.5)
on the SAME natural held-out test set, using the exact cost(r) formula:
    cost(r) = r * error_rate + (1 - hit_rate)
(hit_rate = approve_rate = (TP+FP)/N; error_rate = FP/N -- same definitions
as cost_sensitive_reanalysis.py, so the RATIO_GRID and verdict format are
directly comparable to Sec 5.17's existing table.)

Unlike qcached_nli_fusion_eval.py's 1-D sweep (w only, threshold fixed at
0.5), this script sweeps a full 2-D (w, threshold) grid -- reported as a
grid/frontier, never fit or optimized against the held-out redteam set,
consistent with this paper's established practice of reporting curves at
pre-specified operating points rather than cherry-picking. The redteam
false-accept rate at each grid's cost-optimal point is reported alongside
as a secondary robustness readout, kept separate from the cost(r) number
itself (the redteam set is a small curated adversarial probe, not a sample
of natural production traffic, so it is not mixed into the same cost
formula -- matching how Sec 5.18/5.19's redteam numbers were never fed into
Sec 5.17's cost model either).

Usage:
    python scripts/qcached_nli_fusion_cost_analysis.py
"""

import argparse
import difflib
import json
import time
from pathlib import Path

import numpy as np

TAU_LOW = 0.80
DIFF_TEMPLATE = "{query}\n[diff vs cached_query] {diff}"
W_GRID = [round(x, 2) for x in np.arange(0.0, 1.0001, 0.05)]
THRESHOLD_GRID = [round(x, 2) for x in np.arange(0.1, 0.9001, 0.04)]
RATIO_GRID = [0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500]  # matches cost_sensitive_reanalysis.py


def word_diff_summary(cached_query: str, current_query: str, max_words: int = 12) -> str:
    a, b = cached_query.split(), current_query.split()
    sm = difflib.SequenceMatcher(None, a, b)
    removed, added = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("delete", "replace"):
            removed.extend(a[i1:i2])
        if tag in ("insert", "replace"):
            added.extend(b[j1:j2])
    if not removed and not added:
        return "identical"
    parts = []
    if removed:
        parts.append("removed: " + " ".join(removed[:max_words]))
    if added:
        parts.append("added: " + " ".join(added[:max_words]))
    return "; ".join(parts)


def nli_zero_shot_scores(model, pairs, batch_size=64):
    raw = np.array(model.predict(pairs, batch_size=batch_size, show_progress_bar=False, apply_softmax=False))
    exp = np.exp(raw - raw.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)
    return probs[:, 1] - probs[:, 0]


def cost(error_rate: float, hit_rate: float, r: float) -> float:
    return r * error_rate + (1.0 - hit_rate)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zero-shot-model", default="cross-encoder/nli-MiniLM2-L6-H768")
    parser.add_argument("--finetuned-checkpoint",
                         default="results/finetuned_verifier_model_lmarena_nli_qcached_adv_ablation_qcached_adv_diff")
    parser.add_argument("--natural-stash", default="results/finetune_verifier_qcached_experiment.examples.json")
    parser.add_argument("--redteam-results", default="results/llm_redteam_results.json")
    parser.add_argument("--output", default="results/qcached_nli_fusion_cost_analysis_lmarena.json")
    parser.add_argument("--raw-scores-output", default="results/qcached_nli_fusion_raw_scores_lmarena.json")
    args = parser.parse_args()

    def log(msg: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    import torch
    from sentence_transformers import CrossEncoder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    natural = json.loads(Path(args.natural_stash).read_text(encoding="utf-8"))
    test_rows = natural["test"]
    test_labels = np.array([1 if row[-1] else 0 for row in test_rows])
    natural_zero_pairs = [(row[1], row[3]) for row in test_rows]
    natural_ft_pairs = [(DIFF_TEMPLATE.format(query=row[1], diff=word_diff_summary(row[2], row[1])), row[3])
                         for row in test_rows]
    log(f"Natural test: {len(test_rows)} pairs, {int(test_labels.sum())} positive")

    data = json.loads(Path(args.redteam_results).read_text(encoding="utf-8"))
    triples = [t for t in data["all_triples"] if t["similarity"] >= TAU_LOW]
    redteam_categories = [t["category"] for t in triples]
    redteam_zero_pairs = [(t["query_a"], t["answer_b"]) for t in triples]
    redteam_ft_pairs = [(DIFF_TEMPLATE.format(query=t["query_a"], diff=word_diff_summary(t["query_b"], t["query_a"])),
                          t["answer_b"]) for t in triples]
    log(f"Redteam probes: {len(triples)} (tau_low={TAU_LOW})")

    log(f"Loading zero-shot model {args.zero_shot_model!r}...")
    zero_model = CrossEncoder(args.zero_shot_model, device=device)
    log(f"Loading fine-tuned checkpoint {args.finetuned_checkpoint!r}...")
    ft_model = CrossEncoder(args.finetuned_checkpoint, device=device)

    log("Scoring (4 passes: natural x2, redteam x2)...")
    natural_score_zero_raw = nli_zero_shot_scores(zero_model, natural_zero_pairs)
    natural_score_ft = np.array(ft_model.predict(natural_ft_pairs, batch_size=64, show_progress_bar=False))
    redteam_score_zero_raw = nli_zero_shot_scores(zero_model, redteam_zero_pairs)
    redteam_score_ft = np.array(ft_model.predict(redteam_ft_pairs, batch_size=64, show_progress_bar=False))
    log("Scoring done.")

    natural_score_zero = (natural_score_zero_raw + 1.0) / 2.0
    redteam_score_zero = (redteam_score_zero_raw + 1.0) / 2.0

    Path(args.raw_scores_output).write_text(json.dumps({
        "natural_score_zero_raw": natural_score_zero_raw.tolist(),
        "natural_score_ft": natural_score_ft.tolist(),
        "natural_labels": test_labels.tolist(),
        "redteam_score_zero_raw": redteam_score_zero_raw.tolist(),
        "redteam_score_ft": redteam_score_ft.tolist(),
        "redteam_categories": redteam_categories,
    }), encoding="utf-8")
    log(f"Wrote raw per-example scores to {args.raw_scores_output}")

    log(f"\nSweeping {len(W_GRID)} x {len(THRESHOLD_GRID)} = {len(W_GRID) * len(THRESHOLD_GRID)} (w, threshold) grid points...")
    grid = []
    for w in W_GRID:
        fused_natural = w * natural_score_ft + (1 - w) * natural_score_zero
        fused_redteam = w * redteam_score_ft + (1 - w) * redteam_score_zero
        for thr in THRESHOLD_GRID:
            nat_approved = fused_natural >= thr
            tp = int((nat_approved & (test_labels == 1)).sum())
            fp = int((nat_approved & (test_labels == 0)).sum())
            n = len(test_labels)
            hit_rate = (tp + fp) / n
            error_rate = fp / n

            red_approved = fused_redteam >= thr
            red_total_fa = float(red_approved.mean()) * 100
            red_entity_swap_n = sum(1 for c in redteam_categories if c == "entity_swap")
            red_entity_swap_fa = sum(1 for c, a in zip(redteam_categories, red_approved) if c == "entity_swap" and a)
            red_entity_swap_pct = red_entity_swap_fa / red_entity_swap_n * 100 if red_entity_swap_n else float("nan")

            grid.append({
                "w": w, "threshold": thr,
                "hit_rate": hit_rate, "error_rate": error_rate,
                "redteam_total_fa_pct": red_total_fa,
                "redteam_entity_swap_fa_pct": red_entity_swap_pct,
            })
    log(f"Grid computed: {len(grid)} points.")

    baseline = next(p for p in grid if p["w"] == 1.0 and p["threshold"] == 0.5)
    log(f"Baseline (pure fine-tuned, w=1.0, thr=0.5): hit_rate={baseline['hit_rate']:.4f} "
        f"error_rate={baseline['error_rate']:.4f} redteam_fa={baseline['redteam_total_fa_pct']:.1f}% "
        f"entity_swap={baseline['redteam_entity_swap_fa_pct']:.1f}%")

    # AND-gate veto (accept only if BOTH score_ft>=0.5 AND score_zero_raw>=0) is not expressible as a point
    # on the (w, threshold) weighted-average grid -- a logical AND can be strictly stricter than any convex
    # combination allows (see qcached_nli_fusion_eval.py's finding: AND-gate beats the weighted-average
    # sweep on every single redteam category). Computed separately here so it can compete in the same
    # cost(r) comparison as a third candidate alongside the grid and the baseline.
    and_natural_approved = (natural_score_ft >= 0.5) & (natural_score_zero_raw >= 0.0)
    and_tp = int((and_natural_approved & (test_labels == 1)).sum())
    and_fp = int((and_natural_approved & (test_labels == 0)).sum())
    and_n = len(test_labels)
    and_hit_rate, and_error_rate = (and_tp + and_fp) / and_n, and_fp / and_n
    and_redteam_approved = (redteam_score_ft >= 0.5) & (redteam_score_zero_raw >= 0.0)
    and_gate = {
        "hit_rate": and_hit_rate, "error_rate": and_error_rate,
        "redteam_total_fa_pct": float(and_redteam_approved.mean()) * 100,
        "redteam_entity_swap_fa_pct": (
            sum(1 for c, a in zip(redteam_categories, and_redteam_approved) if c == "entity_swap" and a)
            / sum(1 for c in redteam_categories if c == "entity_swap") * 100
        ),
    }
    log(f"AND-gate veto: hit_rate={and_gate['hit_rate']:.4f} error_rate={and_gate['error_rate']:.4f} "
        f"redteam_fa={and_gate['redteam_total_fa_pct']:.1f}% entity_swap={and_gate['redteam_entity_swap_fa_pct']:.1f}%")

    log("\n=== Cost-ratio sweep: baseline vs. best (w,threshold) grid point vs. AND-gate ===")
    cost_sweep = []
    for r in RATIO_GRID:
        baseline_cost = cost(baseline["error_rate"], baseline["hit_rate"], r)
        and_cost = cost(and_gate["error_rate"], and_gate["hit_rate"], r)
        scored = sorted(grid, key=lambda p: cost(p["error_rate"], p["hit_rate"], r))
        best_grid = scored[0]
        best_grid_cost = cost(best_grid["error_rate"], best_grid["hit_rate"], r)

        candidates = [("baseline", baseline_cost, baseline), ("grid_best", best_grid_cost, best_grid),
                      ("and_gate", and_cost, and_gate)]
        winner_name, winner_cost, winner = min(candidates, key=lambda c: c[1])

        entry = {
            "r": r,
            "baseline_cost": baseline_cost,
            "grid_best_w": best_grid["w"], "grid_best_threshold": best_grid["threshold"], "grid_best_cost": best_grid_cost,
            "and_gate_cost": and_cost,
            "winner": winner_name,
            "winner_cost": winner_cost,
            "cost_improvement_over_baseline_pct": (baseline_cost - winner_cost) / baseline_cost * 100 if baseline_cost > 0 else float("nan"),
            "winner_hit_rate": winner["hit_rate"], "winner_error_rate": winner["error_rate"],
            "winner_redteam_total_fa_pct": winner["redteam_total_fa_pct"],
            "winner_redteam_entity_swap_fa_pct": winner["redteam_entity_swap_fa_pct"],
        }
        cost_sweep.append(entry)
        log(f"  r={r:>6.2f}  baseline={baseline_cost:.4f}  grid_best={best_grid_cost:.4f}(w={best_grid['w']:.2f},"
            f"thr={best_grid['threshold']:.2f})  and_gate={and_cost:.4f}  -> WINNER={winner_name} "
            f"({entry['cost_improvement_over_baseline_pct']:+.1f}% vs baseline)  "
            f"redteam_fa={winner['redteam_total_fa_pct']:.1f}%  entity_swap={winner['redteam_entity_swap_fa_pct']:.1f}%")

    output = {
        "zero_shot_model": args.zero_shot_model,
        "finetuned_checkpoint": args.finetuned_checkpoint,
        "n_natural_test": len(test_rows),
        "n_redteam": len(triples),
        "w_grid": W_GRID,
        "threshold_grid": THRESHOLD_GRID,
        "baseline_pure_finetuned": baseline,
        "and_gate_veto": and_gate,
        "cost_ratio_sweep": cost_sweep,
        "full_grid": grid,
    }
    Path(args.output).write_text(json.dumps(output, indent=2), encoding="utf-8")
    log(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
