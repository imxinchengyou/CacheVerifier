"""Direction 2 (RESEARCH_PROPOSAL.md / memory): cross-methodology validation
against Baral et al. (2026, arXiv:2606.19719) "Closing the Calibration Gap
in Semantic Caching" -- recompute this paper's Group D/E verifier results
using their P-CHR AUC / Calibration Retention Rate (CRR) metrics instead of
this paper's own ROC-AUC.

Exact definitions taken from the paper's Section 3 (equations 2, 3, 6):

    P-CHR AUC = integral_0^1 Precision(CHR^-1(c)) dc
        -- area under the (CHR, Precision) curve as the decision threshold
        tau sweeps; CHR(tau) = fraction of ALL queries the cache would
        serve at that threshold ("cache hit ratio"), Precision(tau) =
        fraction of those served that are actually correct.

    PR-AUC = area under the standard (Recall, Precision) curve, same
        threshold sweep, Recall(tau) = fraction of all TRUE POSITIVES
        served at that threshold.

    CRR = P-CHR AUC / PR-AUC  (in (0, 1])

Methodological note (architectural difference from the source paper):
Baral et al. evaluate a two-stage retrieve-then-rerank pipeline (K=50
candidates reranked, PR-AUC computed on the ground-truth candidate's score
even when it wasn't top-ranked, P-CHR AUC computed on the top-ranked
candidate's score). This paper's own architecture is single-tier: exact/HNSW
similarity search already commits to exactly one candidate before the
verifier ever runs (K=1, no rerank stage), and the `would_be_correct` label
already encodes whether serving THAT ONE candidate would be correct. This
means CHR(tau) and Recall(tau) can both be computed directly from the same
(score, label) pairs already produced by Group D/E -- Precision(tau) is
identical for both curves, only the x-axis normalizer differs (all queries
N vs. all positives P) -- with no additional retrieval/reranking simulation
needed. This is the natural K=1 special case of their framework, not an
approximation, but it is worth stating plainly: it is not evaluating the
same two-stage pipeline shape their numbers describe.

Usage:
    python scripts/compute_crr.py
"""

import json
from pathlib import Path

import numpy as np

CACHE_DIR = Path("results/.cache")


def pr_p_chr_auc(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Returns (PR-AUC, P-CHR AUC) computed from the same threshold sweep."""
    order = np.argsort(-scores)  # descending: sweep tau from high to low
    sorted_labels = labels[order]
    n = len(labels)
    n_pos = int(labels.sum())

    tp = np.cumsum(sorted_labels == 1)
    fp = np.cumsum(sorted_labels == 0)
    served = tp + fp  # number of queries the cache would serve at this tau

    precision = tp / np.maximum(served, 1)
    recall = tp / n_pos
    chr_ = served / n

    # Prepend the (0, 1)-precision point (tau -> +inf, nothing served) so
    # both curves start at x=0, matching the integral's lower bound; by
    # convention an empty-serve precision of 1.0 (vacuously) is standard
    # for PR curves.
    precision = np.concatenate([[1.0], precision])
    recall = np.concatenate([[0.0], recall])
    chr_ = np.concatenate([[0.0], chr_])

    pr_auc = float(np.trapz(precision, recall))
    p_chr_auc = float(np.trapz(precision, chr_))
    return pr_auc, p_chr_auc


def load_pairs(trace_path: Path, scored_path: Path, tau_low: float = 0.80, tau_high: float = 0.97):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from cacheverifier.experiments.verified_sweep import load_match_trace, load_scored

    trace = load_match_trace(trace_path)
    scored = load_scored(scored_path)
    gz_indices = [i for i in scored if trace[i].similarity is not None and tau_low <= trace[i].similarity < tau_high]
    scores = np.array([scored[i].score for i in gz_indices])
    labels = np.array([1 if trace[i].would_be_correct else 0 for i in gz_indices])
    return scores, labels


def report(name: str, scores: np.ndarray, labels: np.ndarray) -> None:
    pr_auc, p_chr_auc = pr_p_chr_auc(scores, labels)
    p = float(labels.mean())
    ceiling = p * (1 - np.log(p)) if 0 < p < 1 else float("nan")
    crr = p_chr_auc / pr_auc if pr_auc > 0 else float("nan")
    print(f"{name}")
    print(f"  n={len(labels)}  positive_rate(p)={p:.4f}  structural_ceiling p(1-ln p)={ceiling:.4f}")
    print(f"  PR-AUC={pr_auc:.4f}  P-CHR AUC={p_chr_auc:.4f}  op_gap={pr_auc-p_chr_auc:.4f}  CRR={crr:.4f}")
    print()


def main() -> None:
    print("=== SearchQueries (corrected) ===")
    trace_path = CACHE_DIR / "search_queries_corrected__precomputed__n150000.trace.json"
    d_scored = CACHE_DIR / "search_queries_corrected__precomputed__n150000__cross_encoder_cross-encoder_ms-marco-MiniLM-L6-v2__lo0.8__hi0.97.scored.json"
    e_scored = CACHE_DIR / "search_queries_corrected__precomputed__n150000__cross_encoder__root_workspace_finetuned_verifier_model_searchqueries_corrected__lo0.8__hi0.97.scored.json"
    scores, labels = load_pairs(trace_path, d_scored)
    report("Group D (off-the-shelf ms-marco-MiniLM-L6-v2)", scores, labels)
    scores, labels = load_pairs(trace_path, e_scored)
    report("Group E (fine-tuned)", scores, labels)

    print("=== LmArena (full 60k) ===")
    trace_path = CACHE_DIR / "lmarena__precomputed__n60000.trace.json"
    d_scored = CACHE_DIR / "lmarena__precomputed__n60000__cross_encoder_cross-encoder_ms-marco-MiniLM-L6-v2__lo0.8__hi0.97.scored.json"
    scores, labels = load_pairs(trace_path, d_scored)
    report("Group D (off-the-shelf ms-marco-MiniLM-L6-v2)", scores, labels)

    print("=== Quora (full 60k) ===")
    trace_path = CACHE_DIR / "quora__sentence-transformer_sentence-transformers_all-MiniLM-L6-v2__n60000.trace.json"
    d_scored = CACHE_DIR / "quora__sentence-transformer_sentence-transformers_all-MiniLM-L6-v2__n60000__cross_encoder_cross-encoder_ms-marco-MiniLM-L6-v2__lo0.8__hi0.97.scored.json"
    scores, labels = load_pairs(trace_path, d_scored)
    report("Group D (off-the-shelf ms-marco-MiniLM-L6-v2)", scores, labels)

    print("Reference point from Baral et al. (2026) Table 2, general cross-encoders,")
    print("ms-marco-MiniLM-L12-v2 (their two-stage K=50 retrieve+rerank pipeline):")
    print("  PR-AUC=0.565  P-CHR AUC=0.241  CRR=0.427")


if __name__ == "__main__":
    main()
