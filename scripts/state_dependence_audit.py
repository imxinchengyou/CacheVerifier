"""Lightweight state-dependence audit (no new model inference): does the
gray-zone population's composition drift over the course of the stream as
the cache grows, or is it roughly stationary? Answers question 10 from the
CRC protocol discussion -- informs whether a chronological calibration/test
split is measuring "transfer across a real distribution shift" or is
statistically equivalent to a random split for practical purposes.

Bins the already-cached match trace + gray-zone scores into deciles of
stream position and reports, per decile: gray-zone rate, incorrect rate
within gray zone, mean verifier score within gray zone.

Usage:
    python scripts/state_dependence_audit.py
"""

from pathlib import Path

import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cacheverifier.experiments.verified_sweep import load_match_trace, load_scored

CACHE_DIR = Path("results/.cache")

DATASETS = {
    "lmarena": (
        "lmarena__precomputed__n60000.trace.json",
        "lmarena__precomputed__n60000__cross_encoder_cross-encoder_ms-marco-MiniLM-L6-v2__lo0.8__hi0.97.scored.json",
    ),
    "quora": (
        "quora__sentence-transformer_sentence-transformers_all-MiniLM-L6-v2__n60000.trace.json",
        "quora__sentence-transformer_sentence-transformers_all-MiniLM-L6-v2__n60000__cross_encoder_cross-encoder_ms-marco-MiniLM-L6-v2__lo0.8__hi0.97.scored.json",
    ),
    "search_queries_corrected": (
        "search_queries_corrected__precomputed__n150000.trace.json",
        "search_queries_corrected__precomputed__n150000__cross_encoder_cross-encoder_ms-marco-MiniLM-L6-v2__lo0.8__hi0.97.scored.json",
    ),
}

N_BINS = 10


def main() -> None:
    for name, (trace_file, scored_file) in DATASETS.items():
        trace = load_match_trace(CACHE_DIR / trace_file)
        scored = load_scored(CACHE_DIR / scored_file)
        n = len(trace)
        bin_edges = np.linspace(0, n, N_BINS + 1).astype(int)

        print(f"\n=== {name} (n={n} total stream positions, gray zone = union band scored) ===")
        print(f"{'decile':>8} {'gz_rate':>9} {'n_gz':>7} {'incorrect_in_gz':>16} {'mean_verifier_score':>20}")
        for b in range(N_BINS):
            lo, hi = bin_edges[b], bin_edges[b + 1]
            idxs = range(lo, hi)
            gz_idxs = [i for i in idxs if i in scored]
            n_total = hi - lo
            n_gz = len(gz_idxs)
            gz_rate = n_gz / n_total if n_total else float("nan")
            if n_gz:
                incorrect = np.mean([0 if trace[i].would_be_correct else 1 for i in gz_idxs])
                mean_score = np.mean([scored[i].score for i in gz_idxs])
            else:
                incorrect, mean_score = float("nan"), float("nan")
            print(f"{b:>8} {gz_rate:>9.3f} {n_gz:>7} {incorrect:>16.3f} {mean_score:>20.3f}")


if __name__ == "__main__":
    main()
