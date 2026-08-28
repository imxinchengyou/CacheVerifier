# Synchronous Online Verification Gating in Semantic Caches

**Chengyou Xin** · LoopDot AI Research · 2026-07-26

English | [简体中文](README_ZH.md)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21703364.svg)](https://doi.org/10.5281/zenodo.21703364)

Semantic caches replace exact matching with vector similarity to reuse an
LLM's past answers, but similarity and answer correctness are not the same
quantity. This repo is the code and full experimental artifacts behind an
empirical study asking one question: under a single-tier semantic cache,
does gating cache hits with a **real** (non-oracle), **synchronous**
verifier — evaluated online against static-threshold and adaptive-threshold
baselines on ~210k real requests across three datasets — actually improve
the hit-rate/error-rate trade-off?

![Hit-rate vs. error-rate Pareto frontier on LmArena: static threshold, adaptive threshold, oracle verifier, off-the-shelf verifier, and domain-fine-tuned verifier](results/lmarena_pareto_full_with_finetune.png)

**TL;DR**
- An **oracle** verifier proves the mechanism has real headroom: +20–28
  percentage points of hit rate at matched error rate on both benchmark
  datasets.
- An **off-the-shelf** cross-encoder verifier cashes in only a small,
  fragile slice of that headroom under the paper's original grid-searched
  evaluation — the paper's Go/No-Go verdict is a **weak Go**, not an
  unqualified win. **[2026-08-15 update]** Retested with an *honest*
  threshold selection (chronological calibration/test split, no peeking at
  the test half), SearchQueries' verdict reverses to a clean win at every
  tested point — how much of the original "net harmful" result reflects a
  genuine SearchQueries-specific weakness versus how the original
  evaluation happened to pick its threshold is now the paper's own
  least-settled open question (§6.1/§5.4).
- **Fine-tuning that same verifier on a dataset's own gray-zone labels**
  closes most of the gap on all three independent datasets tested,
  including turning SearchQueries from a *net-harmful* verifier (AUC 0.60 —
  see the erratum note at the top of [`PAPER.md`](PAPER.md) /
  [`PAPER_EN.md`](PAPER_EN.md): an earlier release of this paper reported
  AUC 0.49 due to a since-corrected data defect) into one that beats the
  static-threshold frontier at 53 of 54 tested points, 1 tie, zero losses
  (AUC 0.71) — and the same zero-loss verdict holds under honest
  calibration too, on all three datasets.
- The recipe tolerates realistic label noise (~30%) and cold start, and
  holds up on real production customer-support traffic — with **one
  genuine counter-example**, traced to a specific, monitorable cause, and
  a working monitor prototype that catches it before it does damage.
- **[2026-08-17 update]** A reproduction bug in the adaptive-threshold
  baseline (Group B) was found and fixed — the official vCache algorithm
  pre-seeds each cache entry with two synthetic bootstrap observations that
  this paper's earlier port omitted. After the fix, Group B's hit rate rises
  **4.4x–29.1x** across all three datasets, with error rate staying below
  the target guarantee throughout.

**Read the paper:** [`PAPER.md`](PAPER.md) (Chinese) · [`PAPER_EN.md`](PAPER_EN.md) ·
[`PAPER_EN.tex`](PAPER_EN.tex) (LaTeX source)

## Results at a glance

| Dataset | Off-the-shelf verifier (Group D) | Domain-fine-tuned verifier (Group E) |
|---|---|---|
| LmArena (conversational) | AUC 0.72 · best reproducible net gain ≈ **+1.9pp** hit rate (grid search) · **+5.66pp** under honest calibration | AUC **0.88** · beats static-threshold frontier at nearly every tested point (grid search) · **6/6** under honest calibration, **+5.66pp** |
| SearchQueries (short keyword) | AUC 0.60 · **net harmful** under grid search (23/36 losses to static threshold) · **reverses to 6/6 wins** (+0.78pp to +3.67pp) under honest calibration | AUC **0.71** · wins 53/54 tested points, 1 tie, 0 losses (grid search) · **6/6** under honest calibration, **+7.74pp** |
| Quora (paraphrase pairs) | — (not in original benchmark) | Smaller-magnitude replication of the same pattern; never worse than the untuned baseline under either grid search or honest calibration (0 losses either way) |

"Grid search" = the paper's original hand-picked threshold grid, best point reported. "Honest calibration" = a threshold chosen via Youden's J on a held-out calibration half only, then measured on the untouched test half (§5.4) — added 2026-08-15/16 specifically to test whether the grid-search numbers above were optimistic; see §5.4/§6.1 for the full account of where the two methods agree and where they don't (Quora is the one dataset where honest calibration is *worse*, traced to the dataset's own score-separability ceiling, not a calibration artifact).

Oracle ceiling (upper bound on the mechanism, both benchmark datasets): **+20–28pp** hit rate at matched error rate. A separate reproduction fix for the adaptive-threshold baseline (Group B) raised its hit rate **4.4x–29.1x** across all three datasets (§5.2) — Group B sits at a different hit-rate scale and isn't part of the Go/No-Go comparison above. Full numbers, confidence intervals, and further robustness/ablation sections (noise, cold start, drift monitor, τ_high sensitivity, reranker capacity vs. training distribution, Conformal Risk Control, rewrite-vs-reject, Top-K cascade, CRC closed-loop self-selection) are in the paper, §5.9–§5.16.

## Further ablations (§5.9–§5.16)

- **Drift monitor (§5.9):** two change-point tests on gray-zone labels alone
  catch the one real-traffic counter-example's degradation before it does
  damage, with no false alarms on the unaffected brand.
- **Action-verb bucketing pre-filter (§5.10):** tested and refuted on all
  three datasets.
- **τ_high sensitivity (§5.11):** dataset-dependent — widening it more than
  triples LmArena's net lead but flips SearchQueries to a net loss.
- **Reranker capacity vs. training distribution (§5.12):** neither a
  larger same-distribution reranker nor a broader-distribution one
  meaningfully closes SearchQueries' gap — in-domain fine-tuning (§5.6)
  remains the only verified remedy.
- **Conformal Risk Control (§5.13):** upgrades the gray-zone reuse
  threshold from a point estimate to a finite-sample risk guarantee, at
  near-oracle efficiency (η≈1.0) across all three datasets.
- **Rewrite instead of reject (§5.14):** a TweakLLM-style rewrite-and-serve
  policy shows no measurable net benefit over the existing binary gate — a
  negative ablation.
- **Top-K candidate cascade (§5.15):** retrieving more than the single
  nearest neighbor is close to a free lunch on LmArena, essentially no
  effect on Quora, and a real hit-rate/error-rate trade-off on SearchQueries
  that fine-tuning mitigates but doesn't eliminate.
- **CRC closed-loop self-selection (§5.16):** a genuine online closed-loop
  test (not a static split) confirms the gate's own reuse/reject decisions
  can feed back into future cache state — harm scales monotonically with
  direct-hit rate, from no detectable effect (Quora) to more than tripling
  realized risk (LmArena); online recalibration fully compensates on two
  of three datasets but not the third.

## Repository layout

| Path | Contents |
|---|---|
| `cacheverifier/` | Cache policies (static/adaptive/synchronous-verified), embedders, verifiers, metrics, experiment runners |
| `scripts/` | Dataset conversion, fine-tuning, drift-monitoring, and plotting scripts referenced throughout the paper |
| `configs/` | Per-dataset YAML configs (LmArena, SearchQueries, Quora, Twitter Amazon/Comcast) |
| `results/` | Every reported metric (JSON) and figure (PNG); see [`results/PRETRAINED_MODELS.md`](results/PRETRAINED_MODELS.md) for the two fine-tuned verifier checkpoints, hosted on Hugging Face rather than committed here |
| `tests/` | Unit tests for `cacheverifier/` |

## Reproducing

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
# Verifier fine-tuning / cross-encoder experiments need the heavier deps:
.venv/bin/pip install -r requirements-embeddings.txt

.venv/bin/pytest tests/ -q
```

Datasets are not redistributed in this repo (see `data/` in `.gitignore`) —
`scripts/convert_*.py` regenerate them from the public sources cited in the
paper's §4.1 (HuggingFace `vCache/SemBenchmarkLmArena` / `SemBenchmarkSearchQueries`,
Quora Question Pairs, the Twitter Customer Support corpus). Each `configs/*.yaml`
then drives `cacheverifier/experiments/run_baselines.py` (Groups A/B) and
`run_verified.py` (Groups C/D) for that dataset.

## Fine-tuned models

The Group E fine-tuned verifiers are on the Hugging Face Hub, not in this
repo — see [`results/PRETRAINED_MODELS.md`](results/PRETRAINED_MODELS.md).

## Citation

Archived on Zenodo with DOI [10.5281/zenodo.21703364](https://doi.org/10.5281/zenodo.21703364)
(this concept DOI always resolves to the latest version; the current version is v1.2.0, DOI [10.5281/zenodo.22055591](https://doi.org/10.5281/zenodo.22055591)).
arXiv listing forthcoming — this will be updated with the arXiv ID once live.

```bibtex
@misc{xin2026synchronous,
  title  = {Synchronous Online Verification Gating in Semantic Caches: An Empirical Study},
  author = {Xin, Chengyou},
  year   = {2026},
  note   = {LoopDot AI Research},
  url    = {https://github.com/imxinchengyou/CacheVerifier},
  doi    = {10.5281/zenodo.21703364}
}
```

## Acknowledgments

Groups A/B of this work build directly on public benchmarks and reference
code from the **vCache** project (L. G. Schroeder, A. Desai, A. Cuadron,
K. Chu, S. Liu, M. Zhao, S. Krusche, A. Kemper, I. Stoica, M. Zaharia, and
J. E. Gonzalez) — the SemCacheLmArena/SemCacheSearchQueries datasets, the
static-threshold grid, and the `VerifiedDecisionPolicy` this paper ports
line-by-line. Sections 5.6 and 5.8 further build on **Quora Question
Pairs** (Iyer, Dandekar, & Csernai, 2017) and the Kaggle **"Customer
Support on Twitter"** dataset (Axelbrooke, 2017). See the paper's own
Acknowledgments section for the full note.

## License

**All rights reserved** — see [`LICENSE`](LICENSE). This repository is
public to support reproducibility of the paper's reported results; no
license is granted for reuse, modification, or redistribution.