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
  fragile slice of that headroom — the paper's Go/No-Go verdict is a
  **weak Go**, not an unqualified win.
- **Fine-tuning that same verifier on a dataset's own gray-zone labels**
  closes most of the gap on all three independent datasets tested,
  including turning SearchQueries from a *net-harmful* verifier (AUC 0.60 —
  see the erratum note at the top of [`PAPER.md`](PAPER.md) /
  [`PAPER_EN.md`](PAPER_EN.md): an earlier release of this paper reported
  AUC 0.49 due to a since-corrected data defect) into one that beats the
  static-threshold frontier at 53 of 54 tested points, 1 tie, zero losses
  (AUC 0.71).
- The recipe tolerates realistic label noise (~30%) and cold start, and
  holds up on real production customer-support traffic — with **one
  genuine counter-example**, traced to a specific, monitorable cause, and
  a working monitor prototype that catches it before it does damage.

**Read the paper:** [`PAPER.md`](PAPER.md) (Chinese) · [`PAPER_EN.md`](PAPER_EN.md) ·
[`PAPER_EN.tex`](PAPER_EN.tex) (LaTeX source)

## Results at a glance

| Dataset | Off-the-shelf verifier (Group D) | Domain-fine-tuned verifier (Group E) |
|---|---|---|
| LmArena (conversational) | AUC 0.72 · best reproducible net gain ≈ **+1.9pp** hit rate | AUC **0.88** · beats static-threshold frontier at nearly every tested point |
| SearchQueries (short keyword) | AUC 0.60 · **net harmful** (23/36 losses to static threshold) | AUC **0.71** · wins 53/54 tested points, 1 tie, 0 losses |
| Quora (paraphrase pairs) | — (not in original benchmark) | Smaller-magnitude replication of the same pattern; never worse than the untuned baseline |

Oracle ceiling (upper bound on the mechanism, both benchmark datasets): **+20–28pp** hit rate at matched error rate. Full numbers, confidence intervals, and four further robustness ablations (noise, cold start, drift, real production traffic) are in the paper, §5.

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
(this concept DOI always resolves to the latest version; the current version is v1.1.0, DOI [10.5281/zenodo.22020647](https://doi.org/10.5281/zenodo.22020647)).
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