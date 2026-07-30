# Synchronous Online Verification Gating in Semantic Caches

**Chengyou Xin** · LoopDot AI Research · 2026-07-26

Semantic caches replace exact matching with vector similarity to reuse an
LLM's past answers, but similarity and answer correctness are not the same
quantity. This repo is the code and full experimental artifacts behind an
empirical study asking one question: under a single-tier semantic cache,
does gating cache hits with a **real** (non-oracle), **synchronous**
verifier — evaluated online against static-threshold and adaptive-threshold
baselines on ~210k real requests across three datasets — actually improve
the hit-rate/error-rate trade-off?

**Read the paper:** [`PAPER.md`](PAPER.md) (Chinese) · [`PAPER_EN.md`](PAPER_EN.md) ·
[`PAPER_EN.tex`](PAPER_EN.tex) (LaTeX source)

**Headline finding:** an oracle verifier shows the mechanism has 20–28
percentage points of theoretical headroom, but an off-the-shelf real
verifier cashes in only a small, fragile slice of it — the paper's Go/No-Go
verdict is a **weak Go**. Fine-tuning that same verifier on a dataset's own
gray-zone labels recovers most of the gap on every one of three independent
datasets tested, and the paper goes on to test whether that recipe survives
label noise, cold start, temporal drift, and real production traffic (with
one genuine counter-example). See the paper's abstract for the full summary.

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

## License

**All rights reserved** — see [`LICENSE`](LICENSE). This repository is
public to support reproducibility of the paper's reported results; no
license is granted for reuse, modification, or redistribution.