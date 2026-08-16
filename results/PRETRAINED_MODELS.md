# Fine-tuned verifier checkpoints (Group E)

The three fine-tuned cross-encoder checkpoints referenced in Section 5.6 of
the paper are hosted on the Hugging Face Hub (large binaries don't belong in
a git repo), under the same all-rights-reserved terms as this repository's
`LICENSE`:

- LmArena: https://huggingface.co/ChengyouXin/cacheverifier-lmarena
- SearchQueries: https://huggingface.co/ChengyouXin/cacheverifier-searchqueries
- Quora: https://huggingface.co/ChengyouXin/cacheverifier-quora

```python
from sentence_transformers import CrossEncoder

model = CrossEncoder("ChengyouXin/cacheverifier-lmarena")
```

**Provenance notes (2026-08-16):**

- The LmArena checkpoint is the exact model that produced this paper's
  original Section 5.6 numbers.
- The SearchQueries checkpoint hosted here is the version trained on the
  *corrected* dataset (see the erratum at the top of `PAPER.md`/`PAPER_EN.md`)
  — it replaces an earlier upload that was trained before that data defect
  was found and fixed.
- The Quora checkpoint was never persisted after its original 2026-07
  training run (not committed to this repo, never previously uploaded here).
  This upload is a faithful retrain from the exact original train/test split
  (`results/finetune_verifier_experiment_quora.examples.json`); the
  retrained baseline AUC matches the original report exactly (0.6309) and
  the fine-tuned AUC is close (0.7407 vs. the original 0.7393). See
  `RESEARCH_PROPOSAL.md` §10 direction 1 for the full account.
