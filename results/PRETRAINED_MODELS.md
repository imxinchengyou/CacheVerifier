# Fine-tuned verifier checkpoints (Group E)

The two fine-tuned cross-encoder checkpoints referenced in Section 5.6 of
the paper were previously committed directly into this repo under
`results/finetuned_verifier_model{,_searchqueries}/`. They're now hosted on
the Hugging Face Hub instead (large binaries don't belong in a git repo),
under the same all-rights-reserved terms as this repository's `LICENSE`:

- LmArena: https://huggingface.co/ChengyouXin/cacheverifier-lmarena
- SearchQueries: https://huggingface.co/ChengyouXin/cacheverifier-searchqueries

```python
from sentence_transformers import CrossEncoder

model = CrossEncoder("ChengyouXin/cacheverifier-lmarena")
```
