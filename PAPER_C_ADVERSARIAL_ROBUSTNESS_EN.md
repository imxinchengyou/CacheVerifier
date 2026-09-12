# Adversarial Robustness of Semantic Cache Verifiers: Gaps Exposed by Red-Teaming and Partial Repair via Training

**Author:** Chengyou Xin
**ORCID:** [0009-0008-2347-5136](https://orcid.org/0009-0008-2347-5136)
**Affiliation:** LoopDot AI Research
**Date:** 2026-08-12
**Note:** This research project was originally released as a single, complete technical report covering all experiments (Zenodo DOI: 10.5281/zenodo.21703364, a concept DOI, always resolving to the latest version); this paper is one of the companion papers distilled from that project into an independent, self-contained unit. This paper is a companion to [Xin, C. (2026). Synchronous Online Verification Gating in Semantic Caches: An Empirical Study — Part I: Core Findings and the Go/No-Go Verdict. Zenodo. https://doi.org/10.5281/zenodo.22660442.](hereafter "the main paper"), reusing the main paper's datasets, gray-zone architecture, and honest-calibration protocol; this paper does not re-derive that infrastructure and only briefly restates it where necessary — see the main paper for full details.
**Code and full experimental artifacts:** this repository, `scripts/llm_redteam_verifier.py`, `scripts/finetune_verifier_train_eval.py`, `results/llm_redteam_results.json`, `results/llm_redteam_train_pool.json`

---

## Abstract

The main paper shows that fine-tuning a lightweight cross-encoder verifier on in-domain gray-zone labels reliably repairs its discriminative gap on natural benchmark data. But the hard cases in "natural benchmark data" are whatever a dataset happens to contain, not cases deliberately designed to fool the verifier — this can only reflect the verifier's average performance against *this particular distribution*, not how badly it fails in the worst case against queries specifically constructed to beat it. This paper measures that worst case systematically, using an LLM to generate adversarial samples covering five known failure axes (negation, action-verb swap, direction reversal, named-entity swap, and quantity/parameter swap): across 306 adversarial samples that clear the gray-zone similarity threshold, an off-the-shelf verifier's false-accept rate reaches 84.0% (95% CI [79.7%, 87.9%]), far beyond anything measured on natural data elsewhere in this project; more importantly, the in-domain fine-tuning the main paper validates as an effective remedy **provides no protection at all** on this batch of adversarial samples (fine-tuned false-accept rate 87.6%, its confidence interval almost fully overlapping the untuned one) — meaning fine-tuning learns what this dataset's natural noise distribution looks like, not a general robustness to deliberately constructed attacks; the two are different capabilities. A direct remedy was tested: mixing adversarial samples targeted at these five failure axes — only 3.8% of the training set — into the original natural training data and re-fine-tuning. This drops the false-accept rate from 84–88% down to 53.6% (95% CI [48.0%, 59.2%], not overlapping either of the other two models' confidence intervals), without sacrificing natural-data discriminative power (AUC rises from 0.7212 to 0.8749, nearly matching the 0.88 that pure natural-data fine-tuning achieves). This suggests the adversarial-robustness gap is not necessarily a fundamental limitation of the fine-tuning mechanism itself, and at least partly reflects a prior absence of this category of sample in the training data (only one data proportion was tested here; whether a higher proportion narrows the gap further remains unverified) — but 53.6% still means more than half of adversarial samples fool the verifier: the problem is substantially mitigated, far from solved.

---

## 1. Introduction

Semantic caching replaces exact matching with vector similarity to reuse an LLM's past answers, and the main paper shows that whether adding a real, lightweight verifier for synchronous online gating to this architecture realizes its theoretical benefit depends heavily on how well the verifier matches the business data domain — in-domain fine-tuning is currently the only remedy validated to work. But that conclusion rests entirely on **natural benchmark data**: the similar query pairs occurring in the LmArena, SearchQueries, and Quora datasets all arise naturally from each dataset itself, not from anyone deliberately designing them to fool the verifier.

This leaves a gap that has never been systematically measured: if someone (or simply the traffic itself) happens to construct query pairs that are superficially highly similar but require sharply different answers — "can I still use my subscription" vs. "can I no longer use my subscription," "is the refund policy limited to this month" vs. "is the refund policy not limited to this month" — how fragile is a verifier that performs well on natural data? Such hard cases have previously appeared only as anecdotes in Reddit discussions (like the classic "cancel"/"pause" subscription example), never systematically measured, and it is unknown whether the remedy the main paper validates (in-domain fine-tuning) can survive this worst case.

This paper fills that gap, answering two questions: (1) how badly does a verifier that performs well on natural data — even one fine-tuned in-domain — fail in the worst case against deliberately constructed adversarial queries; (2) if natural-data fine-tuning is not enough, can adversarial training targeted at known failure axes close the gap?

---

## 2. Relationship to the Main Paper

This paper is not a standalone verifier-robustness study; it is a necessary supplement to the main paper's Go/No-Go verdict. The main paper's "weak Go" verdict rests on error rates measured against natural benchmark data, and this paper shows that those error rates **systematically understate** how fragile the verifier really is under adversarial conditions. That means, for any deployment that might face deliberately misleading queries (a public-facing customer-support bot, say, where users may construct such misleading questions intentionally or not), the main paper's "weak Go" verdict itself needs further qualification from this paper's findings — the mechanism is conditionally worth doing under natural traffic, but deployments with meaningful adversarial exposure need additional remedies, validated here.

The five failure axes tested here (negation, action-verb swap, direction reversal, named-entity swap, quantity/parameter swap) were not designed from scratch; they reuse specific mechanisms already identified in the main paper's own research process: negation comes from this paper's own motivating discussion; action-verb swap follows the classic "cancel/pause" paradigm; direction reversal is the "same verb, opposite object direction" mechanism found in one of the main paper's tested-and-refuted bucketing pre-filter hypotheses; named-entity swap is the "H.W. Bush/W. Bush"-style hard case diagnosed while honestly calibrating the main paper's Quora dataset; quantity/parameter swap is a newly added fifth category. These five axes were later reused and extended by a companion paper (the integration-layer paper, which tests joint decisions and non-stationarity adaptation) into a cross-dataset failure-axis taxonomy — this paper is the first systematic, independent measurement of that taxonomy.

---

## 3. Experimental Setup (Reused from the Main Paper; Restated Only as Necessary)

**Dataset.** All experiments in this paper run only on SemCacheLMArena — a conversational dataset drawn from LM-Arena human preference logs, 60,000 records spanning 3,500 GPT-4o-mini-generated paraphrase classes with 1–23 paraphrases each; full construction details are in the main paper's Section 4.1. LmArena, rather than the other two datasets, was chosen because it is the dataset where the main paper's fine-tuning gain is most robust (held-out AUC rising from 0.72 to 0.88), making it the best test case for whether that fine-tuning gain extends to adversarial conditions.

**Gray-zone architecture.** When a query's similarity to a cached candidate falls in `[tau_low, tau_high)`, the request enters the gray zone and is handed to the verifier to decide whether to reuse; this paper fixes tau_low=0.80, the widest grid setting used throughout the main paper, and the necessary condition an adversarial sample must clear before it can actually reach the verifier at all.

**Verifiers.** Both versions are taken directly from the main paper — (a) **off-the-shelf** (Group D): `cross-encoder/ms-marco-MiniLM-L6-v2`, without any in-domain fine-tuning; (b) **in-domain fine-tuned** (Group E): the same cross-encoder fine-tuned on LmArena's own gray-zone labels (checkpoint: Hugging Face `ChengyouXin/cacheverifier-lmarena`), the same checkpoint the main paper's honest-calibration table uses.

---

## 4. Results

### 4.1 LLM-Driven Automated Red-Teaming: How Does the Verifier Perform on Deliberately Constructed Hard Cases

**Method.** DeepSeek (`deepseek-chat`) generates adversarial samples covering the five specific failure axes listed in Section 2 above, rather than arbitrarily designed ones. Each category generates a number of `(query_a, query_b, answer_b)` triples, where `answer_b` is a specific answer that holds only for `query_b` — wrong if treated as an answer to `query_a`. After generation, `query_a`/`query_b` are encoded with the same live encoder the main paper's Quora dataset uses (`sentence-transformers/all-MiniLM-L6-v2`), keeping only pairs whose cosine similarity clears the tau_low=0.80 threshold — the necessary condition for a pair to actually reach the verifier in the real pipeline. Qualifying pairs are scored with the same off-the-shelf cross-encoder (Group D) and the LmArena fine-tuned checkpoint (Group E). For statistical stability, 4 independent rounds were generated (20 per category each) and pooled, rather than trusting a single round — a single round (15–20 per category) swung the same untuned verifier's overall false-accept rate from 58.2% to 85.2% between two independent trial runs, far too much variance to trust; pooling and reporting a parameter-free bootstrap CI on the aggregate fixes this. Script: `scripts/llm_redteam_verifier.py`; results: `results/llm_redteam_results.json`.

**A diagnosed environment issue that also fixed the comparison method.** Under the sentence-transformers version used for this experiment, the fine-tuned checkpoint's `predict()` returns scores compressed into [0, 1] (suspected cause: this version fails to recognize the `activation_fn` metadata saved by a newer version, silently defaulting to sigmoid) rather than the unbounded logits otherwise assumed — meaning the default threshold=0.0 decision rule is met almost unconditionally in [0,1]-probability space, not a meaningful comparison. The script now auto-probes each verifier's score range with two known-answer sanity pairs, and switches to the mathematically equivalent threshold=0.5 when a [0,1]-bounded range is detected (sigmoid is monotonic, so logit≥0 iff sigmoid(logit)≥0.5 — this substitution only compensates for the loading-environment difference and does not change the actual decision criterion).

**Results** (4 rounds pooled, 306/400 pairs cleared tau_low=0.80):

| Category | Qualifying pairs | Off-the-shelf false-accept rate | Fine-tuned (LmArena) false-accept rate |
|---|---|---|---|
| Negation | 79 | 96.2% | 98.7% |
| Quantity swap | 80 | 96.2% | 90.0% |
| Direction reversal | 76 | 86.8% | 92.1% |
| Named-entity swap | 39 | 48.7% | 66.7% |
| Action-verb swap | 32 | 59.4% | 68.8% |
| **Total** | **306** | **84.0%**, 95% CI [79.7%, 87.9%] | **87.6%**, 95% CI [83.7%, 91.2%] |

Two key findings: (1) The false-accept rate is far higher than anything the main paper measures on natural data (its worst natural error rates are single digits to low tens of percent), meaning natural-benchmark performance systematically overstates the verifier's robustness to real adversarial conditions; inspecting the highest-confidence false-accepts (e.g. "can I use this gift card on sale items" vs. "...actually, can I not use it on sale items," scored 11.32 against a threshold of 0.0) confirms this is not borderline ambiguity — the verifier confidently approves an answer that is the literal opposite of correct. (2) In-domain fine-tuning, which the main paper shows substantially repairs discriminative power on natural data, **provides essentially no protection on adversarial samples** — the two 95% confidence intervals nearly fully overlap (84.0% vs. 87.6%), meaning this fine-tuning recipe learns "what this dataset's natural noise distribution looks like" rather than a general robustness to deliberately constructed attacks — two different capabilities that cannot substitute for one another. Negation is the most severe blind spot measured here — consistent with the verifier's training objective (MS MARCO passage-relevance ranking, never specifically optimized for negation/polarity), not a coincidence.

**Limitations.** (1) The adversarial samples are LLM-generated, and no independent human verification confirmed that every sample's premise (that query_a and query_b genuinely require different answers) holds — though the highest-confidence inspected cases are unambiguous, no systematic manual quality audit was performed. (2) Only tau_low=0.80 was tested; behavior at higher similarity thresholds is untested. (3) Only the LmArena fine-tuned checkpoint was tested; how SearchQueries'/Quora's fine-tuned checkpoints handle the same adversarial samples is unknown (though since LmArena is the most robust fine-tuning success story of the three datasets, SearchQueries/Quora's adversarial robustness is unlikely to be better). (4) Category sample sizes are uneven (39 for named-entity swap vs. 80 for quantity swap) — per-category percentages carry wider uncertainty than the aggregate figure and shouldn't be over-interpreted for fine-grained cross-category ranking. (5) This round of red-teaming did not test the false-reject direction (rejecting a candidate that should have been approved) — only whether the verifier becomes easier to fool, not whether adversarial inputs might also make it overly conservative.

### 4.2 Can Adversarial Training Close This Gap

**Motivation.** Section 4.1 left the sharpest open question: given that in-domain fine-tuning helps on natural noise but provides no protection against deliberately constructed adversarial samples, can training specifically targeted at these five failure axes close the gap? This is the most direct next step — reusing the adversarial data already generated and the existing fine-tuning pipeline, at low cost.

**Method.** Independent of Section 4.1's 306 held-out test samples, a fresh batch of adversarial training samples was generated separately using the same five failure-axis templates (223 triples, `results/llm_redteam_train_pool.json`, non-overlapping with the held-out test set by construction — two independent stochastic generations). Each triple becomes two training rows: (query_a, answer_b) → label 0 (the specific adversarial case Section 4.1 showed both verifiers false-accept), (query_b, answer_b) → label 1 (the contrasting correct pairing). These 446 adversarial rows were merged directly with the 11,271 natural gray-zone training rows the main paper's own LmArena fine-tuning used (adversarial data is 3.8% of the combined set), and fine-tuned with the exact same training procedure (off-the-shelf base, 1 epoch, batch_size=16, `scripts/finetune_verifier_train_eval.py` reused unmodified) — so the only variable between this run and the main paper's original result is the training data itself (pure natural data vs. natural plus adversarial data), not the method. Trained on a remote GPU (Tesla T4), 733 steps in 206.9 seconds. Evaluation: (a) AUC on the main paper's original natural-data held-out test set (4,831 rows), to check whether natural-data discriminative power was sacrificed; (b) re-scoring the new model against Section 4.1's **exact same** 306 held-out adversarial samples (`scripts/llm_redteam_verifier.py --load-triples`, skipping regeneration to guarantee the test set was never touched during this training).

**Results.** Natural-data AUC rose from 0.7212 (untuned baseline, matching the main paper's originally reported baseline) to 0.8749, essentially matching the 0.88 the main paper's pure-natural-data fine-tuning achieved — adding 3.8% adversarial training data **did not cost natural-data discriminative power**. False-accept rate on the adversarial held-out set:

| Verifier | Overall false-accept rate | 95% CI |
|---|---|---|
| Off-the-shelf (Section 4.1) | 84.0% | [79.7%, 87.9%] |
| Naturally fine-tuned (Section 4.1) | 87.6% | [83.7%, 91.2%] |
| **Adversarial + natural fine-tuned (this section)** | **53.6%** | **[48.0%, 59.2%]** |

Adversarial training cut the false-accept rate from 84–88% to 53.6%, with a confidence interval that does not overlap either of the other two — **this is a real, statistically significant improvement, not noise**. By category, most axes improved substantially (negation 96.2%→57.0%, action-verb swap 59.4%→12.5%, quantity swap 96.2%→58.8%, direction reversal 86.8%→57.9%), but **named-entity swap is the one category that got worse** (48.7%→61.5%) — this category also had the fewest training triples of the five (28 of 223), and whether insufficient training volume or some cross-category negative transfer explains this has not been ruled out.

**This is not a "problem solved" result — it is a "substantially mitigated but far from eliminated" result**: 53.6% still means more than half of adversarial samples fool the verifier, a long way from the single-digit-to-low-tens error rates on natural data. But the direction is clear: **this capability gap is not unfixable — a small (only 3.8%), targeted amount of adversarial training on known failure axes produces a large improvement at no natural-data cost** — a step beyond Section 4.1's diagnosis that "fine-tuning learns the natural noise distribution, not adversarial robustness": given the right data in the training set, the same fine-tuning mechanism can in fact learn some adversarial robustness too — this is not a fundamental limitation of the mechanism itself.

**Limitations.** (1) Only one adversarial data proportion (3.8%) was tested; whether a higher proportion closes the gap further, or starts costing natural-data performance, was not tested. (2) The named-entity-swap regression was not further diagnosed — unclear whether it's insufficient sample size or cross-category negative transfer, and whether simply feeding this category more training data would fix it was not tested. (3) Training and evaluation data, while two independent, non-overlapping generations, both come from the same LLM (DeepSeek) and the same family of prompt templates — whether the robustness learned here generalizes to adversarial samples from a different generator was not tested, so this could reflect learning this one generator's particular style rather than true cross-source generalization. (4) Like Section 4.1, only LmArena was tested. (5) The 53.6% figure was not compared against a simpler, more conservative default strategy (e.g., raising the static threshold) — a sufficiently conservative static threshold could in principle also suppress the false-accept rate, at the cost of hit rate, and this trade-off in the adversarial setting specifically was not measured.

---

## 5. Discussion

Together, the two sections above point to a distinction not previously made explicit: **"how good is the verifier's discriminative power" and "how robust is the verifier" are two different capability dimensions, and fine-tuning on the same batch of natural data only improves the former.** This distinction matters practically for any system relying on a verifier for a binary decision — a high AUC on natural benchmark data alone cannot be taken to mean the system is equally reliable against deliberately constructed inputs; conversely, if a deployment genuinely risks deliberately misleading input (whether from malicious users or unintentionally highly ambiguous phrasing), evaluation results on natural data will be systematically too optimistic.

Section 4.2's results also show that this gap is fundamentally a data-coverage problem, not a fundamental limitation of the architecture or training method: as long as the training data includes the corresponding failure pattern, the same fine-tuning mechanism can learn the corresponding robustness, and can do so at almost no cost to natural-data performance (a 3.8% adversarial-data share already produced a statistically significant improvement). This offers a concrete, low-cost recommendation for deployments: if certain failure axes have been identified as a real risk for a given business use case (negation, for instance, may correspond to high-frequency phrasing like "can I" vs. "can I not" in customer support), targeted supplementation with a small amount of adversarial training data is more realistic than waiting for a general, architecture-level robustness breakthrough.

The named-entity-swap category getting worse after adversarial training is this paper's one directional anomaly; the explanation available at the time went no further than the open guess that "adversarial training's own data mix (the sample-size balance across failure axes) may need targeted tuning." A later companion paper (testing joint decisions and non-stationarity adaptation), in an independent investigation, found that the root cause specific to the named-entity-swap axis lies elsewhere — the verifier's pretraining objective (specifically, whether it was trained on semantic entailment) matters more than the training-data mix. This paper's "data mix" guess was therefore not borne out; the real lever lies elsewhere, a point left for the reader to verify against that later work.

---

## 6. Limitations

This section aggregates the limitations already listed in Sections 4.1 and 4.2 without repeating them (see those subsections for details), and adds two cross-cutting limitations: (1) all experiments in this paper run on a single dataset (LmArena) and a single verifier architecture (the `ms-marco-MiniLM-L6-v2` family of cross-encoders); whether the specific adversarial-robustness figures (84.0%, 53.6%, etc.) generalize to other datasets or verifier architectures is unverified; (2) this paper did not test whether adversarial training introduces new fragility on adversarial query types outside its training distribution and outside the five failure axes tested here — a common risk with adversarial training is that it hardens against known attack surfaces while doing nothing, or even causing harm, against unknown ones, and no experiment here was specifically designed to check for this.

---

## 7. Conclusion and Future Work

This paper is the first systematic measurement of a lightweight verifier's worst-case performance, in a semantic-cache gating role, against deliberately constructed adversarial queries: both the off-the-shelf and in-domain fine-tuned versions have false-accept rates in the 84–88% range, far beyond anything seen on natural data, and in-domain fine-tuning provides no protection against this at all. Targeted adversarial training brings this figure down to 53.6% — a real, statistically significant improvement at almost no cost to natural-data performance — but the problem is far from solved.

Future work, in priority order:

1. Test whether a higher adversarial-data proportion narrows the gap further, and at what proportion natural-data performance starts to suffer.
2. Diagnose why the named-entity-swap category gets worse after adversarial training (insufficient sample size vs. cross-category negative transfer), and test whether targeted additional training data fixes it.
3. Generate adversarial samples with an LLM independent of DeepSeek, to verify that the robustness learned here generalizes across generators rather than overfitting to one generator's particular style.
4. Reproduce this paper's red-teaming method and adversarial-training recipe on the SearchQueries and Quora datasets, to check whether the findings are specific to LmArena.
5. Test whether adversarial training introduces new fragility outside this paper's five failure axes.
6. Directly compare the 53.6% residual false-accept rate against a more conservative static-threshold strategy, quantifying the respective hit-rate cost of "add training data" versus "raise the threshold" in the adversarial setting.

---

## Acknowledgments

This paper's experimental setup, datasets, and verifier implementation build directly on the main paper; thanks to the vCache project, cited by the main paper, for publicly releasing the `SemCacheLmArena` dataset and reference implementation.

---

## References

- Xin, C. (2026). Synchronous Online Verification Gating in Semantic Caches: An Empirical Study — Part I: Core Findings and the Go/No-Go Verdict. Zenodo. https://doi.org/10.5281/zenodo.22660442 (the main paper).
- Schroeder, L. G., Desai, A., Cuadron, A., Chu, K., Liu, S., Zhao, M., Krusche, S., Kemper, A., Stoica, I., Zaharia, M., & Gonzalez, J. E. (2025). vCache: Verified semantic prompt caching. *arXiv:2502.03771*.
