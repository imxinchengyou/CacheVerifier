# Semantic-cache hit decisions on real customer-support traffic: CacheVerifier, vCache, Redis static threshold, GPTCache

[中文](ENTERPRISE_COMPARISON_ZH.md)

2026-11-02 to 2026-11-03. Four ways of deciding a semantic-cache hit are compared on the same real enterprise customer-support data, with the same embedding, the same request stream and the same labels. The labels themselves are checked with blind human annotation.

## Summary

- Compared at a matched error rate, the fine-tuned CacheVerifier is the best of the four on this Twitter support data. At strict error rates (1%–3%) it serves roughly 6%–15% more hits (relative) than a static threshold. This holds under both judge-label versions, under recalibration against human labels, and for all three verifier input formats tested.
- The absolute gain on this kind of traffic is small. At a 2% error rate every method serves only about 5%–9% of requests from the cache.
- The off-the-shelf (not fine-tuned) CacheVerifier is worse than a static threshold, as in §5.4 of the paper.
- GPTCache's default configuration (cosine ≥ 0.60) hits 79%–83% of requests, most of them wrong. GPTCache's learned evaluator performs about like a static threshold and adds 60–90 ms per request.
- vCache keeps its error rate, but is very conservative: its hit rate is about one fifth of the others'.
- Calibrated on the first two-plus years and deployed on the following weeks, a static threshold's realised error rate is 1.3–2× the 2% target. The fine-tuned CacheVerifier stays closest to the target.
- 100 blind human labels show that the non-thinking DeepSeek judge is too strict. With thinking mode (effort low), binary agreement with the human rises from 0.68 to 0.77 (paired test, p = 0.049). However, neither retraining on these more accurate labels nor giving the verifier the cache entry's earlier message brought the verifier closer to human judgement.

## 1. Data

Real support conversations of two brands from the Kaggle "Customer Support on Twitter" corpus (Axelbrooke, 2017), replayed in real time order:

| Dataset | Requests | Span | Calibration half | Test half |
|---|---|---|---|---|
| AmazonHelp | 60,000 | 2015-06 → 2017-10 | 2015-06 → 2017-10-14 | 2017-10-14 → 2017-10-27 |
| comcastcares | 30,180 | 2014-07 → 2017-12 | 2014-07 → 2017-11-05 | 2017-11-05 → 2017-12-03 |

The corpus is concentrated in its last few weeks, so each test half spans only 2 or 4 weeks. The corpus is licensed CC BY-NC-SA 4.0, and this repository contains no tweet text. Regenerate the data from Kaggle with `scripts/convert_twitter_cs_dataset.py`; the result files hold only indices, similarities and labels.

## 2. Labels

The dataset's own equivalence label means "both customer messages received the identical canned reply". It is too strict: at a static threshold of 0.80 only about 2% of hits count as correct, and every method's hit precision is 1%–2%, which separates nothing. These results are kept as the `same` label set in the result JSON.

The main results use a DeepSeek judge instead. For every nearest-neighbour pair with similarity ≥ 0.6 (46,295 AmazonHelp pairs and 22,004 comcastcares pairs), it is asked whether the cached support reply could be sent unchanged to the new customer: YES / PARTIAL / NO. YES counts as correct; PARTIAL and NO count as wrong. Pairs below 0.6 are not relabelled and count as wrong; 300 of them per dataset are judged as a check on that assumption.

Two judge-label versions were made:

| Version | Model and setting | Used in |
|---|---|---|
| old judge | `deepseek-chat` (served at the time by DeepSeek-V4.1-Flash), non-thinking | Section 4 |
| new judge | `deepseek-flash`, thinking mode, `reasoning_effort` = low | Section 6 |

## 3. Methods and protocol

All methods share one embedding (all-MiniLM-L6-v2, normalised) and one nearest neighbour (top-1, exact search). The cache starts empty and every request is inserted after it is decided. The nearest neighbour therefore does not depend on the method, and every method is replayed on the same trace.

| Method | Hit rule | Default (converted from the source code) |
|---|---|---|
| Redis static threshold (RedisVL SemanticCache) | cosine ≥ τ | `distance_threshold` 0.1, i.e. cosine ≥ 0.90 |
| GPTCache default | faiss squared L2 + `SearchDistanceEvaluation` | `similarity_threshold` 0.8, i.e. cosine ≥ 0.60 (gptcache 0.1.44) |
| GPTCache learned evaluator | official `OnnxModelEvaluation` (GPTCache/albert-duplicate-onnx) scores the two messages | score ≥ 0.8 |
| vCache | per-entry online threshold (this repo's Group B port, with the official two seed observations) | δ = 0.02 |
| CacheVerifier off-the-shelf | cosine ≥ 0.97 hits directly; 0.80–0.97 goes to the cross-encoder (ms-marco-MiniLM-L6-v2) | threshold 0 |
| CacheVerifier fine-tuned | same, verifier fine-tuned on the judge labels of the earliest 70% of the calibration half's gray zone (warmup 10%, 3 epochs, lr 2e-5) | threshold chosen on the remaining 30% |

Two comparisons:

- **Matched error rate.** Each method's parameter is swept, and the best test-half hit rate with test-half error rate ≤ e is reported. This compares the curves; it is not a number obtainable at deployment.
- **Honest calibration.** Each method's parameter is chosen on the calibration half (highest hit rate with error rate ≤ 2%), and the test half is reported. vCache uses δ = 2% directly.

Metrics:

- **Hit rate, wrong-hit rate** (wrong hits / all requests) and hit precision, with bootstrap 95% intervals.
- **Latency.** Every component is timed in one process, interleaved on the same requests, on CPU at batch size 1: embedding, search, cross-encoder, GPTCache evaluator, vCache decision. A miss adds measured DeepSeek latency (median 1.65 s).
- **Cost.** Verification is priced as **self-hosted compute** (measured CPU time × $0.045 per vCPU-hour). An LLM call is $4.98×10⁻⁵, fitted from real bills. "With error cost" counts each wrong answer as 5 LLM calls. vCache checks the new answer against the cached one on every miss, priced as one judge call.
- **Drift.** The test half is cut into 4 time windows, and hit precision is reported per window.

## 4. Results: old judge labels

### 4.1 Hit rate at matched error rate (test half)

| Dataset | e | Static threshold | GPTCache learned | CacheVerifier off-the-shelf | **CacheVerifier fine-tuned** | vCache |
|---|---|---|---|---|---|---|
| AmazonHelp | 1% | 4.00% | 3.66% | 3.12% | **4.46%** | 1.11% |
| AmazonHelp | 2% | 5.55% | 5.38% | 4.27% | **6.14%** | 3.33% |
| AmazonHelp | 3% | 7.27% | 7.20% | 5.69% | **7.69%** | 3.33% |
| comcastcares | 1% | 4.54% | 3.83% | 4.14% | **5.24%** | 1.41% |
| comcastcares | 2% | 6.71% | 6.28% | 5.49% | **7.56%** | 3.93% |
| comcastcares | 3% | 8.61% | 8.39% | 6.81% | **9.28%** | 3.93% |

### 4.2 Honest calibration (target error rate 2%)

| Dataset | Method (parameter) | Hit rate | Wrong-hit rate | Precision | Decision overhead P50 / P95 | End-to-end P50 / P95 | Cost per verification | Cost per 1M requests | With error cost (r=5) |
|---|---|---|---|---|---|---|---|---|---|
| AmazonHelp | Static threshold (0.84) | 8.52% | 3.95% | 0.54 | 21.5 / 66.5 ms | 1660 / 2033 ms | 0 | $45.94 | $55.79 |
| | GPTCache learned (0.43) | 7.06% | 2.92% | 0.59 | 85.3 / 234.0 ms | 1735 / 2157 ms | $9.2×10⁻⁷ | $47.59 | $54.87 |
| | CacheVerifier off-the-shelf | 6.66% | 3.62% | 0.46 | 23.7 / 78.4 ms | 1666 / 2080 ms | $3.3×10⁻⁷ | $46.91 | $55.93 |
| | **CacheVerifier fine-tuned** | 6.81% | **2.33%** | **0.66** | 23.9 / 77.6 ms | 1664 / 2081 ms | $3.3×10⁻⁷ | $46.83 | **$52.65** |
| | vCache (δ=0.02) | 1.11% | 0.42% | 0.62 | 23.6 / 68.1 ms | 1677 / 2086 ms | $4.5×10⁻⁵ | $93.73 | $94.78 |
| comcastcares | Static threshold (0.86) | 8.10% | 2.69% | 0.67 | 31.6 / 52.2 ms | 1666 / 2091 ms | 0 | $46.20 | $52.90 |
| | GPTCache learned (0.88) | 7.23% | 2.34% | 0.68 | 125.7 / 214.7 ms | 1763 / 2167 ms | $1.2×10⁻⁶ | $47.86 | $53.69 |
| | CacheVerifier off-the-shelf | 7.77% | 3.56% | 0.54 | 33.9 / 71.2 ms | 1670 / 2029 ms | $4.1×10⁻⁷ | $46.43 | $55.29 |
| | **CacheVerifier fine-tuned** | **8.29%** | **2.35%** | **0.72** | 33.9 / 71.7 ms | 1672 / 2086 ms | $4.1×10⁻⁷ | **$46.17** | **$52.01** |
| | vCache (δ=0.02) | 1.41% | 0.44% | 0.69 | 33.3 / 54.6 ms | 1682 / 2090 ms | $4.5×10⁻⁵ | $93.95 | $95.05 |

- The calibration half spans more than two years; the test half is the last few weeks, so there is a time shift between them. The static threshold's test-half error rate is twice the target (3.95%), and the GPTCache evaluator and the off-the-shelf verifier also overshoot. Only the fine-tuned CacheVerifier (2.33% / 2.35%) and vCache land near or below the target. The static threshold serves more hits because it makes more errors.
- End-to-end latency is set by the LLM call on a miss: every method's P50 is 1.66–1.78 s. CacheVerifier calls its verifier on only 13%–15% of requests (the gray zone), so its decision overhead P50 is about 2 ms above the static threshold's. The GPTCache evaluator runs on every request and adds 64–94 ms.
- vCache's cost comes almost entirely from pricing its per-miss answer check as a judge call. With vCache's default exact-string comparison that cost is near zero, but what it learns then matches the strict canned-reply label of Section 2.

### 4.3 Default configurations

GPTCache's default (cosine ≥ 0.60) hits 83% / 79% of requests, and 73% / 63% of all requests get a wrong reply. Redis's default (0.90) hits 4.0% / 5.0% at an error rate of 0.94% / 1.14%. vCache (δ = 0.02) has an error rate of 0.42% / 0.44% at a hit rate of 1.1% / 1.4%.

### 4.4 Drift

The gray-zone positive rate (judge labels) moves only slightly within the test half: AmazonHelp 32.3% in the calibration half and 27.8%–29.9% across the four test windows; comcastcares 38.6% and 38.8%–43.6%.

Hit precision in the last test window:

| Method | AmazonHelp | comcastcares |
|---|---|---|
| CacheVerifier fine-tuned | 0.675 | 0.742 |
| Static threshold | 0.538 | 0.708 |
| GPTCache learned | 0.615 | 0.741 |
| vCache | 0.585 | 0.792 |

The test half spans only weeks, so "after drift" here mostly means the gap between calibration and test (the overshoot in 4.2). This split does not cover the strong drift that §5.8 of the paper measured over comcastcares' full 3.5 years.

## 5. Blind human check

### 5.1 Sample and annotation

From pairs with similarity ≥ 0.8, 50 per dataset were drawn, stratified by the old judge's verdict (YES 20, PARTIAL 15, NO 15), shuffled, and labelled blind by one annotator. The annotator did not see the verdict or the similarity. They saw the new customer message, the earlier customer message and the support reply (Chinese translation with the English original expandable), and answered the same question as the judge.

Of the 100 labels, 57 are "can send as-is", 25 "partly", 17 "no" and 1 "can't tell". A text-free label file is in `results/enterprise_audit_human_labels_notext.jsonl` (item, indices, similarity, split, human label, judge verdict).

### 5.2 Old judge vs human

Share of items the human marked "can send as-is" ("can't tell" dropped):

| Judge verdict | AmazonHelp | comcastcares | All |
|---|---|---|---|
| YES | 16/20 = 0.80 | 17/20 = 0.85 | 33/40 = 0.82 |
| PARTIAL | 3/14 = 0.21 | 10/15 = 0.67 | 13/29 = 0.45 |
| NO | 4/15 = 0.27 | 7/15 = 0.47 | 11/30 = 0.37 |

A YES from the judge is mostly reliable. But the judge is too strict: many pairs it called PARTIAL or NO the human would send as-is, especially on comcastcares.

### 5.3 Re-scoring with the human rates

Here each pair's correctness is replaced by the probability that the human says "can send" given the judge's verdict, and the test-half matched-error frontier is recomputed:

| Expected human error ≤ | AmazonHelp static | AmazonHelp CV fine-tuned | comcastcares static | comcastcares CV fine-tuned |
|---|---|---|---|---|
| 1% | 3.15% | **3.35%** | 4.74% | **5.08%** |
| 2% | 5.23% | **5.58%** | 8.61% | **9.11%** |
| 3% | 6.79% | **7.07%** | 11.83% | **12.54%** |
| 5% | 9.91% | **10.41%** | **18.31%** | 16.99% |

The fine-tuned CacheVerifier is still highest at 1%–3%, but its lead shrinks from about +11%–13% to about +5%–7%; at 5% on comcastcares the static threshold is higher. The re-scoring assumes that, within one verdict, the chance of a human "can send" does not depend on the method's score. Each cell holds only 14–20 items (the 95% interval of 0.45 is about 0.28–0.63). Read it as "same direction, uncertain size".

## 6. Thinking-mode judge

### 6.1 Judge settings on the 99 human labels

Same prompt, four settings, compared with the human as YES vs not YES. No prompt was tuned on these labels.

| Setting | YES verdicts | Binary accuracy | Human agrees when judge says YES | Human "can send" judged YES | 3-class kappa | Mean latency | Mean output tokens |
|---|---|---|---|---|---|---|---|
| non-thinking (`deepseek-chat`) | 39 | 0.68 | 0.82 | 0.56 | 0.34 | 2.4 s | 1 |
| thinking, low | 44 | **0.77** | **0.89** | 0.68 | **0.44** | 3.5 s | 206 |
| thinking, high (default) | 48 | **0.77** | 0.85 | **0.72** | 0.39 | 3.9 s | 279 |
| thinking, max | 51 | 0.70 | 0.76 | 0.68 | 0.33 | 4.8 s | 422 |

- Low and high are significantly better than non-thinking. In paired McNemar tests, low fixed 13 items and broke 4 (p = 0.049); high fixed 12 and broke 3 (p = 0.035). Max is not significantly different (p = 0.82).
- Most of the gain is on comcastcares (binary accuracy 0.62 → 0.80 / 0.76).
- Rerunning the non-thinking judge agrees with its own verdicts from one day earlier on only 87% of items: the same model and setting is not fully deterministic.
- The sample is stratified by the old judge's verdicts, so absolute accuracies do not extrapolate. The four settings are compared paired, on the same items.

### 6.2 Relabelling every candidate pair with the new judge

The new labels are more lenient. The YES share rises from 11.1% to 16.9% on AmazonHelp and from 19.6% to 37.5% on comcastcares. In the 300-pair checks below 0.6, YES rises from 7 to 29 (AmazonHelp) and from 25 to 52 (comcastcares). Counting every pair below 0.6 as wrong therefore penalises methods that can hit at low similarity (the GPTCache evaluator, vCache) more under these labels.

One shift looks odd but holds up: at similarity ≥ 0.97 on AmazonHelp, YES falls from 83% to 61%. Nearly all flipped pairs are short follow-ups that carry no context of their own ("ok", "any update?", "I already did that"). The two messages are almost identical, but each support reply depends on its own conversation: it names the customer, refers to an email already sent, or asks for a specific action. Reusing it as-is is wrong. This is a real failure mode of static thresholds on support traffic. These requests are served directly in the ≥ 0.97 band and never reach the verifier.

### 6.3 Full comparison under the new judge labels

**Hit rate at matched error rate** (test half; bold = row maximum):

| Dataset | e | Static threshold | GPTCache learned | CacheVerifier off-the-shelf | CacheVerifier fine-tuned | vCache |
|---|---|---|---|---|---|---|
| AmazonHelp | 1% | 2.58% | 2.22% | 2.43% | **2.60%** | 1.00% |
| AmazonHelp | 2% | 4.91% | 4.30% | 3.79% | **5.37%** | 1.94% |
| AmazonHelp | 3% | 6.35% | 5.98% | 5.28% | **7.12%** | 3.30% |
| AmazonHelp | 5% | 9.27% | 9.15% | 8.22% | **9.78%** | 6.21% |
| comcastcares | 1% | 4.74% | 3.83% | 3.80% | **4.83%** | 1.81% |
| comcastcares | 2% | 7.10% | 6.60% | 5.77% | **7.86%** | 3.38% |
| comcastcares | 3% | 9.23% | 8.75% | 7.42% | **10.26%** | 5.14% |
| comcastcares | 5% | 12.54% | 12.82% | 11.45% | **13.98%** | 8.37% |

**Honest calibration (target 2%):**

| Dataset | Method (parameter) | Hit rate | Wrong-hit rate | Precision | Cost per 1M requests | With error cost (r=5) | Last-window precision |
|---|---|---|---|---|---|---|---|
| AmazonHelp | Static threshold (0.850) | 7.27% | 3.48% | 0.52 | $46.56 | $55.24 | 0.52 |
| | GPTCache learned (0.720) | 6.17% | 3.08% | 0.50 | $48.03 | $55.70 | 0.51 |
| | CacheVerifier off-the-shelf | 6.24% | 3.54% | 0.43 | $47.12 | $55.94 | 0.43 |
| | CacheVerifier fine-tuned | 6.65% | 2.67% | 0.60 | $46.92 | $53.56 | 0.61 |
| | vCache (δ=0.02) | 1.00% | 0.48% | 0.52 | $243.29 | $244.49 | 0.55 |
| comcastcares | Static threshold (0.855) | 8.61% | 2.62% | 0.70 | $45.95 | $52.48 | 0.69 |
| | GPTCache learned (0.870) | 7.49% | 2.35% | 0.69 | $47.73 | $53.59 | 0.70 |
| | CacheVerifier off-the-shelf | 8.30% | 3.38% | 0.59 | $46.16 | $54.58 | 0.60 |
| | CacheVerifier fine-tuned | 8.61% | 2.23% | 0.74 | $46.01 | $51.57 | 0.71 |
| | vCache (δ=0.02) | 1.81% | 0.47% | 0.74 | $242.14 | $243.31 | 0.77 |

Latencies are as in 4.2 (same components, same call rates). vCache's answer check is priced here as a thinking-mode judge call ($2.0×10⁻⁴ each), hence its higher cost.

**Default configurations:**

| Dataset | Method (default) | Hit rate | Wrong-hit rate | Precision |
|---|---|---|---|---|
| AmazonHelp | Redis default (0.90) | 4.00% | 1.50% | 0.62 |
| | GPTCache default (0.60) | 83.12% | 68.51% | 0.18 |
| | GPTCache learned (0.80) | 5.50% | 2.66% | 0.52 |
| | CacheVerifier off-the-shelf (0.00) | 5.23% | 2.91% | 0.44 |
| | CacheVerifier fine-tuned (0.00) | 5.82% | 2.25% | 0.61 |
| | vCache (δ=0.02) | 1.00% | 0.48% | 0.52 |
| comcastcares | Redis default (0.90) | 5.01% | 1.07% | 0.79 |
| | GPTCache default (0.60) | 78.72% | 48.93% | 0.38 |
| | GPTCache learned (0.80) | 8.51% | 2.84% | 0.67 |
| | CacheVerifier off-the-shelf (0.00) | 3.04% | 0.62% | 0.80 |
| | CacheVerifier fine-tuned (0.00) | 10.19% | 2.85% | 0.72 |
| | vCache (δ=0.02) | 1.81% | 0.47% | 0.74 |

The fine-tuned CacheVerifier uses threshold 0 here: it is trained with binary cross-entropy, so a score of 0 means the model estimates a 50% chance the reply is reusable. This default is loose for it; holding a low error rate still requires choosing the threshold on the calibration half.

### 6.4 Do more accurate labels train a verifier closer to humans?

Under re-scoring with the human rates for the new judge's verdicts, at a 2% error rate:

| | AmazonHelp | comcastcares |
|---|---|---|
| Trained on old labels | 5.21% | 8.48% |
| Trained on new labels | 5.22% | 8.57% |
| Static threshold | 4.91% | 7.58% |

The two models are almost the same.

AUC against the human labels, on the 52 test-half audit items in the gray zone (none used in training):

| Model | AUC (95% CI) |
|---|---|
| Off-the-shelf | 0.46 |
| Trained on old labels | 0.66 (0.51–0.80) |
| Trained on new labels | 0.59 (0.43–0.75) |

New minus old is −0.07 (95% interval −0.16 to +0.02). More accurate judge labels did not train a verifier that agrees better with the human.

## 7. Other verifier changes

### 7.1 Training on the human labels directly

The human labels are also the only ground truth, so training and evaluation never share an item. The 35 gray-zone items in the calibration half are for training; the 52 gray-zone items in the test half are for evaluation (44% human "can send").

| Model | Training data | AUC vs human (52 items, 95% CI) |
|---|---|---|
| Off-the-shelf | none | 0.46 (0.30–0.62) |
| Human labels only | 35 items | 0.45 (0.28–0.61) |
| Old judge labels | 1,579 / 1,038 pairs | 0.66 (0.51–0.80) |
| Old judge + human | same, 10 / 14 labels replaced by human ones, plus the other human items | 0.67 (0.51–0.81) |
| Old judge verdict itself (YES > PARTIAL > NO) | — | 0.65 |

35 human labels are too few to train on alone, and mixing them into the judge labels shows no effect. A verifier trained on judge labels matches the judge itself.

### 7.2 Giving the verifier the cache entry's earlier message

The judge and the human both saw the earlier customer message; the verifier sees only (new message, cached reply). The earlier message was added to the verifier input in the two representations from §5.22 of the paper: raw concatenation, and a word-level diff summary. All three inputs were retrained in one environment (T4 GPU, torch 2.2.2 + sentence-transformers 3.4.1), with the same training set and recipe.

| Training labels | Input | AUC vs human (52 items) | vs message only |
|---|---|---|---|
| old | message only | 0.687 (0.52–0.82) | — |
| old | concatenation | 0.645 (0.49–0.80) | −0.04 (−0.18 to +0.09) |
| old | diff summary | 0.634 (0.48–0.77) | −0.05 (−0.17 to +0.06) |
| new | message only | 0.634 (0.46–0.77) | — |
| new | concatenation | 0.573 (0.41–0.72) | −0.06 (−0.19 to +0.06) |
| new | diff summary | 0.606 (0.44–0.75) | −0.03 (−0.16 to +0.09) |

**Hit rate at matched error rate** (test half, error measured with each model's own training labels; bold = column maximum within each group):

| Labels | Dataset | Method | ≤ 1% | ≤ 2% | ≤ 3% | ≤ 5% |
|---|---|---|---|---|---|---|
| old | AmazonHelp | static threshold | 4.00% | 5.55% | 7.27% | 9.27% |
| old | AmazonHelp | message only | 4.48% | 6.15% | 7.74% | 10.30% |
| old | AmazonHelp | concatenation | 4.60% | **6.35%** | **7.87%** | **10.40%** |
| old | AmazonHelp | diff summary | **4.63%** | 6.31% | 7.85% | 10.32% |
| old | comcastcares | static threshold | 4.54% | 6.71% | 8.61% | 11.83% |
| old | comcastcares | message only | **5.29%** | **7.71%** | 9.32% | 12.25% |
| old | comcastcares | concatenation | 5.18% | 7.51% | **9.36%** | 12.31% |
| old | comcastcares | diff summary | 4.90% | 7.42% | 9.34% | **12.37%** |
| new | AmazonHelp | static threshold | 2.58% | 4.91% | 6.35% | 9.27% |
| new | AmazonHelp | message only | 2.54% | 5.38% | 7.10% | 9.87% |
| new | AmazonHelp | concatenation | **2.83%** | 5.32% | **7.17%** | **9.97%** |
| new | AmazonHelp | diff summary | 2.81% | **5.49%** | 7.10% | 9.82% |
| new | comcastcares | static threshold | 4.74% | 7.10% | 9.23% | 12.54% |
| new | comcastcares | message only | 4.94% | **8.16%** | **10.54%** | **14.08%** |
| new | comcastcares | concatenation | **5.19%** | 8.07% | 10.23% | 13.90% |
| new | comcastcares | diff summary | 4.81% | 7.66% | 10.27% | 13.89% |

Adding the earlier message brought no visible improvement. The AUCs are slightly lower but within their intervals, and the frontiers nearly coincide. The same model (message only, old labels) scores AUC 0.657 trained on CPU and 0.687 trained on GPU, so the training environment alone moves AUC by about 0.03. 52 human labels can resolve only AUC differences of roughly 0.1 or more. The remaining gap more likely comes from the whole conversation, which neither single message contains and which this data does not have.

## 8. Conclusions

1. On real customer-support traffic, compared at a matched error rate, the fine-tuned CacheVerifier is the best of the four approaches. At strict error rates it serves about 6%–15% more hits (relative) than a static threshold, and under time shift it stays closest to its target error rate. This is robust to the label version, to recalibration against human labels, and to the verifier input format.
2. The off-the-shelf CacheVerifier is worse than a static threshold; in-domain fine-tuning is required.
3. The absolute gain on this traffic is small. At a 2% error rate every method serves only 5%–9% of requests from the cache, because support replies mostly depend on the specific conversation.
4. GPTCache's default configuration is unusable on this data, and its learned evaluator matches a static threshold at a higher latency. vCache keeps its error rate at about one fifth of the others' hit rate.
5. The LLM judge itself needs human calibration: the non-thinking judge is too strict, and thinking mode (low) is closer to the human. On this data, however, neither more accurate labels nor more input made the verifier better. Going further needs data with whole conversations, or many more human labels.

## 9. Limitations

- The main labels come from an LLM judge. The human check is one annotator, 100 items, and covers only similarity ≥ 0.8.
- Every pair below similarity 0.6 counts as wrong, which undercounts methods that can hit at low similarity.
- Two English support accounts only; each test half spans weeks.
- One training run and one seed per model; three vCache seeds per δ.
- Exact nearest-neighbour search, not hnswlib. Latency was measured on one desktop CPU, and its P95 is affected by background load.
- Cost uses self-hosted compute at an assumed vCPU price; the error cost r = 5 is an assumption.
- The provider changed the model behind the judge's name (Section 2), so reruns will differ somewhat.

## 10. Reproducing

First generate the data from Kaggle (`scripts/convert_twitter_cs_dataset.py`; see the README's Third-party data section). Then run, in order:

| Step | Script | Notes |
|---|---|---|
| 1 | `enterprise_compare_build.py` | embeddings and the shared nearest-neighbour trace |
| 2 | `enterprise_compare_judge.py` | judge relabelling (needs `DEEPSEEK_API_KEY`; `JUDGE_TAG=judge_thinklow` for the thinking-mode version; about 68k calls) |
| 3 | `enterprise_compare_gptcache.py`, `enterprise_compare_gptcache_gpu.py` | GPTCache evaluator scores (separate gptcache environment; the GPU script only scores, the CPU script checks those scores and measures latency) |
| 4 | `enterprise_compare_verifier.py` | CacheVerifier off-the-shelf scores and fine-tuning (`JUDGE_TAG` as above) |
| 5 | `enterprise_compare_latency.py` | interleaved latency benchmark |
| 6 | `enterprise_compare_analyze.py` | the metrics (`JUDGE_TAG` as above) |
| 7 | `enterprise_compare_audit_sample.py` | regenerates the 100 audit items, with text, locally only |
| 8 | `enterprise_compare_human_reweight.py`, `enterprise_compare_human_finetune.py`, `enterprise_compare_judge_thinking.py`, `enterprise_compare_thinklow_eval.py` | Sections 5–7 |
| 9 | `enterprise_compare_qcached_gpu.py`, `enterprise_compare_qcached.py` | Section 7.2 |

Result files:
- `results/enterprise_comparison_{twitter_amazon,twitter_comcast}[_judge_thinklow].json`
- `results/enterprise_comparison_assumptions.json`
- `results/enterprise_comparison_human_reweight.json`
- `results/enterprise_human_finetune.json`
- `results/enterprise_judge_thinking{,_raw}.json`
- `results/enterprise_thinklow_eval.json`
- `results/enterprise_qcached_gpu.json`
- `results/enterprise_audit_key.json`
- `results/enterprise_audit_human_labels_notext.jsonl`
