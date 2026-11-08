# Results index

Every file in `results/`, grouped by the paper section that reports it. Section numbers refer to the full technical report ([`PAPER_EN.md`](../PAPER_EN.md)) unless another document is named; the sister papers B, C and D reuse the same files for the sections they take over (B §4.1–4.4 = report §5.13, §5.16, §5.17; C §4.1–4.2 = §5.18–5.19; D §4.1–4.6 = §5.20–5.25).

`*` is a wildcard. Dataset suffixes: `lmarena`, `quora`, `search_queries` / `searchqueries` / `sq` (SearchQueries), `twitter_amazon` / `amazon` and `twitter_comcast` / `comcast` (Twitter customer support).

**SearchQueries files without `corrected` in the name are superseded.** They predate the 2026-08-09 erratum (placeholder answers in the converted dataset) and are kept only so the erratum can be checked. Use the `*_corrected*` / `sq_corrected*` versions.

Most files are written by a script of the same or a similar name in [`scripts/`](../scripts/). Fine-tuned verifier checkpoints are on Hugging Face, not here: see [`PRETRAINED_MODELS.md`](PRETRAINED_MODELS.md).

## Main experiment groups (§5.1–5.6)

| Files | What | Section |
|---|---|---|
| `*_groupA.json`, `lmarena_15k_groupA.json` | Group A, static similarity threshold (`15k` = first 15,000 LmArena requests) | 5.1 |
| `*_groupB.json`, `*_groupB_seedfix.json` | Group B, vCache-style adaptive threshold (`seedfix` = after the seed-observation fix) | 5.2 |
| `*_groupC.json` | Group C, oracle verifier (ceiling) | 5.3 |
| `*_groupD_cross_encoder.json`, `search_queries_corrected_groupD.json`, `*_groupD_finetuned_matched.json`, `threshold_calibration_*.json` | Group D, off-the-shelf verifier; `finetuned_matched` = Group E checkpoint on the Group D grid; `threshold_calibration_` = honest calibration (threshold chosen on a calibration half) | 5.4 |
| `*_groupE_finetuned.json`, `*_groupE_honest_calibration*.json` | Group E, domain-fine-tuned verifier, and its honest calibration | 5.6 |
| `groupD_vs_groupA_fair_frontier*.json`, `grid_search_vs_dense_groupA_check.json` | Same-request comparison against Group A (2026-09-24 erratum) | erratum, 5.4 |
| `oracle_judge_latency_deepseek.json` | Judge latency for synchronous verification | 5.5 |
| `finetune_verifier_experiment*.json`, `finetune_verifier_comparison_*.json`, `quora_finetune_retrain_comparison.json`, `auc_comparison_off_the_shelf_vs_finetuned.png` | Fine-tuning runs and off-the-shelf vs fine-tuned AUC (`*.examples.json` = per-pair scores) | 5.6 |
| `*_pareto*.png`, `direction1_sq_honest_vs_groupA.png` | Hit-rate vs error-rate frontiers | 5.1, 5.6 |
| `logs/*.log` | Group E calibration and retraining logs | 5.6 |

## Deployment checks and ablations (§5.7–5.12)

| Files | What | Section |
|---|---|---|
| `noise_sweep_*`, `coldstart_sweep_*`, `drift_experiment*`, `sq_corrected_ablations/` | Label noise, cold start and drift sweeps | 5.7 |
| `twitter_*_group*.json`, `noise_twitter_*`, `coldstart_twitter_*`, `drift_twitter_*`, `twitter_case_study_*.png` | Real customer-support traffic case study | 5.8 |
| `positive_rate_monitor_*.json` | Prototype positive-rate monitor | 5.9 |
| `bucketing_ablation_*.json`, `axis_problem_experiment*.json` | Action-verb bucketing (refuted) and the axis-problem test | 5.10 |
| `*_groupD_tauhigh_grid.json`, `denoised_tau_high.json` | Tuning τ_high (withdrawn after label denoising, 2026-09-27 erratum) | 5.11 |
| `*_reranker_*_auc.json` | Larger off-the-shelf rerankers | 5.12 |

## Risk control and cost (§5.13–5.17; Paper B §4.1–4.5)

| Files | What | Section |
|---|---|---|
| `crc_pilot_*`, `crc_step0_*`, `crc_step2_*`, `crc_step3_*`, `crc_protocol_t_*`, `crc_v2_rq1_frontier*`, `gray_zone_risk_curve_check.json` | Conformal Risk Control on the gray zone: pilots, label checks, repeated splits, time-ordered Protocol T | 5.13 (B §4.1) |
| `rewrite_vs_reject_*.json` | Rewrite instead of reject (negative) | 5.14 |
| `*_cascade_k2_*.json`, `topk_ceiling/` | Top-K candidate cascade | 5.15 |
| `crc_closed_loop_self_selection_*`, `crc_closed_loop_write_probability_sweep_*` | Self-selection closed loop | 5.16 (B §4.2) |
| `cost_sensitive_reanalysis*.json`, `denoised_cost_sensitive.json`, `denoised_cost_curve_for_site.json` | Cost-ratio reanalysis (`_fair` = after the 2026-09-24 erratum, `denoised_` = after 2026-09-27) | 5.17 (B §4.3) |
| `crc_closed_loop_calibration_sources_*` | Which labels online recalibration can use | B §4.4 |
| `audited_aci_*.json`, `iaci_bound_counterexample.json` | Audited ACI bound checks and the I-ACI counterexample | B §4.5, [`AUDITED_ACI_PROOF_DRAFT.md`](../AUDITED_ACI_PROOF_DRAFT.md) |

## Adversarial robustness and verifier inputs (§5.18–5.23; Papers C and D)

| Files | What | Section |
|---|---|---|
| `llm_redteam_*.json`, `openai_redteam_generated*.json`, `generator_c_*.json`, `finetune_adversarial_comparison.json` | LLM red-teaming and adversarial training | 5.18–5.19 (C §4.1–4.2) |
| `selective_deferral_*`, `oracle_judge_accuracy_*`, `human_adjudication_worksheet.*`, `lmarena_fulltext_adjudication.*` | Selective prediction / defer-to-judge (negative), judge accuracy and the manual adjudication of judge-vs-benchmark disagreements | 5.20 (D §4.1) |
| `finetune_verifier_qcached*`, `qcached_*redteam*`, `qcached_diff_answer_attribution*`, `qcached_joint_ablation.json`, `qcached_natural_truncation_check.json` | Adding the cached query to the verifier input (`loo` = leave-one-attack-type-out) | 5.22 (D §4.3) |
| `qcached_nli_*.json` | Why entity_swap still lags: training-trajectory sweep and post-hoc fusion | 5.23 (D §4.4) |

## Online adaptation and joint decisions (§5.21, §5.24–5.25; Paper D)

| Files | What | Section |
|---|---|---|
| `joint_grayzone_*.json`, `topk_cascade_*_analysis.json`, `aci_online_threshold_cascade*.json`, `gray_zone_high_precision_compare.json`, `gray_zone_length_sliced_threshold.json` | Joint decisions with the similarity signal; causal and forward-chain validation | 5.24 (D §4.5) |
| `aci_crc_violation_fix.json`, `aci_real_production_drift_*.json`, `aci_theoretical_bound_verification.json`, `aci_window_sweep_causal_test.*` | ACI on drifting streams and its finite-sample bound | 5.21, 5.25 (D §4.2, §4.6) |

## Label audits (§5.26 and errata)

| Files | What | Section |
|---|---|---|
| `direct_hit_annotation_*`, `direct_hit_error_taxonomy.json`, `gray_zone_annotation_*`, `gray_zone_label_check_llm.json`, `denoised_rescore.json` | Blind human check of benchmark labels and the denoised rescoring | 5.26, 2026-09-27 erratum |

## Real customer-support traffic ([`ENTERPRISE_COMPARISON.md`](../ENTERPRISE_COMPARISON.md))

| Files | What | Section |
|---|---|---|
| `enterprise_comparison_*.json`, `enterprise_qcached_gpu.json` | Four caches compared at matched error rates | §2–4 |
| `enterprise_audit_*`, `enterprise_judge_thinking*.json`, `enterprise_thinklow_eval.json`, `enterprise_comparison_human_reweight.json`, `enterprise_human_finetune.json` | 100 blind human labels, judge settings, reweighting and training on the human labels (labels file carries no tweet text) | §5–7 |
| `enterprise_guarantee_feasibility.json`, `enterprise_allhits_guarantee*.json` | Risk guarantees on the stream and extending them to every hit | §8 |

## Exploratory analyses not reported in a paper section

Kept for reproducibility of negative or preliminary results mentioned in passing (for example Paper B §7 item 11).

| Files | What |
|---|---|
| `history_decay_*.json`, `loosening_failure_*.json` | Stateful closed-loop effects: how long history matters, with and without TTL eviction, and whether sliding-window CRC fails after a large loosening (it does not) |
| `gap_closed_loop_s2_*.json` | Cache-local gap signal (top-1 minus top-2 similarity) in the closed loop |
| `cache_sensitivity_*.json` | How much the closed-loop gray-zone distribution shifts with the threshold, and whether it depends on history |
| `dc_headroom_pilot.json`, `entry_error_clustering.json` | Headroom between the deployed verifier and the oracle; whether errors cluster on cache entries |
| `searchqueries_gray_zone_needed_indices.json` | Indices of SearchQueries gray-zone pairs that needed generated answers (data preparation) |
