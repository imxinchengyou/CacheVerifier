"""Follow-up to the q_cached ablation (finetune_verifier_qcached_train_eval.py):
that experiment found naive text-concatenation of the cached query makes
false-accept rate on the Section 5.18/5.19 adversarial red-team set WORSE
(97.7% vs 86.6% for the text-only control), not better. The likely mechanism
is a lexical-overlap shortcut: in NATURAL gray-zone training data, "current
query text looks almost identical to cached query text" correlates strongly
with label=True (real paraphrases), so the model learns to treat that
textual similarity itself as evidence of safety -- exactly backwards for the
adversarial set, which is manufactured to have near-identical query_a/query_b
text with a genuinely wrong answer.

This script tests whether mixing in Section 5.19's existing adversarial
training pool (results/llm_redteam_train_pool.json, 223 triples that cleared
tau_low, already used once WITHOUT q_cached to produce the 53.6% FA rate in
Section 5.19) breaks that shortcut when q_cached is also present. Each
adversarial triple becomes two rows, exactly as build_adversarial_training_
stash.py already does for the text-only case:
  - (query_a, cached_query=query_b, answer_b) -> label False  (the hard negative)
  - (query_b, cached_query=query_b, answer_b) -> label True   (contrasting positive)

Two arms are trained on IDENTICAL combined data (natural qcached-stash train
rows + these adversarial rows), differing only in text_a construction, same
as the parent ablation:
  - "text_only_adv": text_a = query               (reproduces Sec 5.19's recipe, sanity check)
  - "qcached_adv":   text_a = query + cached_query (this follow-up's actual test)

Held-out evaluation is unchanged from both prior experiments: natural AUC on
the untouched qcached-stash test split, and false-accept rate on the 306-
sample red-team held-out set (results/llm_redteam_results.json) via
qcached_redteam_eval.py pointed at this script's output checkpoints.

Usage:
    python scripts/finetune_verifier_qcached_adversarial_train_eval.py

--input-mode diff (2026-09-05 follow-up): even with the adversarial rows
mixed in, "concat" mode's false-accept rate (69.0%) stayed WORSE than the
text-only control (53.6%) -- the 223 adversarial rows (3.8% of training)
weren't enough to override the lexical-overlap shortcut natural data teaches.
--input-mode diff sidesteps the shortcut at its source: instead of raw
cached_query text, text_a gets a word-level diff summary (via difflib) of
what changed between cached_query and the current query, so raw surface
overlap is never exposed to the tokenizer in the first place.

Usage:
    python scripts/finetune_verifier_qcached_adversarial_train_eval.py --input-mode diff
"""

import argparse
import difflib
import json
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

QCACHED_TEMPLATE = "{query}\n[cached_query] {cached_query}"
DIFF_TEMPLATE = "{query}\n[diff vs cached_query] {diff}"
TAU_LOW = 0.80


def word_diff_summary(cached_query: str, current_query: str, max_words: int = 12) -> str:
    """Word-level diff describing how cached_query differs from current_query,
    e.g. 'removed: pause; added: cancel', or 'identical'. Pure difflib."""
    a, b = cached_query.split(), current_query.split()
    sm = difflib.SequenceMatcher(None, a, b)
    removed, added = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("delete", "replace"):
            removed.extend(a[i1:i2])
        if tag in ("insert", "replace"):
            added.extend(b[j1:j2])
    if not removed and not added:
        return "identical"
    parts = []
    if removed:
        parts.append("removed: " + " ".join(removed[:max_words]))
    if added:
        parts.append("added: " + " ".join(added[:max_words]))
    return "; ".join(parts)


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(scores))
    pos_ranks = ranks[labels == 1]
    n_pos, n_neg = (labels == 1).sum(), (labels == 0).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((pos_ranks.sum() - n_pos * (n_pos - 1) / 2) / (n_pos * n_neg))


def evaluate_scores(scores: np.ndarray, labels: np.ndarray) -> dict:
    percentiles = np.percentile(scores, np.arange(10, 100, 10))
    curve = []
    for t in percentiles:
        approved = scores >= t
        n_approved, n_rejected = approved.sum(), (~approved).sum()
        false_approve = float((labels[approved] == 0).mean()) if n_approved else 0.0
        false_reject = float((labels[~approved] == 1).mean()) if n_rejected else 0.0
        curve.append({
            "threshold": float(t),
            "approve_rate": float(n_approved / len(scores)),
            "false_approve_rate": false_approve,
            "false_reject_rate": false_reject,
        })
    return {
        "auc": roc_auc(scores, labels),
        "score_min": float(scores.min()),
        "score_mean": float(scores.mean()),
        "score_max": float(scores.max()),
        "curve": curve,
    }


def build_adversarial_rows(pool_path: str, exclude_category: str | None = None):
    """(idx, query, cached_query, answer, label) rows from the existing
    Sec 5.19 adversarial training pool -- same filter/expansion as
    build_adversarial_training_stash.py, with cached_query added.

    `exclude_category`, added 2026-09-07 for the leave-one-axis-out check
    promised in social_media/reddit/post_12's reply to Such-Process5697:
    drops every triple of that category BEFORE expansion, so the resulting
    model has zero direct exposure (neither negative nor contrasting
    positive rows) to that axis, while all other axes stay exactly as before."""
    pool = json.loads(Path(pool_path).read_text(encoding="utf-8"))
    qualifying = [t for t in pool["all_triples"] if t.get("similarity", 0.0) >= TAU_LOW]
    if exclude_category is not None:
        before = len(qualifying)
        qualifying = [t for t in qualifying if t["category"] != exclude_category]
        print(f"Adversarial training pool: {before} cleared tau_low={TAU_LOW}, "
              f"{before - len(qualifying)} triples of category={exclude_category!r} excluded, "
              f"{len(qualifying)} remain")
    else:
        print(f"Adversarial training pool: {len(qualifying)}/{len(pool['all_triples'])} triples cleared tau_low={TAU_LOW}")
    rows = []
    for i, t in enumerate(qualifying):
        rows.append((f"adv_neg_{i}", t["query_a"], t["query_b"], t["answer_b"], False))
        rows.append((f"adv_pos_{i}", t["query_b"], t["query_b"], t["answer_b"], True))
    return rows


def train_one_arm(arm_name, texts_a, texts_b, train_labels, base_model, epochs, batch_size,
                   logging_steps, model_out, device):
    from datasets import Dataset
    from sentence_transformers import CrossEncoder
    from sentence_transformers.cross_encoder import CrossEncoderTrainer, CrossEncoderTrainingArguments
    from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
    from transformers import TrainerCallback

    def log(msg: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}][{arm_name}] {msg}", flush=True)

    class VerboseLoggerCallback(TrainerCallback):
        def __init__(self):
            self.t_start = None

        def on_train_begin(self, args, state, control, **kwargs):
            self.t_start = time.time()
            log(f"[train] begin: max_steps={state.max_steps} epochs={args.num_train_epochs} "
                f"batch_size={args.per_device_train_batch_size}")

        def on_log(self, args, state, control, logs=None, **kwargs):
            logs = logs or {}
            if "loss" not in logs:
                return
            elapsed = time.time() - self.t_start
            step, total = state.global_step, state.max_steps
            rate = step / elapsed if elapsed > 0 else 0
            eta = (total - step) / rate if rate > 0 else float("inf")
            log(f"[train] step {step:>4}/{total} ({100 * step / total:5.1f}%)  "
                f"loss={logs['loss']:.4f}  {rate:.3f} step/s  elapsed={elapsed / 60:.1f}m  eta={eta / 60:.1f}m")

        def on_train_end(self, args, state, control, **kwargs):
            log(f"[train] end: {state.global_step} steps in {(time.time() - self.t_start) / 60:.1f}m")

    log(f"Loading fresh copy of {base_model!r} to fine-tune on {device}...")
    t0 = time.time()
    # num_labels=1 + ignore_mismatched_sizes: a no-op for models already saved with a
    # single-output head (e.g. ms-marco-MiniLM-L6-v2), but required for models with a
    # different head shape (e.g. 3-way NLI checkpoints) -- reinitializes just the final
    # classification layer to a single output while keeping the pretrained encoder body,
    # added 2026-09-07 for the NLI-pretrained-base experiment (RESEARCH_PROPOSAL.md
    # direction 16 "再续之十四").
    tuned = CrossEncoder(base_model, num_labels=1, automodel_args={"ignore_mismatched_sizes": True}, device=device)
    log(f"  model loaded in {time.time() - t0:.1f}s")

    train_dataset = Dataset.from_dict({"query": texts_a, "response": texts_b, "label": train_labels})
    n_pos = sum(1 for lbl in train_labels if lbl == 1.0)
    log(f"Built training dataset: {len(train_dataset)} rows ({n_pos} correct / {len(train_labels) - n_pos} incorrect)")

    loss = BinaryCrossEntropyLoss(tuned)
    steps_per_epoch = -(-len(train_dataset) // batch_size)
    total_steps = steps_per_epoch * epochs
    log(f"Training config: epochs={epochs} batch_size={batch_size} steps_per_epoch={steps_per_epoch} "
        f"total_steps={total_steps}")

    training_args = CrossEncoderTrainingArguments(
        output_dir=str(Path(model_out).with_name(Path(model_out).name + "_checkpoints")),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        use_cpu=(device == "cpu"),
        report_to="none",
        save_strategy="no",
        logging_strategy="steps",
        logging_steps=logging_steps,
        logging_first_step=True,
        disable_tqdm=True,
    )
    trainer = CrossEncoderTrainer(model=tuned, args=training_args, train_dataset=train_dataset, loss=loss)
    trainer.add_callback(VerboseLoggerCallback())

    log(f"Fine-tuning on {len(texts_a)} examples for {epochs} epoch(s)...")
    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0
    log(f"  fine-tuning done in {train_time:.1f}s ({train_time / max(1, len(texts_a)):.3f}s/example)")

    Path(model_out).parent.mkdir(parents=True, exist_ok=True)
    tuned.save(model_out)
    log(f"Saved fine-tuned model to {model_out}")
    return tuned, train_time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--natural-stash", default="results/finetune_verifier_qcached_experiment.examples.json")
    parser.add_argument("--adversarial-pool", default="results/llm_redteam_train_pool.json")
    parser.add_argument("--base-model", default="cross-encoder/ms-marco-MiniLM-L6-v2")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--model-out-prefix", default="results/finetuned_verifier_model_lmarena_qcached_adv_ablation")
    parser.add_argument("--output", default="results/finetune_verifier_qcached_adversarial_comparison.json")
    parser.add_argument("--input-mode", choices=["concat", "diff"], default="concat",
                        help="'concat' = raw cached_query text (2026-09-05 result: still worse than text-only "
                             "even with adversarial rows mixed in, 69.0% vs 53.6%). 'diff' = word-level diff "
                             "summary instead of raw text (see module docstring).")
    parser.add_argument("--target-adv-fraction", type=float, default=None,
                        help="If set, downsample natural_train (keeping the front, i.e. earliest stream "
                             "positions) so the adversarial rows make up exactly this fraction of the combined "
                             "training set, instead of using the full natural_train and whatever fraction falls "
                             "out. Added 2026-09-06: SearchQueries' much larger natural train set (60,395 rows) "
                             "diluted the same 446 adversarial rows to 0.7% instead of LmArena/Quora's 3.8%, and "
                             "RESEARCH_PROPOSAL.md direction 16 flagged this as the likely reason its improvement "
                             "was much weaker (61.8%->58.5% vs LmArena's 53.6%->12.4%) -- this flag tests that "
                             "hypothesis directly by matching the ratio instead of generating new adversarial data.")
    parser.add_argument("--truncate-query-words", type=int, default=None,
                        help="If set, truncate `query` and `cached_query` (NOT `answer`) in the natural "
                             "train/test rows to this many words (front of the string, plain str.split()), "
                             "leaving the shared adversarial rows untouched. Added 2026-09-06: RESEARCH_PROPOSAL.md "
                             "direction 16 found query length correlates with the diff+adversarial-training "
                             "improvement across datasets (LmArena 38.6 words/41.2pp, Quora 9.9 words/9.2pp, "
                             "SearchQueries 5.4 words/2.6pp) but with n=3 across datasets, other dataset "
                             "properties (answer length, domain) are confounded with it. This isolates query "
                             "length as a single controlled variable within LmArena alone: truncate LmArena's "
                             "naturally long (~38.6-word) queries down to SearchQueries' ~5-word average and see "
                             "if the improvement collapses toward SearchQueries' weak result.")
    parser.add_argument("--truncate-answer-words", type=int, default=None,
                        help="If set, truncate `answer` (front of the string, plain str.split()) in the natural "
                             "train/test rows to this many words, leaving the shared adversarial rows untouched. "
                             "Added 2026-09-06: the --truncate-query-words=5 experiment found truncating LmArena's "
                             "query to SearchQueries' length only partially collapsed the improvement (41.2pp -> "
                             "27.4pp, still ~10x SearchQueries' 2.6pp) while answer length stayed untouched "
                             "(LmArena ~279 words vs SearchQueries' ~50 words) -- this isolates answer length as "
                             "the next candidate variable, combined with query truncation.")
    parser.add_argument("--truncate-answer-mode", choices=["front", "back"], default="front",
                        help="Which end of `answer` to keep when --truncate-answer-words is set. 'front' (default, "
                             "matches the original --truncate-query-words behavior) keeps the first N words. "
                             "'back' keeps the LAST N words instead. Added 2026-09-06: the front-truncation run "
                             "found the improvement collapsed further (13.7pp, vs 27.4pp with query-only "
                             "truncation) but didn't fully match SearchQueries' 2.6pp -- this checks whether that "
                             "residual gap is an artifact of WHICH part of the answer got kept (LmArena answers "
                             "may put their conclusion/final answer at the end, so front-truncation could be "
                             "discarding the most decision-relevant content) rather than pure length.")
    parser.add_argument("--exclude-category", default=None,
                        choices=["negation", "action_verb", "direction", "entity_swap", "quantity_swap"],
                        help="Leave-one-axis-out check (social_media/reddit/post_12 reply to Such-Process5697): "
                             "drop every adversarial training triple of this category before expansion, so the "
                             "resulting model has zero direct exposure to that axis. Evaluate the held-out axis's "
                             "score with qcached_redteam_eval.py afterward and compare to the in-sample number.")
    args = parser.parse_args()
    qcached_arm_name = "qcached_adv" if args.input_mode == "concat" else f"qcached_adv_{args.input_mode}"

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def log(msg: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    log(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    natural = json.loads(Path(args.natural_stash).read_text(encoding="utf-8"))
    natural_train, test_rows = natural["train"], natural["test"]
    log(f"Natural stash: {len(natural_train)} train, {len(test_rows)} test (test unchanged, natural-only)")

    if args.truncate_query_words is not None:
        n = args.truncate_query_words

        def truncate_query_row(row):
            idx, q, cq, a, label = row
            return (idx, " ".join(q.split()[:n]), " ".join(cq.split()[:n]), a, label)

        before_qw = sum(len(q.split()) for _, q, _, _, _ in natural_train) / len(natural_train)
        natural_train = [truncate_query_row(r) for r in natural_train]
        test_rows = [truncate_query_row(r) for r in test_rows]
        after_qw = sum(len(q.split()) for _, q, _, _, _ in natural_train) / len(natural_train)
        log(f"--truncate-query-words={n}: truncated query/cached_query (answer untouched) in natural "
            f"train+test rows; avg query words {before_qw:.2f} -> {after_qw:.2f}")

    if args.truncate_answer_words is not None:
        n = args.truncate_answer_words
        keep_back = args.truncate_answer_mode == "back"

        def truncate_answer_row(row):
            idx, q, cq, a, label = row
            words = a.split()
            kept = words[-n:] if keep_back else words[:n]
            return (idx, q, cq, " ".join(kept), label)

        before_aw = sum(len(a.split()) for _, _, _, a, _ in natural_train) / len(natural_train)
        natural_train = [truncate_answer_row(r) for r in natural_train]
        test_rows = [truncate_answer_row(r) for r in test_rows]
        after_aw = sum(len(a.split()) for _, _, _, a, _ in natural_train) / len(natural_train)
        log(f"--truncate-answer-words={n} (mode={args.truncate_answer_mode}): truncated answer "
            f"(query/cached_query untouched by this flag) in natural train+test rows; "
            f"avg answer words {before_aw:.2f} -> {after_aw:.2f}")

    adversarial_rows = build_adversarial_rows(args.adversarial_pool, exclude_category=args.exclude_category)
    log(f"Adversarial training rows: {len(adversarial_rows)}")

    if args.target_adv_fraction is not None:
        n_adv = len(adversarial_rows)
        natural_cap = round(n_adv * (1.0 / args.target_adv_fraction - 1.0))
        log(f"--target-adv-fraction={args.target_adv_fraction:.1%}: downsampling natural_train from "
            f"{len(natural_train)} to {natural_cap} rows (front of stream order) to hit this ratio "
            f"with the existing {n_adv} adversarial rows")
        natural_train = natural_train[:natural_cap]

    train_rows = natural_train + adversarial_rows
    n_adv_frac = len(adversarial_rows) / len(train_rows)
    log(f"Combined train set: {len(train_rows)} rows "
        f"({len(natural_train)} natural + {len(adversarial_rows)} adversarial, {n_adv_frac:.1%} adversarial)")

    test_labels = np.array([1 if label else 0 for _, _, _, _, label in test_rows])
    log(f"Natural test label balance: {int(test_labels.sum())} correct / {len(test_labels) - int(test_labels.sum())} incorrect")

    train_labels = [1.0 if label else 0.0 for _, _, _, _, label in train_rows]

    if args.input_mode == "concat":
        qcached_text_fn = lambda q, cq: QCACHED_TEMPLATE.format(query=q, cached_query=cq)
    else:
        qcached_text_fn = lambda q, cq: DIFF_TEMPLATE.format(query=q, diff=word_diff_summary(cq, q))

    results = {}
    for arm_name, texts_a_train, texts_a_test in [
        ("text_only_adv",
         [q for _, q, _, _, _ in train_rows],
         [q for _, q, _, _, _ in test_rows]),
        (qcached_arm_name,
         [qcached_text_fn(q, cq) for _, q, cq, _, _ in train_rows],
         [qcached_text_fn(q, cq) for _, q, cq, _, _ in test_rows]),
    ]:
        texts_b_train = [a for _, _, _, a, _ in train_rows]
        texts_b_test = [a for _, _, _, a, _ in test_rows]

        tuned, train_time = train_one_arm(
            arm_name, texts_a_train, texts_b_train, train_labels,
            args.base_model, args.epochs, args.batch_size, args.logging_steps,
            f"{args.model_out_prefix}_{arm_name}", device,
        )

        test_pairs = list(zip(texts_a_test, texts_b_test))
        t0 = time.time()
        scores = np.array(tuned.predict(test_pairs, batch_size=32, show_progress_bar=False))
        eval_result = evaluate_scores(scores, test_labels)
        log(f"[{arm_name}] natural held-out AUC = {eval_result['auc']:.4f} "
            f"(scored {len(test_pairs)} pairs in {time.time() - t0:.1f}s)")
        results[arm_name] = {"train_time_seconds": train_time, **eval_result}
        del tuned

    output = {
        "input_mode": args.input_mode,
        "target_adv_fraction": args.target_adv_fraction,
        "truncate_query_words": args.truncate_query_words,
        "truncate_answer_words": args.truncate_answer_words,
        "truncate_answer_mode": args.truncate_answer_mode,
        "exclude_category": args.exclude_category,
        "n_train": len(train_rows),
        "n_natural_train": len(natural_train),
        "n_adversarial_train_rows": len(adversarial_rows),
        "n_test": len(test_rows),
        "test_positive_rate": float(test_labels.mean()),
        "finetuned_text_only_adv": results["text_only_adv"],
        "finetuned_qcached_adv": results[qcached_arm_name],
        "auc_delta_qcached_vs_text_only": results[qcached_arm_name]["auc"] - results["text_only_adv"]["auc"],
    }
    Path(args.output).write_text(json.dumps(output, indent=2), encoding="utf-8")
    log(f"Natural AUC: text_only_adv={results['text_only_adv']['auc']:.4f}  "
        f"{qcached_arm_name}={results[qcached_arm_name]['auc']:.4f}")
    log(f"Wrote comparison to {args.output}")
    log("Next: run qcached_redteam_eval.py pointed at these two checkpoints to get the "
        "adversarial false-accept-rate comparison (the decisive metric for this follow-up).")


if __name__ == "__main__":
    main()
