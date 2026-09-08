"""Stage 2 of the q_cached ablation: fine-tune TWO cross-encoders on the
EXACT SAME rows/split from finetune_verifier_qcached_experiment.py's stash,
differing only in text_a construction:

  - "text_only" arm: text_a = current query               (Section 5.6's recipe)
  - "qcached"   arm: text_a = current query + cached query (this ablation)

text_b is the candidate answer in both arms. Everything else (base model,
epochs, batch size, loss, optimizer, train/test split) is identical, so any
AUC difference between the two arms isolates the effect of adding the cached
query, not a confound from different training data or hyperparameters.

Why concatenation into one text-pair segment instead of a true 3-segment
cross-encoder: sentence-transformers' CrossEncoder tokenizes via
`self.tokenizer(*texts, ...)`, and HuggingFace tokenizers only accept two
positional segment args (`text`, `text_pair` -- a third positional arg means
`text_target`, an unrelated seq2seq field, not a third segment). A true
3-segment architecture would need a custom model head; concatenation is the
minimal change that tests the "does the cached query help at all" question
before investing in that larger rebuild.

Usage:
    python scripts/finetune_verifier_qcached_train_eval.py \\
        --stash results/finetune_verifier_qcached_experiment.examples.json --epochs 1

--input-mode diff (2026-09-05 follow-up): the "concat" mode above was found to
BACKFIRE on the Section 5.18 adversarial red-team set (false-accept rate rose
from 86.6% to 97.7% -- see RESEARCH_PROPOSAL.md direction 16). Diagnosis: raw
concatenation exposes surface-level word overlap between query and
cached_query to the tokenizer, and in NATURAL gray-zone data that overlap
correlates strongly with label=True (real paraphrases), so the model learns
"queries look alike -> safe" -- exactly backwards for adversarial pairs that
are near-identical except for the one word that matters. --input-mode diff
replaces the raw cached_query text with a word-level diff summary (via
difflib, no model/LLM involved) describing what changed between cached_query
and the current query (e.g. "removed: pause; added: cancel"), so the input no
longer exposes raw overlap for the model to key off of -- it exposes the
delta directly.
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


def word_diff_summary(cached_query: str, current_query: str, max_words: int = 12) -> str:
    """Word-level diff describing how cached_query differs from current_query,
    e.g. 'removed: pause; added: cancel', or 'identical' if the two tokenize
    to the same words. Pure difflib, no model calls."""
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
    tuned = CrossEncoder(base_model, device=device)
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
    parser.add_argument("--stash", default="results/finetune_verifier_qcached_experiment.examples.json")
    parser.add_argument("--base-model", default="cross-encoder/ms-marco-MiniLM-L6-v2")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--model-out-prefix", default="results/finetuned_verifier_model_lmarena_qcached_ablation")
    parser.add_argument("--output", default="results/finetune_verifier_qcached_comparison.json")
    parser.add_argument("--train-size", type=int, default=None)
    parser.add_argument("--input-mode", choices=["concat", "diff"], default="concat",
                        help="'concat' = raw cached_query text appended (2026-09-05 result: backfires on the "
                             "adversarial red-team set). 'diff' = word-level diff summary instead of raw text "
                             "(see module docstring).")
    args = parser.parse_args()
    qcached_arm_name = "qcached" if args.input_mode == "concat" else f"qcached_{args.input_mode}"

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def log(msg: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    log(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    stash = json.loads(Path(args.stash).read_text(encoding="utf-8"))
    train_rows, test_rows = stash["train"], stash["test"]
    if args.train_size is not None:
        train_rows = train_rows[: args.train_size]
    log(f"Loaded stash from {args.stash}")
    log(f"Train: {len(train_rows)}  Test: {len(test_rows)}")

    # rows are (position, query, cached_query, answer, label)
    test_labels = np.array([1 if label else 0 for _, _, _, _, label in test_rows])
    log(f"Test label balance: {int(test_labels.sum())} correct / {len(test_labels) - int(test_labels.sum())} incorrect "
        f"({test_labels.mean():.1%} positive)")

    log(f"Loading baseline (untuned, text-only) model {args.base_model!r} for reference...")
    from sentence_transformers import CrossEncoder
    baseline = CrossEncoder(args.base_model, device=device)
    test_pairs_text_only = [(q, a) for _, q, _, a, _ in test_rows]
    baseline_scores = np.array(baseline.predict(test_pairs_text_only, batch_size=32, show_progress_bar=False))
    baseline_eval = evaluate_scores(baseline_scores, test_labels)
    log(f"  off-the-shelf (text-only) AUC = {baseline_eval['auc']:.4f}")
    del baseline

    train_labels = [1.0 if label else 0.0 for _, _, _, _, label in train_rows]

    if args.input_mode == "concat":
        qcached_text_fn = lambda q, cq: QCACHED_TEMPLATE.format(query=q, cached_query=cq)
    else:
        qcached_text_fn = lambda q, cq: DIFF_TEMPLATE.format(query=q, diff=word_diff_summary(cq, q))

    results = {}
    for arm_name, texts_a_train, texts_a_test in [
        ("text_only",
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
        log(f"[{arm_name}] fine-tuned AUC = {eval_result['auc']:.4f} "
            f"(scored {len(test_pairs)} pairs in {time.time() - t0:.1f}s)")
        results[arm_name] = {"train_time_seconds": train_time, **eval_result}
        del tuned

    output = {
        "input_mode": args.input_mode,
        "n_train": len(train_rows),
        "n_test": len(test_rows),
        "test_positive_rate": float(test_labels.mean()),
        "off_the_shelf_text_only": baseline_eval,
        "finetuned_text_only": results["text_only"],
        "finetuned_qcached": results[qcached_arm_name],
        "auc_delta_finetuning_text_only": results["text_only"]["auc"] - baseline_eval["auc"],
        "auc_delta_qcached_vs_text_only_finetuned": results[qcached_arm_name]["auc"] - results["text_only"]["auc"],
    }
    Path(args.output).write_text(json.dumps(output, indent=2), encoding="utf-8")
    log(f"AUC: off-the-shelf={baseline_eval['auc']:.4f}  "
        f"finetuned(text_only)={results['text_only']['auc']:.4f}  "
        f"finetuned({qcached_arm_name})={results[qcached_arm_name]['auc']:.4f}")
    log(f"Wrote comparison to {args.output}")


if __name__ == "__main__":
    main()
