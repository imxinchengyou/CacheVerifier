"""Stage 2 of the fine-tuning experiment: train a cross-encoder on the
gray-zone calibration split (from finetune_verifier_experiment.py's stash)
and compare its held-out discriminative power against the untuned,
off-the-shelf model used for Group D in PAPER.md.

Headline metric is ROC-AUC on the held-out test labels: does fine-tuning on
domain-specific gray-zone examples let the model separate "would have been
correct" from "would have been wrong" better than the generic pretrained
model does out of the box? Also reports false-approve/false-reject rate
curves across a percentile-based threshold grid (built from each model's own
score distribution, exactly the lesson learned in PAPER.md Section 5.4 about
not reusing a threshold grid calibrated on a different score distribution).

Usage:
    python scripts/finetune_verifier_train_eval.py \\
        --stash results/finetune_verifier_experiment.examples.json --epochs 1
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Plain rank-based AUC, no sklearn dependency surprises."""
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
        curve.append(
            {
                "threshold": float(t),
                "approve_rate": float(n_approved / len(scores)),
                "false_approve_rate": false_approve,
                "false_reject_rate": false_reject,
            }
        )
    return {
        "auc": roc_auc(scores, labels),
        "score_min": float(scores.min()),
        "score_mean": float(scores.mean()),
        "score_max": float(scores.max()),
        "curve": curve,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stash", default="results/finetune_verifier_experiment.examples.json")
    parser.add_argument("--base-model", default="cross-encoder/ms-marco-MiniLM-L6-v2")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--logging-steps", type=int, default=10,
                        help="How often (in training steps) to print a loss/lr/ETA line")
    parser.add_argument("--model-out", default="results/finetuned_verifier_model")
    parser.add_argument("--output", default="results/finetune_verifier_comparison.json")
    parser.add_argument("--label-noise", type=float, default=0.0,
                        help="Fraction of TRAIN labels to randomly flip before fine-tuning, simulating a noisy "
                             "production feedback signal (e.g. user thumbs-up/down) instead of clean oracle labels. "
                             "Test labels are never touched, so held-out AUC still measures against ground truth.")
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument("--train-size", type=int, default=None,
                        help="Cap training examples (from the front of the split) for faster iteration, "
                             "e.g. when sweeping many --label-noise levels")
    args = parser.parse_args()

    import torch
    from datasets import Dataset
    from sentence_transformers import CrossEncoder
    from sentence_transformers.cross_encoder import CrossEncoderTrainer, CrossEncoderTrainingArguments
    from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
    from transformers import TrainerCallback

    def log(msg: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    class VerboseLoggerCallback(TrainerCallback):
        """Prints one clean text line per logging step (loss/lr/elapsed/ETA)
        instead of relying on a single overwriting tqdm progress bar, which
        is unreadable when the run is monitored via `tail -f` on a log file."""

        def __init__(self):
            self.t_start = None

        def on_train_begin(self, args, state, control, **kwargs):
            self.t_start = time.time()
            log(f"[train] begin: max_steps={state.max_steps} epochs={args.num_train_epochs} "
                f"batch_size={args.per_device_train_batch_size} "
                f"steps_per_epoch={state.max_steps / max(1, args.num_train_epochs):.0f}")

        def on_log(self, args, state, control, logs=None, **kwargs):
            logs = logs or {}
            if "loss" not in logs:
                return
            elapsed = time.time() - self.t_start
            step, total = state.global_step, state.max_steps
            rate = step / elapsed if elapsed > 0 else 0
            eta = (total - step) / rate if rate > 0 else float("inf")
            lr = logs.get("learning_rate")
            lr_str = f"{lr:.2e}" if lr is not None else "n/a"
            log(f"[train] step {step:>4}/{total} ({100 * step / total:5.1f}%)  "
                f"loss={logs['loss']:.4f}  lr={lr_str}  "
                f"{step / max(elapsed, 1e-9):.3f} step/s  "
                f"elapsed={elapsed / 60:.1f}m  eta={eta / 60:.1f}m")

        def on_train_end(self, args, state, control, **kwargs):
            log(f"[train] end: {state.global_step} steps in {(time.time() - self.t_start) / 60:.1f}m")

    stash = json.loads(Path(args.stash).read_text(encoding="utf-8"))
    train_rows, test_rows = stash["train"], stash["test"]
    if args.train_size is not None:
        train_rows = train_rows[: args.train_size]
    log(f"Loaded stash from {args.stash}")
    log(f"Train: {len(train_rows)}  Test: {len(test_rows)}")

    test_pairs = [(q, a) for _, q, a, _ in test_rows]
    test_labels = np.array([1 if label else 0 for _, _, _, label in test_rows])
    log(f"Test label balance: {int(test_labels.sum())} correct / {len(test_labels) - int(test_labels.sum())} incorrect "
        f"({test_labels.mean():.1%} positive)")

    log(f"Loading baseline (untuned) model {args.base_model!r} on {device}...")
    t0 = time.time()
    baseline = CrossEncoder(args.base_model, device=device)
    log(f"  model loaded in {time.time() - t0:.1f}s")

    log(f"Scoring held-out test set ({len(test_pairs)} pairs, batch_size=32) with the UNTUNED baseline...")
    t0 = time.time()
    baseline_scores = np.array(baseline.predict(test_pairs, batch_size=32, show_progress_bar=False))
    log(f"  done in {time.time() - t0:.1f}s ({len(test_pairs) / (time.time() - t0):.1f} pairs/s)")
    baseline_eval = evaluate_scores(baseline_scores, test_labels)
    log(f"  baseline AUC = {baseline_eval['auc']:.4f}  "
        f"(score range [{baseline_eval['score_min']:.3f}, {baseline_eval['score_max']:.3f}], "
        f"mean={baseline_eval['score_mean']:.3f})")
    del baseline

    log(f"Loading fresh copy of {args.base_model!r} to fine-tune...")
    t0 = time.time()
    tuned = CrossEncoder(args.base_model, device=device)
    log(f"  model loaded in {time.time() - t0:.1f}s")

    train_labels_clean = [1.0 if label else 0.0 for _, _, _, label in train_rows]
    n_flipped = 0
    if args.label_noise > 0:
        rng = np.random.default_rng(args.noise_seed)
        flip_mask = rng.random(len(train_labels_clean)) < args.label_noise
        train_labels = [1.0 - lbl if flip else lbl for lbl, flip in zip(train_labels_clean, flip_mask)]
        n_flipped = int(flip_mask.sum())
        log(f"Label noise: flipped {n_flipped}/{len(train_labels_clean)} train labels "
            f"({n_flipped / len(train_labels_clean):.1%}, target={args.label_noise:.1%}, seed={args.noise_seed}) "
            f"to simulate a noisy feedback signal")
    else:
        train_labels = train_labels_clean

    train_dataset = Dataset.from_dict(
        {
            "query": [q for _, q, _, _ in train_rows],
            "response": [a for _, _, a, _ in train_rows],
            "label": train_labels,
        }
    )
    n_train_pos = sum(1 for lbl in train_labels if lbl == 1.0)
    log(f"Built training dataset: {len(train_dataset)} rows "
        f"({n_train_pos} correct / {len(train_rows) - n_train_pos} incorrect, "
        f"{n_train_pos / len(train_rows):.1%} positive"
        f"{f', {n_flipped} labels flipped' if n_flipped else ''})")

    loss = BinaryCrossEntropyLoss(tuned)
    steps_per_epoch = -(-len(train_dataset) // args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    log(f"Training config: epochs={args.epochs} batch_size={args.batch_size} "
        f"steps_per_epoch={steps_per_epoch} total_steps={total_steps} device={device}")
    training_args = CrossEncoderTrainingArguments(
        output_dir=str(Path(args.model_out).with_name(Path(args.model_out).name + "_checkpoints")),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        use_cpu=(device == "cpu"),
        report_to="none",
        save_strategy="no",
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        logging_first_step=True,
        disable_tqdm=True,
    )
    trainer = CrossEncoderTrainer(
        model=tuned,
        args=training_args,
        train_dataset=train_dataset,
        loss=loss,
    )
    trainer.add_callback(VerboseLoggerCallback())

    log(f"Fine-tuning on {len(train_rows)} examples for {args.epochs} epoch(s)  "
        f"(logging every {args.logging_steps} steps)...")
    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0
    log(f"  fine-tuning done in {train_time:.1f}s ({train_time / max(1, len(train_rows)):.3f}s/example)")

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    tuned.save(args.model_out)
    log(f"Saved fine-tuned model to {args.model_out}")

    log(f"Scoring held-out test set ({len(test_pairs)} pairs, batch_size=32) with the FINE-TUNED model...")
    t0 = time.time()
    tuned_scores = np.array(tuned.predict(test_pairs, batch_size=32, show_progress_bar=False))
    log(f"  done in {time.time() - t0:.1f}s ({len(test_pairs) / (time.time() - t0):.1f} pairs/s)")
    tuned_eval = evaluate_scores(tuned_scores, test_labels)
    log(f"  fine-tuned AUC = {tuned_eval['auc']:.4f}  "
        f"(score range [{tuned_eval['score_min']:.3f}, {tuned_eval['score_max']:.3f}], "
        f"mean={tuned_eval['score_mean']:.3f})")

    result = {
        "n_train": len(train_rows),
        "n_test": len(test_rows),
        "test_positive_rate": float(test_labels.mean()),
        "train_time_seconds": train_time,
        "label_noise_target": args.label_noise,
        "label_noise_actual": n_flipped / len(train_labels_clean) if train_labels_clean else 0.0,
        "n_labels_flipped": n_flipped,
        "baseline": baseline_eval,
        "finetuned": tuned_eval,
        "auc_delta": tuned_eval["auc"] - baseline_eval["auc"],
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    log(f"AUC: baseline={baseline_eval['auc']:.4f}  finetuned={tuned_eval['auc']:.4f}  "
        f"delta={result['auc_delta']:+.4f}")
    log(f"Wrote comparison to {args.output}")


if __name__ == "__main__":
    main()
