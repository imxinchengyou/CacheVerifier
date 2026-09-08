"""Direction 16 follow-up: does fine-tuning the NLI-pretrained base ERASE its
zero-shot entity_swap advantage gradually (a classic forgetting curve) or all
at once (an early, unavoidable overwrite)?

Context (RESEARCH_PROPOSAL.md direction 16 series, PAPER.md Sec 5.22): the
zero-shot NLI checkpoint (cross-encoder/nli-MiniLM2-L6-H768, score =
P(entailment) - P(contradiction), no training at all) gets entity_swap's
false-accept rate down to 5.1%. After the SAME diff+adversarial fine-tuning
recipe used for the ms-marco base, entity_swap is back up to 25.6% on
LmArena -- a clear regression from the zero-shot number, though still far
better than the ms-marco pipeline's stuck-at ~46-49%. This script traces the
FULL trajectory across a single training run: natural-data AUC and
per-category adversarial false-accept rate (LmArena's 306-sample held-out
set, Sec 5.18/5.19) at ~12 checkpoints from step 0 (pure zero-shot) through
the full epoch, instead of just comparing the two endpoints.

This directly distinguishes two hypotheses about WHY entity_swap regresses:
  (a) gradual forgetting -- entity_swap FA rate rises smoothly as training
      proceeds, implying an early-stopping sweet spot could keep most of the
      zero-shot advantage while still fixing the other four axes/natural AUC.
  (b) fast overwrite -- entity_swap FA rate jumps to near its final value
      within the first few percent of steps, implying early stopping alone
      cannot help; a different strategy (layer freezing, rehearsal on NLI
      data, a much lower learning rate) would be needed instead.

Evaluation happens on the LIVE in-memory model at each checkpoint (no
save/reload), reusing the exact diff-mode input construction and 0.5
decision threshold already established for this line of experiments
(qcached_redteam_eval.py / finetune_verifier_qcached_adversarial_train_eval.py).
Only the FINAL checkpoint (end of the single epoch, matching the already-
published number) is written to disk.

Usage:
    python scripts/qcached_nli_forgetting_sweep.py
"""

import argparse
import difflib
import json
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

TAU_LOW = 0.80
DIFF_TEMPLATE = "{query}\n[diff vs cached_query] {diff}"
CHECKPOINT_FRACTIONS = [0.0, 0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.5, 0.65, 0.8, 1.0]


def word_diff_summary(cached_query: str, current_query: str, max_words: int = 12) -> str:
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


def build_adversarial_rows(pool_path: str):
    pool = json.loads(Path(pool_path).read_text(encoding="utf-8"))
    qualifying = [t for t in pool["all_triples"] if t.get("similarity", 0.0) >= TAU_LOW]
    rows = []
    for i, t in enumerate(qualifying):
        rows.append((f"adv_neg_{i}", t["query_a"], t["query_b"], t["answer_b"], False))
        rows.append((f"adv_pos_{i}", t["query_b"], t["query_b"], t["answer_b"], True))
    return rows


def load_redteam_probes(redteam_path: str):
    """Returns list of (category, diff_text, answer_b, label=False) -- every
    row in this set is a false-accept trap (query_a should NOT match
    answer_b), matching qcached_redteam_eval.py's construction."""
    data = json.loads(Path(redteam_path).read_text(encoding="utf-8"))
    triples = [t for t in data["all_triples"] if t["similarity"] >= TAU_LOW]
    probes = []
    for t in triples:
        diff_text = DIFF_TEMPLATE.format(query=t["query_a"], diff=word_diff_summary(t["query_b"], t["query_a"]))
        probes.append((t["category"], diff_text, t["answer_b"]))
    return probes


def evaluate_checkpoint(tuned, test_pairs, test_labels, redteam_probes, natural_batch_size=64, redteam_batch_size=64):
    was_training = tuned.model.training
    tuned.model.eval()

    natural_scores = np.array(tuned.predict(test_pairs, batch_size=natural_batch_size, show_progress_bar=False))
    natural_auc = roc_auc(natural_scores, test_labels)

    redteam_pairs = [(diff_text, answer_b) for _, diff_text, answer_b in redteam_probes]
    redteam_scores = np.array(tuned.predict(redteam_pairs, batch_size=redteam_batch_size, show_progress_bar=False))
    approved = redteam_scores >= 0.5

    by_category = {}
    for (category, _, _), is_fa in zip(redteam_probes, approved):
        by_category.setdefault(category, {"n": 0, "fa": 0})
        by_category[category]["n"] += 1
        by_category[category]["fa"] += int(is_fa)
    total_n = len(redteam_probes)
    total_fa = int(approved.sum())

    if was_training:
        tuned.model.train()

    return {
        "natural_auc": natural_auc,
        "natural_score_mean": float(natural_scores.mean()),
        "redteam_total_fa": total_fa,
        "redteam_total_n": total_n,
        "redteam_total_fa_pct": total_fa / total_n * 100 if total_n else float("nan"),
        "redteam_by_category": {
            c: {"n": v["n"], "fa": v["fa"], "fa_pct": v["fa"] / v["n"] * 100 if v["n"] else float("nan")}
            for c, v in sorted(by_category.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--natural-stash", default="results/finetune_verifier_qcached_experiment.examples.json")
    parser.add_argument("--adversarial-pool", default="results/llm_redteam_train_pool.json")
    parser.add_argument("--redteam-results", default="results/llm_redteam_results.json")
    parser.add_argument("--base-model", default="cross-encoder/nli-MiniLM2-L6-H768")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--model-out", default="results/finetuned_verifier_model_lmarena_nli_forgetting_sweep_final")
    parser.add_argument("--output", default="results/qcached_nli_forgetting_sweep_lmarena.json")
    args = parser.parse_args()

    import torch
    from datasets import Dataset
    from sentence_transformers import CrossEncoder
    from sentence_transformers.cross_encoder import CrossEncoderTrainer, CrossEncoderTrainingArguments
    from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
    from transformers import TrainerCallback

    device = "cuda" if torch.cuda.is_available() else "cpu"

    def log(msg: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    log(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    natural = json.loads(Path(args.natural_stash).read_text(encoding="utf-8"))
    natural_train, test_rows = natural["train"], natural["test"]
    log(f"Natural stash: {len(natural_train)} train, {len(test_rows)} test")

    adversarial_rows = build_adversarial_rows(args.adversarial_pool)
    log(f"Adversarial training rows: {len(adversarial_rows)}")

    train_rows = natural_train + adversarial_rows
    log(f"Combined train set: {len(train_rows)} rows "
        f"({len(natural_train)} natural + {len(adversarial_rows)} adversarial, "
        f"{len(adversarial_rows) / len(train_rows):.1%} adversarial)")

    def diff_text(q, cq):
        return DIFF_TEMPLATE.format(query=q, diff=word_diff_summary(cq, q))

    texts_a_train = [diff_text(q, cq) for _, q, cq, _, _ in train_rows]
    texts_b_train = [a for _, _, _, a, _ in train_rows]
    train_labels = [1.0 if label else 0.0 for _, _, _, _, label in train_rows]

    texts_a_test = [diff_text(q, cq) for _, q, cq, _, _ in test_rows]
    texts_b_test = [a for _, _, _, a, _ in test_rows]
    test_pairs = list(zip(texts_a_test, texts_b_test))
    test_labels = np.array([1 if label else 0 for _, _, _, _, label in test_rows])
    log(f"Natural test: {len(test_pairs)} pairs, {int(test_labels.sum())} positive")

    redteam_probes = load_redteam_probes(args.redteam_results)
    log(f"Redteam probes: {len(redteam_probes)} (tau_low={TAU_LOW})")

    log(f"Loading fresh copy of {args.base_model!r}...")
    t0 = time.time()
    tuned = CrossEncoder(args.base_model, num_labels=1, automodel_args={"ignore_mismatched_sizes": True}, device=device)
    log(f"  loaded in {time.time() - t0:.1f}s")

    train_dataset = Dataset.from_dict({"query": texts_a_train, "response": texts_b_train, "label": train_labels})
    loss = BinaryCrossEntropyLoss(tuned)
    steps_per_epoch = -(-len(train_dataset) // args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    checkpoint_steps = sorted(set(round(f * total_steps) for f in CHECKPOINT_FRACTIONS))
    log(f"Training config: epochs={args.epochs} batch_size={args.batch_size} "
        f"steps_per_epoch={steps_per_epoch} total_steps={total_steps}")
    log(f"Checkpoint steps ({len(checkpoint_steps)}): {checkpoint_steps}")

    sweep_results = []

    log("Evaluating step 0 (pure zero-shot, before any training)...")
    t0 = time.time()
    r0 = evaluate_checkpoint(tuned, test_pairs, test_labels, redteam_probes)
    sweep_results.append({"step": 0, "step_frac": 0.0, "eval_seconds": time.time() - t0, **r0})
    log(f"  step 0: natural_auc={r0['natural_auc']:.4f}  redteam_fa={r0['redteam_total_fa_pct']:.1f}%  "
        f"entity_swap={r0['redteam_by_category'].get('entity_swap', {}).get('fa_pct', float('nan')):.1f}%  "
        f"({time.time() - t0:.1f}s)")

    class SweepCallback(TrainerCallback):
        def __init__(self):
            self.t_train_start = None
            self.done_steps = {0}  # step 0 already evaluated above

        def on_train_begin(self, args_, state, control, **kwargs):
            self.t_train_start = time.time()

        def on_log(self, args_, state, control, logs=None, **kwargs):
            logs = logs or {}
            if "loss" in logs:
                elapsed = time.time() - self.t_train_start
                log(f"[train] step {state.global_step:>4}/{state.max_steps} "
                    f"loss={logs['loss']:.4f} elapsed={elapsed / 60:.1f}m")

        def on_step_end(self, args_, state, control, **kwargs):
            step = state.global_step
            if step in checkpoint_steps and step not in self.done_steps:
                self.done_steps.add(step)
                t0_eval = time.time()
                r = evaluate_checkpoint(tuned, test_pairs, test_labels, redteam_probes)
                eval_s = time.time() - t0_eval
                sweep_results.append({"step": step, "step_frac": step / total_steps, "eval_seconds": eval_s, **r})
                ent = r["redteam_by_category"].get("entity_swap", {}).get("fa_pct", float("nan"))
                log(f"  [checkpoint step={step} frac={step / total_steps:.2f}] "
                    f"natural_auc={r['natural_auc']:.4f}  redteam_fa={r['redteam_total_fa_pct']:.1f}%  "
                    f"entity_swap={ent:.1f}%  (eval {eval_s:.1f}s)")

    training_args = CrossEncoderTrainingArguments(
        output_dir=str(Path(args.model_out).with_name(Path(args.model_out).name + "_checkpoints")),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        use_cpu=(device == "cpu"),
        report_to="none",
        save_strategy="no",
        logging_strategy="steps",
        logging_steps=25,
        logging_first_step=True,
        disable_tqdm=True,
    )
    trainer = CrossEncoderTrainer(model=tuned, args=training_args, train_dataset=train_dataset, loss=loss)
    trainer.add_callback(SweepCallback())

    log(f"Fine-tuning on {len(texts_a_train)} examples for {args.epochs} epoch(s), "
        f"with {len(checkpoint_steps) - 1} in-flight checkpoints...")
    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0
    log(f"Fine-tuning done in {train_time / 60:.1f}m")

    # Make sure the final step (== total_steps) was captured even if on_step_end's
    # last firing landed on a slightly different global_step count than expected.
    if total_steps not in {r["step"] for r in sweep_results}:
        log("Evaluating final step explicitly (not captured by callback)...")
        r_final = evaluate_checkpoint(tuned, test_pairs, test_labels, redteam_probes)
        sweep_results.append({"step": total_steps, "step_frac": 1.0, "eval_seconds": 0.0, **r_final})

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    tuned.save(args.model_out)
    log(f"Saved final checkpoint to {args.model_out}")

    sweep_results.sort(key=lambda r: r["step"])
    output = {
        "base_model": args.base_model,
        "total_steps": total_steps,
        "steps_per_epoch": steps_per_epoch,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "n_train": len(train_rows),
        "n_natural_train": len(natural_train),
        "n_adversarial_train_rows": len(adversarial_rows),
        "n_test": len(test_pairs),
        "n_redteam": len(redteam_probes),
        "train_time_seconds": train_time,
        "sweep": sweep_results,
    }
    Path(args.output).write_text(json.dumps(output, indent=2), encoding="utf-8")
    log(f"Wrote {args.output}")

    log("\nSummary (step_frac | natural_auc | redteam_total_fa% | entity_swap_fa% | negation_fa% | direction_fa%):")
    for r in sweep_results:
        cats = r["redteam_by_category"]
        log(f"  {r['step_frac']:.2f} | {r['natural_auc']:.4f} | {r['redteam_total_fa_pct']:5.1f}% | "
            f"{cats.get('entity_swap', {}).get('fa_pct', float('nan')):5.1f}% | "
            f"{cats.get('negation', {}).get('fa_pct', float('nan')):5.1f}% | "
            f"{cats.get('direction', {}).get('fa_pct', float('nan')):5.1f}%")


if __name__ == "__main__":
    main()
