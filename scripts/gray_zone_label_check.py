"""RESEARCH_PROPOSAL.md 方向 26.18: how noisy are the gray-zone labels
(similarity in [0.80, 0.97))? Unlike the direct-hit check, noise can go both
ways, so wrong- and correct-labelled requests are sampled separately.

Step 1: random 300 wrong + 200 correct gray-zone requests per dataset,
classified by DeepSeek with the 26.15 prompt.
Step 2: a RANDOM (not LLM-selected) subset of 25 wrong + 25 correct per
dataset for blind human review.
Outputs: results/gray_zone_label_check_llm.json and
results/gray_zone_annotation_sample.json."""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from direct_hit_error_taxonomy import DATA, call, parse  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "results" / ".cache"
LO, HI, N_WRONG, N_RIGHT, H_WRONG, H_RIGHT, SEED = 0.80, 0.97, 300, 200, 25, 25, 1


def main():
    load_dotenv(ROOT / ".env")
    key = os.environ["DEEPSEEK_API_KEY"]
    rng = np.random.default_rng(SEED)
    llm, human = {}, []
    for ds, (prefix, path, n) in DATA.items():
        T = json.loads((CACHE / f"{prefix}.trace.json").read_text())
        gray = [i for i, t in enumerate(T) if t[0] is not None and LO <= t[0] < HI]
        wrong = [i for i in gray if not T[i][1]]
        right = [i for i in gray if T[i][1]]
        pick = [(int(i), False) for i in rng.choice(wrong, N_WRONG, replace=False)] + \
               [(int(i), True) for i in rng.choice(right, N_RIGHT, replace=False)]
        need = {i for i, _ in pick} | {int(T[i][2]) for i, _ in pick}
        text = {}
        with open(ROOT / path, encoding="utf-8") as f:
            for j, line in enumerate(f):
                if j >= n or j > max(need):
                    break
                if j in need:
                    o = json.loads(line)
                    text[j] = (o["query"], o["answer"])
        jobs = [(i, lab, text[i][0], text[int(T[i][2])][0], text[int(T[i][2])][1]) for i, lab in pick]
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=60) as ex:
            replies = list(ex.map(lambda j: call(key, j[2], j[3], j[4]), jobs))
        rows = []
        for (i, lab, qn, qc, ans), rep in zip(jobs, replies):
            cat, acc = parse(rep)
            rows.append({"pos": i, "label_correct": lab, "sim": T[i][0], "q_new": qn, "q_cached": qc, "answer": ans,
                         "category": cat, "acceptable": acc, "raw": rep})
        llm[ds] = rows
        print(f"{ds}: {len(rows)} classified in {time.time() - t0:.0f}s, parse errors {sum(r['category'] == '?' for r in rows)}", flush=True)
        # random human subset, independent of the LLM verdicts
        w_rows = [r for r in rows if not r["label_correct"]]
        c_rows = [r for r in rows if r["label_correct"]]
        for r in [w_rows[k] for k in rng.choice(len(w_rows), H_WRONG, replace=False)] + \
                 [c_rows[k] for k in rng.choice(len(c_rows), H_RIGHT, replace=False)]:
            human.append({"dataset": ds, "stratum": "wrong" if not r["label_correct"] else "correct", "pos": r["pos"],
                          "sim": round(r["sim"], 4), "q_new": r["q_new"], "q_cached": r["q_cached"], "answer": r["answer"],
                          "llm_category": r["category"], "llm_acceptable": r["acceptable"]})
    (ROOT / "results/gray_zone_label_check_llm.json").write_text(json.dumps(llm, ensure_ascii=False, indent=1))
    order = rng.permutation(len(human))
    human = [human[k] for k in order]
    for k, it in enumerate(human):
        it["id"] = f"g{k + 1:03d}"
    (ROOT / "results/gray_zone_annotation_sample.json").write_text(json.dumps(human, ensure_ascii=False, indent=1))
    print("human sample:", len(human))


if __name__ == "__main__":
    main()
