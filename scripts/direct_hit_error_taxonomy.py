"""RESEARCH_PROPOSAL.md 方向 26.15: why are direct hits (similarity >= tau_high,
served without the verifier) wrong? In particular, how often is the cause a
generality mismatch between the cached question and the new one -- the one
kind of error a symmetric similarity cannot represent?

Samples wrong and (as a control) correct direct hits from the insert-always
traces and asks DeepSeek to classify each pair without seeing the dataset
label. Answers are truncated to 800 characters.

Usage: python -u scripts/direct_hit_error_taxonomy.py
"""

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "results" / ".cache"
URL = "https://api.deepseek.com/chat/completions"
TAU_HIGH, N_WRONG, N_RIGHT, SEED = 0.97, 400, 150, 0
DATA = {
    "lmarena": ("lmarena__precomputed__n60000", "data/processed/lmarena.jsonl", 60000),
    "search_queries_corrected": ("search_queries_corrected__precomputed__n150000", "data/processed/search_queries_corrected.jsonl", 150000),
    "quora": ("quora__sentence-transformer_sentence-transformers_all-MiniLM-L6-v2__n60000", "data/processed/quora.jsonl", 60000),
}
SYSTEM = (
    "A semantic cache stored an answer to a CACHED QUESTION and is about to reuse it for a NEW QUESTION. "
    "Classify how the NEW QUESTION relates to the CACHED QUESTION. Pick the single best category:\n"
    "A = the cached question is MORE GENERAL; the new question asks for something narrower or more specific\n"
    "B = the cached question is MORE SPECIFIC; the new question is broader\n"
    "C = a different entity, object, number, version, or parameter\n"
    "D = negation, opposite direction, or contradictory\n"
    "E = a different intent or topic\n"
    "F = essentially the same question, just phrased differently\n"
    "G = other\n"
    "Then say whether the CACHED ANSWER would be an acceptable answer to the NEW QUESTION: YES, PARTIAL, or NO.\n"
    "Reply in exactly this format and nothing else: CATEGORY=<letter>; ACCEPTABLE=<YES|PARTIAL|NO>"
)


def call(api_key, q_new, q_cached, answer, retries=5):
    payload = {
        "model": "deepseek-chat", "temperature": 0.0, "max_tokens": 20,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": f"NEW QUESTION:\n{q_new}\n\nCACHED QUESTION:\n{q_cached}\n\nCACHED ANSWER:\n{answer[:800]}"}],
    }
    last = None
    for attempt in range(retries):
        try:
            r = requests.post(URL, headers={"Authorization": f"Bearer {api_key}"}, json=payload, timeout=60)
            if r.status_code == 429 or r.status_code >= 500:
                last = RuntimeError(f"HTTP {r.status_code}")
                time.sleep(2.0 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.5 * (attempt + 1))
    return f"ERROR {last}"


def parse(text):
    c = re.search(r"CATEGORY\s*=\s*([A-G])", text.upper())
    a = re.search(r"ACCEPTABLE\s*=\s*(YES|PARTIAL|NO)", text.upper())
    return (c.group(1) if c else "?"), (a.group(1) if a else "?")


def main():
    load_dotenv(ROOT / ".env")
    key = os.environ["DEEPSEEK_API_KEY"]
    rng = np.random.default_rng(SEED)
    out = {}
    for ds, (prefix, path, n) in DATA.items():
        T = json.loads((CACHE / f"{prefix}.trace.json").read_text())
        hi = [i for i, t in enumerate(T) if t[0] is not None and t[0] >= TAU_HIGH]
        wrong = [i for i in hi if not T[i][1]]
        right = [i for i in hi if T[i][1]]
        pick = [(i, False) for i in rng.choice(wrong, min(N_WRONG, len(wrong)), replace=False)] + \
               [(i, True) for i in rng.choice(right, min(N_RIGHT, len(right)), replace=False)]
        need = {int(i) for i, _ in pick} | {int(T[i][2]) for i, _ in pick}
        text = {}
        with open(ROOT / path, encoding="utf-8") as f:
            for j, line in enumerate(f):
                if j >= n or j > max(need):
                    break
                if j in need:
                    o = json.loads(line)
                    text[j] = (o["query"], o["answer"])
        jobs = [(int(i), lab, text[int(i)][0], text[int(T[i][2])][0], text[int(T[i][2])][1]) for i, lab in pick]
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=60) as ex:
            replies = list(ex.map(lambda j: call(key, j[2], j[3], j[4]), jobs))
        rows = []
        for (i, lab, qn, qc, _), rep in zip(jobs, replies):
            cat, acc = parse(rep)
            rows.append({"pos": i, "label_correct": lab, "sim": T[i][0], "q_new": qn, "q_cached": qc,
                         "category": cat, "acceptable": acc, "raw": rep})
        out[ds] = rows
        print(f"{ds}: {len(rows)} classified in {time.time() - t0:.0f}s, errors {sum(r['category'] == '?' for r in rows)}", flush=True)
        (ROOT / "results/direct_hit_error_taxonomy.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
