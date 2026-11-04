"""Enterprise comparison: human audit sample for the DeepSeek judge labels.

Draws 100 nearest-neighbour pairs with sim >= 0.8 (the range where the compared
methods actually serve hits): per dataset 50, stratified by the judge's verdict
(YES 20, PARTIAL 15, NO 15), shuffled together. Each text is translated into
Chinese on its own (literal translation, same rules as
translate_annotation_sample.py) for the reviewer; the English original stays
alongside.

The annotation page sees only results/enterprise_audit_sample.json (no verdict,
no similarity, no dataset). The verdicts and positions are kept in
results/enterprise_audit_key.json and are joined back only when scoring.

Usage:
    .venv/bin/python -u scripts/enterprise_compare_audit_sample.py
"""

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from translate_annotation_sample import translate  # noqa: E402

CACHE = ROOT / "results" / ".cache"
DATASETS = {"twitter_amazon": "data/processed/twitter_amazon.jsonl",
            "twitter_comcast": "data/processed/twitter_comcast.jsonl"}
SIM_MIN, PER_VERDICT, SEED = 0.8, {"YES": 20, "PARTIAL": 15, "NO": 15}, 0


def main() -> None:
    load_dotenv(ROOT / ".env")
    key = os.environ["DEEPSEEK_API_KEY"]
    rng = np.random.default_rng(SEED)
    picks = []
    for ds, path in DATASETS.items():
        z = np.load(CACHE / f"enterprise_{ds}.npz")
        sim, nn = z["sim"], z["nn"]
        rows = [json.loads(line) for line in open(ROOT / path, encoding="utf-8")]
        judged = {}
        for line in open(CACHE / f"enterprise_{ds}_judge.jsonl"):
            r = json.loads(line)
            if r["kind"] == "above" and r["sim"] >= SIM_MIN:
                judged[r["i"]] = r["verdict"]
        for verdict, k in PER_VERDICT.items():
            pool = sorted(i for i, v in judged.items() if v == verdict)
            for i in rng.choice(pool, k, replace=False):
                i, j = int(i), int(nn[i])
                picks.append({"ds": ds, "i": i, "nn": j, "sim": float(sim[i]), "judge": verdict,
                              "q_new": rows[i]["query"], "q_old": rows[j]["query"], "reply": rows[j]["answer"]})
    rng.shuffle(picks)

    texts = [p[f] for p in picks for f in ("q_new", "q_old", "reply")]
    with ThreadPoolExecutor(max_workers=60) as ex:
        zh = list(ex.map(lambda t: translate(key, t), texts))

    sample, answer_key = [], {}
    for n, p in enumerate(picks, 1):
        item = f"a{n:03d}"
        q_new_zh, q_old_zh, reply_zh = zh[3 * (n - 1): 3 * n]
        sample.append({"id": item, "q_new": p["q_new"], "q_old": p["q_old"], "reply": p["reply"],
                       "q_new_zh": q_new_zh, "q_old_zh": q_old_zh, "reply_zh": reply_zh})
        answer_key[item] = {k: p[k] for k in ("ds", "i", "nn", "sim", "judge")}
    (ROOT / "results/enterprise_audit_sample.json").write_text(json.dumps(sample, ensure_ascii=False, indent=1))
    (ROOT / "results/enterprise_audit_key.json").write_text(json.dumps(answer_key, indent=1))
    print(f"{len(sample)} items; untranslated: {sum(1 for t in zh if t.startswith('ERROR'))}")


if __name__ == "__main__":
    main()
