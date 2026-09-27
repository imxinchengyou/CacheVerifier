"""RESEARCH_PROPOSAL.md 方向 26.16: translate the annotation sample into
Chinese for the human reviewer. Each text is translated on its own (never
the two questions together), literally, keeping numbers, versions, proper
nouns and code as written, so the translation can't erase a difference the
reviewer is supposed to judge. Output: results/direct_hit_annotation_zh.json
keyed by item id."""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
URL = "https://api.deepseek.com/chat/completions"
SYSTEM = (
    "Translate the user's text into Simplified Chinese. Rules: translate literally and faithfully, sentence by "
    "sentence; do not summarize, add, omit, correct or explain anything; keep the original word order of lists "
    "and qualifiers where possible; keep numbers, dates, versions, units, product and brand names, proper nouns, "
    "code, URLs and formulas exactly as written (you may add a Chinese gloss in parentheses after a brand name); "
    "if the text is a terse search query, translate it as a terse search query. If the text is already Chinese, "
    "return it unchanged. Output only the translation."
)


def translate(key, text, retries=5):
    if not text.strip():
        return text
    payload = {"model": "deepseek-chat", "temperature": 0.0, "max_tokens": 4000,
               "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]}
    for attempt in range(retries):
        try:
            r = requests.post(URL, headers={"Authorization": f"Bearer {key}"}, json=payload, timeout=180)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(2.0 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except Exception:  # noqa: BLE001
            time.sleep(2.0 * (attempt + 1))
    return None


def main():
    load_dotenv(ROOT / ".env")
    key = os.environ["DEEPSEEK_API_KEY"]
    items = json.loads((ROOT / "results/direct_hit_annotation_sample.json").read_text())
    jobs = [(it["id"], f, it[src][:4000] if f == "a" else it[src]) for it in items
            for f, src in (("qn", "q_new"), ("qc", "q_cached"), ("a", "answer"))]
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=60) as ex:
        out = list(ex.map(lambda j: translate(key, j[2]), jobs))
    zh = {}
    for (iid, f, _), t in zip(jobs, out):
        zh.setdefault(iid, {})[f] = t
    failed = sum(t is None for t in out)
    (ROOT / "results/direct_hit_annotation_zh.json").write_text(json.dumps(zh, ensure_ascii=False, indent=1))
    print(f"{len(jobs)} texts in {time.time() - t0:.0f}s, failed {failed}")


if __name__ == "__main__":
    main()
