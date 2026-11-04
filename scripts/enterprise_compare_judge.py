"""Enterprise comparison, step 2: answer-level relabel with a DeepSeek judge.

The datasets' own label (`same`: the two customer messages got a
template-identical support reply) is very strict. For every nearest-neighbour
pair with sim >= 0.6 in results/.cache/enterprise_{ds}.npz, this asks whether
the cached support reply could be sent unchanged to the new customer message.
YES counts as a correct reuse; PARTIAL and NO count as wrong (conservative,
same convention as PAPER §5.26).

Pairs below 0.6 are not relabelled (they are assumed not reusable); a random
sample of 300 per dataset is judged as well to check that assumption.

Resumable: verdicts are appended to results/.cache/enterprise_{ds}_judge.jsonl
and already-judged indices are skipped.

JUDGE_TAG selects the judge setting and the output file
(results/.cache/enterprise_{ds}_{JUDGE_TAG}.jsonl):
  judge           deepseek-chat, non-thinking (the 2026-11-02 run)
  judge_thinklow  deepseek-flash, thinking enabled, reasoning_effort low; agreed
                  best with the 100 human labels (enterprise_compare_judge_thinking.py)

Usage:
    [JUDGE_TAG=judge_thinklow] .venv/bin/python -u scripts/enterprise_compare_judge.py
"""

import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "results" / ".cache"
DATASETS = {"twitter_amazon": "data/processed/twitter_amazon.jsonl",
            "twitter_comcast": "data/processed/twitter_comcast.jsonl"}
URL = "https://api.deepseek.com/chat/completions"
SIM_MIN, N_BELOW, SEED = 0.6, 300, 0
JUDGE_TAG = os.environ.get("JUDGE_TAG", "judge")
JUDGE_SETTINGS = {
    "judge": ({"model": "deepseek-chat", "temperature": 0.0, "max_tokens": 5}, 95),
    "judge_thinklow": ({"model": "deepseek-flash", "thinking": {"type": "enabled"}, "reasoning_effort": "low",
                        "max_tokens": 16000}, 200),
}
SETTING, WORKERS = JUDGE_SETTINGS[JUDGE_TAG]
SYSTEM = (
    "You are auditing a semantic cache for a company's customer-support account on Twitter. "
    "A support agent wrote the REPLY for an EARLIER CUSTOMER MESSAGE. The cache wants to send the same REPLY, "
    "unchanged, to a NEW CUSTOMER MESSAGE. Would a support agent be fine sending exactly this reply to the new "
    "customer? Answer YES, PARTIAL (relevant, but misses or misstates something the new message needs), or NO. "
    "Reply with exactly one word: YES, PARTIAL or NO."
)


_local = threading.local()


def _session() -> requests.Session:
    if not hasattr(_local, "session"):
        _local.session = requests.Session()
    return _local.session


def call(key: str, q_new: str, q_old: str, reply: str, retries: int = 6) -> tuple[str, dict]:
    payload = {
        **SETTING,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": f"NEW CUSTOMER MESSAGE:\n{q_new}\n\n"
                                                 f"EARLIER CUSTOMER MESSAGE:\n{q_old}\n\nREPLY:\n{reply[:800]}"}],
    }
    last = None
    for attempt in range(retries):
        try:
            r = _session().post(URL, headers={"Authorization": f"Bearer {key}"}, json=payload, timeout=300)
            if r.status_code == 429 or r.status_code >= 500:
                last = f"HTTP {r.status_code}"
                time.sleep(2.0 * (attempt + 1))
                continue
            r.raise_for_status()
            body = r.json()
            text = (body["choices"][0]["message"].get("content") or "").upper()
            m = re.findall(r"\b(YES|PARTIAL|NO)\b", text)
            return (m[-1] if m else f"?{text[:20]}"), body.get("usage", {})
        except Exception as e:  # noqa: BLE001
            last = str(e)[:80]
            time.sleep(1.5 * (attempt + 1))
    return f"ERROR {last}", {}


def main() -> None:
    load_dotenv(ROOT / ".env")
    key = os.environ["DEEPSEEK_API_KEY"]
    rng = np.random.default_rng(SEED)
    for ds, path in DATASETS.items():
        z = np.load(CACHE / f"enterprise_{ds}.npz")
        sim, nn = z["sim"], z["nn"]
        rows = [json.loads(line) for line in open(ROOT / path, encoding="utf-8")]
        above = [int(i) for i in np.where(sim >= SIM_MIN)[0]]
        below_pool = np.where(sim < SIM_MIN)[0]
        below = [int(i) for i in rng.choice(below_pool, min(N_BELOW, len(below_pool)), replace=False)]
        out = CACHE / f"enterprise_{ds}_{JUDGE_TAG}.jsonl"
        done = set()
        if out.exists():
            done = {r["i"] for r in map(json.loads, open(out)) if r["verdict"] in ("YES", "PARTIAL", "NO")}
        todo = [(i, "above") for i in above if i not in done] + [(i, "below_check") for i in below if i not in done]
        print(f"{ds}: {len(above)} pairs >= {SIM_MIN}, {len(below)} below-check; {len(todo)} to judge", flush=True)
        t0 = time.time()

        def work(job):
            i, kind = job
            j = int(nn[i])
            verdict, usage = call(key, rows[i]["query"], rows[j]["query"], rows[j]["answer"])
            return {"i": i, "kind": kind, "sim": float(sim[i]), "verdict": verdict,
                    "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
                    "prompt_cache_hit_tokens": usage.get("prompt_cache_hit_tokens")}

        with open(out, "a") as f, ThreadPoolExecutor(max_workers=WORKERS) as ex:
            for k, res in enumerate(ex.map(work, todo), 1):
                f.write(json.dumps(res) + "\n")
                if k % 500 == 0:
                    f.flush()
                    print(f"  {k}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
        print(f"{ds}: done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
