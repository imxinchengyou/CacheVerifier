"""Does a thinking-mode DeepSeek judge agree better with the 100 human labels?

Re-judges the 100 audit pairs (results/enterprise_audit_human_labels.jsonl)
with the same system prompt as enterprise_compare_judge.py under four settings:

  chat_nonthinking   model deepseek-chat, as in the original run (non-thinking)
  flash_think_low    model deepseek-flash, thinking enabled, reasoning_effort low
  flash_think_high   ... effort high (the API default)
  flash_think_max    ... effort max

and compares each to the human label ("can send as-is" = yes; "unsure" dropped):
3-class agreement, Cohen's kappa, and the binary YES-vs-not view the main
analysis uses (precision of YES, recall of human yes). No prompt was tuned on
these labels, so all 99 usable items are evaluation data.

Usage:
    .venv/bin/python -u scripts/enterprise_compare_judge_thinking.py
"""

import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from enterprise_compare_judge import SYSTEM, URL  # noqa: E402

SETTINGS = {
    "chat_nonthinking": {"model": "deepseek-chat", "max_tokens": 5},
    "flash_think_low": {"model": "deepseek-flash", "thinking": {"type": "enabled"}, "reasoning_effort": "low", "max_tokens": 16000},
    "flash_think_high": {"model": "deepseek-flash", "thinking": {"type": "enabled"}, "reasoning_effort": "high", "max_tokens": 32000},
    "flash_think_max": {"model": "deepseek-flash", "thinking": {"type": "enabled"}, "reasoning_effort": "max", "max_tokens": 64000},
}
CLASSES = ["YES", "PARTIAL", "NO"]
HUMAN = {"yes": "YES", "partial": "PARTIAL", "no": "NO"}
_local = threading.local()


def session() -> requests.Session:
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
    return _local.s


def call(key: str, setting: dict, item: dict, retries: int = 6) -> dict:
    payload = {**setting, "messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"NEW CUSTOMER MESSAGE:\n{item['new_message']}\n\n"
                                    f"EARLIER CUSTOMER MESSAGE:\n{item['cached_message']}\n\nREPLY:\n{item['cached_reply'][:800]}"}]}
    last = None
    for attempt in range(retries):
        try:
            t0 = time.time()
            r = session().post(URL, headers={"Authorization": f"Bearer {key}"}, json=payload, timeout=600)
            if r.status_code == 429 or r.status_code >= 500:
                last = f"HTTP {r.status_code}"
                time.sleep(2.0 * (attempt + 1))
                continue
            r.raise_for_status()
            body = r.json()
            msg = body["choices"][0]["message"]
            text = (msg.get("content") or "").upper()
            m = re.findall(r"\b(YES|PARTIAL|NO)\b", text)
            return {"verdict": m[-1] if m else f"?{text[:30]}", "latency_s": time.time() - t0, "model": body.get("model"),
                    "usage": body.get("usage", {}), "reasoning_chars": len(msg.get("reasoning_content") or "")}
        except Exception as e:  # noqa: BLE001
            last = str(e)[:80]
            time.sleep(1.5 * (attempt + 1))
    return {"verdict": f"ERROR {last}"}


def load_items() -> list[dict]:
    """The 100 audit pairs with their text. The text file stays local (the Twitter corpus is
    CC BY-NC-SA 4.0); without it the text is rebuilt from data/processed and the text-free labels."""
    with_text = ROOT / "results/enterprise_audit_human_labels.jsonl"
    if with_text.exists():
        return [json.loads(line) for line in open(with_text, encoding="utf-8")]
    items = [json.loads(line) for line in open(ROOT / "results/enterprise_audit_human_labels_notext.jsonl")]
    rows = {ds: [json.loads(line) for line in open(ROOT / f"data/processed/{ds}.jsonl", encoding="utf-8")]
            for ds in {it["dataset"] for it in items}}
    for it in items:
        new, old = rows[it["dataset"]][it["request_index"]], rows[it["dataset"]][it["cached_index"]]
        it.update(new_message=new["query"], cached_message=old["query"], cached_reply=old["answer"])
    return items


def kappa(a: list, b: list) -> float:
    a, b = np.array(a), np.array(b)
    po = (a == b).mean()
    pe = sum((a == c).mean() * (b == c).mean() for c in CLASSES)
    return float((po - pe) / (1 - pe))


def main() -> None:
    load_dotenv(ROOT / ".env")
    key = os.environ["DEEPSEEK_API_KEY"]
    items = load_items()
    items = [it for it in items if it["human_label"] in HUMAN]
    human = [HUMAN[it["human_label"]] for it in items]
    out = {"n": len(items), "settings": {}}
    raw = {}
    for name, setting in SETTINGS.items():
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=50) as ex:
            res = list(ex.map(lambda it: call(key, setting, it), items))
        raw[name] = res
        v = [r["verdict"] for r in res]
        ok = [k for k, x in enumerate(v) if x in CLASSES]
        hv, jv = [human[k] for k in ok], [v[k] for k in ok]
        yes_j = np.array([x == "YES" for x in jv])
        yes_h = np.array([x == "YES" for x in hv])
        cond = {c: {"n": int(sum(x == c for x in jv)),
                    "p_human_yes": float(np.mean([hv[k] == "YES" for k in range(len(jv)) if jv[k] == c])) if c in jv else None}
                for c in CLASSES}
        usage = [r.get("usage", {}) for r in res if "usage" in r]
        s = {"valid": len(ok), "verdict_counts": {c: int(sum(x == c for x in jv)) for c in CLASSES},
             "agreement_3class": float(np.mean(np.array(hv) == np.array(jv))), "kappa_3class": kappa(hv, jv),
             "binary_accuracy": float((yes_j == yes_h).mean()),
             "yes_precision": float(yes_h[yes_j].mean()) if yes_j.any() else None,
             "human_yes_recall": float(yes_j[yes_h].mean()),
             "p_human_yes_given_verdict": cond,
             "mean_latency_s": float(np.mean([r["latency_s"] for r in res if "latency_s" in r])),
             "mean_completion_tokens": float(np.mean([u.get("completion_tokens", 0) for u in usage])),
             "mean_prompt_tokens": float(np.mean([u.get("prompt_tokens", 0) for u in usage])),
             "resolved_models": sorted({r.get("model") for r in res if r.get("model")}),
             "wall_s": time.time() - t0}
        out["settings"][name] = s
        print(f"{name:18s} valid {s['valid']}/{len(items)} counts {s['verdict_counts']} | 3-class agree {s['agreement_3class']:.2f} "
              f"kappa {s['kappa_3class']:.2f} | binary acc {s['binary_accuracy']:.2f} YES-prec {s['yes_precision']} "
              f"human-yes recall {s['human_yes_recall']:.2f} | {s['mean_latency_s']:.1f}s, {s['mean_completion_tokens']:.0f} out tok "
              f"| models {s['resolved_models']}", flush=True)
    original = [it["judge_verdict"] for it in items]
    out["chat_nonthinking_same_as_original"] = float(np.mean([a == b for a, b in zip(original, [r["verdict"] for r in raw["chat_nonthinking"]])]))
    out["original_run_kappa"] = kappa(human, original)
    print(f"original run (2026-11-02) kappa {out['original_run_kappa']:.2f}; rerun agrees with original on "
          f"{out['chat_nonthinking_same_as_original']:.2f} of items", flush=True)
    (ROOT / "results/enterprise_judge_thinking.json").write_text(json.dumps(out, indent=1))
    (ROOT / "results/enterprise_judge_thinking_raw.json").write_text(json.dumps(
        {name: [{"id": it["id"], **r} for it, r in zip(items, res)] for name, res in raw.items()}, indent=1))


if __name__ == "__main__":
    main()
