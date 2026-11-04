"""Enterprise comparison: re-score every method against the 100-item human audit.

Each judged pair (sim >= 0.6) gets an expected human-correct probability
P(human says "can send as-is" | judge verdict, dataset), estimated from the
blind labels in results/enterprise_audit_human_labels_notext.jsonl; "unsure"
answers are dropped. Pairs below
0.6 keep probability 0, as in the main analysis. The audit sampled only
sim >= 0.8, so for 0.6 <= sim < 0.8 these rates are an extrapolation.

Reports, on the test half: (1) the highest hit rate at expected error <= e for
each method's full parameter curve; (2) expected error and precision at each
method's alpha=2% operating point from the main analysis. vCache is replayed
once (seed 0, delta 0.02) with the judge labels it learned from in the main run.

Usage:
    .venv/bin/python -u scripts/enterprise_compare_human_reweight.py
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import enterprise_compare_analyze as A  # noqa: E402

CACHE = ROOT / "results" / ".cache"
LO, HI = 0.80, 0.97


def human_rates() -> dict:
    rows = [json.loads(line) for line in open(ROOT / "results/enterprise_audit_human_labels_notext.jsonl")]
    rates = {}
    for ds in A.DATASETS:
        for v in ("YES", "PARTIAL", "NO"):
            ans = [r["human_label"] for r in rows if r["dataset"] == ds and r["judge_verdict"] == v and r["human_label"] != "unsure"]
            rates[(ds, v)] = (sum(a == "yes" for a in ans) / len(ans), len(ans))
    return rates


def main() -> None:
    rates = human_rates()
    out = {"rates": {f"{ds}/{v}": {"p_human_yes": p, "n": n} for (ds, v), (p, n) in rates.items()}}
    for ds in A.DATASETS:
        z = np.load(CACHE / f"enterprise_{ds}.npz")
        sim, nn = np.nan_to_num(z["sim"], nan=-1.0), z["nn"]
        n = len(sim)
        half = n // 2
        test = np.arange(n) >= half
        p = np.zeros(n)
        judge = np.zeros(n, dtype=bool)
        for line in open(CACHE / f"enterprise_{ds}_judge.jsonl"):
            r = json.loads(line)
            if r["kind"] == "above" and r["verdict"] in ("YES", "PARTIAL", "NO"):
                p[r["i"]] = rates[(ds, r["verdict"])][0]
                judge[r["i"]] = r["verdict"] == "YES"
        g = np.nan_to_num(np.load(CACHE / f"enterprise_{ds}_gptcache.npz")["score"], nan=-1.0)
        v = np.load(CACHE / f"enterprise_{ds}_verifier.npz")
        gray = v["gray"]
        in_gray = np.zeros(n, dtype=bool)
        in_gray[gray] = True

        def cv(keyname):
            s = np.full(n, -np.inf)
            s[gray] = v[keyname]
            return lambda t: (sim >= HI) | (in_gray & (s >= t))

        methods = {"redis_static": lambda t: sim >= t, "gptcache_onnx": lambda t: g >= t,
                   "cv_stock": cv("stock"), "cv_ft": cv("ft_judge")}
        main_json = json.loads((ROOT / f"results/enterprise_comparison_{ds}.json").read_text())["label_sets"]["judge"]["methods"]
        res = {"frontier": {}, "alpha_0.02": {}}
        for name, fn in methods.items():
            pts = []
            for row in main_json[name]["frontier_test"]:
                h = fn(row["param"])[test]
                pts.append((float(((1 - p[test]) * h).mean()), float(h.mean())))
            res["frontier"][name] = {f"{e}": max([hr for er, hr in pts if er <= e] or [0.0]) for e in (0.01, 0.02, 0.03, 0.05)}
            op = main_json[name]["operating_points"]["alpha_0.02"]
            h = fn(op["param"])[test]
            res["alpha_0.02"][name] = {"hit_rate": float(h.mean()), "expected_error": float(((1 - p[test]) * h).mean()),
                                       "expected_precision": float(p[test][h].mean()), "judge_error": op["error_rate"]}
        h, _, _ = A.replay_vcache(sim, nn, judge, 0.02, 0)
        h = h[test]
        res["alpha_0.02"]["vcache"] = {"hit_rate": float(h.mean()), "expected_error": float(((1 - p[test]) * h).mean()),
                                       "expected_precision": float(p[test][h].mean())}
        out[ds] = res
        print(f"\n{ds}  highest hit rate at expected human error <= e")
        for name, fr in res["frontier"].items():
            print(f"  {name:14s} " + "  ".join(f"e<={e}: {hr:.4f}" for e, hr in fr.items()))
        print("  alpha=2% operating points (judge error -> expected human error, expected precision)")
        for name, r in res["alpha_0.02"].items():
            print(f"  {name:14s} hit {r['hit_rate']:.4f}  err {r.get('judge_error', float('nan')):.4f} -> {r['expected_error']:.4f}  prec {r['expected_precision']:.3f}")
    (ROOT / "results/enterprise_comparison_human_reweight.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
