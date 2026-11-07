"""Extend the CRC guarantee from the gray zone to every served hit.

Today the guarantee covers only the gray zone [0.80, 0.97): pairs at sim >= 0.97 are served
without verification, and on the support traffic they produce 2-7x more wrong replies per
request than the guaranteed gray zone (results/enterprise_guarantee_feasibility.json, D).
Risk here is per request (denominator = all requests), loss = wrong reply served.

  P0  gray_only      today's setup: direct hits served unverified, CRC on the gray zone with
                     budget alpha. Guarantee covers only the gray part.
  P0b gray_plugin    same, but the gray budget is alpha minus the direct-hit error estimated
                     on the calibration data. Plug-in, no guarantee.
  P1  verify_all     every candidate (sim >= 0.80) goes through the verifier; one CRC
                     threshold on the per-request loss. Guarantee covers all hits.
  P2  split_budget   two CRC thresholds, one for the direct zone (budget f*alpha) and one for
                     the gray zone ((1-f)*alpha). By linearity of expectation the total is
                     <= alpha. f fixed in advance at 0.25 / 0.5 / 0.75 (choosing it on the data
                     would void the guarantee).

Protocols: R = uniform random 50/50 splits of the evaluation requests (M=200; the theorem
applies), T = calibration part of the stream -> test half in time order. Evaluation requests
are those after the fine-tuning window. Scores: scripts/enterprise_allhits_verifier.py.

Usage:
    .venv/bin/python -u scripts/enterprise_allhits_guarantee.py
    MODEL=grayonly .venv/bin/python -u scripts/enterprise_allhits_guarantee.py   # model trained on gray zone only
                                                                                # (scripts/enterprise_grayonly_on_direct.py)
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

CACHE = ROOT / "results/.cache"
DATASETS = ["twitter_amazon", "twitter_comcast"]
LO, HI = 0.80, 0.97
ALPHAS = [0.01, 0.005, 0.0025]
LABEL_SETS = {"judge": "", "judge_thinklow": "_judge_thinklow"}
FRACTIONS = [0.25, 0.5, 0.75]
M_RANDOM = 200
MODEL = os.environ.get("MODEL", "")
if MODEL:
    LABEL_SETS = {"judge": f"_{MODEL}"}  # only trained on the old judge's labels


def load_labels(ds: str, tag: str, n: int) -> np.ndarray:
    os.environ["JUDGE_TAG"] = tag
    import importlib
    import enterprise_compare_analyze as an
    importlib.reload(an)
    return an.load_judge(ds, n)[0]


def crc(scores: np.ndarray, loss: np.ndarray, alpha: float) -> float:
    """CRC over requests: accept iff score > lam; loss[i] is paid only when accepted. Requests
    that can never be served carry score -inf. Returns lam_hat (inf = accept nothing)."""
    n = len(scores)
    order = np.argsort(scores)
    ss, ll = scores[order], loss[order]
    suffix = np.concatenate([np.cumsum(ll[::-1])[::-1], [0.0]])
    risk = suffix[np.searchsorted(ss, ss, side="right")] / n
    ok = np.where((n / (n + 1)) * risk + 1 / (n + 1) <= alpha)[0]
    return float(ss[ok[0]]) if len(ok) else np.inf


def evaluate(policy: str, cal: np.ndarray, test: np.ndarray, sc: np.ndarray, wrong: np.ndarray,
             gray: np.ndarray, direct: np.ndarray, alpha: float, f: float = 0.5) -> dict:
    w = wrong.astype(float)
    s_gray = np.where(gray, sc, -np.inf)
    s_dir = np.where(direct, sc, -np.inf)
    if policy in ("gray_only", "gray_plugin"):
        a_g = alpha
        if policy == "gray_plugin":
            a_g = max(alpha - float(np.mean(w[cal] * direct[cal])), 0.0)
        lam = crc(s_gray[cal], w[cal], a_g) if a_g > 0 else np.inf
        acc = direct[test] | (s_gray[test] > lam)
        verified = gray[test]
    elif policy == "verify_all":
        s_all = np.where(gray | direct, sc, -np.inf)
        lam = crc(s_all[cal], w[cal], alpha)
        acc = s_all[test] > lam
        verified = gray[test] | direct[test]
    else:  # split_budget
        lam_d = crc(s_dir[cal], w[cal], f * alpha)
        lam_g = crc(s_gray[cal], w[cal], (1 - f) * alpha)
        acc = (s_dir[test] > lam_d) | (s_gray[test] > lam_g)
        verified = gray[test] | direct[test]
    wt = w[test]
    return {"risk": float(np.mean(wt * acc)), "risk_direct": float(np.mean(wt * acc * direct[test])),
            "risk_gray": float(np.mean(wt * acc * gray[test])), "hit_rate": float(acc.mean()),
            "direct_hit_rate": float(np.mean(acc & direct[test])), "verifier_calls": float(verified.mean())}


def main() -> None:
    out = {}
    policies = [("gray_only", None), ("gray_plugin", None), ("verify_all", None)] + [("split_budget", f) for f in FRACTIONS]
    for ds in DATASETS:
        sim = np.nan_to_num(np.load(CACHE / f"enterprise_{ds}.npz")["sim"], nan=-1.0)
        n = len(sim)
        half = n // 2
        for tag, suffix in LABEL_SETS.items():
            lab = load_labels(ds, tag, n)
            v = np.load(CACHE / f"enterprise_{ds}_verifier_allhits{suffix}.npz")
            sc = np.full(n, -np.inf)
            sc[v["cand"]] = v["ft"]
            idx = np.arange(int(v["train_until"]) + 1, n)  # evaluation requests: after the fine-tuning window
            sim_e, sc_e, wrong_e = sim[idx], sc[idx], ~lab[idx]
            gray = (sim_e >= LO) & (sim_e < HI)
            direct = sim_e >= HI
            wrong_e = wrong_e & (gray | direct)  # requests below 0.80 are never served from cache
            cal_t, test_t = idx < half, idx >= half
            key = f"{ds}/{tag}"
            res = {"n_eval": int(len(idx)), "n_cal_temporal": int(cal_t.sum()), "n_test_temporal": int(test_t.sum()),
                   "test_unverified_direct_error": float(np.mean(wrong_e[test_t] & direct[test_t])),
                   "test_direct_precision": float(1 - wrong_e[test_t & direct].mean()),
                   "test_direct_auc": None, "R": {}, "T": {}}
            from sklearn.metrics import roc_auc_score
            m = test_t & direct
            res["test_direct_auc"] = float(roc_auc_score(~wrong_e[m], sc_e[m]))
            m = test_t & gray
            res["test_gray_auc"] = float(roc_auc_score(~wrong_e[m], sc_e[m]))
            rng = np.random.default_rng(0)
            splits = []
            for _ in range(M_RANDOM):
                perm = rng.permutation(len(idx))
                c = np.zeros(len(idx), bool)
                c[perm[: len(idx) // 2]] = True
                splits.append(c)
            for a in ALPHAS:
                for pol, f in policies:
                    name = pol if f is None else f"{pol}_f{f}"
                    runs = [evaluate(pol, c, ~c, sc_e, wrong_e, gray, direct, a, f or 0.5) for c in splits]
                    agg = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
                    agg["p_exceed"] = float(np.mean([r["risk"] > a for r in runs]))
                    res["R"][f"{a}/{name}"] = agg
                    t = evaluate(pol, cal_t, test_t, sc_e, wrong_e, gray, direct, a, f or 0.5)
                    t["ratio_to_alpha"] = t["risk"] / a
                    res["T"][f"{a}/{name}"] = t
            out[key] = res
            print(key, {k: res[k] for k in res if k not in ("R", "T")}, flush=True)
            for k in res["R"]:
                r, t = res["R"][k], res["T"][k]
                print(f"  {k:28s} R: risk {r['risk']:.5f} (dir {r['risk_direct']:.5f}) p_exc {r['p_exceed']:.2f} "
                      f"hit {r['hit_rate']:.4f} | T: risk {t['risk']:.5f} x{t['ratio_to_alpha']:.2f} "
                      f"hit {t['hit_rate']:.4f} dir_hit {t['direct_hit_rate']:.4f} calls {t['verifier_calls']:.3f}", flush=True)
    (ROOT / f"results/enterprise_allhits_guarantee{'_' + MODEL if MODEL else ''}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
