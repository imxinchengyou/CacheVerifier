"""Do CacheVerifier's formal guarantees hold on a real, time-stamped support stream?

Reuses the enterprise comparison caches (scripts/enterprise_compare_*.py): the fine-tuned
verifier's gray-zone scores and the DeepSeek judge labels (wrong = not YES). Only gray-zone
pairs after the fine-tuning window (gray > train_until) are used, so no score was trained
on its own label. Four checks, per dataset and label set:

  A  Protocol R: CRC on uniform random 50/50 splits (M=300). The theorem applies here;
     mean realized risk should sit at alpha.
  B  Protocol T: CRC fitted on the held-out calibration part, applied to the test half
     (real time order). The theorem does not apply; reports realized risk overall and per
     time window.
  C  Audited ACI (PAPER_B section 4.5) run online over the test half with audit rate p:
     realized marginal risk, Theorem bound at delta=0.05, hit rate, extra LLM calls.
  E  High-probability version (Learn-then-Test style, one-sided Clopper-Pearson UCB at
     delta=0.1): the loosest threshold whose upper confidence bound on calibration risk is
     <= alpha. Holds per deployment with prob >= 1-delta instead of only in expectation.
  D  What the guarantee leaves out: the direct-hit zone (sim >= 0.97) is served without
     verification, so its wrong-reply rate adds to the per-request error unchecked.

Usage:
    .venv/bin/python -u scripts/enterprise_guarantee_feasibility.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import beta

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from cacheverifier.metrics.core import crc_select_threshold  # noqa: E402

CACHE = ROOT / "results/.cache"
DATASETS = ["twitter_amazon", "twitter_comcast"]
LO, HI = 0.80, 0.97
ALPHAS = [0.05, 0.02, 0.01]
LABEL_SETS = {"judge": "", "judge_thinklow": "_judge_thinklow"}
N_WINDOWS = 4
M_RANDOM = 300


def load_labels(ds: str, tag: str, n: int) -> np.ndarray:
    os.environ["JUDGE_TAG"] = tag
    import importlib
    import enterprise_compare_analyze as an
    importlib.reload(an)
    lab, _ = an.load_judge(ds, n)
    return lab


def ucb_select_threshold(scores: np.ndarray, wrong: np.ndarray, alpha: float, delta: float = 0.1) -> float:
    order = np.argsort(scores)
    ss, ww = scores[order], wrong[order].astype(int)
    n = len(ss)
    k = np.concatenate([np.cumsum(ww[::-1])[::-1], [0]])[np.searchsorted(ss, ss, side="right")]
    ucb = np.where(k >= n, 1.0, beta.ppf(1 - delta, k + 1, n - k))
    # fixed-sequence testing (Learn-then-Test): walk from the strictest threshold down and stop
    # at the first one whose UCB exceeds alpha; picking any feasible point would need a correction
    bad = np.where(ucb > alpha)[0]
    i = bad[-1] + 1 if len(bad) else 0
    return float(ss[i]) if i < n else float(ss[-1])


def risk_at(scores: np.ndarray, wrong: np.ndarray, lam: float) -> tuple[float, float]:
    acc = scores > lam
    return float(np.mean(wrong & acc)), float(np.mean(acc))


def audited_aci(s: np.ndarray, w: np.ndarray, alpha: float, p: float, gamma: float, seed: int,
                theta1: float = 0.5, theta_lo: float = 0.0, delta: float = 0.05) -> dict:
    rng = np.random.default_rng(seed)
    T = len(s)
    theta = theta1
    e = np.zeros(T)
    served_wrong = np.zeros(T)
    acc_n = audits = 0
    V = 0.0
    for t in range(T):
        a = s[t] >= theta
        z = a and rng.random() < p
        e[t] = a * w[t]
        served_wrong[t] = a * (not z) * w[t]
        ehat = (z / p) * w[t] if a else 0.0
        V += a * w[t] ** 2 * (1 - p) / p
        acc_n += a and not z
        audits += z
        theta = max(theta_lo, theta + gamma * (ehat - alpha))
    K = int(np.ceil(np.log2(T / p)))
    L = np.log((K + 1) / delta)
    bound = alpha + (1 - theta1 + gamma / p) / (gamma * T) + (2 * np.sqrt(max(V, 1) * L) + 2 / (3 * p) * L) / T
    return {"risk": float(e.mean()), "served_risk": float(served_wrong.mean()), "bound": float(bound),
            "hit_rate": acc_n / T, "audit_rate_of_gray": audits / T,
            "risk_second_half": float(e[T // 2:].mean())}


def main() -> None:
    out = {}
    for ds in DATASETS:
        base = np.load(CACHE / f"enterprise_{ds}.npz")
        sim, same = base["sim"], base["same"]
        n = len(sim)
        half = n // 2
        for tag, suffix in LABEL_SETS.items():
            lab = load_labels(ds, tag, n)
            v = np.load(CACHE / f"enterprise_{ds}_verifier{suffix}.npz")
            gray, sc, until = v["gray"], v["ft_judge"], int(v["train_until"])
            keep = gray > until
            g, s = gray[keep], sc[keep]
            wrong = ~lab[g]
            cal, test = g < half, g >= half
            key = f"{ds}/{tag}"
            r = {"n_eval_gray": int(keep.sum()), "n_cal": int(cal.sum()), "n_test": int(test.sum()),
                 "gray_error_rate_cal": float(wrong[cal].mean()), "gray_error_rate_test": float(wrong[test].mean()),
                 "gray_share_of_requests_test": float(test.sum() / (n - half))}
            # D: direct-hit zone, served without verification
            direct = np.where(np.nan_to_num(sim, nan=-1) >= HI)[0]
            direct_test = direct[direct >= half]
            dw = ~lab[direct_test]
            r["direct_hit_rate_test"] = float(len(direct_test) / (n - half))
            r["direct_hit_wrong_per_request_test"] = float(dw.sum() / (n - half))
            r["direct_hit_precision_test"] = float(1 - dw.mean()) if len(dw) else None
            r["A_random"], r["B_temporal"], r["C_audited_aci"] = {}, {}, {}
            rng = np.random.default_rng(0)
            for a in ALPHAS:
                risks, covs, urisks, ucovs = [], [], [], []
                for _ in range(M_RANDOM):
                    perm = rng.permutation(len(s))
                    c, t = perm[: len(s) // 2], perm[len(s) // 2:]
                    lam = crc_select_threshold(s[c], (~wrong[c]).astype(float), a)
                    rr, cc = risk_at(s[t], wrong[t], lam)
                    risks.append(rr)
                    covs.append(cc)
                    # the high-probability claim is about the population risk; the pool itself is the
                    # population a random split samples from, so score it on the whole pool
                    rr, cc = risk_at(s, wrong, ucb_select_threshold(s[c], wrong[c], a))
                    urisks.append(rr)
                    ucovs.append(cc)
                r["A_random"][str(a)] = {"mean_risk": float(np.mean(risks)), "p_exceed": float(np.mean(np.array(risks) > a)),
                                         "mean_accept_gray": float(np.mean(covs))}
                r.setdefault("E_high_prob", {})[str(a)] = {
                    "random_mean_risk": float(np.mean(urisks)), "random_p_exceed": float(np.mean(np.array(urisks) > a)),
                    "random_mean_accept_gray": float(np.mean(ucovs))}
                lam = crc_select_threshold(s[cal], (~wrong[cal]).astype(float), a)
                rr, cc = risk_at(s[test], wrong[test], lam)
                edges = np.linspace(half, n, N_WINDOWS + 1)
                win = []
                for k in range(N_WINDOWS):
                    m = (g >= edges[k]) & (g < edges[k + 1])
                    win.append(risk_at(s[m], wrong[m], lam)[0])
                ur, uc = risk_at(s[test], wrong[test], ucb_select_threshold(s[cal], wrong[cal], a))
                r["E_high_prob"][str(a)].update(temporal_risk=ur, temporal_accept_gray=uc)
                r["B_temporal"][str(a)] = {"risk": rr, "ratio_to_alpha": rr / a, "accept_gray": cc,
                                           "window_risks": win, "windows_over_alpha": int(sum(x > a for x in win))}
                # C: online audited ACI on the test stream, scores squashed to [0, 1]
                st = 1 / (1 + np.exp(-s[test]))
                wt = wrong[test].astype(float)
                for p in (0.05, 0.10, 0.20):
                    runs = [audited_aci(st, wt, a, p, gamma=0.005, seed=k) for k in range(50)]
                    agg = {k2: float(np.mean([x[k2] for x in runs])) for k2 in runs[0]}
                    agg["p_risk_over_bound"] = float(np.mean([x["risk"] > x["bound"] for x in runs]))
                    agg["p_risk_over_alpha"] = float(np.mean([x["risk"] > a for x in runs]))
                    r["C_audited_aci"][f"alpha={a},p={p}"] = agg
            out[key] = r
            print(key, json.dumps({k2: r[k2] for k2 in r if not isinstance(r[k2], dict)}), flush=True)
            for part in ("A_random", "B_temporal", "C_audited_aci", "E_high_prob"):
                for k2, x in r[part].items():
                    print(f"  {part} {k2}: " + ", ".join(f"{a2}={b:.4f}" if isinstance(b, float) else f"{a2}={b}"
                                                        for a2, b in x.items()), flush=True)
    (ROOT / "results/enterprise_guarantee_feasibility.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
