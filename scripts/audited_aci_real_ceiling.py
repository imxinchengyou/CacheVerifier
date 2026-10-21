"""Direction A, step 2.5 (AUDITED_ACI_PROOF_DRAFT.md §10 item 1): on REAL
gray-zone score caches, how much audit budget can the control variate
(Prop 1) and Neyman audit allocation (Prop 2) actually save?

Read-only: uses existing `.scored.json` / trace caches (and the comcast
scored JSON). No model calls, no GPU.

For each (dataset, verifier, alpha):
  1. w = 1 - would_be_correct over the full gray zone [0.80, 0.97).
  2. FORWARD-ONLY (deployable): g(s) = P(w=1 | s) via decreasing isotonic
     regression fit on the chronologically FIRST half; everything below is
     evaluated on the SECOND half only. A constant control variate
     c = mean w among first-half requests above the same threshold is the
     trivial baseline (centering) that any score-dependent f must beat.
  3. Static accept region on the evaluation half: the lowest threshold whose
     marginal risk mean(A*w) is <= alpha (the full-information oracle
     operating point the online algorithm tracks).
  4. Per-accepted-request variance term of the theorem,
         V/n_acc = mean_acc[ (1-p_i)/p_i * (w_i - f_i)^2 ],
     for five schemes at the same mean audit rate BUDGET among accepted:
       uniform p, f=0 | uniform p, f=c (constant) | uniform p, f=g |
       Neyman p~sqrt(g), f=0 | Neyman p~sqrt(g(1-g)), f=g
     and the audit budget each scheme needs to MATCH the uniform/f=0
     variance at BUDGET (the product-relevant number: fewer LLM calls for
     the same guarantee width).
"""
import json
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression

from cacheverifier.experiments.verified_sweep import load_match_trace, load_scored

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "results" / ".cache"
CE = "cross_encoder_cross-encoder_ms-marco-MiniLM-L6-v2"
ALPHAS = (0.02, 0.05, 0.10)
BUDGET = 0.10
P_MIN = 0.01

SOURCES = {
    ("lmarena", "D_offshelf"): ("lmarena__precomputed__n60000.trace.json",
                                f"lmarena__precomputed__n60000__{CE}__lo0.8__hi0.97.scored.json"),
    ("lmarena", "E_finetuned*"): ("lmarena__precomputed__n60000.trace.json",
                                  "lmarena__precomputed__n60000__cross_encoder_results_finetuned_verifier_model__lo0.8__hi0.97.scored.json"),
    ("search_queries_corrected", "D_offshelf"): (
        "search_queries_corrected__precomputed__n150000.trace.json",
        f"search_queries_corrected__precomputed__n150000__{CE}__lo0.8__hi0.97.scored.json"),
    ("search_queries_corrected", "E_finetuned*"): (
        "search_queries_corrected__precomputed__n150000.trace.json",
        "search_queries_corrected__precomputed__n150000__cross_encoder__root_workspace_finetuned_verifier_model_searchqueries_corrected__lo0.8__hi0.97.scored.json"),
    ("quora", "D_offshelf"): (
        "quora__sentence-transformer_sentence-transformers_all-MiniLM-L6-v2__n60000.trace.json",
        f"quora__sentence-transformer_sentence-transformers_all-MiniLM-L6-v2__n60000__{CE}__lo0.8__hi0.97.scored.json"),
}


def load_stream(key):
    if key == ("comcast", "D_offshelf"):
        d = json.loads((ROOT / "results" / "aci_real_production_drift_scored_comcast.json").read_text())
        order = np.argsort(d["positions"], kind="stable")
        return np.asarray(d["scores"])[order], 1 - np.asarray(d["labels"])[order]
    trace_f, scored_f = SOURCES[key]
    trace = load_match_trace(CACHE / trace_f)
    scored = load_scored(CACHE / scored_f)
    idx = sorted(i for i in scored if i < len(trace))
    s = np.array([scored[i].score for i in idx])
    w = np.array([0 if trace[i].would_be_correct else 1 for i in idx])
    return s, w


def forward_split(s, w):
    half = len(s) // 2
    iso = IsotonicRegression(increasing=False, y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(s[:half], w[:half])
    return (s[:half], w[:half]), (s[half:], w[half:], iso.predict(s[half:]))


def neyman(v, budget):
    """p = clip(lam*sqrt(v), P_MIN, 1) with mean(p) == budget (over accepted)."""
    rv = np.sqrt(np.maximum(v, 0.0))
    lo, hi = 0.0, 1e6
    for _ in range(200):
        lam = 0.5 * (lo + hi)
        if np.clip(lam * rv, P_MIN, 1.0).mean() > budget:
            hi = lam
        else:
            lo = lam
    return np.clip(lam * rv, P_MIN, 1.0)


def var_term(p, w, f):
    return float(np.mean((1 - p) / p * (w - f) ** 2))


def budget_to_match(target, make_p, w, f):
    """Smallest mean audit rate at which scheme `make_p(budget)` reaches
    variance <= target (bisection; variance is decreasing in budget)."""
    lo, hi = P_MIN, 1.0
    if var_term(make_p(hi), w, f) > target:
        return None
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if var_term(make_p(mid), w, f) <= target:
            hi = mid
        else:
            lo = mid
    return hi


def analyze(key):
    s_all, w_all = load_stream(key)
    (s_fit, w_fit), (s, w, g) = forward_split(s_all, w_all)
    out = {"n_gray_zone": int(len(s_all)), "n_eval_half": int(len(s)),
           "gray_zone_error_rate_fit_half": float(w_fit.mean()),
           "gray_zone_error_rate": float(w.mean()), "alphas": {}}
    order = np.argsort(-s, kind="stable")
    cum_risk = np.cumsum(w[order]) / len(s)  # marginal risk when accepting the top-k
    for alpha in ALPHAS:
        k = int(np.searchsorted(cum_risk, alpha, side="right"))  # largest k with risk <= alpha
        if k == 0:
            out["alphas"][str(alpha)] = {"note": "no feasible accept region"}
            continue
        acc = order[:k]
        wa, ga = w[acc].astype(float), g[acc]
        zero = np.zeros_like(ga)
        theta = s[acc].min()
        above = w_fit[s_fit >= theta]
        c = float(above.mean()) if len(above) else float(w_fit.mean())
        const = np.full_like(ga, c)
        base = var_term(np.full_like(ga, BUDGET), wa, zero)
        schemes = {
            "uniform_f0": (lambda b: np.full_like(ga, b), zero),
            "uniform_const": (lambda b: np.full_like(ga, b), const),
            "uniform_cv": (lambda b: np.full_like(ga, b), ga),
            "neyman_f0": (lambda b: neyman(ga, b), zero),
            "neyman_cv": (lambda b: neyman(ga * (1 - ga), b), ga),
        }
        res = {}
        for name, (make_p, f) in schemes.items():
            v = var_term(make_p(BUDGET), wa, f)
            need = budget_to_match(base, make_p, wa, f)
            res[name] = {
                "var_per_accepted": v,
                "var_ratio_vs_uniform_f0": v / base if base > 0 else None,
                "budget_needed_to_match_uniform_f0_at_10pct": need,
                "audit_savings": None if need is None else 1 - need / BUDGET,
            }
        out["alphas"][str(alpha)] = {
            "accept_rate": k / len(s),
            "const_cv_c_from_fit_half": c,
            "error_rate_among_accepted": float(wa.mean()),
            "max_g_on_accept": float(ga.max()),
            "prop1_ceiling_E[g(1-g)]/E[g]": float((ga * (1 - ga)).mean() / max(ga.mean(), 1e-12)),
            "prop2_ideal_ratio_(E sqrt g)^2/E g": float(np.sqrt(ga).mean() ** 2 / max(ga.mean(), 1e-12)),
            "frac_accepted_with_g_below_0.01": float((ga < 0.01).mean()),
            "schemes": res,
        }
    return out


def main():
    keys = list(SOURCES) + [("comcast", "D_offshelf")]
    results = {}
    for key in keys:
        name = f"{key[0]}/{key[1]}"
        r = analyze(key)
        results[name] = r
        print(f"\n== {name}  n={r['n_gray_zone']}  gray-zone error rate fit/eval="
              f"{r['gray_zone_error_rate_fit_half']:.3f}/{r['gray_zone_error_rate']:.3f}")
        for a, x in r["alphas"].items():
            if "schemes" not in x:
                print(f"  alpha={a}: {x['note']}")
                continue
            sc = x["schemes"]
            fmt = lambda n: (f"{sc[n]['var_ratio_vs_uniform_f0']:.2f}/"
                             + ("n/a" if sc[n]["audit_savings"] is None else f"{sc[n]['audit_savings']:+.0%}"))
            print(f"  alpha={a}: accept={x['accept_rate']:.3f} err|acc={x['error_rate_among_accepted']:.3f} "
                  f"max_g={x['max_g_on_accept']:.3f} g<0.01={x['frac_accepted_with_g_below_0.01']:.0%} | "
                  f"var-ratio/savings: const {fmt('uniform_const')}  CV(g) {fmt('uniform_cv')}  "
                  f"Neyman {fmt('neyman_f0')}  both {fmt('neyman_cv')}")
    (ROOT / "results" / "audited_aci_real_ceiling.json").write_text(json.dumps({
        "config": {"alphas": ALPHAS, "budget": BUDGET, "p_min": P_MIN,
                   "g_estimator": "decreasing isotonic fit on chronological first half; evaluated on second half only",
                   "note": "E_finetuned* scores cover the whole gray zone incl. the fine-tune training split (in-sample optimism)"},
        "results": results,
    }, indent=2), encoding="utf-8")
    print("\nwrote results/audited_aci_real_ceiling.json")


if __name__ == "__main__":
    main()
