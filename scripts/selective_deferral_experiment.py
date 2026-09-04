"""Direction 15 (RESEARCH_PROPOSAL.md §10) v1: does verifier-score proximity
to the decision threshold predict error propensity better than random
selection -- i.e. is there a usable *selective* structure in the verifier's
uncertainty, such that deferring a small fraction of gray-zone requests to an
oracle captures a large share of the oracle's risk reduction?

Three-way gate in the gray zone: ACCEPT (score >= theta) / REJECT (miss) /
DEFER (band around theta -> oracle judges). delta=0 is Group D/E; delta=1
(whole gray zone deferred) is Group C (oracle). This sweeps the deferral rate
delta and produces the risk-vs-deferral frontier for several deferral
signals.

Pure post-hoc reanalysis of the cached match trace + gray-zone scores from
`cacheverifier.experiments.run_verified` -- no ANN index, no verifier model,
no API calls. Oracle imperfection enters ANALYTICALLY as an expectation:

    E[hit | DEFER, label y] = (1 - eps_fr) if y == 1 else eps_fa
    E[err | DEFER, label y] = eps_fa       if y == 0 else 0

so every number is an exact expectation over the oracle's error process; the
only sampling uncertainty is over which requests land in the test split
(bootstrap) and over eps itself (Beta posterior from the ~200 judge calls
measured by scripts/measure_oracle_judge_accuracy.py).

Locked v1 protocol (RESEARCH_PROPOSAL.md "方向 15 严谨性修正" A-E):
  A. every curve computed on the SAME test-record subset.
  B. delta=0 baseline uses the COST-OPTIMAL single threshold per r (not
     Youden's J -- §5.17). P1 (structure) uses Youden's J as a neutral
     reference theta; P2 (economics) uses cost-optimal theta(r).
  C. eps_oracle measured stratified by |s - theta| (near/far); near dominates
     the deferred set and drives the gate.
  E. narrow RQ: "does threshold-relative score predict error propensity
     better than random?", NOT "does the verifier know when it doesn't know".

Primary statistic P1 (preregistered), per dataset:
    kappa = (A_signal - A_random) / (A_oracle_informed - A_random)  in [0, 1]
A_* = area under that signal's error_rate-vs-realized-delta curve over
delta in [0, 0.30] (delta = fraction of the GRAY ZONE deferred), interpolated
to a common grid. kappa ~ 0 => no usable selective structure (clean negative
result). kappa ~ 1 => the score already pinpoints its own errors (suspicious:
may mean the operating point, not the gate, is the problem).

Usage:
    python scripts/selective_deferral_experiment.py --config configs/lmarena.yaml \
        --output results/selective_deferral_lmarena_groupD.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression

_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))  # np.trapz removed in numpy 2.0

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cacheverifier.config import load_dataset_config
from cacheverifier.data.loaders import load_jsonl
from cacheverifier.experiments.run_verified import CACHE_DIR, _slug
from cacheverifier.experiments.verified_sweep import load_match_trace, load_scored

DEFAULT_TAU_HIGH = 0.97
DEFAULT_TAU_LOW = 0.80
DEFAULT_DELTAS = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.0]
KAPPA_DELTA_CEILING = 0.30            # preregistered: only product-meaningful regime
KAPPA_GRID = np.linspace(0.0, KAPPA_DELTA_CEILING, 31)
SIGNALS = ("margin", "calibrated", "random", "oracle_informed")
COST_RATIOS = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0)   # r = C_false_accept / C_miss  (P2)


def youden_j_threshold(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """Youden's J -- verbatim rule from cacheverifier-service
    verifier_core/finetune.py::select_threshold. Neutral reference theta (P1)."""
    n_pos, n_neg = int((labels == 1).sum()), int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0 or (n_pos + n_neg) < 10:
        return None
    order = np.argsort(-scores)
    s_sorted, y_sorted = scores[order], labels[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    j = tp / n_pos - fp / n_neg
    return float(s_sorted[int(np.argmax(j))])


def cost_optimal_threshold(scores: np.ndarray, labels: np.ndarray, r: float) -> float:
    """Single gray-zone threshold minimising the §5.17 gray-zone cost: a
    gray-zone request costs r if served-and-wrong, 1 if rejected (fresh LLM
    call), 0 if served-and-right. Mirrors cacheverifier-service's
    threshold_by_cost_ratio (k=0..n incl. reject-everything). Accept rule
    `score >= t` matches replay()."""
    order = np.argsort(scores)
    s, y = scores[order], labels[order]
    n = len(s)
    wrong = (y == 0).astype(np.int64)
    suffix_wrong = np.concatenate([np.cumsum(wrong[::-1])[::-1], [0]])   # len n+1
    rejected = np.arange(n + 1)
    cost = r * suffix_wrong + rejected
    k = int(np.argmin(cost))
    return float("inf") if k == n else float(s[k])


def area_under(realized: np.ndarray, risk: np.ndarray) -> float:
    """Trapezoid area of risk vs realized-delta over [0, ceiling], interpolated
    onto KAPPA_GRID."""
    order = np.argsort(realized)
    x, yv = realized[order], risk[order]
    keep = np.concatenate([[True], np.diff(x) > 1e-12])
    x, yv = x[keep], yv[keep]
    if x[0] > 1e-12:
        x = np.concatenate([[0.0], x])
        yv = np.concatenate([[yv[0]], yv])
    return float(_trapz(np.interp(KAPPA_GRID, x, yv), KAPPA_GRID))


class Split:
    """Fixed per-(dataset, tau_low) structure. All the arrays the sweep needs,
    precomputed so each (signal, delta, eps, bootstrap) step is vectorised."""

    def __init__(self, trace, scored: dict[int, float], tau_low: float, tau_high: float):
        gz_idx = [
            i for i, t in enumerate(trace)
            if t.similarity is not None and tau_low <= t.similarity < tau_high and i in scored
        ]
        half = len(gz_idx) // 2
        self.calib_idx, self.test_idx = gz_idx[:half], gz_idx[half:]
        self.ok = len(self.calib_idx) >= 10 and len(self.test_idx) >= 10
        if not self.ok:
            return

        self.calib_scores = np.array([scored[i] for i in self.calib_idx])
        self.calib_labels = np.array([1 if trace[i].would_be_correct else 0 for i in self.calib_idx])
        self.test_scores = np.array([scored[i] for i in self.test_idx])
        self.test_labels = np.array([1 if trace[i].would_be_correct else 0 for i in self.test_idx])
        self.n_test_gz = len(self.test_idx)

        self.theta_youden = youden_j_threshold(self.calib_scores, self.calib_labels)
        self.ok = self.theta_youden is not None
        if not self.ok:
            return
        self.theta_cost = {r: cost_optimal_threshold(self.calib_scores, self.calib_labels, r)
                           for r in COST_RATIOS}

        self._iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self._iso.fit(self.calib_scores, self.calib_labels)

        # test SCOPE = every non-gray-zone request (whole stream) + test-half
        # gray-zone requests. Same denominator convention as
        # threshold_calibration_ablation.py / run_verified.py.
        test_set = set(self.test_idx)
        scope_idx = [
            i for i, t in enumerate(trace)
            if t.similarity is None
            or not (tau_low <= t.similarity < tau_high)
            or i in test_set
        ]
        self.n_scope = len(scope_idx)

        sim = np.array([trace[i].similarity if trace[i].similarity is not None else -1.0 for i in scope_idx])
        wbc = np.array([1 if trace[i].would_be_correct else 0 for i in scope_idx])
        is_gz = np.array([i in test_set for i in scope_idx])
        self.is_gz = is_gz
        # shortcut branch (fixed, threshold-independent): direct hit iff sim >= tau_high.
        self.shortcut_hit = np.where(~is_gz & (sim >= tau_high), 1.0, 0.0)
        self.shortcut_err = np.where(~is_gz & (sim >= tau_high) & (wbc == 0), 1.0, 0.0)

        # map each scope position that is gz -> its index in test_idx order
        pos_in_test = {ti: j for j, ti in enumerate(self.test_idx)}
        self.gz_scope_mask = is_gz
        self.gz_test_j = np.array([pos_in_test[i] for i in scope_idx if i in test_set])
        # scope positions that are gz, in scope order
        self.gz_scope_pos = np.where(is_gz)[0]

    # ---- deferral masks (over test_idx order), fixed given the split --------

    def deferral_mask(self, signal: str, target_delta: float, theta: float) -> np.ndarray:
        if target_delta <= 0.0:
            return np.zeros(self.n_test_gz, dtype=bool)
        if target_delta >= 1.0:
            return np.ones(self.n_test_gz, dtype=bool)
        if signal in ("margin", "calibrated"):
            if signal == "margin":
                cvar = np.abs(self.calib_scores - theta)
                tvar = np.abs(self.test_scores - theta)
            else:
                cvar = np.abs(self._iso.predict(self.calib_scores) - 0.5)
                tvar = np.abs(self._iso.predict(self.test_scores) - 0.5)
            w = float(np.quantile(cvar, target_delta))          # honest: width from calib
            return tvar <= w
        if signal == "oracle_informed":                          # cheat anchor: top-k
            served = self.test_scores >= theta
            fa = served & (self.test_labels == 0)                # deferring kills an error
            fr = (~served) & (self.test_labels == 1)             # deferring: miss -> hit
            pr = fa * 2.0 + fr * 1.0 - 1e-6 * np.abs(self.test_scores - theta)
            k = int(round(target_delta * self.n_test_gz))
            m = np.zeros(self.n_test_gz, dtype=bool)
            if k > 0:
                m[np.argsort(-pr)[:k]] = True
            return m
        raise ValueError(signal)

    # ---- vectorised metric assembly --------------------------------------

    def curve(self, signal: str, deltas: list[float], theta: float,
              eps_fa: float, eps_fr: float, resample: np.ndarray | None) -> dict:
        """error_rate / hit_rate / realized_delta vs delta, all end-to-end over
        the test scope. `resample` is scope-position indices for bootstrap."""
        y = self.test_labels
        v_hit_gz = (self.test_scores >= theta).astype(np.float64)
        v_err_gz = ((v_hit_gz == 1.0) & (y == 0)).astype(np.float64)
        o_hit_gz = np.where(y == 1, 1.0 - eps_fr, eps_fa)
        o_err_gz = np.where(y == 0, eps_fa, 0.0)

        realized, hr, er = [], [], []
        for d in deltas:
            hit = self.shortcut_hit.copy()
            err = self.shortcut_err.copy()
            if signal == "random":
                f = d
                hit[self.gz_scope_pos] = (1 - f) * v_hit_gz[self.gz_test_j] + f * o_hit_gz[self.gz_test_j]
                err[self.gz_scope_pos] = (1 - f) * v_err_gz[self.gz_test_j] + f * o_err_gz[self.gz_test_j]
                realized_d = d
            else:
                mask = self.deferral_mask(signal, d, theta)          # over test_idx order
                mask_scope = mask[self.gz_test_j]                    # over gz_scope_pos order
                hit[self.gz_scope_pos] = np.where(
                    mask_scope, o_hit_gz[self.gz_test_j], v_hit_gz[self.gz_test_j])
                err[self.gz_scope_pos] = np.where(
                    mask_scope, o_err_gz[self.gz_test_j], v_err_gz[self.gz_test_j])
                realized_d = float(mask.sum()) / self.n_test_gz
            if resample is not None:
                hit, err = hit[resample], err[resample]
                # realized delta within the resample (design: report realized, not nominal)
                gz_r = self.is_gz[resample]
                if signal != "random" and gz_r.sum() > 0:
                    deferred_r = np.zeros(self.n_scope, dtype=bool)
                    deferred_r[self.gz_scope_pos] = mask_scope
                    realized_d = float(deferred_r[resample].sum()) / float(gz_r.sum())
            realized.append(realized_d)
            hr.append(float(hit.mean()))
            er.append(float(err.mean()))
        return {"realized_delta": realized, "hit_rate": hr, "error_rate": er}

    def baseline_at(self, theta: float, eps_fa: float, eps_fr: float) -> dict:
        """delta=0 (no deferral), single threshold `theta`. Used for the
        cost-optimal-threshold baseline (must-fix B) so 'does deferral help' is
        measured against an already well-tuned single-threshold policy."""
        c = self.curve("margin", [0.0], theta, eps_fa, eps_fr, None)
        return {"theta": theta, "hit_rate": c["hit_rate"][0], "error_rate": c["error_rate"][0]}

    def evaluate(self, deltas: list[float], theta: float, eps_fa: float, eps_fr: float,
                 resample: np.ndarray | None = None) -> dict:
        curves = {s: self.curve(s, deltas, theta, eps_fa, eps_fr, resample) for s in SIGNALS}
        a_rand = area_under(np.array(curves["random"]["realized_delta"]),
                            np.array(curves["random"]["error_rate"]))
        a_oi = area_under(np.array(curves["oracle_informed"]["realized_delta"]),
                          np.array(curves["oracle_informed"]["error_rate"]))
        denom = a_oi - a_rand
        kappa = {}
        for s in ("margin", "calibrated"):
            a_s = area_under(np.array(curves[s]["realized_delta"]),
                             np.array(curves[s]["error_rate"]))
            kappa[s] = float((a_s - a_rand) / denom) if abs(denom) > 1e-12 else float("nan")
        return {"curves": curves, "area_random": a_rand, "area_oracle_informed": a_oi, "kappa": kappa}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--scored-cache", default=None)
    ap.add_argument("--verifier-label", default="groupD")
    ap.add_argument("--tau-high", type=float, default=DEFAULT_TAU_HIGH)
    ap.add_argument("--tau-low", type=float, default=DEFAULT_TAU_LOW)
    ap.add_argument("--tau-low-secondary", default="0.86,0.92")
    ap.add_argument("--eps-fa-near", type=float, default=0.0)
    ap.add_argument("--eps-fr-near", type=float, default=0.0)
    ap.add_argument("--eps-fa-far", type=float, default=0.0)
    ap.add_argument("--eps-fr-far", type=float, default=0.0)
    ap.add_argument("--eps-n-near", type=int, default=100)
    ap.add_argument("--eps-n-far", type=int, default=100)
    ap.add_argument("--bootstrap", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    def log(m: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

    cfg = load_dataset_config(args.config)
    dataset_path = Path(cfg.processed_path)
    records = load_jsonl(dataset_path)
    if cfg.max_samples is not None:
        records = records[: cfg.max_samples]
    n_records = len(records)
    embedder_key = (cfg.embedder if cfg.embedder != "sentence-transformer"
                    else f"sentence-transformer_{_slug(cfg.embedder_model)}")
    trace_path = CACHE_DIR / f"{_slug(dataset_path.stem)}__{embedder_key}__n{n_records}.trace.json"
    if not trace_path.exists():
        raise SystemExit(f"match trace not cached: {trace_path}")
    trace = load_match_trace(trace_path)
    log(f"loaded trace {trace_path.name} ({len(trace)} rows)")

    if args.scored_cache:
        scored_path = Path(args.scored_cache)
    else:
        vk = f"cross_encoder_{_slug('cross-encoder/ms-marco-MiniLM-L6-v2')}"
        scored_path = (CACHE_DIR / f"{_slug(dataset_path.stem)}__{embedder_key}__n{n_records}__{vk}"
                       f"__lo{args.tau_low}__hi{args.tau_high}.scored.json")
    if not scored_path.exists():
        raise SystemExit(f"gray-zone scores not cached: {scored_path}")
    scored = {i: sc.score for i, sc in load_scored(scored_path).items()}
    log(f"loaded scores {scored_path.name} ({len(scored)} gray-zone candidates)")

    rng = np.random.default_rng(args.seed)
    k_fa = round(args.eps_fa_near * args.eps_n_near)
    k_fr = round(args.eps_fr_near * args.eps_n_near)

    def eps_draw() -> tuple[float, float]:
        return (float(rng.beta(k_fa + 1, args.eps_n_near - k_fa + 1)),
                float(rng.beta(k_fr + 1, args.eps_n_near - k_fr + 1)))

    tau_lows = [("primary", args.tau_low)] + [
        ("secondary", float(x)) for x in args.tau_low_secondary.split(",") if x.strip()
    ]
    out: dict = {
        "config": args.config, "dataset": cfg.name, "verifier_label": args.verifier_label,
        "scored_cache": scored_path.name, "n_records": n_records, "tau_high": args.tau_high,
        "eps_oracle": {"fa_near": args.eps_fa_near, "fr_near": args.eps_fr_near,
                       "fa_far": args.eps_fa_far, "fr_far": args.eps_fr_far,
                       "n_near": args.eps_n_near, "n_far": args.eps_n_far},
        "deltas_nominal": DEFAULT_DELTAS, "kappa_delta_ceiling": KAPPA_DELTA_CEILING,
        "splits": {},
    }

    for tier, tau_low in tau_lows:
        log(f"--- tau_low={tau_low} ({tier}) ---")
        sp = Split(trace, scored, tau_low, args.tau_high)
        if not sp.ok:
            log(f"  tau_low={tau_low}: split too small / single-class, skipping")
            continue
        theta = sp.theta_youden
        point = sp.evaluate(DEFAULT_DELTAS, theta, args.eps_fa_near, args.eps_fr_near)
        baseline_cost_opt = {
            str(r): sp.baseline_at(sp.theta_cost[r], args.eps_fa_near, args.eps_fr_near)
            for r in COST_RATIOS
        }
        oracle_call_rate_by_delta = [  # fraction of ALL test-scope requests that call the judge
            rd * sp.n_test_gz / sp.n_scope for rd in point["curves"]["margin"]["realized_delta"]
        ]
        # must-fix B, done properly: for the economic (P2) question, anchor the
        # deferral band at the COST-OPTIMAL theta(r), not Youden's J -- so
        # "does deferral help" is not confounded by Youden's J being a poor
        # operating point at that r.
        curves_by_cost_theta = {}
        for r in COST_RATIOS:
            tc = sp.theta_cost[r]
            if tc == float("inf"):        # reject-everything: deferral band undefined
                continue
            curves_by_cost_theta[str(r)] = {
                s: sp.curve(s, DEFAULT_DELTAS, tc, args.eps_fa_near, args.eps_fr_near, None)
                for s in SIGNALS
            }

        boot = {"margin": [], "calibrated": []}
        b_hr0, b_er0 = [], []
        for _ in range(args.bootstrap):
            ridx = rng.integers(0, sp.n_scope, size=sp.n_scope)
            fa, fr = eps_draw()
            bp = sp.evaluate(DEFAULT_DELTAS, theta, fa, fr, resample=ridx)
            for s in ("margin", "calibrated"):
                boot[s].append(bp["kappa"][s])
            b_hr0.append(bp["curves"]["margin"]["hit_rate"][0])
            b_er0.append(bp["curves"]["margin"]["error_rate"][0])

        def ci(v: list[float]) -> list[float]:
            a = np.array([x for x in v if np.isfinite(x)])
            return [float("nan"), float("nan")] if len(a) < 10 else \
                [float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))]

        out["splits"][f"tau_low={tau_low}"] = {
            "tier": tier,
            "n_calib_gz": len(sp.calib_idx), "n_test_gz": sp.n_test_gz, "n_test_scope": sp.n_scope,
            "test_gz_pos_rate": float(sp.test_labels.mean()),
            "theta_youden": theta,
            "theta_cost_optimal": {str(r): v for r, v in sp.theta_cost.items()},
            "baseline_cost_optimal": baseline_cost_opt,
            "oracle_call_rate_by_delta": oracle_call_rate_by_delta,
            "curves": point["curves"],
            "curves_by_cost_theta": curves_by_cost_theta,
            "area_random": point["area_random"], "area_oracle_informed": point["area_oracle_informed"],
            "kappa": point["kappa"],
            "kappa_ci": {s: ci(boot[s]) for s in ("margin", "calibrated")},
            "kappa_boot_median": {s: (float(np.median([x for x in boot[s] if np.isfinite(x)]))
                                      if any(np.isfinite(boot[s])) else float("nan"))
                                  for s in ("margin", "calibrated")},
            "delta0_hit_rate_ci": ci(b_hr0), "delta0_error_rate_ci": ci(b_er0),
        }
        km, kc = point["kappa"]["margin"], point["kappa"]["calibrated"]
        log(f"  kappa margin={km:.3f} {out['splits'][f'tau_low={tau_low}']['kappa_ci']['margin']}  "
            f"calibrated={kc:.3f} {out['splits'][f'tau_low={tau_low}']['kappa_ci']['calibrated']}")
        c = point["curves"]
        for s in SIGNALS:
            row = "  ".join(f"{rd:.2f}->{er:.4f}" for rd, er in
                            zip(c[s]["realized_delta"], c[s]["error_rate"]))
            log(f"    {s:16s} err@delta: {row}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
    log(f"wrote {args.output}")


if __name__ == "__main__":
    main()
