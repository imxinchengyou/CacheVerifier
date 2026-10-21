"""RESEARCH_PROPOSAL.md 方向 26.13 analysis: after alpha is loosened from 0.01
to 0.03 at the midpoint, does the current sliding-window CRC exceed the new
target in the transition window, beyond both noise and its own steady state
(constant alpha = 0.03)? Criteria are the preregistered ones."""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ALPHA_NEW, TRANSITION, SEG = 0.03, 10000, 3000


def risk(run, lo, hi):
    o = np.array([[p, a, c] for p, a, c in run["obs"] if lo <= p < hi], dtype=float)
    if len(o) == 0:
        return float("nan"), 0.0, 0
    x = (o[:, 1] == 1) & (o[:, 2] == 0)
    r = x.mean()
    return r, np.sqrt(max(r * (1 - r), 1e-12) / len(x)), len(x)


def main(ds):
    d = json.loads((ROOT / f"results/loosening_failure_{ds}.json").read_text())
    n, half = d["n_records"], d["n_records"] // 2
    runs = {r["label"]: r for r in d["runs"]}
    res = {}
    print(f"\n== {ds}")
    for src in ("S2", "S0"):
        sw, hi_, lo_ = runs[f"{src}_switch"], runs[f"{src}_fixed_hi"], runs[f"{src}_fixed_lo"]
        rs, ss, ns = risk(sw, half, half + TRANSITION)
        rh, sh, nh = risk(hi_, half, half + TRANSITION)
        sd = np.hypot(ss, sh)
        c1 = rs > ALPHA_NEW + 2 * ss
        c2 = (rs - rh) > 2 * sd
        segs = []
        for lo in range(half, n, SEG):
            a, _, _ = risk(sw, lo, lo + SEG)
            b, _, _ = risk(hi_, lo, lo + SEG)
            segs.append((lo - half, round(a, 4), round(b, 4)))
        thr_sw = [t for i, t in sw["thresholds"] if half <= i < half + TRANSITION]
        thr_hi = [t for i, t in hi_["thresholds"] if half <= i < half + TRANSITION]
        res[src] = {"switch_risk": rs, "switch_se": ss, "switch_n": ns, "fixed_hi_risk": rh, "fixed_hi_se": sh,
                    "crit1_exceeds_alpha": bool(c1), "crit2_exceeds_steady_state": bool(c2), "segments": segs,
                    "whole_second_half": [risk(sw, half, n)[0], risk(hi_, half, n)[0], risk(lo_, half, n)[0]],
                    "median_threshold_transition": [float(np.median(thr_sw)) if thr_sw else None,
                                                    float(np.median(thr_hi)) if thr_hi else None],
                    "store": [sw["store"], hi_["store"], lo_["store"]]}
        print(f"  {src}: transition window (first {TRANSITION} requests after switch)")
        print(f"      switch risk {rs:.4f} (se {ss:.4f}, n {ns}) | fixed_hi {rh:.4f} (se {sh:.4f}, n {nh})")
        print(f"      crit1 (> 0.03 + 2se): {c1} | crit2 (switch - fixed_hi > 2se): {c2}  [diff {rs - rh:+.4f}, 2se {2 * sd:.4f}]")
        print(f"      median threshold in transition: switch {res[src]['median_threshold_transition'][0]} vs fixed_hi {res[src]['median_threshold_transition'][1]}")
        print(f"      segments (requests after switch, switch risk, fixed_hi risk): {segs}")
        print(f"      whole second half risk: switch {res[src]['whole_second_half'][0]:.4f} fixed_hi {res[src]['whole_second_half'][1]:.4f} fixed_lo {res[src]['whole_second_half'][2]:.4f}; store {res[src]['store']}")
    return res


if __name__ == "__main__":
    dss = sys.argv[1:] or ["lmarena", "quora"]
    out = {ds: main(ds) for ds in dss}
    lm, qu = out.get("lmarena", {}).get("S2"), out.get("quora", {}).get("S2")
    if lm and qu:
        verdict = lm["crit1_exceeds_alpha"] and lm["crit2_exceeds_steady_state"] and not (qu["crit1_exceeds_alpha"] and qu["crit2_exceeds_steady_state"])
        print(f"\nPREREGISTERED VERDICT (LmArena S2, Quora negative control): current method fails due to statefulness = {verdict}")
        out["verdict_current_method_fails"] = verdict
    (ROOT / "results/loosening_failure_summary.json").write_text(json.dumps(out, indent=1))
