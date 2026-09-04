"""Direction 15 v1, P2: is selective deferral ECONOMICALLY worth it?

Reads scripts/selective_deferral_experiment.py output and, for a grid of
(r, c_j), finds the cost-minimising deferral rate delta* for each signal and
checks whether it beats the two delta=0 baselines:
  - delta=0 @ Youden's J           (the curve's own delta=0 point)
  - delta=0 @ cost-optimal theta(r) (must-fix B: an already well-tuned single
                                     threshold -- so deferral is not just
                                     recovering a bad operating point)

    cost(r, c_j, delta) = r * error_rate(delta)
                        + (1 - hit_rate(delta))
                        + c_j * oracle_call_rate(delta)

r    = C_false_accept / C_miss     (§5.17)
c_j  = C_judge        / C_miss     (a judge call vs a fresh LLM generation)
oracle_call_rate = fraction of ALL test-scope requests that invoke the judge.

Note (interpreting): error_rate is insensitive to the judge's false-reject
rate eps_fr (a rejected gray-zone hit becomes a miss, not an error), so the
eps_fr cost shows up only through (1 - hit_rate). P1's kappa lives on the
error curve and barely moves with eps; P2 is where a realistic (high-eps_fr)
judge actually gets penalised.

Usage:
    python scripts/analyze_selective_deferral_cost.py \
        results/selective_deferral_lmarena_groupD.json:LmArena-D ... \
        --tau-low 0.8 --out results/selective_deferral_cost_summary.json
"""
import argparse
import json
from pathlib import Path

R_GRID = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0)
CJ_GRID = (0.3, 0.7)


def cost_curve(err, hit, ocr, r, c_j):
    return [r * e + (1.0 - h) + c_j * o for e, h, o in zip(err, hit, ocr)]


def analyse_split(sp: dict) -> dict:
    ocr = sp["oracle_call_rate_by_delta"]
    cbct = sp.get("curves_by_cost_theta", {})
    out = {"by_r_cj": {}}
    for r in R_GRID:
        bco_key = min(sp["baseline_cost_optimal"].keys(), key=lambda k: abs(float(k) - r))
        bco = sp["baseline_cost_optimal"][bco_key]
        # must-fix B: prefer the deferral band anchored at cost-optimal theta(r).
        r_key = min(cbct.keys(), key=lambda k: abs(float(k) - r)) if cbct else None
        curves = cbct[r_key] if r_key is not None else sp["curves"]
        anchored = "cost_optimal_theta" if r_key is not None else "youden_fallback"
        for c_j in CJ_GRID:
            e0 = curves["margin"]["error_rate"][0]
            h0 = curves["margin"]["hit_rate"][0]
            cost0_anchor = r * e0 + (1.0 - h0)                       # ocr=0 at delta=0
            cost0_bco = r * bco["error_rate"] + (1.0 - bco["hit_rate"])
            baseline = min(cost0_anchor, cost0_bco)

            rows = {}
            for sig in ("margin", "calibrated", "random", "oracle_informed"):
                c = curves[sig]
                cc = cost_curve(c["error_rate"], c["hit_rate"], ocr, r, c_j)
                i_star = min(range(len(cc)), key=lambda i: cc[i])
                rows[sig] = {
                    "delta_star_realized": c["realized_delta"][i_star],
                    "cost_at_star": cc[i_star],
                    "hit_at_star": c["hit_rate"][i_star],
                    "err_at_star": c["error_rate"][i_star],
                    "ocr_at_star": ocr[i_star],
                    "beats_baseline": cc[i_star] < baseline - 1e-9,
                    "improvement_vs_baseline": baseline - cc[i_star],
                    "defers_at_all": i_star > 0,
                }
            out["by_r_cj"][f"r={r},cj={c_j}"] = {
                "deferral_anchored_at": anchored,
                "cost0_anchor": cost0_anchor,
                "cost0_cost_optimal_theta": cost0_bco,
                "theta_cost_optimal_used": bco["theta"],
                "signals": rows,
            }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="+")
    ap.add_argument("--tau-low", default="0.8")
    ap.add_argument("--out", default="results/selective_deferral_cost_summary.json")
    args = ap.parse_args()

    summary = {}
    for spec in args.specs:
        path, label = spec.rsplit(":", 1)
        d = json.loads(Path(path).read_text())
        sp = d["splits"].get(f"tau_low={args.tau_low}")
        if sp is None:
            continue
        summary[label] = {
            "eps_oracle": d["eps_oracle"],
            "n_test_gz": sp["n_test_gz"], "n_test_scope": sp["n_test_scope"],
            **analyse_split(sp),
        }

    Path(args.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # compact console table
    print(f"\n{'dataset':14s} {'r':>5s} {'cj':>4s}  {'best signal @ delta*':28s} "
          f"{'cost*':>7s} {'base':>7s} {'win?':>5s}  {'delta*':>6s} {'hit*':>6s} {'err*':>6s}")
    for label, s in summary.items():
        for key, blk in s["by_r_cj"].items():
            r = key.split(",")[0].split("=")[1]
            cj = key.split(",")[1].split("=")[1]
            best = min(("margin", "calibrated", "random", "oracle_informed"),
                       key=lambda sg: blk["signals"][sg]["cost_at_star"])
            b = blk["signals"][best]
            base = min(blk["cost0_anchor"], blk["cost0_cost_optimal_theta"])
            win = "yes" if b["cost_at_star"] < base - 1e-9 else "no"
            print(f"{label:14s} {r:>5s} {cj:>4s}  {best:16s} d*={b['delta_star_realized']:.2f}      "
                  f"{b['cost_at_star']:7.4f} {base:7.4f} {win:>5s}  "
                  f"{b['delta_star_realized']:6.2f} {b['hit_at_star']:6.3f} {b['err_at_star']:6.4f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
