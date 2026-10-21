"""RESEARCH_PROPOSAL.md 方向 29: D→C headroom pilot on the 26.18 human gray-zone labels.

Each annotated row is assigned to exactly one bucket:
  info  - human q1 in {partial, unsure}      (not decidable from I_obs)
  ver   - human q1 in {yes, no} and V != Y*  (verifier-limited)
  ok    - human q1 in {yes, no} and V == Y*
Proportions are reweighted from the 25/25 benchmark-label strata to the gray-zone base
rates; CIs come from a stratified bootstrap. Quora is excluded (no real answers).
"""
import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "results" / ".cache"
OUT = ROOT / "results" / "dc_headroom_pilot.json"
N_BOOT, SEED, TRAIN_FRACTION = 5000, 0, 0.7

DATA = {
    "lmarena": {
        "base_rate_correct": 0.766923363557322,
        "theta": -1.703418254852295,
        "ots": "lmarena__precomputed__n60000__cross_encoder_cross-encoder_ms-marco-MiniLM-L6-v2__lo0.8__hi0.97.scored.json",
        "ft": "lmarena__precomputed__n60000__cross_encoder_results_finetuned_verifier_model__lo0.8__hi0.97.scored.json",
    },
    "search_queries_corrected": {
        "base_rate_correct": 0.484144645340751,
        "theta": 6.840526103973389,
        "ots": "search_queries_corrected__precomputed__n150000__cross_encoder_cross-encoder_ms-marco-MiniLM-L6-v2__lo0.8__hi0.97.scored.json",
        "ft": "search_queries_corrected__precomputed__n150000__cross_encoder__root_workspace_finetuned_verifier_model_searchqueries_corrected__lo0.8__hi0.97.scored.json",
    },
}


def auc(scores, labels):
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0) + 0.5 * (diff == 0)).mean())


def weighted_shares(rows, base_rate_correct):
    """Bucket shares reweighted from the equal-size strata to the population."""
    out = {}
    for stratum, w in (("correct", base_rate_correct), ("wrong", 1 - base_rate_correct)):
        sub = [r for r in rows if r["stratum"] == stratum]
        for b in ("info", "ver", "ok"):
            out[b] = out.get(b, 0.0) + w * np.mean([r["bucket"] == b for r in sub])
    return out


def main():
    sample = {x["id"]: x for x in json.loads((ROOT / "results/gray_zone_annotation_sample.json").read_text())}
    labels = {Path(f).stem: json.loads(Path(f).read_text())["q1"]
              for f in glob.glob(str(ROOT / "results/gray_zone_annotation_labels/*.json"))}
    rng = np.random.default_rng(SEED)
    report = {}
    for ds, cfg in DATA.items():
        ots = json.loads((CACHE / cfg["ots"]).read_text())
        ft = json.loads((CACHE / cfg["ft"]).read_text())
        gray_order = sorted(int(k) for k in ots)  # gray-zone rows in stream order
        ft_test_start = gray_order[int(len(gray_order) * TRAIN_FRACTION)]
        rows = []
        for i, s in sample.items():
            if s["dataset"] != ds:
                continue
            h = labels[i]
            y = {"yes": 1, "no": 0}.get(h)
            v = ots[str(s["pos"])][0]
            accept = int(v >= cfg["theta"])
            bucket = "info" if y is None else ("ok" if accept == y else "ver")
            rows.append({"id": i, "stratum": s["stratum"], "human": h, "y": y, "v": v, "accept": accept,
                         "bucket": bucket, "v_ft": ft[str(s["pos"])][0], "ft_test": s["pos"] >= ft_test_start})
        shares = weighted_shares(rows, cfg["base_rate_correct"])
        boots = []
        strata = {k: [r for r in rows if r["stratum"] == k] for k in ("correct", "wrong")}
        for _ in range(N_BOOT):
            res = [strata[k][j] for k in strata for j in rng.integers(0, len(strata[k]), len(strata[k]))]
            boots.append(weighted_shares(res, cfg["base_rate_correct"]))
        ci = {b: [float(np.percentile([x[b] for x in boots], q)) for q in (2.5, 97.5)] for b in ("info", "ver", "ok")}
        ratio = [x["ver"] / x["info"] if x["info"] > 0 else float("inf") for x in boots]

        clear = [r for r in rows if r["y"] is not None]
        vs, ys = np.array([r["v"] for r in clear]), np.array([r["y"] for r in clear])
        # in-sample best threshold on the human labels: optimistic lower bound on threshold-limited error
        best_err = min(np.mean((vs >= t).astype(int) != ys) for t in np.concatenate([vs, [np.inf]]))
        ft_rows = [r for r in clear if r["ft_test"]]
        e_ver, e_info = shares["ver"], shares["info"]
        if e_ver >= 2 * e_info and ci["ver"][0] > 0.10:
            case = "A"
        elif e_info >= 2 * e_ver:
            case = "C"
        else:
            case = "B"
        report[ds] = {
            "n": len(rows), "n_clear": len(clear),
            "human_counts": {k: sum(r["human"] == k for r in rows) for k in ("yes", "no", "partial", "unsure")},
            "shares_reweighted": shares, "ci95": ci,
            "ver_over_info_ci95": [float(np.percentile(ratio, q)) for q in (2.5, 97.5)],
            "case": case,
            "unweighted_ver_rate_on_clear": float(np.mean([r["bucket"] == "ver" for r in clear])),
            "auc_ots_on_clear": auc(vs, ys),
            "best_threshold_error_on_clear_insample": float(best_err),
            "ft_test_rows_clear": len(ft_rows),
            "auc_ft_on_clear_test_rows": auc(np.array([r["v_ft"] for r in ft_rows]), np.array([r["y"] for r in ft_rows])) if ft_rows else None,
            "rows": rows,
        }
        print(f"{ds}: n={len(rows)} clear={len(clear)} info={e_info:.3f} {ci['info']} ver={e_ver:.3f} {ci['ver']} "
              f"ok={shares['ok']:.3f} -> Case {case}; AUC_ots(clear)={report[ds]['auc_ots_on_clear']:.3f} "
              f"best-thr err={best_err:.3f}; ft test rows={len(ft_rows)}")
    OUT.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
