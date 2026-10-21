"""RESEARCH_PROPOSAL.md 方向 26.11 analysis: with a TTL on cache entries, is
the risk gap between a switch run and its constant-threshold control confined
to the first TTL requests after the switch (bounded memory), or does it
outlive the TTL (cascading memory)? Criteria are the preregistered ones."""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MARGIN = 1000


def seg(run, lo, hi, lam0):
    o = np.array(run["obs"], dtype=float)
    w = o[(o[:, 0] >= lo) & (o[:, 0] < hi)]
    x = (w[:, 1] > lam0) & ~w[:, 2].astype(bool)
    return x.mean(), len(w)


def diff(a, b, lo, hi, lam0):
    ra, na = seg(a, lo, hi, lam0)
    rb, nb = seg(b, lo, hi, lam0)
    se = max(np.sqrt(ra * (1 - ra) / na + rb * (1 - rb) / nb), 1e-4)
    return ra - rb, se, na, nb


def main():
    d = json.loads((ROOT / "results/history_decay_ttl_lmarena.json").read_text())
    n, half = d["n_records"], d["n_records"] // 2
    runs = {(r["ttl"], r["label"]): r for r in d["runs"]}
    out, bounded = {}, True
    for ttl in d["ttls"]:
        out[ttl] = {}
        print(f"\nTTL = {ttl}")
        for switch, ref, lam_own, name in (("A_then_B", "fixed_B", d["lamB"], "A->B vs B"),
                                           ("B_then_A", "fixed_A", d["lamA"], "B->A vs A")):
            for lab, lam0 in (("q0.95", d["lam95"]), ("own", lam_own)):
                db, sb, n1, _ = diff(runs[(ttl, switch)], runs[(ttl, ref)], half, half + ttl, lam0)
                da, sa, n2, _ = diff(runs[(ttl, switch)], runs[(ttl, ref)], half + ttl + MARGIN, n, lam0)
                before_sig = abs(db) > 2 * sb
                after_sig = abs(da) > 2 * sa
                cascade = before_sig and after_sig and np.sign(da) == np.sign(db)
                if before_sig and after_sig:
                    bounded = False
                out[ttl][f"{name} @ {lab}"] = {"D_before": db, "se_before": sb, "n_before": n1, "D_after": da,
                                               "se_after": sa, "n_after": n2, "before_sig": bool(before_sig),
                                               "after_sig": bool(after_sig), "cascade_same_sign": bool(cascade)}
                print(f"  {name} @ {lab:5s}: D_before {db:+.4f} (2se {2 * sb:.4f}, n {n1}) {'SIG' if before_sig else 'ns '} | "
                      f"D_after {da:+.4f} (2se {2 * sa:.4f}, n {n2}) {'SIG' if after_sig else 'ns '}"
                      f"{'  <- cascade' if cascade else ''}")
        live = {lab: runs[(ttl, lab)]["live_final"] for lab in ("fixed_A", "fixed_B", "A_then_B", "B_then_A")}
        print(f"  live entries at end: {live}")
    print(f"\nbounded memory (no significant D_after wherever D_before is significant): {bounded}")
    (ROOT / "results/history_decay_ttl_summary.json").write_text(json.dumps({"bounded_memory": bounded, "by_ttl": out}, indent=1))


if __name__ == "__main__":
    main()
