"""Enterprise comparison, step 5: replay every method and compute the six metrics.

Reads the outputs of enterprise_compare_{build,judge,gptcache,verifier}.py and
reports, on the test half (second half of each stream, in time order):
hit rate, wrong-hit rate, P50/P95 latency, cost per verification, total cost
per million requests, and accuracy across later time windows (drift).
Protocol: ENTERPRISE_COMPARISON.md.

Methods (all on the same top-1 neighbour from all-MiniLM-L6-v2):
  redis_static      sim >= tau                       default tau = 0.90 (RedisVL distance_threshold 0.1)
  gptcache_default  sim >= tau                       default tau = 0.60 (faiss L2^2 <= 0.8, threshold 0.8 x max_distance 4)
  gptcache_onnx     albert-duplicate score >= t      default t = 0.8
  vcache            AdaptiveThresholdPolicy(delta)   default delta = 0.02
  cv_stock          sim >= 0.97, or gray zone and stock cross-encoder score >= t   default t = 0.0
  cv_ft             same, with the verifier fine-tuned on the calibration half's labels

JUDGE_TAG (default "judge") picks the judge label file and the matching
enterprise_{ds}_verifier[_{tag}].npz; with another tag only the judge label set is
analysed and results go to results/enterprise_comparison_{ds}_{tag}.json.

Usage:
    [JUDGE_TAG=judge_thinklow] .venv/bin/python -u scripts/enterprise_compare_analyze.py
"""

import json
import os
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cacheverifier.cache.adaptive_threshold import AdaptiveThresholdPolicy  # noqa: E402

CACHE = ROOT / "results" / ".cache"
OUT = ROOT / "results"
DATASETS = ["twitter_amazon", "twitter_comcast"]
LO, HI, SIM_MIN = 0.80, 0.97, 0.6
ALPHAS = [0.01, 0.02, 0.05]
MAIN_ALPHA, R_MAIN, R_ALL = 0.02, 5, [1, 5, 20]
VCACHE_SEEDS = [0, 1, 2]
VCACHE_WORKERS = 10
VCACHE_FRONTIER_DELTAS = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
N_WINDOWS, N_BOOT, SEED = 4, 2000, 0
VCPU_USD_PER_HOUR = 0.045
JUDGE_TAG = os.environ.get("JUDGE_TAG", "judge")
SUFFIX = "" if JUDGE_TAG == "judge" else f"_{JUDGE_TAG}"
LABEL_SETS = ("judge", "same") if not SUFFIX else ("judge",)
USD_PER_CPU_MS = VCPU_USD_PER_HOUR / 3.6e6


# ---------- labels, costs, latency inputs ----------

def load_judge(ds: str, n: int) -> tuple[np.ndarray, dict]:
    lab = np.zeros(n, dtype=bool)
    seen = set()
    below, tokens = [], []
    for line in open(CACHE / f"enterprise_{ds}_{JUDGE_TAG}.jsonl"):
        r = json.loads(line)
        if r["verdict"] not in ("YES", "PARTIAL", "NO"):  # failed calls (ERROR, unparsed) are not labels
            continue
        if r["kind"] == "above":
            lab[r["i"]] = r["verdict"] == "YES"
            seen.add(r["i"])
        else:
            below.append(r["verdict"])
        if r.get("prompt_tokens") is not None:
            tokens.append((r["prompt_tokens"], r["completion_tokens"], r.get("prompt_cache_hit_tokens") or 0))
    sim = np.load(CACHE / f"enterprise_{ds}.npz")["sim"]
    missing = int(np.sum(np.nan_to_num(sim, nan=-1.0) >= 0.6)) - len(seen)
    if missing:
        raise SystemExit(f"{ds}: {missing} pairs >= 0.6 have no valid {JUDGE_TAG} verdict; finish enterprise_compare_judge.py first")
    check = {"n": len(below), "yes": sum(v == "YES" for v in below), "partial": sum(v == "PARTIAL" for v in below)}
    return lab, {"below_threshold_check": check, "tokens": tokens}


def deepseek_prices() -> tuple[float, float, float]:
    """Per-token prices fitted from the real billed calls in realworld_integration_test (cost = a*miss_in + b*out).

    That file holds tweet text and is not in the public repository; without it the
    values fitted from it (1,646 deepseek-v4-flash calls) are used."""
    path = ROOT / "realworld_integration_test/data/deepseek_costs.jsonl"
    if not path.exists():
        return 2.2e-07, 6.6e-07, 4.981476306196841e-05
    rows = [json.loads(line) for line in open(path)]
    X = np.array([[r["prompt_cache_miss_tokens"], r["completion_tokens"]] for r in rows], dtype=float)
    y = np.array([r["cost_usd"] for r in rows])
    a, b = np.linalg.lstsq(X, y, rcond=None)[0]
    return float(a), float(b), float(y.mean())


# ---------- metrics ----------

def rates(hit: np.ndarray, correct: np.ndarray) -> dict:
    n = len(hit)
    wrong = hit & ~correct
    return {"hit_rate": float(hit.mean()), "error_rate": float(wrong.sum() / n),
            "precision": float((hit & correct).sum() / hit.sum()) if hit.any() else float("nan"),
            "n": int(n), "hits": int(hit.sum()), "wrong_hits": int(wrong.sum())}


def boot_ci(hit: np.ndarray, correct: np.ndarray, rng: np.random.Generator) -> dict:
    n = len(hit)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    h = hit[idx]
    w = (hit & ~correct)[idx]
    hr, er = h.mean(1), w.mean(1)
    return {"hit_rate_ci": [float(np.percentile(hr, 2.5)), float(np.percentile(hr, 97.5))],
            "error_rate_ci": [float(np.percentile(er, 2.5)), float(np.percentile(er, 97.5))]}


def pick_param(params: list, hits_fn, correct: np.ndarray, mask: np.ndarray, alpha: float):
    """Highest calibration hit rate with calibration error rate <= alpha; else the most conservative param."""
    best, best_hr, most_cons, cons_hr = None, -1.0, None, 2.0
    for p in params:
        hit = hits_fn(p)[mask]
        hr = hit.mean()
        er = (hit & ~correct[mask]).mean()
        if er <= alpha and hr > best_hr:
            best, best_hr = p, hr
        if hr < cons_hr:
            most_cons, cons_hr = p, hr
    return best if best is not None else most_cons


# ---------- vCache replay ----------

class _Entry:
    __slots__ = ("query_id",)

    def __init__(self, qid: str) -> None:
        self.query_id = qid


class _Match:
    __slots__ = ("entry", "similarity")

    def __init__(self, qid: str, s: float) -> None:
        self.entry, self.similarity = _Entry(qid), s


def _vcache_job(args):
    ds, label_name, delta, seed = args
    z = np.load(CACHE / f"enterprise_{ds}.npz")
    sim, nn = np.nan_to_num(z["sim"], nan=-1.0), z["nn"]
    correct = z["same"] if label_name == "same" else load_judge(ds, len(sim))[0]
    return args, replay_vcache(sim, nn, correct, delta, seed)


def replay_vcache(sim: np.ndarray, nn: np.ndarray, correct: np.ndarray, delta: float, seed: int):
    random.seed(seed)
    policy = AdaptiveThresholdPolicy(target_error_rate=delta)
    n = len(sim)
    hit = np.zeros(n, dtype=bool)
    label_check = np.zeros(n, dtype=bool)  # explore misses on an entry: vCache compares answers to learn
    decide_ms = np.zeros(n, dtype=np.float32)
    for i in range(1, n):
        m = _Match(str(nn[i]), float(sim[i]))
        s = time.perf_counter()
        d = policy.decide(None, None, m)
        decide_ms[i] = (time.perf_counter() - s) * 1000
        hit[i] = d.action == "hit"
        if d.action == "miss":
            label_check[i] = True
        policy.observe(None, m, bool(correct[i]), d.action)
    return hit, label_check, decide_ms


# ---------- main ----------

def main() -> None:
    rng = np.random.default_rng(SEED)
    price_in, price_out, llm_cost = deepseek_prices()
    lat_common = json.loads((CACHE / "enterprise_latency.json").read_text())
    llm_lat = np.array(json.loads((ROOT / "results/oracle_judge_latency_deepseek.json").read_text())["raw_latencies_ms"], float)
    report = {"assumptions": {"vcpu_usd_per_hour": VCPU_USD_PER_HOUR, "llm_cost_per_call_usd": llm_cost,
                              "deepseek_price_per_input_token": price_in, "deepseek_price_per_output_token": price_out,
                              "llm_latency_source": "results/oracle_judge_latency_deepseek.json",
                              "main_alpha": MAIN_ALPHA, "r_values": R_ALL}}

    jobs = [(ds, lab, d, sd) for ds in DATASETS for lab in LABEL_SETS for d in VCACHE_FRONTIER_DELTAS
            for sd in (VCACHE_SEEDS if d in ALPHAS else VCACHE_SEEDS[:1])]
    t0 = time.time()
    with Pool(VCACHE_WORKERS) as pool:
        vruns = dict(pool.imap_unordered(_vcache_job, jobs))
    print(f"vCache: {len(jobs)} replays in {time.time() - t0:.0f}s", flush=True)

    for ds in DATASETS:
        z = np.load(CACHE / f"enterprise_{ds}.npz")
        sim, nn, same = z["sim"], z["nn"], z["same"]
        n = len(sim)
        half = n // 2
        sim0 = np.nan_to_num(sim, nan=-1.0)
        judge, jinfo = load_judge(ds, n)
        g = np.load(CACHE / f"enterprise_{ds}_gptcache.npz")
        gscore = np.nan_to_num(g["score"], nan=-1.0)
        v = np.load(CACHE / f"enterprise_{ds}_verifier{SUFFIX}.npz")
        gray, train_until = v["gray"], int(v["train_until"])
        in_gray = np.zeros(n, dtype=bool)
        in_gray[gray] = True

        def cv_scores(key):
            s = np.full(n, -np.inf, dtype=np.float32)
            s[gray] = v[key]
            return s

        jt = jinfo["tokens"]
        judge_call_cost = (float(np.mean([p - c for p, _, c in jt])) * price_in + float(np.mean([o for _, o, _ in jt])) * price_out) if jt else llm_cost
        lc = lat_common[ds]
        # all from one interleaved benchmark (enterprise_compare_latency.py)
        embed_ms, search_ms, ce_ms, glat, vdec_ms = (np.array(lc[k], float) for k in
                                                     ("embed", "search", "cross_encoder", "gptcache_onnx", "vcache_decide"))

        test = np.arange(n) >= half
        calib = ~test
        calib_ft = calib & (np.arange(n) > train_until)
        out_ds = {"n": n, "test_n": int(test.sum()), "judge_below_threshold_check": jinfo["below_threshold_check"],
                  "judge_call_cost_usd": judge_call_cost, "label_sets": {}}

        for label_name, correct in [(k, {"judge": judge, "same": same}[k]) for k in LABEL_SETS]:
            methods = {}
            tau_grid = [round(x, 3) for x in np.arange(0.50, 0.9951, 0.005)] + [1.01]
            methods["redis_static"] = dict(params=tau_grid, hits=lambda t: sim0 >= t, default=0.90, mask=calib, calls=None)
            methods["gptcache_default"] = dict(params=tau_grid, hits=lambda t: sim0 >= t, default=0.60, mask=calib, calls=None)
            t_grid = [round(x, 3) for x in np.arange(0.0, 1.0001, 0.01)] + [1.01]
            methods["gptcache_onnx"] = dict(params=t_grid, hits=lambda t: gscore >= t, default=0.8, mask=calib,
                                            calls=np.arange(n) >= 1)
            for key, vkey in (("cv_stock", "stock"), ("cv_ft", f"ft_{label_name}")):
                sc = cv_scores(vkey)
                fit_mask = calib if key == "cv_stock" else calib_ft
                grid = sorted(set(np.percentile(sc[gray][gray[:] < half] if key == "cv_stock" else sc[fit_mask & in_gray],
                                                np.arange(0, 100.5, 1)).tolist())) + [np.inf]
                if key == "cv_stock":
                    grid = sorted(set(grid + [0.0]))
                methods[key] = dict(params=grid, hits=(lambda sc_: (lambda t: (sim0 >= HI) | (in_gray & (sc_ >= t))))(sc),
                                    default=(0.0 if key == "cv_stock" else None), mask=fit_mask, calls=in_gray)

            res = {}
            for name, m in methods.items():
                r = {"frontier_test": [], "operating_points": {}}
                for p in m["params"]:
                    h = m["hits"](p)
                    r["frontier_test"].append({"param": float(p), **rates(h[test], correct[test])})
                chosen = {f"alpha_{a}": pick_param(m["params"], m["hits"], correct, m["mask"], a) for a in ALPHAS}
                if m["default"] is not None:
                    chosen["default"] = m["default"]
                for k, p in chosen.items():
                    h = m["hits"](p)
                    r["operating_points"][k] = {"param": float(p), **rates(h[test], correct[test]),
                                                **boot_ci(h[test], correct[test], rng),
                                                "call_rate": float(m["calls"][test].mean()) if m["calls"] is not None else 0.0,
                                                "_hit": h, "_calls": m["calls"]}
                res[name] = r

            # vCache: online, delta = alpha; report mean over seeds
            vc = {"frontier_test": [], "operating_points": {}}
            for delta in VCACHE_FRONTIER_DELTAS:
                seeds = VCACHE_SEEDS if delta in ALPHAS else VCACHE_SEEDS[:1]
                runs = [vruns[(ds, label_name, delta, s)] for s in seeds]
                per = [rates(h[test], correct[test]) for h, _, _ in runs]
                summary = {"param": delta, "seeds": len(seeds),
                           **{k: float(np.mean([p[k] for p in per])) for k in ("hit_rate", "error_rate", "precision")},
                           "hit_rate_range": [min(p["hit_rate"] for p in per), max(p["hit_rate"] for p in per)],
                           "error_rate_range": [min(p["error_rate"] for p in per), max(p["error_rate"] for p in per)]}
                vc["frontier_test"].append(summary)
                if delta in ALPHAS or delta == 0.02:
                    h, chk, dms = runs[0]
                    op = {**summary, **boot_ci(h[test], correct[test], rng),
                          "call_rate": float(chk[test].mean()), "_hit": h, "_calls": chk, "_decide_ms": dms}
                    for key in ([f"alpha_{delta}"] if delta in ALPHAS else []) + (["default"] if delta == 0.02 else []):
                        vc["operating_points"][key] = dict(op)
                print(f"  {ds}/{label_name} vcache delta={delta}: hit {summary['hit_rate']:.4f} err {summary['error_rate']:.4f}", flush=True)
            res["vcache"] = vc

            # latency, cost, drift for every operating point
            win_edges = np.linspace(half, n, N_WINDOWS + 1).astype(int)
            for name, r in res.items():
                for k, op in r["operating_points"].items():
                    h, calls = op.pop("_hit"), op.pop("_calls")
                    dms = op.pop("_decide_ms", None)
                    idx = np.where(test)[0]
                    total = rng.choice(embed_ms, len(idx)) + rng.choice(search_ms, len(idx))
                    if name == "gptcache_onnx":
                        total = total + rng.choice(glat, len(idx))
                        per_call_cpu_ms = float(glat.mean())
                    elif name in ("cv_stock", "cv_ft"):
                        total = total + np.where(calls[idx], rng.choice(ce_ms, len(idx)), 0.0)
                        per_call_cpu_ms = float(ce_ms.mean())
                    elif name == "vcache":
                        total = total + rng.choice(vdec_ms, len(idx))
                        per_call_cpu_ms = None
                    else:
                        per_call_cpu_ms = None
                    overhead = total.copy()
                    total = total + np.where(~h[idx], rng.choice(llm_lat, len(idx)), 0.0)
                    op["latency_ms"] = {"p50": float(np.percentile(total, 50)), "p95": float(np.percentile(total, 95)),
                                        "decision_overhead_p50": float(np.percentile(overhead, 50)),
                                        "decision_overhead_p95": float(np.percentile(overhead, 95))}
                    if name == "vcache":
                        per_check = judge_call_cost
                    elif per_call_cpu_ms is not None:
                        per_check = per_call_cpu_ms * USD_PER_CPU_MS
                    else:
                        per_check = 0.0
                    embed_cost = float(embed_ms.mean() + search_ms.mean()) * USD_PER_CPU_MS
                    miss = 1 - op["hit_rate"]
                    infra = 1e6 * (miss * llm_cost + op["call_rate"] * per_check + embed_cost)
                    op["cost"] = {"per_verification_usd": per_check, "verifications_per_million": op["call_rate"] * 1e6,
                                  "total_per_million_usd": infra,
                                  "wrong_answers_per_million": op["error_rate"] * 1e6,
                                  **{f"total_per_million_usd_r{rr}": infra + 1e6 * op["error_rate"] * rr * llm_cost for rr in R_ALL}}
                    op["drift_windows"] = []
                    for a, b in zip(win_edges[:-1], win_edges[1:]):
                        w = np.zeros(n, dtype=bool)
                        w[a:b] = True
                        op["drift_windows"].append(rates(h[w], correct[w]))
            gray_pos = {"calibration_half": float(correct[calib & in_gray].mean()),
                        "test_windows": [float(correct[(np.arange(n) >= a) & (np.arange(n) < b) & in_gray].mean())
                                         for a, b in zip(win_edges[:-1], win_edges[1:])]}
            out_ds["label_sets"][label_name] = {"methods": res, "gray_zone_positive_rate": gray_pos}
        report[ds] = out_ds
        (OUT / f"enterprise_comparison_{ds}{SUFFIX}.json").write_text(json.dumps(out_ds, indent=1))
        print_summary(ds, out_ds)
    (OUT / "enterprise_comparison_assumptions.json").write_text(json.dumps(report["assumptions"], indent=1))


def print_summary(ds: str, out_ds: dict) -> None:
    for label_name, block in out_ds["label_sets"].items():
        print(f"\n== {ds} | labels={label_name} | gray-zone positive rate {block['gray_zone_positive_rate']}")
        for key in ("default", f"alpha_{MAIN_ALPHA}"):
            print(f"-- operating point: {key}")
            for name, r in block["methods"].items():
                op = r["operating_points"].get(key)
                if not op:
                    continue
                c, l = op["cost"], op["latency_ms"]
                last = op["drift_windows"][-1]
                print(f"{name:17s} p={op['param']:<8.3f} hit {op['hit_rate']:.4f} err {op['error_rate']:.4f} "
                      f"prec {op['precision']:.3f} | p50 {l['p50']:.0f} p95 {l['p95']:.0f} ovh95 {l['decision_overhead_p95']:.1f}ms | "
                      f"$/verif {c['per_verification_usd']:.2e} $/1M {c['total_per_million_usd']:.2f} r5 {c['total_per_million_usd_r5']:.2f} | "
                      f"last-window prec {last['precision']:.3f} err {last['error_rate']:.4f}")


if __name__ == "__main__":
    main()
