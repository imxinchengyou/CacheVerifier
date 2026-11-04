"""GPU batch scoring for enterprise_compare_gptcache.py (same tokenizer, model and output column).

Checks 300 random pairs against GPTCache's official CPU evaluation() before saving."""
import gzip, json, sys, time
import numpy as np, onnxruntime
from gptcache.similarity_evaluation import OnnxModelEvaluation
from huggingface_hub import hf_hub_download

BATCH, N_CHECK = 128, 300
ev = OnnxModelEvaluation()
sess = onnxruntime.InferenceSession(hf_hub_download(repo_id="GPTCache/albert-duplicate-onnx", filename="model.onnx"),
                                    providers=["CUDAExecutionProvider"])
print("providers", sess.get_providers(), flush=True)
assert sess.get_providers()[0] == "CUDAExecutionProvider"


def score(pairs):
    tok = ev.tokenizer
    out = np.zeros(len(pairs), dtype=np.float32)
    enc = [tok(a, b) for a, b in pairs]
    order = np.argsort([len(e["input_ids"]) for e in enc])
    for s in range(0, len(order), BATCH):
        idx = order[s:s + BATCH]
        w = max(len(enc[k]["input_ids"]) for k in idx)
        feed = {}
        for name in ("input_ids", "attention_mask", "token_type_ids"):
            arr = np.full((len(idx), w), tok.pad_token_id, dtype=np.int64)
            for r, k in enumerate(idx):
                arr[r, :len(enc[k][name])] = enc[k][name]
            feed[name] = arr
        out[idx] = sess.run(None, feed)[0][:, 1]
    out[np.array([a.lower() == b.lower() for a, b in pairs])] = 1.0
    return out


rng = np.random.default_rng(0)
for ds in sys.argv[1:]:
    pairs = json.load(gzip.open(f"pairs_{ds}.json.gz", "rt"))
    chk = rng.choice(len(pairs), N_CHECK, replace=False)
    official = np.array([float(np.asarray(ev.evaluation({"question": pairs[k][0]}, {"question": pairs[k][1]})).ravel()[0]) for k in chk])
    diff = np.abs(score([pairs[k] for k in chk]) - official).max()
    print(f"{ds}: GPU batched vs official CPU evaluation(), max abs diff on {N_CHECK} pairs = {diff:.2e}", flush=True)
    if diff > 1e-3:
        raise SystemExit("mismatch; not saving")
    t0 = time.time()
    np.save(f"scores_{ds}.npy", score(pairs))
    print(f"{ds}: {len(pairs)} pairs scored in {time.time() - t0:.0f}s", flush=True)
