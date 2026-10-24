"""
ResolvAI - Business Entity Resolution
CROSS-ENCODER RESCORING (GPU) + BLEND WITH LightGBM

A multilingual transformer reads both records side by side ("name ; address" of S1 vs S2/S3)
and scores whether they are the same business. It is trained on full-scale train candidate pairs,
applied only to pairs the LightGBM model did not already rule out (p >= --lo), and blended with
the LightGBM probability. Blend weight and threshold are tuned on held-out train entities with the
same global one-to-one rule used at test time.

Inputs (found recursively under --data_root):
  raw TSVs, train_probs.parquet + test_probs.parquet (outputs of route_b.py), route_b.py (helpers)
Model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (Apache-2.0, ~118M params)

Usage (GPU):
  python ce.py --smoke      (3-minute end-to-end check)
  python ce.py              (full run)
"""
import os, sys, gc, json, time, argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import route_b as rb                                                   # shared helpers (loading, scoring, output)

parser = argparse.ArgumentParser()
parser.add_argument("--data_root", default="data")
parser.add_argument("--out_dir", default="output")
parser.add_argument("--model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
parser.add_argument("--lo", type=float, default=0.01)                 # rescore pairs with LightGBM p >= lo
parser.add_argument("--n_train", type=int, default=1_500_000)
parser.add_argument("--n_val", type=int, default=50_000)
parser.add_argument("--eval_entities", type=int, default=250_000)
parser.add_argument("--max_test_rows", type=int, default=15_000_000)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--lr", type=float, default=4e-5)
parser.add_argument("--bs_train", type=int, default=128)
parser.add_argument("--bs_score", type=int, default=1024)
parser.add_argument("--max_len", type=int, default=128)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--smoke", action="store_true")
parser.add_argument("--mock", action="store_true")                    # CPU logic test without torch
args, _ = parser.parse_known_args()
rb.args.data_root = args.data_root
if args.smoke:
    args.n_train, args.n_val, args.eval_entities, args.max_test_rows = 20_000, 2_000, 5_000, 50_000
    args.out_dir = os.path.join(args.out_dir, "smoke")
log = rb.log


# ----------------------------------------------------------------------------- data
def texts(split):
    """Raw 'name ; address' strings, in the same row order route_b / baseline used."""
    def t(df):
        return (df.business_name.str.strip() + " ; " + df.business_address.str.strip()).values
    s1 = rb.load(f"{split}_source1.tsv")
    s23 = pd.concat([rb.load(f"{split}_source2.tsv"), rb.load(f"{split}_source3.tsv")], ignore_index=True)
    return s1.entity_id.values, t(s1), s23.entity_id.values, t(s23)


# ----------------------------------------------------------------------------- model (GPU)
def train_model(A, B, y, vA, vB, vy):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=1).to(dev)
    net = model                                  # single GPU: fp16 autocast works reliably (DataParallel breaks it)
    log(f"device={dev} gpus={torch.cuda.device_count()} params={sum(p.numel() for p in model.parameters())/1e6:.0f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * int(np.ceil(len(y) / args.bs_train))
    sch = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
    scaler = torch.cuda.amp.GradScaler(enabled=(dev == "cuda"))
    lossf = torch.nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(args.seed)
    step, t0 = 0, time.time()
    for ep in range(args.epochs):
        net.train()
        order = rng.permutation(len(y))
        run = 0.0
        for s in range(0, len(order), args.bs_train):
            b = order[s:s + args.bs_train]
            enc = tok(list(A[b]), list(B[b]), truncation=True, max_length=args.max_len,
                      padding=True, return_tensors="pt").to(dev)
            tgt = torch.tensor(y[b], dtype=torch.float32, device=dev)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(dev == "cuda")):
                logits = net(**enc).logits.squeeze(-1)
            loss = lossf(logits.float(), tgt)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sch.step()
            run = 0.98 * run + 0.02 * loss.item() if step else loss.item()
            step += 1
            if step % 1000 == 0 or step == steps:
                log(f"  epoch {ep+1} step {step}/{steps} loss {run:.4f}  ({(time.time()-t0)/step*1000:.0f} ms/step)")
    vp = score_model((tok, model, net), vA, vB)
    eps = 1e-6
    vll = -np.mean(vy * np.log(vp + eps) + (1 - vy) * np.log(1 - vp + eps))
    log(f"val logloss={vll:.4f}  val acc@0.5={np.mean((vp > 0.5) == vy):.4f}")
    return tok, model, net


def score_model(bundle, A, B):
    import torch
    tok, model, net = bundle
    dev = next(model.parameters()).device
    net.eval()
    out = np.zeros(len(A), np.float32)
    order = np.argsort([len(a) + len(b) for a, b in zip(A, B)])     # similar lengths per batch = less padding
    t0 = time.time()
    with torch.no_grad():
        for n, s in enumerate(range(0, len(order), args.bs_score)):
            b = order[s:s + args.bs_score]
            enc = tok(list(A[b]), list(B[b]), truncation=True, max_length=args.max_len,
                      padding=True, return_tensors="pt").to(dev)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(dev.type == "cuda")):
                logits = net(**enc).logits.squeeze(-1)
            out[b] = torch.sigmoid(logits.float()).cpu().numpy()
            if n % 2000 == 0 and n:
                log(f"  scored {s:,}/{len(order):,}  ({len(order[:s])/(time.time()-t0):.0f} pairs/s)")
    return out


# ----------------------------------------------------------------------------- mock (CPU logic test only)
def mock_train(*a):
    return None


def mock_score(bundle, A, B):
    from rapidfuzz.process import cpdist
    from rapidfuzz import fuzz
    return (cpdist(list(A), list(B), scorer=fuzz.token_set_ratio, workers=-1) / 100.0).astype(np.float32)


# ----------------------------------------------------------------------------- main
def main():
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    do_train, do_score = (mock_train, mock_score) if args.mock else (train_model, score_model)

    # ---- train side
    ids1, T1, ids23, T23 = texts("train")
    tp = pd.read_parquet(rb.find("train_probs.parquet"))
    I, J, p2, y, part = tp.i.values, tp.j.values, tp.p2.values, tp.y.values, tp.part.values
    del tp; gc.collect()
    live = p2 >= args.lo
    fit = np.where(live & (part == 0))[0]
    fit = rng.choice(fit, min(args.n_train, len(fit)), replace=False)
    val = np.where(live & (part == 1))[0]
    val = rng.choice(val, min(args.n_val, len(val)), replace=False)
    log(f"train pairs={len(fit):,} (pos rate {y[fit].mean():.3f})  val pairs={len(val):,}")
    bundle = do_train(T1[I[fit]], T23[J[fit]], y[fit], T1[I[val]], T23[J[val]], y[val].astype(np.float32))

    # ---- held-out entities: rescore, then tune blend weight + threshold with global one-to-one
    ev_ents = np.unique(I[part == 2])
    ev_ents = rng.choice(ev_ents, min(args.eval_entities, len(ev_ents)), replace=False)
    in_ev = np.zeros(len(ids1), bool); in_ev[ev_ents] = True
    ev_rows = np.where(live & in_ev[I])[0]
    log(f"eval entities={len(ev_ents):,}  rescored pairs={len(ev_rows):,}")
    ce_ev = do_score(bundle, T1[I[ev_rows]], T23[J[ev_rows]])

    gt = rb.load("train_ground_truth.tsv")
    true_sets = {k: set(x for x in v.split(",") if x) for k, v in zip(gt.source1_entity_id, gt.matched_entity_ids)}
    eval_ids = list(ids1[ev_ents])
    res = {}
    for a in [1.0, 0.75, 0.5, 0.25, 0.0]:
        s = p2.copy()
        s[ev_rows] = a * p2[ev_rows] + (1 - a) * ce_ev
        for thr in np.round(np.arange(0.30, 0.951, 0.05), 2):
            f = rb.score_rows(I, J, rb.one_to_one(J, s, thr), true_sets, eval_ids, ids1, ids23)
            res[(a, float(thr))] = f
        best_t = max((t for (aa, t) in res if aa == a), key=lambda t: res[(a, t)])
        log(f"  blend a={a:.2f} (1.0 = LightGBM only): best thr={best_t}  F0.5={res[(a, best_t)]:.4f}")
    (a_best, thr_best) = max(res, key=res.get)
    log(f"BEST blend a={a_best}  thr={thr_best}  held-out F0.5={res[(a_best, thr_best)]:.4f}  "
        f"(LightGBM only best: {max(v for (aa, t), v in res.items() if aa == 1.0):.4f})")
    json.dump({"a": a_best, "thr": thr_best, "grid": {f"{k[0]}_{k[1]}": v for k, v in res.items()}},
              open(os.path.join(args.out_dir, "ce_report.json"), "w"), indent=1)
    del I, J, p2, y, part, T1, T23, ids1, ids23, gt, true_sets; gc.collect()

    # ---- test side
    ids1, T1, ids23, T23 = texts("test")
    ts = pd.read_parquet(rb.find("test_probs.parquet"))
    I, J, p = ts.i.values, ts.j.values, ts.prob.values
    del ts; gc.collect()
    rows = np.where(p >= args.lo)[0]
    if len(rows) > args.max_test_rows:                      # keep the most likely ones if over budget
        rows = rows[np.argsort(-p[rows])[:args.max_test_rows]]
    log(f"test pairs={len(p):,}  rescored={len(rows):,}")
    ce = do_score(bundle, T1[I[rows]], T23[J[rows]])
    pd.DataFrame({"row": rows.astype(np.int64), "ce": ce}).to_parquet(os.path.join(args.out_dir, "test_ce.parquet"),
                                                                       index=False)
    s = p.copy()
    s[rows] = a_best * p[rows] + (1 - a_best) * ce
    kept = rb.one_to_one(J, s, thr_best)
    log(f"test matches: {len(kept):,}  entities with >=1 match: {len(np.unique(I[kept])):,} / {len(ids1):,}")
    mpath = os.path.join(args.out_dir, "matching_results.tsv")
    cpath = os.path.join(args.out_dir, "candidate_pairs.tsv")
    rb.write_lists(mpath, "matched_entity_ids", ids1, ids23, I[kept], J[kept])
    rb.write_lists(cpath, "candidate_entity_ids", ids1, ids23, I, J)
    log("files written")
    rb.self_check(mpath, cpath, ids1, ids23)
    if not args.mock and bundle is not None:
        bundle[1].save_pretrained(os.path.join(args.out_dir, "ce_model"))
        bundle[0].save_pretrained(os.path.join(args.out_dir, "ce_model"))
        log("model saved")


if __name__ == "__main__":
    main()
