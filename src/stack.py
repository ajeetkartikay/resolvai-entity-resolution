"""
ResolvAI - Business Entity Resolution
STACK: stronger cross-encoder + learned stacker over (LightGBM v3, cross-encoder, competition context)

1. Train the multilingual cross-encoder on hard train pairs (LightGBM p >= lo) of 'fit' entities.
2. Score it on the live train pairs of the OTHER entities (never seen by the cross-encoder) and on
   all live test pairs (both GPUs if available). Scores are saved as soon as they exist.
3. Train a LightGBM stacker on those out-of-sample train rows: LightGBM v3 probabilities, cross-encoder
   score, and their competition context (rank / gap vs. the entity's other candidates and vs. the other
   entities competing for the same record, sibling agreement).
4. Tune the threshold on held-out entities with the same global one-to-one rule as at test time,
   compare against the fixed 50/50 blend, keep the better one, write the submission files.

Inputs (found recursively under --data_root):
  raw TSVs; train_probs.parquet + test_probs.parquet (route_b.py v3 output); route_b.py + ce.py (er-code)
Resume options: --ce_model DIR (skip training), --train_ce FILE / --test_ce FILE (skip scoring).
"""
import os, sys, gc, json, time, argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import route_b as rb
import ce as cem
import lightgbm as lgb

parser = argparse.ArgumentParser()
parser.add_argument("--data_root", default="data")
parser.add_argument("--out_dir", default="output")
parser.add_argument("--lo", type=float, default=0.01)
parser.add_argument("--n_train", type=int, default=2_500_000)
parser.add_argument("--n_val", type=int, default=50_000)
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--max_len", type=int, default=96)
parser.add_argument("--max_test_rows", type=int, default=15_000_000)
parser.add_argument("--ce_model", default="")
parser.add_argument("--train_ce", default="")
parser.add_argument("--test_ce", default="")
parser.add_argument("--skip_ce", action="store_true")           # use --train_ce/--test_ce only, no model at all
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--smoke", action="store_true")
parser.add_argument("--mock", action="store_true")
args, _ = parser.parse_known_args()
if args.smoke:
    args.n_train, args.n_val, args.max_test_rows = 20_000, 2_000, 50_000
    args.out_dir = os.path.join(args.out_dir, "smoke")
rb.args.data_root = cem.args.data_root = args.data_root
cem.args.lr, cem.args.max_len, cem.args.n_train, cem.args.seed = args.lr, args.max_len, args.n_train, args.seed
cem.args.epochs = 1
log = rb.log
os.makedirs(args.out_dir, exist_ok=True)


# ----------------------------------------------------------------------------- cross-encoder helpers
def load_model(path):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).to("cuda" if torch.cuda.is_available() else "cpu")
    return tok, model, model


def score(bundle, A, B, label):
    """Score pairs; split across 2 GPUs when available (falls back to 1 GPU on any problem)."""
    t0 = time.time()
    if args.mock:
        out = cem.mock_score(None, A, B)
    else:
        import torch
        out = None
        if torch.cuda.device_count() > 1 and len(A) > 200_000:
            try:
                import copy
                from concurrent.futures import ThreadPoolExecutor
                tok, model, _ = bundle
                m2 = copy.deepcopy(model).to("cuda:1")
                h = len(A) // 2
                with ThreadPoolExecutor(2) as ex:
                    f1 = ex.submit(cem.score_model, (tok, model, model), A[:h], B[:h])
                    f2 = ex.submit(cem.score_model, (copy.deepcopy(tok), m2, m2), A[h:], B[h:])  # own tokenizer per thread
                    out = np.concatenate([f1.result(), f2.result()])
                del m2
                torch.cuda.empty_cache()
            except Exception as e:
                log(f"2-GPU scoring failed ({e!r}), using 1 GPU")
                out = None
        if out is None:
            out = cem.score_model(bundle, A, B)
    log(f"{label}: scored {len(A):,} pairs in {time.time()-t0:.0f}s")
    return out.astype(np.float32)


# ----------------------------------------------------------------------------- stacker features
SFEAT = ["p1", "p2", "ce", "b", "ce_rank_i", "ce_gap_i", "ce_rank_j", "ce_gap_j", "b_rank_i", "b_gap_i",
         "b_rank_j", "b_gap_j", "n_ce_hi_i", "n_ce_hi_j", "n_b_hi_i", "sib1_ce", "sib2_ce", "sib1_b", "n_live_i"]


def stack_features(I, J, p1, p2, ce):
    """All arrays are over the SCORED rows only (competition is among scored rows)."""
    b = 0.5 * p2 + 0.5 * ce
    F = {"p1": p1, "p2": p2, "ce": ce, "b": b}
    F["ce_rank_i"], F["ce_gap_i"], F["n_live_i"] = rb.group_stats(I, ce)
    F["ce_rank_j"], F["ce_gap_j"], _ = rb.group_stats(J, ce)
    F["b_rank_i"], F["b_gap_i"], _ = rb.group_stats(I, b)
    F["b_rank_j"], F["b_gap_j"], _ = rb.group_stats(J, b)
    for name, key, w in [("n_ce_hi_i", I, ce), ("n_ce_hi_j", J, ce), ("n_b_hi_i", I, b)]:
        _, inv = np.unique(key, return_inverse=True)
        F[name] = np.bincount(inv, weights=(w > 0.5).astype(np.float32)).astype(np.float32)[inv]
    s1, s2 = rb.sibling_rows(I, b)
    F["sib1_ce"] = np.where(s1 >= 0, ce[np.maximum(s1, 0)], -1).astype(np.float32)
    F["sib2_ce"] = np.where(s2 >= 0, ce[np.maximum(s2, 0)], -1).astype(np.float32)
    F["sib1_b"] = np.where(s1 >= 0, b[np.maximum(s1, 0)], -1).astype(np.float32)
    return np.column_stack([np.asarray(F[k], dtype=np.float32) for k in SFEAT])


def expf_select(I, J, s, floor, gamma):
    """Per-entity expected-F0.5 decision. After one-to-one, for each entity keep the top-k candidates
    maximising E[F0.5] ~ 1.25*sum(p_top_k) / (k + 0.25*E[#true]); predict empty if P(singleton) is larger."""
    k = np.where(s >= floor)[0]
    k = k[np.argsort(-s[k], kind="stable")]
    _, first = np.unique(J[k], return_index=True)
    keep = k[first]
    if len(keep) == 0:
        return keep
    p = np.clip(s[keep], 0, 1) ** gamma
    ki = I[keep]
    o = np.lexsort((-p, ki))
    keep, p, ki = keep[o], p[o], ki[o]
    uniq, st, cnt = np.unique(ki, return_index=True, return_counts=True)
    gid = np.repeat(np.arange(len(uniq)), cnt)
    ET = np.bincount(gid, weights=p)
    logP0 = np.bincount(gid, weights=np.log1p(-np.minimum(p, 1 - 1e-6)))
    cum = np.cumsum(p)
    cum = cum - np.r_[0, cum[st[1:] - 1]][gid]
    rank = np.arange(len(p)) - st[gid] + 1
    f = 1.25 * cum / (rank + 0.25 * ET[gid])
    best_f = np.full(len(uniq), -1.0); np.maximum.at(best_f, gid, f)
    is_best = f >= best_f[gid] - 1e-12
    kbest = np.full(len(uniq), 10**9); np.minimum.at(kbest, gid[is_best], rank[is_best])
    ok = best_f > np.exp(logP0)
    sel = (rank <= kbest[gid]) & ok[gid]
    return keep[sel]


def full_scores(n, rows, s):
    out = np.full(n, -1.0, np.float32)
    out[rows] = s
    return out


# ----------------------------------------------------------------------------- main
def main():
    rng = np.random.default_rng(args.seed)

    # ---------------- train side
    ids1, T1, ids23, T23 = cem.texts("train")
    tp = pd.read_parquet(rb.find("train_probs.parquet"))
    I, J = tp.i.values.astype(np.int64), tp.j.values.astype(np.int64)
    p1, p2, y, part = tp.p1.values, tp.p2.values, tp.y.values, tp.part.values
    del tp; gc.collect()
    live = p2 >= args.lo
    log(f"train pairs={len(I):,}  live={live.sum():,}  parts={np.bincount(part[part >= 0]).tolist()}")

    # 1) cross-encoder
    if args.skip_ce:
        bundle = None
        log("skip_ce: using precomputed cross-encoder scores")
    elif args.ce_model:
        bundle = None if args.mock else load_model(args.ce_model)
        log(f"loaded cross-encoder from {args.ce_model}")
    else:
        fit = np.where(live & (part == 0))[0]
        fit = rng.choice(fit, min(args.n_train, len(fit)), replace=False)
        val = np.where(live & (part == 1))[0]
        val = rng.choice(val, min(args.n_val, len(val)), replace=False)
        log(f"cross-encoder train pairs={len(fit):,} (pos {y[fit].mean():.3f})")
        trainer = cem.mock_train if args.mock else cem.train_model
        bundle = trainer(T1[I[fit]], T23[J[fit]], y[fit], T1[I[val]], T23[J[val]], y[val].astype(np.float32))
        if bundle is not None:
            d = os.path.join(args.out_dir, "ce_model")
            bundle[1].save_pretrained(d); bundle[0].save_pretrained(d)
            log("cross-encoder saved")

    # 2) out-of-sample scores on train rows of entities in parts 1 and 2
    srows = np.where(live & ((part == 1) | (part == 2)))[0]
    if args.train_ce:
        d = pd.read_parquet(rb.find(args.train_ce)); srows, ce_tr = d.row.values, d.ce.values
    else:
        if args.smoke:
            ents = rng.choice(np.unique(I[srows]), 5_000, replace=False)
            srows = srows[np.isin(I[srows], ents)]
        ce_tr = score(bundle, T1[I[srows]], T23[J[srows]], "train")
        pd.DataFrame({"row": srows, "ce": ce_tr}).to_parquet(os.path.join(args.out_dir, "train_ce.parquet"), index=False)
    del T1, T23; gc.collect()

    # 3) stacker: fit on part-1 + half of part-2 entities, evaluate on the other half of part 2
    Is, Js = I[srows], J[srows]
    X = stack_features(Is, Js, p1[srows], p2[srows], ce_tr)
    ys = y[srows]
    ent_half = rng.random(len(ids1)) < 0.5
    pe = part[srows]
    fit_m = (pe == 1) | ((pe == 2) & ent_half[Is])
    ev_ents = np.unique(Is[(pe == 2) & ~ent_half[Is]])
    es_m = fit_m & (rng.random(len(srows)) < 0.1)
    tr_m = fit_m & ~es_m
    log(f"stacker rows: fit={tr_m.sum():,} es={es_m.sum():,}  eval entities={len(ev_ents):,}")
    prm = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.9,
               bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=args.seed, num_threads=4)
    dtr = lgb.Dataset(X[tr_m], ys[tr_m], feature_name=SFEAT)
    st = lgb.train(prm, dtr, 3000, valid_sets=[lgb.Dataset(X[es_m], ys[es_m], feature_name=SFEAT, reference=dtr)],
                   callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])
    st = lgb.Booster(model_str=st.model_to_string(num_iteration=st.best_iteration))
    st.save_model(os.path.join(args.out_dir, "stacker.txt"))
    imp = pd.Series(st.feature_importance("gain"), index=SFEAT).sort_values(ascending=False)
    log("stacker top features:\n" + (imp / imp.sum()).round(3).head(12).to_string())
    ps = st.predict(X, num_threads=4).astype(np.float32)
    del X; gc.collect()

    gt = rb.load("train_ground_truth.tsv")
    true_sets = {k: set(x for x in v.split(",") if x) for k, v in zip(gt.source1_entity_id, gt.matched_entity_ids)}
    eval_ids = list(ids1[ev_ents])
    n = len(I)
    cands = {"p2": full_scores(n, srows, p2[srows]),
             "blend": full_scores(n, srows, 0.5 * p2[srows] + 0.5 * ce_tr),
             "stack": full_scores(n, srows, ps)}
    res = {}
    for name, s in cands.items():
        grid = {float(t): rb.score_rows(I, J, rb.one_to_one(J, s, t), true_sets, eval_ids, ids1, ids23)
                for t in np.round(np.arange(0.30, 0.901, 0.025), 3)}
        bt = max(grid, key=grid.get)
        res[name] = (bt, grid[bt], grid)
        log(f"{name:6s}: best thr={bt}  held-out F0.5={grid[bt]:.4f}")
    # expected-F0.5 decision rule (per entity), on the two strongest scores
    for name in ("blend", "stack"):
        best = (None, -1)
        for fl in (0.05, 0.1, 0.2):
            for gm in (1.0, 1.5, 2.0):
                f = rb.score_rows(I, J, expf_select(I, J, cands[name], fl, gm), true_sets, eval_ids, ids1, ids23)
                if f > best[1]:
                    best = ((fl, gm), f)
        res[name + "_expf"] = (best[0], best[1], {})
        log(f"{name}_expf: best (floor, gamma)={best[0]}  held-out F0.5={best[1]:.4f}")
    method = max(("blend", "stack", "blend_expf", "stack_expf"), key=lambda k: res[k][1])
    thr = res[method][0]
    log(f"BEST {method} param={thr}  held-out F0.5={res[method][1]:.4f}  (v3 alone {res['p2'][1]:.4f}, "
        f"blend {res['blend'][1]:.4f}, stack {res['stack'][1]:.4f}, blend_expf {res['blend_expf'][1]:.4f}, "
        f"stack_expf {res['stack_expf'][1]:.4f})")
    json.dump({"method": method, "thr": thr, **{k: {"thr": v[0], "f": v[1]} for k, v in res.items()},
               "importance": (imp / imp.sum()).round(4).to_dict()},
              open(os.path.join(args.out_dir, "stack_report.json"), "w"), indent=1)
    del I, J, p1, p2, y, part, cands, ps, gt, true_sets; gc.collect()

    # ---------------- test side
    ids1, T1, ids23, T23 = cem.texts("test")
    ts = pd.read_parquet(rb.find("test_probs.parquet"))
    I, J = ts.i.values.astype(np.int64), ts.j.values.astype(np.int64)
    p1 = ts.p1.values if "p1" in ts else ts.prob.values
    p2 = ts.prob.values
    del ts; gc.collect()
    rows = np.where(p2 >= args.lo)[0]
    if len(rows) > args.max_test_rows:
        rows = rows[np.argsort(-p2[rows])[:args.max_test_rows]]
    rows = np.sort(rows)
    log(f"test pairs={len(I):,}  live={len(rows):,}")
    if args.test_ce:
        d = pd.read_parquet(rb.find(args.test_ce)); rows, ce_te = d.row.values, d.ce.values
    else:
        ce_te = score(bundle, T1[I[rows]], T23[J[rows]], "test")
        pd.DataFrame({"row": rows, "ce": ce_te}).to_parquet(os.path.join(args.out_dir, "test_ce.parquet"), index=False)
    del T1, T23; gc.collect()
    if method.startswith("stack"):
        s_live = st.predict(stack_features(I[rows], J[rows], p1[rows], p2[rows], ce_te), num_threads=4)
    else:
        s_live = 0.5 * p2[rows] + 0.5 * ce_te
    s = full_scores(len(I), rows, s_live.astype(np.float32))
    kept = expf_select(I, J, s, thr[0], thr[1]) if method.endswith("_expf") else rb.one_to_one(J, s, thr)
    log(f"test matches: {len(kept):,}  entities with >=1 match: {len(np.unique(I[kept])):,} / {len(ids1):,}")
    mpath = os.path.join(args.out_dir, "matching_results.tsv")
    cpath = os.path.join(args.out_dir, "candidate_pairs.tsv")
    rb.write_lists(mpath, "matched_entity_ids", ids1, ids23, I[kept], J[kept])
    rb.write_lists(cpath, "candidate_entity_ids", ids1, ids23, I, J)
    pd.DataFrame({"row": rows, "score": s_live.astype(np.float32)}).to_parquet(
        os.path.join(args.out_dir, "test_final_scores.parquet"), index=False)
    log("files written")
    rb.self_check(mpath, cpath, ids1, ids23)


if __name__ == "__main__":
    main()
