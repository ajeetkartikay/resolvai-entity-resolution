"""
ResolvAI - Business Entity Resolution
ROUTE B: LightGBM pair matcher, trained at FULL scale.

Inputs (found recursively under --data_root):
  - raw TSVs: train_source{1,2,3}.tsv, train_ground_truth.tsv, test_source{1,2,3}.tsv
  - train_cand_scored.parquet  (output of baseline_pipeline.py --split train)
  - test_cand_scored.parquet   (output of baseline_pipeline.py --split test)

Steps:
  train: features for a sample of S1 entities -> LightGBM (early stopping on a 2nd sample)
         -> predict ALL train pairs -> tune threshold on the remaining (held-out) entities,
            with global one-to-one assignment, exactly as at test time
  test:  features for all test pairs (chunked) -> predict -> threshold + one-to-one
         -> output/matching_results.tsv + output/candidate_pairs.tsv (+ self-check)
"""
import os, re, csv, gc, glob, json, time, argparse
import numpy as np
import pandas as pd
from unidecode import unidecode
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist
from rapidfuzz.distance import JaroWinkler, Levenshtein
import lightgbm as lgb

parser = argparse.ArgumentParser()
parser.add_argument("--data_root", default="data")
parser.add_argument("--out_dir", default="output")
parser.add_argument("--mode", default="both")            # train | test | both
parser.add_argument("--fit_frac", type=float, default=0.12)
parser.add_argument("--es_frac", type=float, default=0.03)
parser.add_argument("--chunk", type=int, default=5_000_000)
parser.add_argument("--n_threads", type=int, default=4)
parser.add_argument("--seed", type=int, default=42)
args, _ = parser.parse_known_args()


def log(msg, t0=[time.time()]):
    try:
        import psutil
        mem = f" | RAM {psutil.Process().memory_info().rss / 2**30:5.1f} GB"
    except Exception:
        mem = ""
    print(f"[{time.time() - t0[0]:7.0f}s{mem}] {msg}", flush=True)


def find(name):
    hits = sorted(glob.glob(os.path.join(args.data_root, "**", name), recursive=True))
    if not hits:
        raise FileNotFoundError(name)
    return hits[0]


def load(name):
    df = pd.read_csv(find(name), sep="\t", dtype=str, keep_default_na=False,
                     quoting=csv.QUOTE_NONE, on_bad_lines="warn")
    for c in df.columns:
        df[c] = df[c].str.strip()
    return df


# ----------------------------------------------------------------------------- normalization
# (identical rules to baseline_pipeline.py; this version also keeps initials / numbers / script flag)
US_STATES = {"alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga", "hawaii": "hi",
    "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks", "kentucky": "ky",
    "louisiana": "la", "maine": "me", "maryland": "md", "massachusetts": "ma", "michigan": "mi",
    "minnesota": "mn", "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne",
    "nevada": "nv", "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny",
    "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok", "oregon": "or",
    "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc", "south dakota": "sd",
    "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa",
    "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy", "district of columbia": "dc"}
IN_STATES = {"andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br",
    "chhattisgarh": "cg", "goa": "ga", "gujarat": "gj", "haryana": "hr", "himachal pradesh": "hp",
    "jharkhand": "jh", "karnataka": "ka", "kerala": "kl", "keralam": "kl", "madhya pradesh": "mp",
    "maharashtra": "mh", "manipur": "mn", "meghalaya": "ml", "mizoram": "mz", "nagaland": "nl",
    "odisha": "od", "orissa": "od", "punjab": "pb", "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn",
    "telangana": "ts", "tripura": "tr", "uttar pradesh": "up", "uttarakhand": "uk", "west bengal": "wb",
    "delhi": "dl", "new delhi": "dl", "jammu and kashmir": "jk", "ladakh": "la", "chandigarh": "ch",
    "puducherry": "py", "pondicherry": "py"}
CITY_ALIAS = {"bangalore": "bengaluru", "madras": "chennai", "bombay": "mumbai", "calcutta": "kolkata",
              "gurgaon": "gurugram", "poona": "pune", "trivandrum": "thiruvananthapuram", "cochin": "kochi"}
ADDR_MAP = {"street": "st", "str": "st", "saint": "st", "road": "rd", "avenue": "ave", "av": "ave",
    "drive": "dr", "boulevard": "blvd", "bd": "blvd", "bvd": "blvd", "lane": "ln", "court": "ct",
    "place": "pl", "highway": "hwy", "sector": "sec", "block": "blk", "near": "nr", "opposite": "opp",
    "centre": "center", "north": "n", "south": "s", "east": "e", "west": "w"}
ORDINALS = {"first": "1st", "second": "2nd", "third": "3rd", "fourth": "4th", "fifth": "5th", "sixth": "6th",
    "seventh": "7th", "eighth": "8th", "ninth": "9th", "tenth": "10th", "eleventh": "11th", "twelfth": "12th",
    "twelth": "12th", "thirteenth": "13th", "fourteenth": "14th", "fifteenth": "15th", "sixteenth": "16th",
    "seventeenth": "17th", "eighteenth": "18th", "nineteenth": "19th", "twentieth": "20th",
    "thirtieth": "30th", "fortieth": "40th", "fiftieth": "50th", "sixtieth": "60th", "seventieth": "70th",
    "eightieth": "80th", "ninetieth": "90th", "hundredth": "100th"}
ADDR_MAP.update(ORDINALS)
ADDR_DROP = {"unit", "no", "house", "hno", "h", "num", "number", "flat", "door", "plot", "apt",
             "apartment", "suite", "ste", "floor", "fl", "flr", "n", "o"}
LEGAL = {"inc", "incorporated", "llc", "lp", "llp", "ltd", "limited", "pvt", "private", "corp",
         "corporation", "co", "company", "plc", "pllc", "pc", "sa", "sas", "sasu", "sarl", "eurl", "sci",
         "snc", "gmbh", "opc", "the", "and"}
JUNK = {"m", "s", "ms", "smt", "shri", "sri", "dr", "mr", "mrs", "www", "com", "net", "org",
        "center", "centre", "group"}
STATE_TABLES = {"US": US_STATES, "India": IN_STATES}


def squash(t):
    """collapse repeated letters: praaivett -> praivet, limittedd -> limited"""
    return re.sub(r"(.)\1+", r"\1", t)


def skel(t):
    """consonant skeleton, robust to transliteration: praaivett / piraiveett / private -> prvt"""
    t = squash(t)
    return t[:1] + re.sub(r"[aeiouy]", "", t[1:])


# transliterated legal words (e.g. 'praaivett limittedd') are caught through their skeletons
LEGAL_SKEL = {"prvt", "lmtd", "lmtt", "crprtn", "incrprtd", "cmpn"}   # private/limited/corporation/...


def _ascii(s):
    return unidecode(s or "").lower().replace("&", " and ")


def _replace_phrases(s, table):
    for k in sorted(table, key=len, reverse=True):
        if " " in k and k in s:
            s = re.sub(rf"\b{k}\b", table[k], s)
    return s


def norm_name(name):
    s = re.sub(r"\.(com|in|net|org|co)\b", " ", _ascii(name))
    seen, full = set(), []
    for t in re.findall(r"[a-z0-9]+", s):
        if t not in seen:
            seen.add(t)
            full.append(t)
    core = [t for t in full if t not in LEGAL and t not in JUNK
            and not (len(t) >= 4 and skel(t) in LEGAL_SKEL)] or full
    return " ".join(core), "".join(t[0] for t in core), " ".join(skel(t) for t in core)


def norm_addr(addr, country):
    table = STATE_TABLES.get(country, {})
    s = _replace_phrases(_ascii(addr).replace("n°", " "), table)
    out, skip = [], False
    for t in re.findall(r"[a-z0-9]+", s):
        if skip:                                   # number after 'pmb' (private mailbox) is noise
            skip = False
            if t.isdigit():
                continue
        if t == "pmb":
            skip = True
            continue
        t = CITY_ALIAS.get(t, t)
        t = table.get(t, t)
        if country == "France" and t == "r":
            t = "rue"
        t = ADDR_MAP.get(t, t)
        if t.isdigit():
            t = t.lstrip("0") or "0"
        if t not in ADDR_DROP:
            out.append(t)
    nums = sorted({t for t in out if any(c.isdigit() for c in t)})
    hnum = next((t for t in out if t.isdigit()), "")          # primary (house) number, in address order
    return " ".join(out), " ".join(nums), hnum


def normalize_full(df):
    nn = [norm_name(x) for x in df.business_name.values]
    aa = [norm_addr(a, c) for a, c in zip(df.business_address.values, df.country.values)]
    return pd.DataFrame({
        "entity_id": df.entity_id.values,
        "name_core": [x[0] for x in nn], "initials": [x[1] for x in nn], "name_skel": [x[2] for x in nn],
        "addr_n": [x[0] for x in aa], "addr_nums": [x[1] for x in aa], "hnum": [x[2] for x in aa],
        "nonlatin": df.business_name.str.contains(r"[^\x00-\x7F]", regex=True).values.astype(np.int8),
    })


def load_split(sp):
    s1 = normalize_full(load(f"{sp}_source1.tsv"))
    s23 = normalize_full(pd.concat([load(f"{sp}_source2.tsv"), load(f"{sp}_source3.tsv")], ignore_index=True))
    cand = pd.read_parquet(find(f"{sp}_cand_scored.parquet"), columns=["i", "j", "name", "addr", "combo"])
    cand["i"] = cand.i.astype(np.int32); cand["j"] = cand.j.astype(np.int32)
    for c in ["name", "addr", "combo"]:
        cand[c] = cand[c].astype(np.float32)
    assert cand.i.max() < len(s1) and cand.j.max() < len(s23), "candidate file does not match this split"
    log(f"{sp}: S1={len(s1):,} S2+S3={len(s23):,} candidate pairs={len(cand):,}")
    return s1, s23, cand


# ----------------------------------------------------------------------------- features
FEATURES = ["name", "addr", "combo", "n_ratio", "n_tsort", "n_tset", "n_partial", "n_jw", "n_nospace",
            "sk_ratio", "sk_tset", "a_ratio", "a_tset", "init_eq", "acronym",
            "num_inter", "num_jacc", "num_a", "num_b", "num_only_a", "num_only_b", "num_conflict",
            "hnum_both", "hnum_eq", "hnum_close",
            "b_nonlatin", "b_addr_empty", "b_is_s3", "len_a", "len_b",
            "m", "gap_i", "rank_i", "gap_j", "rank_j", "n_cand_i"]
STAGE2 = ["p1", "rank_ip", "gap_ip", "rank_jp", "gap_jp", "n_hi_i", "n_hi_j",
          # sibling agreement: does this record agree with the entity's OTHER strongest records?
          "sib1_p", "sib1_nratio", "sib1_nskel", "sib1_aset", "sib1_heq", "sib1_hconf",
          "sib2_p", "sib2_nratio", "sib2_aset", "sib2_heq", "sib2_hconf"]


def sibling_rows(I, p1):
    """For every pair row r of entity i: the rows of i's best and second-best OTHER candidates (by p1)."""
    order = np.lexsort((-p1, I))
    Is = I[order]
    st = np.r_[0, np.flatnonzero(Is[1:] != Is[:-1]) + 1]
    cnt = np.diff(np.r_[st, len(Is)])
    gid = np.repeat(np.arange(len(st)), cnt)
    t1 = order[st]
    t2 = np.where(cnt > 1, order[np.minimum(st + 1, len(order) - 1)], -1)
    t3 = np.where(cnt > 2, order[np.minimum(st + 2, len(order) - 1)], -1)
    g = np.empty(len(I), np.int64); g[order] = gid
    r = np.arange(len(I))
    T1, T2, T3 = t1[g], t2[g], t3[g]
    s1 = np.where(r == T1, T2, T1)
    s2 = np.where((r == T1) | (r == T2), T3, T2)
    return s1.astype(np.int32), s2.astype(np.int32)


def group_stats(key, m):
    """Per group of `key`: rank of m (1 = best), m - group max, group size. Pure numpy, memory-light."""
    order = np.lexsort((-m, key))
    ks = key[order]
    starts = np.r_[0, np.flatnonzero(ks[1:] != ks[:-1]) + 1]
    sizes = np.diff(np.r_[starts, len(ks)])
    gid = np.repeat(np.arange(len(starts)), sizes)
    rank = np.empty(len(m), np.float32); rank[order] = np.arange(len(ks)) - starts[gid] + 1
    gap = np.empty(len(m), np.float32);  gap[order] = m[order] - m[order][starts][gid]
    size = np.empty(len(m), np.float32); size[order] = sizes[gid]
    return rank, gap, size


def competition(cand):
    m = cand[["name", "addr", "combo"]].to_numpy(np.float32).mean(1)
    rank_i, gap_i, n_i = group_stats(cand.i.values, m)
    rank_j, gap_j, _ = group_stats(cand.j.values, m)
    return {"m": m, "rank_i": rank_i, "gap_i": gap_i, "n_cand_i": n_i, "rank_j": rank_j, "gap_j": gap_j}


def stage2_features(I, J, p1):
    """Context from stage-1 probabilities of ALL competitors on both sides."""
    rank_ip, gap_ip, _ = group_stats(I, p1)
    rank_jp, gap_jp, _ = group_stats(J, p1)
    hi = (p1 > 0.5).astype(np.float32)
    n_hi_i = np.bincount(I, weights=hi, minlength=int(I.max()) + 1).astype(np.float32)[I]
    n_hi_j = np.bincount(J, weights=hi, minlength=int(J.max()) + 1).astype(np.float32)[J]
    sib1, sib2 = sibling_rows(I, p1)
    return {"p1": p1.astype(np.float32, copy=False), "rank_ip": rank_ip, "gap_ip": gap_ip, "rank_jp": rank_jp,
            "gap_jp": gap_jp, "n_hi_i": n_hi_i, "n_hi_j": n_hi_j, "_sib1": sib1, "_sib2": sib2}


class FeatureBuilder:
    def __init__(self, s1, s23, cand):
        self.s1, self.s23, self.cand = s1, s23, cand
        self.comp = competition(cand)
        self.num1 = [frozenset(s.split()) for s in s1.addr_nums.values]
        self.num23 = [frozenset(s.split()) for s in s23.addr_nums.values]
        self.b_is_s3 = s23.entity_id.str.startswith("S3").values.astype(np.int8)
        self.len1 = s1.name_core.str.len().values.astype(np.float32)
        self.len23 = s23.name_core.str.len().values.astype(np.float32)
        self.stage2 = None                                   # set to stage2_features(...) for stage 2

    def with_cand(self, cand):
        """Same records / precomputed lookups, different candidate subset (e.g. one country)."""
        fb = object.__new__(FeatureBuilder)
        fb.__dict__.update(self.__dict__)
        fb.cand, fb.comp, fb.stage2 = cand, competition(cand), None
        return fb

    def build(self, rows):
        c, s1, s23 = self.cand, self.s1, self.s23
        I, J = c.i.values[rows], c.j.values[rows]
        an, bn = s1.name_core.values[I], s23.name_core.values[J]
        aa, ba = s1.addr_n.values[I], s23.addr_n.values[J]
        ak, bk = s1.name_skel.values[I], s23.name_skel.values[J]
        cp = lambda x, z, sc: cpdist(list(x), list(z), scorer=sc, workers=-1).astype(np.float32)
        X = {"name": c.name.values[rows], "addr": c.addr.values[rows], "combo": c.combo.values[rows]}
        X["n_ratio"] = cp(an, bn, fuzz.ratio)
        X["n_tsort"] = cp(an, bn, fuzz.token_sort_ratio)
        X["n_tset"] = cp(an, bn, fuzz.token_set_ratio)
        X["n_partial"] = cp(an, bn, fuzz.partial_ratio)
        X["n_jw"] = cp(an, bn, JaroWinkler.normalized_similarity)
        an_ns = np.array([s.replace(" ", "") for s in an], dtype=object)
        bn_ns = np.array([s.replace(" ", "") for s in bn], dtype=object)
        X["n_nospace"] = cp(an_ns, bn_ns, fuzz.ratio)
        X["sk_ratio"] = cp(ak, bk, fuzz.ratio)                # transliteration-robust name similarity
        X["sk_tset"] = cp(ak, bk, fuzz.token_set_ratio)
        X["a_ratio"] = cp(aa, ba, fuzz.ratio)
        X["a_tset"] = cp(aa, ba, fuzz.token_set_ratio)
        ai, bi = s1.initials.values[I], s23.initials.values[J]
        X["init_eq"] = (ai == bi).astype(np.int8)
        X["acronym"] = ((bn_ns == ai) | (an_ns == bi)).astype(np.int8)
        inter = np.fromiter((len(self.num1[i] & self.num23[j]) for i, j in zip(I, J)), np.int16, len(I))
        na = np.fromiter((len(self.num1[i]) for i in I), np.int16, len(I))
        nb = np.fromiter((len(self.num23[j]) for j in J), np.int16, len(J))
        X["num_inter"], X["num_a"], X["num_b"] = inter, na, nb
        X["num_only_a"], X["num_only_b"] = na - inter, nb - inter
        X["num_jacc"] = (inter / np.maximum(na + nb - inter, 1)).astype(np.float32)
        X["num_conflict"] = ((na > 0) & (nb > 0) & (inter == 0)).astype(np.int8)
        ha, hb = s1.hnum.values[I], s23.hnum.values[J]
        both = (ha != "") & (hb != "")
        X["hnum_both"] = both.astype(np.int8)
        X["hnum_eq"] = (both & (ha == hb)).astype(np.int8)
        lev = cpdist(list(ha), list(hb), scorer=Levenshtein.distance, workers=-1)
        X["hnum_close"] = (both & (ha != hb) & (lev == 1)).astype(np.int8)   # 6421 vs 6424: near-miss
        X["b_nonlatin"] = s23.nonlatin.values[J]
        X["b_addr_empty"] = (ba == "").astype(np.int8)
        X["b_is_s3"] = self.b_is_s3[J]
        X["len_a"], X["len_b"] = self.len1[I], self.len23[J]
        for k, v in self.comp.items():
            X[k] = v[rows]
        cols = FEATURES
        if self.stage2 is not None:
            for k, v in self.stage2.items():
                if not k.startswith("_"):
                    X[k] = v[rows]
            p1 = self.stage2["p1"]
            for tag, key, full in [("sib1", "_sib1", True), ("sib2", "_sib2", False)]:
                srow = self.stage2[key][rows]
                ok = srow >= 0
                sj = c.j.values[np.where(ok, srow, 0)]
                X[f"{tag}_p"] = np.where(ok, p1[np.where(ok, srow, 0)], -1).astype(np.float32)
                sn, sa, sh = s23.name_core.values[sj], s23.addr_n.values[sj], s23.hnum.values[sj]
                X[f"{tag}_nratio"] = np.where(ok, cp(bn, sn, fuzz.ratio), -1).astype(np.float32)
                if full:
                    X[f"{tag}_nskel"] = np.where(ok, cp(s23.name_skel.values[J], s23.name_skel.values[sj], fuzz.ratio),
                                                 -1).astype(np.float32)
                X[f"{tag}_aset"] = np.where(ok, cp(ba, sa, fuzz.token_set_ratio), -1).astype(np.float32)
                both = ok & (hb != "") & (sh != "")
                X[f"{tag}_heq"] = np.where(ok, (both & (hb == sh)).astype(np.int8), -1).astype(np.int8)
                X[f"{tag}_hconf"] = np.where(ok, (both & (hb != sh)).astype(np.int8), -1).astype(np.int8)
            cols = FEATURES + STAGE2
        return pd.DataFrame(X)[cols]

    def build_matrix(self, rows):
        """Features for `rows`, built in chunks into one float32 matrix (low peak memory)."""
        cols = FEATURES + (STAGE2 if self.stage2 is not None else [])
        M = np.empty((len(rows), len(cols)), dtype=np.float32)
        for s in range(0, len(rows), args.chunk):
            M[s:s + args.chunk] = self.build(rows[s:s + args.chunk]).to_numpy(np.float32)
            gc.collect()
        return M, cols

    def predict_all(self, bst, label=""):
        n = len(self.cand)
        prob = np.zeros(n, np.float32)
        for s in range(0, n, args.chunk):
            rows = np.arange(s, min(s + args.chunk, n))
            prob[rows] = bst.predict(self.build(rows), num_threads=args.n_threads)
            log(f"  {label} predicted {min(s + args.chunk, n):,}/{n:,}")
            gc.collect()
        return prob


# ----------------------------------------------------------------------------- decision + scoring
def one_to_one(J, s, thr):
    """Rows kept: s >= thr, and each S2/S3 record (J) only for its highest-scoring S1 entity."""
    k = np.where(s >= thr)[0]
    k = k[np.argsort(-s[k], kind="stable")]
    _, first = np.unique(J[k], return_index=True)
    return k[first]


def f05_entity(pred, true):
    if not true and not pred: return 1.0
    if not true or not pred:  return 0.0
    tp = len(pred & true)
    if tp == 0: return 0.0
    p, r = tp / len(pred), tp / len(true)
    return 1.25 * p * r / (0.25 * p + r)


def score_rows(I, J, kept, true_sets, eval_ids, s1_ids, s23_ids):
    pred = {}
    for i, j in zip(I[kept], J[kept]):
        pred.setdefault(s1_ids[i], set()).add(s23_ids[j])
    return float(np.mean([f05_entity(pred.get(k, set()), true_sets[k]) for k in eval_ids]))


# ----------------------------------------------------------------------------- output
def write_lists(path, header, s1_ids, s23_ids, I, J):
    order = np.lexsort((J, I))
    I, J = I[order], J[order]
    rows = np.arange(len(s1_ids))
    starts, ends = np.searchsorted(I, rows, "left"), np.searchsorted(I, rows, "right")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"source1_entity_id\t{header}\n")
        for r in rows:
            f.write(s1_ids[r] + "\t" + ",".join(s23_ids[J[starts[r]:ends[r]]]) + "\n")


def self_check(match_path, cand_path, s1_ids, s23_ids):
    """Streams both files line by line (low memory): format, coverage, duplicates, valid ids, matches in candidates."""
    valid, expected = set(s23_ids), list(s1_ids)
    with open(match_path, encoding="utf-8") as fm, open(cand_path, encoding="utf-8") as fc:
        assert fm.readline().rstrip("\n") == "source1_entity_id\tmatched_entity_ids"
        assert fc.readline().rstrip("\n") == "source1_entity_id\tcandidate_entity_ids"
        n = 0
        for lm, lc, k in zip(fm, fc, expected):
            km, vm = lm.rstrip("\n").split("\t")
            kc, vc = lc.rstrip("\n").split("\t")
            assert km == k and kc == k, f"row order / id mismatch at {k}"
            m = vm.split(",") if vm else []
            c = set(vc.split(",")) if vc else set()
            assert len(m) == len(set(m)) and all(x in valid for x in m) and set(m) <= c, k
            n += 1
        assert n == len(expected) and not fm.readline() and not fc.readline(), "row count mismatch"
    log("self-check PASSED")

# ----------------------------------------------------------------------------- training (shared by dev + full)
PARAMS = dict(objective="binary", learning_rate=0.1, num_leaves=127, min_data_in_leaf=200,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1)


def labels(s1_ids, s23_ids, true_sets, I, J):
    pos1 = dict(zip(s1_ids, range(len(s1_ids))))
    pos23 = dict(zip(s23_ids, range(len(s23_ids))))
    n23 = np.int64(len(s23_ids))
    keys = np.fromiter((pos1[k] * n23 + pos23[m] for k, v in true_sets.items() if k in pos1
                        for m in v if m in pos23), np.int64)
    return np.isin(I.astype(np.int64) * n23 + J, keys).astype(np.int8)


def _train(Xa, ya, cols, Xb=None, yb=None, rounds=2000):
    p = dict(PARAMS, num_threads=args.n_threads, seed=args.seed)
    da = lgb.Dataset(Xa, ya, feature_name=cols, free_raw_data=True)
    if Xb is None:
        return lgb.train(p, da, num_boost_round=rounds)
    b = lgb.train(p, da, num_boost_round=rounds, valid_sets=[lgb.Dataset(Xb, yb, feature_name=cols, reference=da)],
                  callbacks=[lgb.early_stopping(50), lgb.log_evaluation(200)])
    return lgb.Booster(model_str=b.model_to_string(num_iteration=b.best_iteration))


def fit_models(fb, y, part, s1_ids, s23_ids, true_sets):
    """part[i] per S1 entity: 0 = fit, 1 = early stopping, 2 = held-out evaluation."""
    I, J = fb.cand.i.values, fb.cand.j.values
    fit_rows, es_rows = np.where(part[I] == 0)[0], np.where(part[I] == 1)[0]
    eval_ids = [k for k in s1_ids[part == 2] if k in true_sets]
    log(f"fit pairs={len(fit_rows):,}  es pairs={len(es_rows):,}  eval entities={len(eval_ids):,}")

    # ---- stage 1
    fb.stage2 = None
    (Xf, cols), (Xe, _) = fb.build_matrix(fit_rows), fb.build_matrix(es_rows)
    log("stage-1 matrices built")
    bst1 = _train(Xf, y[fit_rows], cols, Xe, y[es_rows])
    n1 = bst1.current_iteration()
    log(f"stage 1 trained ({n1} trees)")
    # out-of-fold stage-1 probabilities for the fit rows (2 halves by entity), so stage 2 learns from
    # honest probabilities, like the ones it will see at test time
    half = (np.random.default_rng(args.seed + 1).random(len(s1_ids)) < 0.5)[I[fit_rows]]
    oof = np.zeros(len(fit_rows), np.float32)
    for h in (False, True):
        b = _train(Xf[half != h], y[fit_rows][half != h], cols, rounds=n1)
        oof[half == h] = b.predict(Xf[half == h], num_threads=args.n_threads)
    del Xf, Xe; gc.collect()
    p1 = fb.predict_all(bst1, "stage1")
    p1[fit_rows] = oof

    # ---- stage 2
    fb.stage2 = stage2_features(I, J, p1)
    log("stage-2 context features ready")
    (Xf, cols), (Xe, _) = fb.build_matrix(fit_rows), fb.build_matrix(es_rows)
    log("stage-2 matrices built")
    bst2 = _train(Xf, y[fit_rows], cols, Xe, y[es_rows])
    log(f"stage 2 trained ({bst2.current_iteration()} trees)")
    imp = pd.Series(bst2.feature_importance("gain"), index=FEATURES + STAGE2).sort_values(ascending=False)
    log("stage-2 top features:\n" + (imp / imp.sum()).round(3).head(15).to_string())
    del Xf, Xe; gc.collect()
    p2 = fb.predict_all(bst2, "stage2")

    # ---- threshold search on held-out entities, global one-to-one (as at test time)
    report = {"stage1": {}, "stage2": {}}
    for name, s in [("stage1", p1), ("stage2", p2)]:
        for thr in np.round(np.arange(0.40, 0.951, 0.05), 2):
            report[name][float(thr)] = score_rows(I, J, one_to_one(J, s, thr), true_sets, eval_ids,
                                                  s1_ids, s23_ids)
        b = max(report[name], key=report[name].get)
        log(f"{name}: best thr={b}  held-out F0.5={report[name][b]:.4f}   " +
            " ".join(f"{t}:{v:.4f}" for t, v in report[name].items()))
    report["baseline_mean055"] = score_rows(I, J, one_to_one(J, fb.comp["m"], 0.55), true_sets, eval_ids,
                                            s1_ids, s23_ids)
    thr = max(report["stage2"], key=report["stage2"].get)
    log(f"BEST stage2 thr={thr}  held-out F0.5={report['stage2'][thr]:.4f}  "
        f"(stage1 best {max(report['stage1'].values()):.4f}, baseline {report['baseline_mean055']:.4f})")
    report["best_thr"] = thr
    report["importance"] = (imp / imp.sum()).round(4).to_dict()
    fb.last_probs = (p1, p2)                 # kept so train() can save them for later blending
    return bst1, bst2, thr, report


def train():
    s1, s23, cand = load_split("train")
    fb = FeatureBuilder(s1, s23, cand)
    s1_ids, s23_ids = s1.entity_id.values, s23.entity_id.values
    gt = load("train_ground_truth.tsv")
    true_sets = {k: set(x for x in v.split(",") if x) for k, v in zip(gt.source1_entity_id, gt.matched_entity_ids)}
    y = labels(s1_ids, s23_ids, true_sets, cand.i.values, cand.j.values)
    log(f"labels: {y.sum():,} positives of {len(y):,} pairs")
    u = np.random.default_rng(args.seed).random(len(s1_ids))
    part = np.where(u < args.fit_frac, 0, np.where(u < args.fit_frac + args.es_frac, 1, 2))
    blocked = np.zeros(len(s1_ids), bool); blocked[np.unique(cand.i.values)] = True
    part[~blocked] = 3                     # entities not blocked (s1_frac < 1): excluded from fit AND scoring
    log(f"entities used: {blocked.sum():,} of {len(s1_ids):,}")
    bst1, bst2, thr, report = fit_models(fb, y, part, s1_ids, s23_ids, true_sets)
    os.makedirs(args.out_dir, exist_ok=True)
    bst1.save_model(os.path.join(args.out_dir, "lgb_stage1.txt"))
    bst2.save_model(os.path.join(args.out_dir, "lgb_stage2.txt"))
    with open(os.path.join(args.out_dir, "train_report.json"), "w") as f:
        json.dump(report, f, indent=1)
    p1, p2 = fb.last_probs
    pd.DataFrame({"i": cand.i.values.astype(np.int32), "j": cand.j.values.astype(np.int32),
                  "p1": p1, "p2": p2, "y": y, "part": part[cand.i.values].astype(np.int8)}).to_parquet(
        os.path.join(args.out_dir, "train_probs.parquet"), index=False)
    log("train_probs.parquet saved")
    del fb, cand, y, p1, p2; gc.collect()
    return bst1, bst2, thr


# ----------------------------------------------------------------------------- test
def test(bst1, bst2, thr):
    """Country by country: candidate pairs never cross countries, so all competition / sibling features are
    identical to a single pass, while peak memory is roughly halved."""
    s1, s23, cand = load_split("test")
    cty = load("test_source1.tsv").country.values
    I, J = cand.i.values, cand.j.values
    p1 = np.zeros(len(I), np.float32)
    prob = np.zeros(len(I), np.float32)
    base = FeatureBuilder(s1, s23, cand.iloc[:1].reset_index(drop=True))
    pair_cty = cty[I]
    for c in pd.unique(cty):
        rows = np.where(pair_cty == c)[0]
        if len(rows) == 0:
            continue
        fb = base.with_cand(cand.iloc[rows].reset_index(drop=True))
        sI, sJ = fb.cand.i.values, fb.cand.j.values
        a = fb.predict_all(bst1, f"{c} stage1")
        fb.stage2 = stage2_features(sI, sJ, a)
        b = fb.predict_all(bst2, f"{c} stage2")
        p1[rows], prob[rows] = a, b
        del fb, a, b, sI, sJ
        gc.collect()
        log(f"{c}: {len(rows):,} pairs done")
    kept = one_to_one(J, prob, thr)
    s1_ids, s23_ids = s1.entity_id.values, s23.entity_id.values
    log(f"test matches: {len(kept):,}  entities with >=1 match: {len(np.unique(I[kept])):,} / {len(s1_ids):,}")
    os.makedirs(args.out_dir, exist_ok=True)
    pd.DataFrame({"i": I.astype(np.int32), "j": J.astype(np.int32), "p1": p1, "prob": prob}).to_parquet(
        os.path.join(args.out_dir, "test_probs.parquet"), index=False)
    log("test_probs.parquet saved")
    mpath = os.path.join(args.out_dir, "matching_results.tsv")
    cpath = os.path.join(args.out_dir, "candidate_pairs.tsv")
    write_lists(mpath, "matched_entity_ids", s1_ids, s23_ids, I[kept], J[kept])
    write_lists(cpath, "candidate_entity_ids", s1_ids, s23_ids, I, J)
    log("files written")
    self_check(mpath, cpath, s1_ids, s23_ids)


def main():
    if args.mode in ("train", "both"):
        bst1, bst2, thr = train()
    else:
        bst1 = lgb.Booster(model_file=find("lgb_stage1.txt"))
        bst2 = lgb.Booster(model_file=find("lgb_stage2.txt"))
        thr = json.load(open(find("train_report.json")))["best_thr"]
    if args.mode in ("test", "both"):
        test(bst1, bst2, thr)


if __name__ == "__main__":
    main()
