"""
ResolvAI - Business Entity Resolution
BASELINE PIPELINE (Route A)

  raw TSVs -> normalize names/addresses -> TF-IDF blocking (3 channels, per country)
  -> cosine similarity scoring -> threshold on mean similarity -> one-to-one assignment
  -> output/matching_results.tsv + output/candidate_pairs.tsv

Usage:
  python baseline_pipeline.py --data_root data --split test --out_dir output
"""
import os, re, csv, gc, glob, time, argparse
import numpy as np
import pandas as pd
from unidecode import unidecode
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

# ----------------------------------------------------------------------------- config
parser = argparse.ArgumentParser()
parser.add_argument("--data_root", default="data")   # searched recursively for the TSVs
parser.add_argument("--split", default="test")                # "test" or "train"
parser.add_argument("--out_dir", default="output")
parser.add_argument("--K", type=int, default=30)              # candidates kept per channel
parser.add_argument("--block_max_df", type=float, default=0.01)
parser.add_argument("--threshold", type=float, default=0.55)  # tuned on dev: mean-sim >= 0.55
parser.add_argument("--n_threads", type=int, default=4)
parser.add_argument("--s1_frac", type=float, default=1.0)   # block only a random share of S1 (train speed-up)
args, _ = parser.parse_known_args()                           # ignores Jupyter's own args


def log(msg, t0=[time.time()]):
    print(f"[{time.time() - t0[0]:7.0f}s] {msg}", flush=True)


# ----------------------------------------------------------------------------- loading
def find(name):
    hits = glob.glob(os.path.join(args.data_root, "**", name), recursive=True)
    if not hits:
        raise FileNotFoundError(name)
    return hits[0]


def load(name):
    # QUOTE_NONE is essential: names containing '"' otherwise swallow thousands of rows
    df = pd.read_csv(find(name), sep="\t", dtype=str, keep_default_na=False,
                     quoting=csv.QUOTE_NONE, on_bad_lines="warn")
    for c in df.columns:
        df[c] = df[c].str.strip()
    return df


# ----------------------------------------------------------------------------- normalization
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
    "centre": "center", "first": "1st", "second": "2nd", "third": "3rd", "fourth": "4th",
    "north": "n", "south": "s", "east": "e", "west": "w"}
ADDR_DROP = {"unit", "no", "house", "hno", "h", "num", "number", "flat", "door", "plot", "apt",
             "apartment", "suite", "ste", "floor", "fl", "flr", "n", "o"}
LEGAL = {"inc", "incorporated", "llc", "lp", "llp", "ltd", "limited", "pvt", "private", "corp",
         "corporation", "co", "company", "plc", "pllc", "pc", "sa", "sas", "sasu", "sarl", "eurl", "sci",
         "snc", "gmbh", "opc", "the", "and"}
JUNK = {"m", "s", "ms", "smt", "shri", "sri", "dr", "mr", "mrs", "www", "com", "net", "org",
        "center", "centre", "group"}
STATE_TABLES = {"US": US_STATES, "India": IN_STATES}


def _ascii(s):
    return unidecode(s or "").lower().replace("&", " and ")


def _replace_phrases(s, table):
    for k in sorted(table, key=len, reverse=True):     # "west bengal" before "west"
        if " " in k and k in s:
            s = re.sub(rf"\b{k}\b", table[k], s)
    return s


def norm_name(name):
    s = re.sub(r"\.(com|in|net|org|co)\b", " ", _ascii(name))
    seen, full = set(), []
    for t in re.findall(r"[a-z0-9]+", s):             # drop repeated words
        if t not in seen:
            seen.add(t)
            full.append(t)
    core = [t for t in full if t not in LEGAL and t not in JUNK] or full
    return " ".join(core)


def norm_addr(addr, country):
    table = STATE_TABLES.get(country, {})
    s = _replace_phrases(_ascii(addr).replace("n°", " "), table)
    out = []
    for t in re.findall(r"[a-z0-9]+", s):
        t = CITY_ALIAS.get(t, t)
        t = table.get(t, t)
        if country == "France" and t == "r":
            t = "rue"
        t = ADDR_MAP.get(t, t)
        if t.isdigit():
            t = t.lstrip("0") or "0"
        if t not in ADDR_DROP:
            out.append(t)
    return " ".join(out)


def normalize_df(df):
    out = pd.DataFrame({"entity_id": df.entity_id.values, "country": df.country.values})
    out["name_core"] = [norm_name(x) for x in df.business_name.values]
    out["addr_n"] = [norm_addr(a, c) for a, c in zip(df.business_address.values, df.country.values)]
    out["combo_text"] = out.name_core + " " + out.addr_n
    return out


# ----------------------------------------------------------------------------- vectorizers
def block_vectorizers(max_df):
    """Pruned vectorizers used ONLY to find candidates fast (common n-grams removed)."""
    char = lambda n: TfidfVectorizer(analyzer="char_wb", ngram_range=(n, n), min_df=2, max_df=max_df,
                                     sublinear_tf=True, dtype=np.float32)
    word = lambda: TfidfVectorizer(token_pattern=r"[a-z0-9]+", max_df=max_df,
                                   sublinear_tf=True, dtype=np.float32)
    return [("name", char(3), "name_core"), ("addr", word(), "addr_n"), ("combo", char(4), "combo_text")]


def score_vectorizers():
    """Full vectorizers used to SCORE candidate pairs (same as the dev experiments)."""
    char = lambda: TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=2,
                                   sublinear_tf=True, dtype=np.float32)
    word = lambda: TfidfVectorizer(token_pattern=r"[a-z0-9]+", sublinear_tf=True, dtype=np.float32)
    return [("name", char(), "name_core"), ("addr", word(), "addr_n"), ("combo", char(), "combo_text")]


def fit_transform(vec, a_txt, b_txt):
    try:
        vec.fit(pd.concat([a_txt, b_txt], ignore_index=True))
        return vec.transform(a_txt), vec.transform(b_txt)
    except ValueError:                                   # empty vocabulary (tiny/odd country)
        return None, None


# ----------------------------------------------------------------------------- per-country work
def run_country(a, b):
    """a, b: normalized frames for one country (index = global row position).
    Returns global (I, J) candidate pairs and their 3 cosine sims."""
    small = len(a) + len(b) < 20000
    max_df = 1.0 if small else args.block_max_df

    # 1) blocking: top-K per channel, union
    nb = len(b)
    keys = []
    for ch, vec, col in block_vectorizers(max_df):
        XA, XB = fit_transform(vec, a[col], b[col])
        if XA is None:
            continue
        C = sp_matmul_topn(XA, XB.T.tocsr(), top_n=args.K, threshold=0.0, sort=False,
                           n_threads=args.n_threads).tocoo()
        keys.append(C.row.astype(np.int64) * nb + C.col.astype(np.int64))
        del XA, XB, C
        gc.collect()
    if not keys:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros((0, 3), np.float32)
    keys = np.unique(np.concatenate(keys))
    ra, rb = keys // nb, keys % nb
    del keys

    # 2) scoring: exact cosine for all 3 channels on every candidate pair
    sims = np.zeros((len(ra), 3), dtype=np.float32)
    for c, (ch, vec, col) in enumerate(score_vectorizers()):
        XA, XB = fit_transform(vec, a[col], b[col])
        if XA is None:
            continue
        for s in range(0, len(ra), 2_000_000):
            sims[s:s + 2_000_000, c] = np.asarray(
                XA[ra[s:s + 2_000_000]].multiply(XB[rb[s:s + 2_000_000]]).sum(1)).ravel()
        del XA, XB
        gc.collect()
    return a.index.values[ra], b.index.values[rb], sims


# ----------------------------------------------------------------------------- output
def write_lists(path, header, s1_ids, s23_ids, I, J):
    """One row per S1 entity (all of them, in file order); comma-separated S2/S3 ids; empty if none."""
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

# ----------------------------------------------------------------------------- main
def main():
    os.makedirs(args.out_dir, exist_ok=True)
    sp = args.split
    s1 = normalize_df(load(f"{sp}_source1.tsv"))
    s23 = normalize_df(pd.concat([load(f"{sp}_source2.tsv"), load(f"{sp}_source3.tsv")], ignore_index=True))
    log(f"loaded + normalized: S1={len(s1):,}  S2+S3={len(s23):,}")

    keep = np.random.default_rng(123).random(len(s1)) < args.s1_frac   # all True when s1_frac = 1.0
    log(f"blocking {keep.sum():,} of {len(s1):,} S1 entities (s1_frac={args.s1_frac})")
    Is, Js, Ss = [], [], []
    for cty in s1.country.unique():                     # open set of countries (France included)
        a, b = s1[(s1.country == cty).values & keep], s23[s23.country == cty]
        if len(b) == 0:
            log(f"{cty!r}: no S2/S3 records, all singletons")
            continue
        I, J, S = run_country(a, b)
        Is.append(I); Js.append(J); Ss.append(S)
        log(f"{cty!r}: S1={len(a):,} pool={len(b):,} candidates={len(I):,} ({len(I)/max(len(a),1):.1f}/entity)")
        gc.collect()

    I = np.concatenate(Is) if Is else np.zeros(0, np.int64)
    J = np.concatenate(Js) if Js else np.zeros(0, np.int64)
    S = np.concatenate(Ss) if Ss else np.zeros((0, 3), np.float32)
    pd.DataFrame({"i": I.astype(np.int32), "j": J.astype(np.int32), "name": S[:, 0], "addr": S[:, 1],
                  "combo": S[:, 2]}).to_parquet(os.path.join(args.out_dir, f"{sp}_cand_scored.parquet"),
                                                index=False)

    # decision rule: mean similarity >= threshold, then each S2/S3 record goes to its best S1 only
    score = S.mean(1)
    keep = np.where(score >= args.threshold)[0]
    keep = keep[np.argsort(-score[keep], kind="stable")]
    _, first = np.unique(J[keep], return_index=True)
    MI, MJ = I[keep[first]], J[keep[first]]
    log(f"matches: {len(MI):,}  entities with >=1 match: {len(np.unique(MI)):,} / {len(s1):,}")

    s1_ids, s23_ids = s1.entity_id.values, s23.entity_id.values
    mpath = os.path.join(args.out_dir, "matching_results.tsv")
    cpath = os.path.join(args.out_dir, "candidate_pairs.tsv")
    write_lists(mpath, "matched_entity_ids", s1_ids, s23_ids, MI, MJ)
    write_lists(cpath, "candidate_entity_ids", s1_ids, s23_ids, I, J)
    log("files written")
    self_check(mpath, cpath, s1_ids, s23_ids)


if __name__ == "__main__":
    main()
