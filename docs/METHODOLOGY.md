# ResolvAI: Methodology

**Team:** ByteForce  
**Members:** Ajeet Kumar, Anu Priya (IIIT Kottayam)  
**Date:** September 2026

---

## 1. Executive Summary

We resolve Source 2/3 records to Source 1 entities with a four-stage pipeline: per-country **three-channel TF-IDF blocking**
(~75 candidates per entity, ~99% pair recall), a **two-stage LightGBM matcher** with competition and *sibling-agreement*
features, a fine-tuned **multilingual transformer cross-encoder**, and a **LightGBM stacker** that combines them under a
global one-to-one assignment. Every design decision was validated at **full data density** on held-out training entities
(held-out F0.5 = 0.9696; hidden test set F0.5 = 0.951).

---

## 2. Methodology

### 2.1 Problem Analysis

- **Scale:** 2.2M S1 / 10.3M S2+S3 train records, 1.73M / 10.0M test records. 5.6% of S1 entities are singletons; a typical
  entity has 2–5 matches; ~27% of S2/S3 records match nothing (pure distractors).
- **Every true pair shares the country** and ≥1 name or address token; **no S2/S3 record belongs to more than one S1 entity**
  (exploited via one-to-one assignment).
- **Noise:** abbreviations and legal suffixes (Pvt/Private, LLC/(LLC)/-Corporation), repeated words, squashed web-style
  names (`fortunefinance.com`), acronyms (`DAC` = Dorelle's Ace Cargo), full vs. abbreviated states (Missouri/MO, Kerala/KL),
  zero-padded house numbers (`00930`), mailbox noise (`PMB 2140`), ordinal words (`Ninth St`), component reordering.
- **Scripts:** 13.5% of S2/S3 names in India are in native scripts (Hindi, Bengali, Kannada, Malayalam); their romanisations
  are inconsistent (`praaivett limittedd` = Private Limited). S1 names are always Latin.
- **Look-alikes are adversarial:** same name at a different address (chains, e.g. `828` vs `826 William St`) and different
  businesses at the same address. A simple similarity rule tuned on a 10% sample scored 0.92, but only **0.72 at full density**,
  because sampling removes most look-alikes. From then on, all tuning was done at full density.
- **Data pitfall:** names containing `"` break default CSV parsing (thousands of rows silently merged); all files are read with
  `quoting=QUOTE_NONE`.

### 2.2 Solution Strategy

**Approach Type:** Blocking + learned pairwise matcher + transformer cross-encoder + stacking (hybrid)  
**Core Innovation:** (1) *sibling-agreement* and *competition* features that judge a candidate against the entity's other strong
records and against rival entities claiming the same record; (2) out-of-sample cross-encoder scores fed into a stacker with the
same context features; (3) all decisions tuned at the real test density with global one-to-one assignment.

---

## 3. Candidate Generation (Blocking)

- **Normalisation:** Unicode→ASCII transliteration, legal-form and honorific stripping, deduplicated tokens, state/city alias
  tables (US, India), address abbreviation map, number normalisation, French `R`→`rue`.
- **Blocking keys:** per country (country treated as an open set, so France is handled automatically), three TF-IDF channels,
  each returning the top-30 most similar S2/S3 records by sparse cosine (`sparse_dot_topn`):
  1. character 3-grams of the core name (typos, transliterations, squashed names)
  2. word tokens of the normalised address (rare tokens such as locality names and house numbers dominate)
  3. character 4-grams of name + address
  Very frequent n-grams (document frequency > 1%) are pruned for the search only (6× faster, recall loss < 0.2%);
  candidates are then scored with the full vectorisers.
- **Candidate pairs generated:** 130.7M for test (~75.5 per S1 entity); 84.3M for the 50% training sample.
- **How true matches were not lost:** the union of three complementary channels reaches **98.8% pair recall** and a
  **ceiling F0.5 of 0.996** (perfect matching on the candidates), measured on held-out training entities.

---

## 4. Matching Model

**Features used (per candidate pair):**
- *Name:* TF-IDF cosines, Levenshtein ratio, token-sort/set, partial ratio, Jaro-Winkler, space-free ratio, similarity of
  **consonant skeletons** (robust to transliteration), initials / acronym match, name lengths.
- *Address:* token cosine, fuzzy ratios, address-number overlap (count, Jaccard, conflicts, numbers present on one side only),
  primary house number equal / **off by one digit** (near-miss look-alikes), empty-address and native-script flags.
- *Competition context:* rank and gap of the pair versus the entity's best candidate, and versus the best rival entity for the
  same record; stage-1 probabilities of all competitors (stage 2).
- *Sibling agreement (stage 2):* similarity of the candidate's name, address and house number to the entity's two strongest
  *other* records. A true record agrees with its siblings; a distractor disagrees.

**Model type:**
1. Two-stage LightGBM (stage 2 trained on out-of-fold stage-1 probabilities).
2. Cross-encoder: `paraphrase-multilingual-MiniLM-L12-v2` (Apache-2.0, 118M params) fine-tuned as a binary classifier on
   2.5M hard pairs (`name ; address` of both records), fp16, one epoch.
3. LightGBM stacker over LightGBM probabilities, cross-encoder score, their ranks/gaps within entity and within record,
   counts of confident candidates and sibling cross-encoder scores, trained only on entities the cross-encoder never saw.

**Threshold selection method:** F0.5 macro-averaged over held-out training entities, computed exactly like the official
evaluation (singletons included) after **global one-to-one assignment** (each S2/S3 record goes only to its highest-scoring S1 entity).
We compared a fixed threshold, a 50/50 blend, the stacker and a per-entity expected-F0.5 decision rule, and kept the best
(stacker, threshold 0.725).

---

## 5. Results & Error Analysis

| Stage (held-out training entities, full density) | F0.5 |
|---|---|
| TF-IDF similarity rule | 0.718 |
| LightGBM stage 1 | 0.932 |
| + stage 2 context features | 0.937 |
| + sibling-agreement features (v3) | 0.943 |
| + cross-encoder (50/50 blend) | 0.967 |
| + stacker | **0.9696** |

Hidden test set: 0.641 (rule) → 0.917 (LightGBM) → 0.948 (cross-encoder blend) → **0.951** (stacker).

- **Common false positives (wrong merges):** about two thirds are near-identical distractors that belong to no entity
  (`Nexeus Printing, 1521 8th Ave` vs `Nexeas Printing, #1526 Eighth Ave`; `Nowak Liquor, 262` vs `283 Arnold Rd`), and
  empty-address records of chain names shared by several S1 entities in different cities.
- **Common false negatives (missed matches):** records whose name is in a native script with extra words
  (`হরি ইম্পেক্স কনস্ট্রাকশনস`), squashed or heavily abbreviated names with empty addresses, and ~4% of true pairs never
  retrieved by blocking. India (0.916) was consistently harder than the US (0.950).

---

## 6. Conclusion

A carefully validated blocking + matching pipeline, evaluated at the true data density, lifted the score from 0.64 to 0.95.
The largest single gains came from full-scale evaluation (exposing look-alikes), sibling-agreement features, and a
multilingual cross-encoder, which also transferred well to the unseen French data. The main lesson was that sampling a
matching problem hides its hardest cases, so validation has to preserve competition density.

---

## Appendix

### A. Code Artefacts
The repository contains `src/` (five scripts), `README.md` (exact commands, hardware and runtimes), `requirements.txt`
and `tests/` (synthetic end-to-end run). Entry points in order: `baseline_pipeline.py` (blocking, test and train) → `route_b.py --mode train` →
`route_b.py --mode test` → `stack.py` (writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`).

### B. Additional Results
- Blocking channel recall (held-out): name 0.741, address 0.923, name+address 0.971, union **0.988**.
- Stacker decision comparison (held-out): v3 alone 0.9429, blend 0.9671, stack 0.9696, blend expected-F0.5 0.9618,
  stack expected-F0.5 0.9696.
