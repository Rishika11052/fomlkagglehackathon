"""
Step 1: Data exploration for the FoML 2026 dependency parsing competition.

Run from the repo root:
    python src/eda.py

Reads   data/train.csv, data/test.csv, data/sample.csv
Writes  reports/eda_summary.md   (all numbers, for the report)
        reports/figures/*.png    (figures, for the report)
"""
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # save figures without needing a display
import matplotlib.pyplot as plt

DATA = Path("data")
OUT = Path("reports")
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

lines = []  # collected markdown for the summary file


def log(text=""):
    print(text)
    lines.append(text)


def table(series_or_df, title):
    log(f"\n### {title}\n")
    df = series_or_df.to_frame() if isinstance(series_or_df, pd.Series) else series_or_df
    log(df.to_markdown())


def savefig(name):
    plt.tight_layout()
    plt.savefig(FIG / name, dpi=130)
    plt.close()


# ---------------------------------------------------------------- load
train = pd.read_csv(DATA / "train.csv", keep_default_na=False)
test = pd.read_csv(DATA / "test.csv", keep_default_na=False)
sample = pd.read_csv(DATA / "sample.csv", keep_default_na=False)
# keep_default_na=False so tokens like "NA", "null", "nan" stay as strings

# head id
train["head"] = train["target"].str.split("|").str[0].astype(int)
# dependency relation
train["rel"] = train["target"].str.split("|").str[1]
# base relation before ":" if it exists
train["base_rel"] = train["rel"].str.split(":").str[0]
# lowercased token to do vocabulary checks
train["lower"] = train["token"].str.lower()
test["lower"] = test["token"].str.lower()

log("# Data exploration summary\n")

# ---------------------------------------------------------------- 1. sizes and integrity
log("## 1. Sizes and integrity\n")
# iterate through both datasets
for name, df in [("train", train), ("test", test)]:
    #number of unique sentences, number of tokens, min/max IDs
    log(f"- {name}: {len(df):,} tokens, {df.sentence_id.nunique():,} sentences, "
        f"IDs {df.ID.min()}..{df.ID.max()}")
# Checks the number of rows and column names in the (sample)submission template.
log(f"- sample: {len(sample):,} rows, columns {list(sample.columns)}")
# checks whether all sample IDs are present in either the train or test datasets.
log(f"- sample IDs == train IDs + test IDs: {set(sample.ID) == set(train.ID) | set(test.ID)}")
# finds sentence IDs that occur in both training and test datasets(which should be 0)
log(f"- sentence_id overlap train/test: {len(set(train.sentence_id) & set(test.sentence_id))}")

#check token positioning within sentences
for name, df in [("train", train), ("test", test)]:
    # group tokens belonging to the same sentence and calculate min, max and count number of tokens
    g = df.groupby("sentence_id", sort=False).token_id.agg(["min", "max", "count"])
    # check if min=1 and max = count for all sentences
    ok = ((g["min"] == 1) & (g["max"] == g["count"])).all()
    log(f"- {name}: token_id runs 1..n in every sentence: {ok}")



def check_tree(heads):
    """heads[i] = head of word i+1. 0 is root.
    Returns a dict of pcontaing bad_head, n_roots, cycle and unreachable.
    n is number of tokens in the sentence.
    A valid tree has: every head in 0..n, no self loops, exactly one root,
    no cycles, and every word reaching the root."""
    n = len(heads)
    # checks if head_id is outside 0-n and if token is pointing to itself(self loop)
    bad_head = [i + 1 for i, h in enumerate(heads) if not (0 <= h <= n) or h == i + 1]
    # counts number of tokens with head_id = 0 (only roots have head_id = 0) 
    # n_roots should be 1
    n_roots = sum(h == 0 for h in heads)
    #in_cycle is a set for tokens involved in cycle
    # unreachable is set for tokens that cannot reach the root
    in_cycle, unreachable = set(), set()
    # if there ar eno bad heads then start at each token and recursively follow its head
    # till you either reach the root or reach a token already visitied 
    if not bad_head:
        for start in range(1, n + 1):
            seen, a = [], start
            while a != 0 and a not in seen:
                seen.append(a)
                a = heads[a - 1]
            if a != 0:  # reched a visited node - cycle
                in_cycle.update(seen[seen.index(a):])
                unreachable.add(start)
    return {"bad_head": bad_head, "n_roots": n_roots,
            "cycle": sorted(in_cycle), "unreachable": sorted(unreachable)}

# calls check_tree and returns True if tree i
def tree_is_valid(heads):
    """ calls check_tree and returns True if
    there are no invalid heads
    there ar eno slef loops
    no cycles exist
    every token can reach the root
    """
    p = check_tree(heads)
    return not p["bad_head"] and p["n_roots"] == 1 and not p["cycle"] and not p["unreachable"]


# Manually check check_tree() function 
# using a valid tree, 
assert tree_is_valid([2, 0, 2]) 
# tree with cycle
assert check_tree([2, 1, 0])["cycle"] == [1, 2]
# tree with multiple roots  
assert check_tree([0, 0, 2])["n_roots"] == 2
# tree with invalid head (head 4 doesn't exist)
assert check_tree([4, 0])["bad_head"] == [1] 
# tree with self loop     
assert check_tree([1, 0])["bad_head"] == [1]     

#apply check_tree to all sentences in train
# store the results in a dict-problems indexed by sentence_id
problems = {sid: check_tree(s["head"].tolist())
            for sid, s in train.groupby("sentence_id", sort=False)}
# extracts the number of roots for each sentence and stores it in a pandas Series
n_roots = pd.Series({sid: p["n_roots"] for sid, p in problems.items()})
# prints summary statistics about dependency trees 
log(f"- train: root count per sentence: {n_roots.value_counts().to_dict()}  (key = number of roots)")

# checks each type of problem (bad_head, cycle, unreachable) and logs the number of sentences with that problem
for key, label in [("bad_head", "heads outside 0..n or self-loops"),
                   ("cycle", "cycles"), ("unreachable", "words that cannot reach the root")]:
    bad = [sid for sid, p in problems.items() if p[key]]
    log(f"- train: sentences with {label}: {len(bad)}" + (f"  e.g. {bad[:5]}" if bad else ""))
bad_roots = n_roots[n_roots != 1]
if len(bad_roots):
    log(f"- train: sentences with zero or several roots: {list(bad_roots.index[:10])}")
valid_ids = {sid for sid, p in problems.items()
             if not p["bad_head"] and p["n_roots"] == 1 and not p["cycle"] and not p["unreachable"]}
log(f"- train: valid trees: {len(valid_ids)} / {len(problems)}  (later tree statistics use only these)")

# exact duplicate sentences between train and test (worth knowing, not to exploit blindly)
#convert each sentence into a single string by joining its tokens with spaces
sent_text = lambda df: df.groupby("sentence_id", sort=False).token.apply(" ".join)
#Generate sentence strings for train and test
tr_text, te_text = sent_text(train), sent_text(test)
# check duplicates by checking if each test sentence string is in the set of train sentence strings
dup = te_text.isin(set(tr_text))
#print duplicate statistics
log(f"- test sentences whose exact text also appears in train: {dup.sum()} "
    f"({dup.mean():.1%}); tokens covered: "
    f"{test.sentence_id.isin(te_text[dup].index).mean():.1%}")

# ---------------------------------------------------------------- 2. sentence lengths
#Compare the length of sentences in the train and test datasets

log("\n## 2. Sentence lengths\n")
#no of tokens in each sentence for train and test datasets
len_tr = train.groupby("sentence_id").size()
len_te = test.groupby("sentence_id").size()
#creates a table comparing train and test based on count, avg sentence length, std deviation , 25 , 50 , 75 percentiles and longest sentence
table(pd.DataFrame({"train": len_tr.describe(), "test": len_te.describe()}).round(2),
      "Sentence length (tokens)")

#sentence length groups
bins = [0, 5, 10, 20, 30, 50, 1000]
labels = ["1-5", "6-10", "11-20", "21-30", "31-50", "51+"]


def tok_share(L):
    """share of all tokens that sit in sentences of each length bin
    no of tokens in each bucket/ total no of tokens in the dataset"""
    return L.groupby(pd.cut(L, bins, labels=labels), observed=False).sum() / L.sum()

#table for length bucket comparison
table(pd.DataFrame({
    "train sents %": pd.cut(len_tr, bins, labels=labels).value_counts(normalize=True, sort=False) * 100,
    "test sents %": pd.cut(len_te, bins, labels=labels).value_counts(normalize=True, sort=False) * 100,
    "train tokens %": tok_share(len_tr) * 100,
    "test tokens %": tok_share(len_te) * 100,
}).round(1), "Length buckets (share of sentences and of tokens)")

#plot for sentence length distribution
# overlapping histograms for train and test
plt.figure(figsize=(7, 4))
plt.hist([len_tr.clip(upper=60), len_te.clip(upper=60)], bins=30, density=True,
         label=["train", "test"], alpha=0.8)
plt.xlabel("sentence length (clipped at 60)"); plt.ylabel("density"); plt.legend()
plt.title("Sentence length: train vs test")
savefig("01_sentence_length.png")

# ---------------------------------------------------------------- 3. relations
log("\n## 3. Dependency relations\n")
#counts number of unique dependency relation labels and base lables(before :)
log(f"- {train.rel.nunique()} relation labels, {train.base_rel.nunique()} base labels (before ':')")
rel_counts = train.rel.value_counts()
table((rel_counts / len(train) * 100).round(2).rename("% of tokens"), "Relation frequency")
log(f"\n- top 10 relations cover {rel_counts.head(10).sum() / len(train):.1%} of tokens")
log(f"- relations with < 100 examples: {list(rel_counts[rel_counts < 100].index)}")

#assign length buckets to sentences in train based on their sentence length
train["len_bucket"] = train.sentence_id.map(pd.cut(len_tr, bins, labels=labels))
# create a crosstab of relation labels vs sentence length buckets, normalized by columns to get percentages
rel_by_len = (pd.crosstab(train.rel, train.len_bucket, normalize="columns") * 100) \
    .loc[rel_counts.head(15).index].round(1)
#this table shows how relation frequencies change with sentence length
table(rel_by_len, "Top 15 relations: % of tokens within each sentence-length bucket")


#plot relational frequencies
#bar chart
plt.figure(figsize=(10, 4))
rel_counts.plot.bar()
plt.yscale("log"); plt.ylabel("count (log)"); plt.title("Relation frequency (train)")
savefig("02_relation_frequency.png")

# ---------------------------------------------------------------- 4. head distance and direction
log("\n## 4. Head distance and direction\n")
#how far the head of a token is from the token itself and whether it is in the left or right direction

#selects tokens whose head > 0
nonroot = train[train["head"] > 0].copy()
# calculate signed head distance
nonroot["dist"] = nonroot["head"] - nonroot["token_id"]  # >0 meaning head to the right
nonroot["absdist"] = nonroot["dist"].abs()
#print the head distance statistics
log(f"- head is the adjacent word: {(nonroot.absdist == 1).mean():.1%}")
log(f"- head within 2 words: {(nonroot.absdist <= 2).mean():.1%}")
log(f"- head within 5 words: {(nonroot.absdist <= 5).mean():.1%}")
log(f"- head within 10 words: {(nonroot.absdist <= 10).mean():.1%}")
log(f"- head to the right of the word: {(nonroot.dist > 0).mean():.1%}")

#groups tokens based on dependency relations and calculates
# count of tokens, percentage tokens to the right , median abs dist to the head and avg dist to the head for each relation
per_rel = nonroot.groupby("rel").agg(
    count=("dist", "size"),
    head_right_pct=("dist", lambda d: (d > 0).mean() * 100),
    median_absdist=("absdist", "median"),
    mean_absdist=("absdist", "mean"),
).sort_values("count", ascending=False).round(1)
table(per_rel, "Per relation: direction and distance")

#plot bar chart of signed head distances
plt.figure(figsize=(8, 4))
nonroot["dist"].clip(-15, 15).value_counts().sort_index().plot.bar()
plt.xlabel("head position - word position (clipped to ±15)"); plt.ylabel("count")
plt.title("Signed head distance (train)")
savefig("03_head_distance.png")

#select root tokens
root_rows = train[train["head"] == 0].merge(len_tr.rename("n"), left_on="sentence_id", right_index=True)
#calculates relative position of root in sentence and prints median value
log(f"\n- root relative position in sentence (0=start, 1=end): "
    f"median {((root_rows.token_id - 1) / (root_rows.n - 1).clip(lower=1)).median():.2f}")
table(root_rows.lower.value_counts().head(15).rename("count"), "Most common root words")

# ---------------------------------------------------------------- 5. tree shape
log("\n## 5. Tree shape\n")
#structure of dependency tree: projective vs non-projective, depth of tree, number of dependents per head

def is_nonprojective_arc(heads):
    """heads: list where heads[i] is the head of word i+1.
    Returns per-arc flags: True if some word between dependent and head
    is not a descendant of the head."""
    n = len(heads)
    flags = []
    # Iterate through each dependent token and get its head
    for d in range(1, n + 1):
        h = heads[d - 1]
        #find words betweenthe dependent and its head
        lo, hi = sorted((d, h))
        bad = False
        #check every word to see if it is a descendant of the head
        for k in range(lo + 1, hi):
            a = k  # go upwards to reach the root
            #if h is reached before root then projective arc
            while a != 0 and a != h:
                a = heads[a - 1]
            #if root is reached before h then non-projective arc
            if a != h:
                bad = True
                break
        flags.append(bad)
    return flags


# manually check the is_nonprojective_arc() function
# a simple projective tree
assert is_nonprojective_arc([2, 3, 0]) == [False, False, False]
# A projective tree with nested dependencies.
assert is_nonprojective_arc([2, 0, 2, 3]) == [False, False, False, False]
#A tree containing crossing/non-projective dependencies.
assert is_nonprojective_arc([0, 4, 1, 1]) == [False, True, False, False]
# a non-projective tree with a single crossing dependency.
assert is_nonprojective_arc([3, 3, 0, 1]) == [False, False, False, True]

#check for all valid trees
np_flags = []
np_sent = 0
max_depth = []
for sid, s in train.groupby("sentence_id", sort=False):
    if sid not in valid_ids:  
        continue
    heads = s["head"].tolist()
    f = is_nonprojective_arc(heads)
    np_flags.extend(f)
    np_sent += any(f)
    depths = []
    for d in range(1, len(heads) + 1):
        a, k = d, 0
        while a != 0:
            a, k = heads[a - 1], k + 1
        depths.append(k)
    max_depth.append(max(depths))
log(f"- non-projective arcs: {np.mean(np_flags):.2%} of tokens")
log(f"- sentences with at least one non-projective arc: {np_sent / len(valid_ids):.2%}")
log("  (low values mean Eisner, a projective decoder, loses almost nothing versus Chu-Liu-Edmonds)")
log(f"- tree depth: median {np.median(max_depth):.0f}, max {max(max_depth)}")

children = train[train["head"] > 0].groupby(["sentence_id", "head"]).size()
log(f"- dependents per head (among words that have any): mean {children.mean():.2f}, max {children.max()}")

# ---------------------------------------------------------------- 6. tokens, vocabulary, ambiguity
log("\n## 6. Tokens and vocabulary\n")

# Calculate the number of unique lowercased words in the training dataset.
vocab = set(train.lower)
log(f"- train vocabulary (lowercased): {len(vocab):,} types")
log(f"- test tokens unseen in train: {(~test.lower.isin(vocab)).mean():.1%}")
log(f"- test types unseen in train: {len(set(test.lower) - vocab) / test.lower.nunique():.1%}")

# OOV(Out of Vocabulary) rate = percentage of unique test words that are missing from the training vocabulary.
# Estimate OOV rate for validation
#estimates how well the model might do on unseen data
rng = np.random.default_rng(42)
sids = train.sentence_id.unique()
val_ids = set(rng.choice(sids, size=len(sids) // 10, replace=False))
tr_part = train[~train.sentence_id.isin(val_ids)]
va_part = train[train.sentence_id.isin(val_ids)]
log(f"- (preview) OOV rate of a random 10% sentence split of train: "
    f"{(~va_part.lower.isin(set(tr_part.lower))).mean():.1%}  <- should be close to the test rate")

#classify token shape
def shape(t):
    if not t:
        return "empty"
    if all(not c.isalnum() for c in t):
        return "punct"
    if t.isdigit() or t.replace(".", "").replace(",", "").isdigit():
        return "number"
    if t.isupper() and len(t) > 1:
        return "ALLCAPS"
    if t[0].isupper():
        return "Capitalised"
    if t.islower():
        return "lower"
    return "mixed"

#compare token shapes in train and test datasets
table(pd.DataFrame({
    "train %": train.token.map(shape).value_counts(normalize=True) * 100,
    "test %": test.token.map(shape).value_counts(normalize=True) * 100,
}).round(1), "Token shape")

# groups tokens by their lowercased word and finds the frequency of the most common relation for each word.
# Estimate relation accuracy when assigning each word its most frequent relation (seen in train and scored on train as well)
word_major = train.groupby("lower").rel.agg(lambda r: r.value_counts().iloc[0])
log(f"\n- if every word got its most frequent relation (seen in train, scored on train): "
    f"{word_major.sum() / len(train):.1%} relation accuracy (an optimistic upper bound for word identity alone)")
# amiguous words
amb = (train.groupby("lower").rel.nunique()).sort_values(ascending=False)
#how often each ambiguous word occurs
freq = train.lower.value_counts()
# Creates a table containing word frequency and number of relations.
amb_tbl = pd.DataFrame({"count": freq, "n_relations": amb}).query("count >= 200") \
    .sort_values("n_relations", ascending=False).head(15)
amb_tbl["top_relations"] = [", ".join(train[train.lower == w].rel.value_counts().head(3).index)
                            for w in amb_tbl.index]
table(amb_tbl, "Most ambiguous frequent words")

# ---------------------------------------------------------------- 7. rare and unseen words
#This section checks how well we can predict dependency relations for words that appear rarely or do not appear at all in the training portion.
# It uses the 90/10 validation split from Section 6.

log("\n## 7. Rare and unseen words\n")
tr_freq = tr_part.lower.value_counts()
va = va_part.copy()
va["train_freq"] = va.lower.map(tr_freq).fillna(0).astype(int)
va["freq_bucket"] = pd.cut(va.train_freq, [-1, 0, 2, 10, 100, 10**9],
                           labels=["unseen", "1-2", "3-10", "11-100", "100+"])
word_rel = tr_part.groupby("lower").rel.agg(lambda r: r.value_counts().index[0])
overall_major = tr_part.rel.value_counts().index[0]
va["word_major_ok"] = va.lower.map(word_rel).fillna(overall_major) == va.rel
va["head_absdist"] = (va["head"] - va["token_id"]).abs().where(va["head"] > 0)
table(va.groupby("freq_bucket", observed=False).agg(
    tokens_pct=("rel", lambda r: len(r) / len(va) * 100),
    word_majority_rel_acc=("word_major_ok", lambda x: x.mean() * 100),
    median_head_dist=("head_absdist", "median"),
).round(1), "Validation tokens by how often the word was seen in training")

unseen = va[va.freq_bucket == "unseen"]
table(pd.DataFrame({
    "unseen words %": unseen.rel.value_counts(normalize=True) * 100,
    "all words %": va.rel.value_counts(normalize=True) * 100,
}).fillna(0).sort_values("unseen words %", ascending=False).head(12).round(1),
      "Relations of unseen words vs all words")
log("\n- unseen words are mostly content words (names, nouns, adjectives), "
    "so word identity can't help them: suffix and shape features must")

# ---------------------------------------------------------------- 8. suffixes and prefixes
log("\n## 8. Suffixes and prefixes\n")
# checks whether the beginning or ending of a word can help predict its dependency relation
# especially for words rare and unseen dwords uring training.
alpha_tr = tr_part[tr_part.lower.str.isalpha()]
alpha_unseen = unseen[unseen.lower.str.isalpha()]
overall_major = alpha_tr.rel.value_counts().index[0]  # fallback: most common relation of alphabetic words
rows = []
for kind in ["suffix", "prefix"]:
    for k in [1, 2, 3, 4]:
        cut = (lambda s: s.str[-k:]) if kind == "suffix" else (lambda s: s.str[:k])
        aff_rel = alpha_tr.groupby(cut(alpha_tr.lower)).rel.agg(lambda r: r.value_counts().index[0])
        pred = cut(alpha_unseen.lower).map(aff_rel).fillna(overall_major)
        rows.append({"feature": f"{kind} len {k}",
                     "rel acc on unseen alphabetic words %": round((pred == alpha_unseen.rel).mean() * 100, 1)})
rows.append({"feature": f"always predict '{overall_major}' (baseline)",
             "rel acc on unseen alphabetic words %":
                 round((alpha_unseen.rel == overall_major).mean() * 100, 1)})
table(pd.DataFrame(rows).set_index("feature"),
      f"How well one affix predicts the relation of unseen words (n={len(alpha_unseen)})")

long_tr = alpha_tr[alpha_tr.lower.str.len() >= 5]  
suf3 = long_tr.assign(suf=long_tr.lower.str[-3:]).groupby("suf")
suf_tbl = pd.DataFrame({"count": suf3.size(),
                        "word_types": suf3.lower.nunique(),
                        "top_relations": suf3.rel.agg(lambda r: ", ".join(r.value_counts().index[:3])),
                        "top_%": suf3.rel.agg(lambda r: r.value_counts(normalize=True).iloc[0] * 100).round(1)})
table(suf_tbl.query("count >= 300 and word_types >= 20").sort_values("count", ascending=False).head(15),
      "Common 3-letter suffixes of words with 5+ letters (at least 20 different words each)")

# ---------------------------------------------------------------- 9. train vs test side by side
log("\n## 9. Train vs test statistics\n")


def corpus_stats(df, ref_vocab=None):
    shp = df.token.map(shape)
    first = df[df.token_id == 1].token
    last = df[df.token_id == df.groupby("sentence_id").token_id.transform("max")]
    out = {
        "sentences": df.sentence_id.nunique(),
        "tokens": len(df),
        "mean sentence length": df.groupby("sentence_id").size().mean(),
        "% one-token sentences": (df.groupby("sentence_id").size() == 1).mean() * 100,
        "mean token length (chars)": df.token.str.len().mean(),
        "% punctuation tokens": (shp == "punct").mean() * 100,
        "% capitalised tokens": (shp == "Capitalised").mean() * 100,
        "% numbers": (shp == "number").mean() * 100,
        "% sentences starting capitalised": first.map(shape).isin(["Capitalised", "ALLCAPS"]).mean() * 100,
        "% sentences ending in punctuation": (last.token.map(shape) == "punct").mean() * 100,
    }
    if ref_vocab is not None:
        out["% tokens unseen in reference vocab"] = (~df.lower.isin(ref_vocab)).mean() * 100
    return out


table(pd.DataFrame({
    "train (all)": corpus_stats(train),
    "val preview (vs train part)": corpus_stats(va_part, set(tr_part.lower)),
    "test (vs all train)": corpus_stats(test, vocab),
}).round(2), "Corpus statistics")

# ---------------------------------------------------------------- 10. punctuation
log("\n## 7. Punctuation\n")
p = train[train.rel == "punct"]
log(f"- punct is {len(p) / len(train):.1%} of train tokens; every one of them counts for the score")
log(f"- punct tokens whose shape is all-symbols: {(p.token.map(shape) == 'punct').mean():.1%}")
log(f"- all-symbol tokens labelled punct: "
    f"{(train[train.token.map(shape) == 'punct'].rel == 'punct').mean():.1%}")

# ---------------------------------------------------------------- write summary
(OUT / "eda_summary.md").write_text("\n".join(lines), encoding="utf-8")
print(f"\nWrote {OUT / 'eda_summary.md'} and {len(list(FIG.glob('*.png')))} figures in {FIG}")