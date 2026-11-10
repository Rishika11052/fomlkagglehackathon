"""
    Two simple reference baselines.

  A. word_majority : Assign each word its most frequent relation in the training data
                     choose the most common head position for that relation.
  B. rules         : manually defined rules to find the head
                     finx any problems to ensure output is valid tree

Run from the repo root:
    python src/baselines.py
When evaluating on validation data, statistics are calculated using only the training split to avoid data leakage.
When making predictions on the test set statistics are calculated using the entire train.csv file.
Writes: experiments/results.csv (2 rows), experiments/preds/baseline_*.csv,
        submissions/baseline_<best>.csv
"""
import pandas as pd

from data import train_val, load_train, load_test
from evaluate import evaluate, print_report, log_result, save_val_preds, check_tree
from submit import make_submission

RARE_MAX = 2      # words seen <= 2 times act as unseen words
SMOOTH = 5        # pseudo-counts when estimating how often a word is the root

# relation groups used by the rules in baseline B
NOMINAL_MOD = {"det", "det:predet", "amod", "nummod", "compound", "nmod:poss", "nmod:desc", "case"}
CLAUSE_DEP = {"nsubj", "nsubj:pass", "nsubj:outer", "csubj", "csubj:pass", "aux", "aux:pass",
              "cop", "mark", "expl", "advmod", "cc", "cc:preconj"}
NOMINAL_HEAD = {"nsubj", "nsubj:pass", "nsubj:outer", "obj", "iobj", "obl", "obl:unmarked",
                "obl:agent", "nmod", "nmod:unmarked", "appos", "conj", "root", "list", "vocative"}
CLAUSE_HEAD = {"root", "ccomp", "xcomp", "advcl", "advcl:relcl", "acl", "acl:relcl",
               "parataxis", "csubj", "conj"}
ATTACH_LEFT_CLAUSE = {"obj", "iobj", "xcomp", "ccomp", "advcl", "obl", "obl:agent",
                      "obl:unmarked", "compound:prt"}            
ATTACH_LEFT_NOMINAL = {"nmod", "nmod:unmarked", "appos", "acl", "acl:relcl"}  
ATTACH_ROOT = {"parataxis", "list", "vocative", "discourse", "dislocated", "orphan", "dep", "reparandum"}


def shape(t):
    """Return a string describing the shape of a token, used for unseen words.
    Punctuation, numbers, capitalisation and length are the main features."""
    if not t:
        return "empty"
    if not any(c.isalnum() for c in t):
        return "punct"
    if t.replace(".", "").replace(",", "").isdigit():
        return "number"
    if t.isupper() and len(t) > 1:
        return "ALLCAPS"
    if t[0].isupper():
        return "Capitalised"
    return "lower"


def majority(keys, rels):
    # most frequent relation for each key
    # ties are broken alphabetically
    df = pd.DataFrame({"k": keys, "rel": rels})
    counts = df.groupby(["k", "rel"]).size().rename("n").reset_index()
    counts = counts.sort_values(["k", "n", "rel"], ascending=[True, False, True])
    return counts.drop_duplicates("k").set_index("k")["rel"]


# ---------------------------------------------------------------- learn statistics
def fit_stats(train):
    """Everything the baselines need, counted from `train` only."""
    tr = train.assign(lower=train["token"].str.lower())
    freq = tr["lower"].map(tr["lower"].value_counts())        # how often each row's word occurs
    rare = tr[freq <= RARE_MAX]                                 # rare word to model unseen ones
    nonroot = tr[tr["rel"] != "root"]
    rare_nonroot = rare[rare["rel"] != "root"]

    st = {
        "word": majority(tr["lower"], tr["rel"]),                          # word -> relation
        "word_nonroot": majority(nonroot["lower"], nonroot["rel"]),        # same but root excluded
        "suf": majority(rare["lower"].str[-3:], rare["rel"]),              # unseen: 3-letter suffix
        "suf_nonroot": majority(rare_nonroot["lower"].str[-3:], rare_nonroot["rel"]),
        "shape": majority(rare["token"].map(shape), rare["rel"]),          # unseen: word shape
        "shape_nonroot": majority(rare_nonroot["token"].map(shape), rare_nonroot["rel"]),
        "fallback": rare["rel"].value_counts().index[0],                   # last fallback
    }

    
    arcs = tr[tr["head"] > 0]
    off = {}
    for rel, s in (arcs["head"] - arcs["token_id"]).groupby(arcs["rel"]):
        vc = s.value_counts()
        off[rel] = min(vc[vc == vc.max()].index, key=lambda o: (abs(o), o))
    st["offset"] = off

    
    is_word = tr["token"].map(shape) != "punct"
    prior = (tr.loc[is_word, "rel"] == "root").mean()                     
    g = tr[is_word].groupby("lower")["rel"]
    st["root_score"] = ((g.apply(lambda r: (r == "root").sum()) + SMOOTH * prior)
                        / (g.size() + SMOOTH))
    st["root_prior"] = prior
    return st


def predict_rel(tokens, st, nonroot=False):
    #Check the word table first.
    #If the word is rare or unseen, check its suffix.
    # If that doesn't help then use its word shape.
    sfx = "_nonroot" if nonroot else ""
    lower = tokens.str.lower()
    rel = lower.map(st["word" + sfx])                                        # seen words
    by_suf = lower.str[-3:].map(st["suf" + sfx]).where(lower.str.len() >= 4)  # unseen long enough
    rel = rel.fillna(by_suf)
    rel = rel.fillna(tokens.map(shape).map(st["shape" + sfx]))             # unseen short or odd
    return rel.fillna(st["fallback"])


# ---------------------------------------------------------------- baseline A
# head = word + relation's most common offset to the word's position
# selected head is within the sentence and not the word itself.
def offset_head(i, rel, n, st):
    
    h = min(max(i + st["offset"].get(rel, 1), 1), n)
    if h == i:
        h = i + 1 if i < n else i - 1
    return h


def baseline_word_majority(df, st):
    out = df.copy()
    out["pred_rel"] = predict_rel(out["token"], st).to_numpy()
    heads = []
    for _, s in out.groupby("sentence_id", sort=False):
        n = len(s)
        for i, rel in zip(s["token_id"], s["pred_rel"]):
            if n == 1 or rel == "root":
                heads.append(0)                     
            else:
                heads.append(offset_head(i, rel, n, st))
    out["pred_head"] = heads
    out.loc[out.groupby("sentence_id")["token_id"].transform("size") == 1, "pred_rel"] = "root"
    return out


# ---------------------------------------------------------------- baseline B
#Starting from word i, move in the given direction (+1 for right, -1 for left)
# pick the first word whose relation belongs to allowed.
def nearest(rels, i, step, allowed, skip=()):
    j = i + step
    while 1 <= j <= len(rels):
        if rels[j - 1] in allowed and rels[j - 1] not in skip:
            return j
        j += step
    return None


# repeatedly re attach one word of each cycle to the root until the tree is valid
def break_cycles(heads, root):
    heads = list(heads)
    while True:
        cyc = check_tree(heads)["cycle"]
        if not cyc:
            return heads
        heads[cyc[0] - 1] = root


def rules_sentence(tokens, rels, rels_nonroot, st):
    n = len(tokens)
    if n == 1:
        return [0], ["root"]                                 # one word then it is the root

    # choose exactly one root
    # choose the non punctuation word that is most frequently the root in the training data.
    scores = [st["root_score"].get(t.lower(), st["root_prior"]) if shape(t) != "punct" else -1
              for t in tokens]
    root = max(range(n), key=lambda k: scores[k]) + 1       # a tie then leftmost
    rels = [r2 if r == "root" else r for r, r2 in zip(rels, rels_nonroot)]  # only one root allowed
    rels[root - 1] = "root"

    heads = []
    for i, r in enumerate(rels, start=1):
        if i == root:
            h = 0
        elif r == "punct":
            h = root                                          # punctuation - main verb
        elif r in NOMINAL_MOD:                                # the/big/of - next noun-like word
            h = nearest(rels, i, +1, NOMINAL_HEAD)
        elif r in CLAUSE_DEP:                                 # subject/aux/mark - next verb-like word
            h = nearest(rels, i, +1, CLAUSE_HEAD)
        elif r in ATTACH_LEFT_CLAUSE:                         # object/oblique - previous verb-like word
            h = nearest(rels, i, -1, CLAUSE_HEAD)
        elif r in ATTACH_LEFT_NOMINAL:                        # nmod/relative clause - previous noun-like word
            h = nearest(rels, i, -1, NOMINAL_HEAD)
        elif r == "conj":                                     # conj -> first conjunct
            h = nearest(rels, i, -1, NOMINAL_HEAD | CLAUSE_HEAD, skip={"conj"})
        elif r == "flat":                                    
            h = i - 1
            while h > 1 and rels[h - 1] == "flat":
                h -= 1
        elif r in ATTACH_ROOT:
            h = root
        else:
            h = offset_head(i, r, n, st)                      # anything else: typical offset
        if i != root and (h is None or h == i or h < 1):
            h = root                                          # no valid candidate - root
        heads.append(h)

    return break_cycles(heads, root), rels


def baseline_rules(df, st):
    out = df.copy()
    rel_all = predict_rel(out["token"], st).to_numpy()
    rel_nonroot = predict_rel(out["token"], st, nonroot=True).to_numpy()
    heads, rels, pos = [], [], 0
    for _, s in out.groupby("sentence_id", sort=False):
        n = len(s)
        h, r = rules_sentence(s["token"].tolist(), list(rel_all[pos:pos + n]),
                              list(rel_nonroot[pos:pos + n]), st)
        heads += h
        rels += r
        pos += n
    out["pred_head"], out["pred_rel"] = heads, rels
    return out


# ---------------------------------------------------------------- run
BASELINES = {"baseline_word_majority": baseline_word_majority,
             "baseline_rules": baseline_rules}

if __name__ == "__main__":
    tr, va = train_val()
    st = fit_stats(tr)                          # learned on the training part only

    scores = {}
    for name, fn in BASELINES.items():
        pred = fn(va, st)
        m, by_len, per_rel = evaluate(pred)
        print(f"\n===== {name} =====")
        print_report(m, by_len, per_rel, top=10)
        log_result(name, m, params={"rare_max": RARE_MAX, "smooth": SMOOTH, "fit_on": "train_part"})
        save_val_preds(name, pred)
        scores[name] = m["LAS"]

    #Use the better-performing baseline to make predictions on the test set, after retraining it on the entire train.csv file.
    best = max(scores, key=scores.get)
    print(f"\nbest baseline: {best} (val LAS {scores[best]:.4f}); writing test submission")
    test_pred = BASELINES[best](load_test(), fit_stats(load_train()))
    make_submission(test_pred, best)