"""
Step 5: pseudo-POS "supertagger" = averaged structured perceptron + Viterbi.

Each word gets a supertag = its relation + where its head is:
    "det/R"  (head to the Right),  "obj/L"  (head to the Left),  "root"
Mapping used by the parser (explicit, see split_tag):
    tag "obj/L" -> rel_class "obj" (replaces the word-majority class), sdir "L"
    tag "root"  -> rel_class "root",                                     sdir "ROOT"

Run ONCE from the repo root (about 2-4 minutes):
    python src/tagger.py
Writes tags/tags_train_oof.csv, tags/tags_val.csv, tags/tags_test.csv (ID, tag, rel_class, sdir)
and tags/T1_metrics.json. Training-part tags are out-of-fold (jackknifed): each
fold is tagged by a tagger that never saw that fold's sentences.

Self-test only:
    python src/tagger.py --test
"""
import itertools
import json
import sys
import time

import numpy as np
import pandas as pd

from data import ROOT, SEED, train_val, load_test
from baselines import shape

TAGS_DIR = ROOT / "tags"
EPOCHS = 6           # passes over the training sentences
BOS, EOS = "<S>", "</S>"


# ---------------------------------------------------------------- tags <-> parser features
def make_tag(rel, head, token_id):
    # gold supertag: relation + direction of the head ("root" has no direction)
    if head == 0:
        return "root"
    return f"{rel}/{'R' if head > token_id else 'L'}"


def split_tag(tag):
    # the explicit mapping: supertag -> (rel_class for the parser, sdir for --dirfeats)
    if tag == "root":
        return "root", "ROOT"
    rel, d = tag.rsplit("/", 1)       # rsplit: relations like "nmod:poss" keep their ":"
    return rel, d


def gold_tags(df):
    return [make_tag(r, h, t) for r, h, t in zip(df["rel"], df["head"], df["token_id"])]


# ---------------------------------------------------------------- features
def token_features(tokens):
    """String features for every word of one sentence (same number per word)."""
    low = [t.lower() for t in tokens]
    shp = [shape(t) for t in tokens]
    n = len(tokens)
    W = lambda i: low[i] if 0 <= i < n else (BOS if i < 0 else EOS)        # word at i (padded)
    S = lambda i: shp[i] if 0 <= i < n else (BOS if i < 0 else EOS)        # shape at i
    X = lambda i: low[i][-3:] if 0 <= i < n else (BOS if i < 0 else EOS)   # suffix-3 at i
    feats = []
    for i, t in enumerate(tokens):
        w = low[i]
        feats.append([
            "b",                                   # bias
            "w=" + w,                              # the word
            "p1=" + w[:1], "p2=" + w[:2], "p3=" + w[:3],               # prefixes
            "s1=" + w[-1:], "s2=" + w[-2:], "s3=" + w[-3:], "s4=" + w[-4:],  # suffixes
            "sh=" + shp[i],                        # shape
            "dig=%d" % any(c.isdigit() for c in t),
            "hyp=%d" % ("-" in t),
            "cap=%d" % (t[:1].isupper()),
            "pos=" + ("first" if i == 0 else "last" if i == n - 1 else "mid"),
            "len=%d" % min(len(t), 8),
            "w-1=" + W(i - 1), "w+1=" + W(i + 1), "w-2=" + W(i - 2), "w+2=" + W(i + 2),  # neighbours
            "x-1=" + X(i - 1), "x+1=" + X(i + 1),                      # neighbour suffixes
            "sh-1=" + S(i - 1), "sh+1=" + S(i + 1),                    # neighbour shapes
            "ww-=" + W(i - 1) + "|" + w, "ww+=" + w + "|" + W(i + 1),  # word pairs
            "shh=" + S(i - 1) + "|" + shp[i] + "|" + S(i + 1),         # shape trigram
        ])
    return feats


# ---------------------------------------------------------------- Viterbi
def viterbi(E, A, start):
    """E: (n, T) emission scores, A: (T, T) transition scores (prev -> next),
    start: (T,) scores for the first tag. Returns the best tag index sequence."""
    n, T = E.shape
    delta = start + E[0]                          # best score ending in each tag at position 0
    back = np.zeros((n, T), dtype=np.int64)
    for i in range(1, n):
        cand = delta[:, None] + A                 # cand[prev, next]
        back[i] = cand.argmax(0)                  # best previous tag for each next tag
        delta = cand.max(0) + E[i]
    path = [int(delta.argmax())]
    for i in range(n - 1, 0, -1):                 # follow back-pointers
        path.append(int(back[i, path[-1]]))
    return path[::-1]


# ---------------------------------------------------------------- the tagger
class Tagger:
    """Averaged structured perceptron with first-order (tag bigram) transitions."""
    featurize = staticmethod(token_features)    # feature function; tagger2.py plugs in its own

    def fit(self, sentences, tags, epochs=EPOCHS, seed=SEED, on_epoch=None):
        # sentences: list of token lists; tags: list of tag lists (same shapes)
        # on_epoch: optional callback(epoch, tagger) run with the averaged weights after each epoch
        # sentences: list of token lists; tags: list of tag lists (same shapes)
        self.tags = sorted({t for ts in tags for t in ts})                  # tag set from training only
        self.tag_id = {t: k for k, t in enumerate(self.tags)}
        feats = [self.featurize(s) for s in sentences]
        vocab = {}                                                          # feature string -> row
        for fs in feats:
            for f_word in fs:
                for f in f_word:
                    vocab.setdefault(f, len(vocab))
        self.vocab = vocab
        F = [np.array([[vocab[f] for f in fw] for fw in fs], dtype=np.int64) for fs in feats]
        Y = [np.array([self.tag_id[t] for t in ts]) for ts in tags]

        T, D = len(self.tags), len(vocab)
        W = np.zeros((D + 1, T), np.float32)        # emission weights (+1 row for unseen features)
        U = np.zeros((D + 1, T), np.float32)        # running sum of (step x update), for averaging
        A = np.zeros((T, T)); UA = np.zeros((T, T))     # transitions prev -> next
        st = np.zeros(T); Ust = np.zeros(T)             # start-of-sentence tag scores
        c = 1.0                                          # update counter
        rng = np.random.default_rng(seed)

        for ep in range(epochs):
            errors = 0
            for k in rng.permutation(len(F)):            # shuffled sentence order, seeded
                f, y = F[k], Y[k]
                pred = np.array(viterbi(W[f].sum(1), A, st))
                if (pred != y).any():
                    errors += 1
                    wrong = np.flatnonzero(pred != y)
                    for i in wrong:                      # emission update at wrong positions only
                        np.add.at(W, (f[i], y[i]), 1.0); np.add.at(U, (f[i], y[i]), c)
                        np.add.at(W, (f[i], pred[i]), -1.0); np.add.at(U, (f[i], pred[i]), -c)
                    st[y[0]] += 1; Ust[y[0]] += c; st[pred[0]] -= 1; Ust[pred[0]] -= c
                    np.add.at(A, (y[:-1], y[1:]), 1.0); np.add.at(UA, (y[:-1], y[1:]), c)
                    np.add.at(A, (pred[:-1], pred[1:]), -1.0); np.add.at(UA, (pred[:-1], pred[1:]), -c)
                c += 1
            print(f"  epoch {ep + 1}/{epochs}: {errors / len(F):.1%} sentences with an error")
            if on_epoch is not None:                     # e.g. measure dev accuracy at this epoch
                self.W, self.A, self.st = W - U / c, A - UA / c, st - Ust / c
                self.W[D] = 0.0
                on_epoch(ep + 1, self)

        # averaged weights = w - U / c  (standard lazy-averaging trick)
        self.W = W - U / c
        self.A = A - UA / c
        self.st = st - Ust / c
        self.W[D] = 0.0                                  # unseen features contribute nothing
        return self

    def predict(self, sentences):
        out = []
        unk = len(self.vocab)
        for s in sentences:
            f = np.array([[self.vocab.get(x, unk) for x in fw] for fw in self.featurize(s)])
            out.append([self.tags[k] for k in viterbi(self.W[f].sum(1), self.A, self.st)])
        return out


def by_sentence(df, col):
    # list of per-sentence lists, in df order
    return [g.tolist() for _, g in df.groupby("sentence_id", sort=False)[col]]


def flat(lists):
    return [x for l in lists for x in l]


def tag_frame(df, tags):
    # the tag file: ID + tag + the two parser features derived from it
    rel, sdir = zip(*[split_tag(t) for t in tags])
    return pd.DataFrame({"ID": df["ID"].to_numpy(), "tag": tags, "rel_class": rel, "sdir": sdir})


# ---------------------------------------------------------------- self-test
def self_test():
    rng = np.random.default_rng(0)
    # 1. Viterbi = brute force on tiny random problems
    for n in range(1, 5):
        for _ in range(20):
            T = 3
            E, A, st = rng.normal(size=(n, T)), rng.normal(size=(T, T)), rng.normal(size=T)
            score = lambda p: st[p[0]] + sum(E[i, p[i]] for i in range(n)) + sum(A[p[i - 1], p[i]] for i in range(1, n))
            best = max(itertools.product(range(T), repeat=n), key=score)
            assert abs(score(viterbi(E, A, st)) - score(best)) < 1e-9
    print("Viterbi = brute force: passed")

    # 2. tag mapping round trip
    for t in ["det/R", "nmod:poss/R", "obj/L", "root", "compound:prt/L"]:
        rel, d = split_tag(t)
        assert (make_tag(rel, 0, 1) if d == "ROOT" else make_tag(rel, 2 if d == "R" else 1, 1 if d == "R" else 2)) == t
    print("tag <-> (rel_class, sdir) mapping round trip: passed")

    # 3. a learnable toy task: tag depends on the word and on the previous word
    sents = [["the", "dog", "runs"], ["a", "cat", "sleeps"], ["the", "cat", "runs"], ["a", "dog", "sleeps"]] * 5
    tags = [["det/R", "nsubj/R", "root"]] * len(sents)
    tg = Tagger().fit(sents, tags, epochs=3)
    assert tg.predict(sents) == tags
    print("toy task learned perfectly: passed")

    # 4. no leakage + determinism, on a slice of real data
    tr, _ = train_val()
    ids = tr["sentence_id"].unique()
    A_ids, B_ids = set(ids[:600]), set(ids[600:900])
    a, b = tr[tr["sentence_id"].isin(A_ids)], tr[tr["sentence_id"].isin(B_ids)].copy()
    assert not A_ids & B_ids
    p1 = Tagger().fit(by_sentence(a, "token"), by_sentence(a.assign(t=gold_tags(a)), "t"), epochs=2) \
        .predict(by_sentence(b, "token"))
    b[["head", "rel"]] = b[["head", "rel"]].sample(frac=1, random_state=1).to_numpy()   # shuffle B's labels
    p2 = Tagger().fit(by_sentence(a, "token"), by_sentence(a.assign(t=gold_tags(a)), "t"), epochs=2) \
        .predict(by_sentence(b, "token"))
    assert p1 == p2
    print("tags of unseen sentences don't depend on their labels, runs are identical: passed")


# ---------------------------------------------------------------- run
if __name__ == "__main__":
    self_test()
    if "--test" in sys.argv:
        sys.exit()
    t0 = time.time()
    tr, va = train_val()
    te = load_test()
    tr_tags = gold_tags(tr)
    tr = tr.assign(tag=tr_tags)

    # out-of-fold tags for the training part
    oof = pd.Series(index=tr.index, dtype=object)
    for k in sorted(tr["fold"].unique()):
        fit_part, tag_part = tr[tr["fold"] != k], tr[tr["fold"] == k]
        assert not set(fit_part["sentence_id"]) & set(tag_part["sentence_id"])   # sentence-level separation
        print(f"[{time.time() - t0:.0f}s] fold {k}: training on {fit_part['sentence_id'].nunique():,} sentences")
        tg = Tagger().fit(by_sentence(fit_part, "token"), by_sentence(fit_part, "tag"))
        oof[tag_part.index] = flat(tg.predict(by_sentence(tag_part, "token")))

    # validation and test tags: one tagger trained on the whole training part
    print(f"[{time.time() - t0:.0f}s] full training part: {tr['sentence_id'].nunique():,} sentences")
    tg = Tagger().fit(by_sentence(tr, "token"), by_sentence(tr, "tag"))
    va_tags = flat(tg.predict(by_sentence(va, "token")))
    te_tags = flat(tg.predict(by_sentence(te, "token")))

    TAGS_DIR.mkdir(exist_ok=True)
    tag_frame(tr, oof.tolist()).to_csv(TAGS_DIR / "tags_train_oof.csv", index=False)
    tag_frame(va, va_tags).to_csv(TAGS_DIR / "tags_val.csv", index=False)
    tag_frame(te, te_tags).to_csv(TAGS_DIR / "tags_test.csv", index=False)

    # ---- T1: how good are the tags? (compared with the word-majority classes used so far)
    from features import word_classes
    va_gold = gold_tags(va)
    vrel = np.array([split_tag(t)[0] for t in va_tags])
    vdir = np.array([split_tag(t)[1] for t in va_tags])
    grel = np.array([split_tag(t)[0] for t in va_gold])
    gdir = np.array([split_tag(t)[1] for t in va_gold])
    unseen = ~va["token"].str.lower().isin(set(tr["token"].str.lower())).to_numpy()
    majority = word_classes(tr, va)
    oof_rel = np.array([split_tag(t)[0] for t in oof])
    m = {
        "val_tag_acc": float((np.array(va_tags) == np.array(va_gold)).mean()),
        "val_rel_acc": float((vrel == grel).mean()),
        "val_dir_acc": float((vdir == gdir).mean()),
        "val_rel_acc_unseen": float((vrel == grel)[unseen].mean()),
        "majority_rel_acc": float((majority == grel).mean()),
        "majority_rel_acc_unseen": float((majority == grel)[unseen].mean()),
        "oof_rel_acc_train": float((oof_rel == tr["rel"].to_numpy()).mean()),
        "n_tags": len(tg.tags), "epochs": EPOCHS, "seed": SEED,
    }
    (TAGS_DIR / "T1_metrics.json").write_text(json.dumps(m, indent=2))
    print("\n===== T1: tagger on validation =====")
    print(f"supertag accuracy {m['val_tag_acc']:.4f} | relation part {m['val_rel_acc']:.4f} "
          f"| direction part {m['val_dir_acc']:.4f}")
    print(f"relation part on unseen words {m['val_rel_acc_unseen']:.4f}")
    print(f"word-majority classes (used so far): {m['majority_rel_acc']:.4f} overall, "
          f"{m['majority_rel_acc_unseen']:.4f} on unseen words")
    print(f"out-of-fold relation accuracy on the training part: {m['oof_rel_acc_train']:.4f} "
          f"(should be close to validation)")
    print(f"[{time.time() - t0:.0f}s] wrote tags/ (tags_train_oof.csv, tags_val.csv, tags_test.csv, T1_metrics.json)")