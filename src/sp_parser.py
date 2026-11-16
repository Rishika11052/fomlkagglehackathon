"""
#2: structured perceptron arc scorer (global training with Eisner in the loop).

Unlike the logistic-regression arc scorer (which classifies word pairs one by one),
this model learns from WHOLE trees: for each training sentence it decodes the
best tree with single-root Eisner under the current weights, and if the tree is
wrong it adds the gold arcs' features and subtracts the predicted arcs' features.
Weights are averaged over all updates (averaged perceptron). Same hashed arc
features as parser.py.

Run from the repo root (needs the tag folder of --tags):
    python src/sp_parser.py --tags tagger2near --dirfeats     # trains, reports val UAS, saves model
    python src/sp_parser.py --test                            # self-test only
"""
import argparse
import time

import numpy as np

from data import ROOT, SEED, train_val
from eisner import eisner, NEG
from features import Sentences, arc_features, D_ARC
import parser as P

EPOCHS = 10


def row_of(h, d, n):
    # row of pair (head h, dependent d) inside one sentence's all_candidates block:
    # dependents in order 1..n, each with candidate heads 0..n except itself
    return (d - 1) * n + np.where(h < d, h, h - 1)


class StructPerceptronArc:
    """Averaged structured perceptron over arc features; decision_function() returns arc scores,
    so it plugs into parser.score_matrices / decode / the ensemble like the LR scorer."""

    def __init__(self, epochs=EPOCHS, seed=SEED, scale=1.0):
        self.epochs, self.seed, self.scale = epochs, seed, scale

    def fit(self, S, gold, on_epoch=None):
        w = np.zeros(D_ARC)                      # current weights
        u = np.zeros(D_ARC)                      # running sum of (step x update) for averaging
        c = 1.0                                  # step counter
        rng = np.random.default_rng(self.seed)
        for ep in range(self.epochs):
            wrong_trees = 0
            for ks in P.chunks(S.n_sent, rng.permutation(S.n_sent)):     # shuffled, in chunks
                sent, h, d = P.all_candidates(S, ks)
                X = arc_features(S, sent, h, d)                         # features of every candidate arc
                pos = 0
                for k in ks:
                    n = S.lens[k]
                    rows = slice(pos, pos + n * n)
                    m = np.full((n + 1, n + 1), NEG)
                    m[h[rows], d[rows]] = X[rows] @ w                    # current arc scores
                    pred = np.array(eisner(m))                           # best tree now
                    g = gold[S.row_start[k]:S.row_start[k] + n]
                    diff = np.flatnonzero(pred != g)                     # words with a wrong head
                    if len(diff):
                        wrong_trees += 1
                        dep = diff + 1
                        for heads_, sign in ((g[diff], 1.0), (pred[diff], -1.0)):
                            r = pos + row_of(heads_, dep, n)            # rows of these arcs
                            idx = np.concatenate([X.indices[X.indptr[i]:X.indptr[i + 1]] for i in r])
                            np.add.at(w, idx, sign)                      # + gold arcs, - predicted arcs
                            np.add.at(u, idx, sign * c)
                    c += 1
                    pos += n * n
            self.w = w - u / c                                           # averaged weights so far
            print(f"  epoch {ep + 1}/{self.epochs}: {wrong_trees / S.n_sent:.1%} training trees wrong")
            if on_epoch is not None:
                on_epoch(ep + 1, self)
        return self

    def decision_function(self, X):
        # arc scores. In an ensemble every member gets the same per-word log-softmax, and a common
        # scale factor does not change Eisner's choice, so no calibration against LR is needed.
        return (X @ self.w) * self.scale


def self_test():
    # a perceptron must be able to fit a small training set almost perfectly
    tr, _ = train_val()
    small = tr[tr["sentence_id"].isin(tr["sentence_id"].unique()[:60])].reset_index(drop=True)
    from features import word_classes
    S = Sentences(small, word_classes(small, small))
    sp = StructPerceptronArc(epochs=8).fit(S, small["head"].to_numpy())
    uas = (P.decode(P.score_matrices(S, sp), "eisner") == small["head"].to_numpy()).mean()
    assert uas > 0.97, uas
    assert (np.array(row_of(np.array([0, 2, 3]), np.array([1, 1, 1]), 3)) == [0, 1, 2]).all()
    print(f"self-test: training UAS on 60 sentences {uas:.3f} (> 0.97): passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="tagger2near", choices=list(P.TAG_DIRS))
    ap.add_argument("--dirfeats", action="store_true")
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    if args.test:
        self_test()
        raise SystemExit
    t0 = time.time()
    tr, va = train_val()
    d = P.TAG_DIRS[args.tags]
    (ct, st), (cv, sv) = P.load_tags(tr, "train_oof", d), P.load_tags(va, "val", d)
    S_tr = Sentences(tr, ct, st if args.dirfeats else None)
    S_va = Sentences(va, cv, sv if args.dirfeats else None)
    gold_va = va["head"].to_numpy()

    def report(ep, sp):  # diagnostic only: validation UAS after each epoch (not used for choices)
        uas = (P.decode(P.score_matrices(S_va, sp), "eisner") == gold_va).mean()
        print(f"[{time.time() - t0:.0f}s] after epoch {ep}: val UAS {uas:.4f}")

    sp = StructPerceptronArc().fit(S_tr, tr["head"].to_numpy(), on_epoch=report)
    name = f"sp_{args.tags}{'_dir' if args.dirfeats else ''}"
    (ROOT / "models").mkdir(exist_ok=True)
    np.save(ROOT / "models" / f"{name}_weights.npy", sp.w)          # plain weights, easy to reload
    print(f"[{time.time() - t0:.0f}s] saved models/{name}_weights.npy")