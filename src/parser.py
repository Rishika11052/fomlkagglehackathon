"""
graph-based parser = arc scorer + single root Eisner + relation classifier.

Run from the repo root:
    python src/parser.py
Runs experiments E0-E3 on the fixed validation split, logs them to
experiments/results.csv, saves validation predictions and models, and writes
submissions/graph_lr_eisner.csv.

  E0      Eisner on gold scores                      -> Shows the best possible score the decoder can achieve.
  E1      arc scorer + argmax head (self masked)     -> Predicts the head with the highest score without allowing a word to be its own head. Also provides diagnostics.
  E2      arc scorer + single-root Eisner            -> Checks whether using Eisner decoding improves the predictions.
  E3      E2 + relation classifier on PREDICTED heads ->  Predicts relations using the heads predicted by Eisner. This is the actual parser score.
  E3-gold relation classifier on GOLD heads          -> For comparison: Uses the correct heads instead of predicted ones.
"""
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.linear_model import SGDClassifier

from data import ROOT, SEED, train_val, load_test
from eisner import eisner, NEG
from evaluate import evaluate, print_report, log_result, save_val_preds
from features import (Sentences, arc_features, rel_features,
                      word_classes, word_classes_oof, dist_bucket)
from submit import make_submission

MODELS = ROOT / "models"
N_NEAR = 6        # nearest wrong heads used as negatives
N_RAND = 2        # extra random wrong heads (so far away heads are seen too)
ARC_PARAMS = dict(loss="log_loss", alpha=1e-6, max_iter=10, tol=None, average=True, random_state=SEED)
REL_PARAMS = dict(loss="log_loss", alpha=1e-6, max_iter=10, tol=None, average=True,
                  random_state=SEED, n_jobs=-1)


# ---------------------------------------------------------------- candidates
def train_candidates(S, heads, rng):
    """For every word: its gold head (label 1) + sampled wrong heads (label 0).
    Wrong heads = N_NEAR nearest words, ROOT, and N_RAND random others."""
    sent, hs, ds, ys = [], [], [], []
    for k in range(S.n_sent):
        n = S.lens[k]
        gold = heads[S.row_start[k]:S.row_start[k] + n]       # gold heads of this sentence
        for d in range(1, n + 1):
            g = gold[d - 1]
            words = np.array([h for h in range(1, n + 1) if h != d and h != g])  # wrong word heads
            near = words[np.argsort(np.abs(words - d), kind="stable")[:N_NEAR]] if len(words) else words
            rest = np.setdiff1d(words, near)
            rand = rng.choice(rest, size=min(N_RAND, len(rest)), replace=False) if len(rest) else rest
            neg = np.r_[near, rand, [0] if g != 0 else []].astype(int)  
            cand = np.r_[g, neg]
            sent += [k] * len(cand); hs += cand.tolist(); ds += [d] * len(cand)
            ys += [1] + [0] * len(neg)
    return np.array(sent), np.array(hs), np.array(ds), np.array(ys)


def all_candidates(S):
    # every possible head 0..n for every word 1..n, except the word itself
    sent, hs, ds = [], [], []
    for k in range(S.n_sent):
        n = S.lens[k]
        h, d = np.meshgrid(np.arange(n + 1), np.arange(1, n + 1))   # rows = dependents
        keep = h != d                                               # no self-loops
        sent.append(np.full(keep.sum(), k)); hs.append(h[keep]); ds.append(d[keep])
    return np.concatenate(sent), np.concatenate(hs), np.concatenate(ds)


# ---------------------------------------------------------------- scoring + decoding
def score_matrices(S, model):
    """Score every arc, then log-softmax over each word's candidate heads, so
    every word's scores are on the same scale before decoding."""
    sent, h, d = all_candidates(S)
    raw = model.decision_function(arc_features(S, sent, h, d))
    mats, pos = [], 0
    for k in range(S.n_sent):
        n = S.lens[k]
        m = np.full((n + 1, n + 1), NEG)                  # m[h, d]; impossible arcs stay NEG
        cnt = n * n                                       # n dependents x n candidate heads each
        m[h[pos:pos + cnt], d[pos:pos + cnt]] = raw[pos:pos + cnt]
        pos += cnt
        cols = m[:, 1:]
        valid = cols > NEG / 2
        lse = logsumexp(np.where(valid, cols, -np.inf), axis=0)   # per-word normaliser
        m[:, 1:] = np.where(valid, cols - lse, NEG)               # log P(head | word)
        mats.append(m)
    return mats


def decode(mats, method):
    # one head per word, in df row order
    out = []
    for m in mats:
        if method == "argmax":
            out += m[:, 1:].argmax(0).tolist()            # best head per word independently
        else:
            out += eisner(m)                              # best single root projective tree
    return np.array(out)


def head_diagnostics(S, mats, gold):
    """Checks for the sampled-negative mismatch, scored over ALL candidates."""
    rank, g_all = [], []
    for k, m in enumerate(mats):
        n = S.lens[k]
        g = gold[S.row_start[k]:S.row_start[k] + n]
        gold_score = m[g, np.arange(1, n + 1)]
        rank += ((m[:, 1:] > gold_score).sum(0) + 1).tolist()   # 1 = gold head scored highest
        g_all += g.tolist()
    rank, g_all = np.array(rank), np.array(g_all)
    d = S.local
    dist = np.where(g_all == 0, -1, np.abs(g_all - d))
    bucket = pd.cut(dist, [-2, 0, 1, 3, 6, 1000], labels=["root", "1", "2-3", "4-6", "7+"])
    tbl = pd.DataFrame({"top1": rank == 1, "top3": rank <= 3, "bucket": bucket}) \
        .groupby("bucket", observed=False).agg(tokens=("top1", "size"), top1=("top1", "mean"),
                                                top3=("top3", "mean"))
    print(f"gold head rank over all candidates: top-1 {np.mean(rank == 1):.4f}, "
          f"top-3 {np.mean(rank <= 3):.4f}")
    print("by gold head distance:")
    print(tbl.round(4).to_string())

    # are argmax mistakes mostly heads that were rarely negatives in training (far away)?
    pred = decode(mats, "argmax")
    wrong = pred != g_all
    far = (pred > 0) & (np.abs(pred - d) > N_NEAR)
    print(f"argmax errors whose predicted head is > {N_NEAR} words away: "
          f"{far[wrong].mean():.1%} of errors (vs {(np.abs(g_all - d) > N_NEAR)[g_all > 0].mean():.1%} "
          f"of gold heads that far)")


# ---------------------------------------------------------------- relations
def predict_rels(S, heads, model):
    # best non root relation for each word
    # words attached to ROOT get "root"
    scores = model.decision_function(rel_features(S, heads))
    rel = model.classes_[scores.argmax(1)]
    return np.where(np.asarray(heads) == 0, "root", rel)


# ---------------------------------------------------------------- run
def run_and_log(name, df, heads, rels, params, notes=""):
    out = df.assign(pred_head=heads, pred_rel=rels)
    m, by_len, per_rel = evaluate(out)
    print(f"\n===== {name} =====")
    print_report(m, by_len, per_rel, top=10)
    log_result(name, m, params=params, notes=notes)
    save_val_preds(name, out)
    return m


if __name__ == "__main__":
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    tr, va = train_val()
    te = load_test()

    # Use out-of-fold predictions for training data and classes from the full training split for validation and test data.
    cls_tr = word_classes_oof(tr)
    cls_va = word_classes(tr, va)
    cls_te = word_classes(tr, te)
    S_tr, S_va, S_te = Sentences(tr, cls_tr), Sentences(va, cls_va), Sentences(te, cls_te)
    gold_tr, gold_va = tr["head"].to_numpy(), va["head"].to_numpy()
    print(f"[{time.time() - t0:.0f}s] features ready")

    params = {"arc": {k: v for k, v in ARC_PARAMS.items() if k != "random_state"},
              "n_near": N_NEAR, "n_rand": N_RAND}

    # ---- E0 ----
    gold_mats = []
    for k in range(S_va.n_sent):
        n = S_va.lens[k]
        m = np.zeros((n + 1, n + 1))
        m[gold_va[S_va.row_start[k]:S_va.row_start[k] + n], np.arange(1, n + 1)] = 1.0
        gold_mats.append(m)
    run_and_log("E0_eisner_gold_scores", va, decode(gold_mats, "eisner"), va["rel"],
                {}, notes="decoder ceiling, not a model")

    # ---- arc scorer
    sent, h, d, y = train_candidates(S_tr, gold_tr, rng)
    X = arc_features(S_tr, sent, h, d)
    print(f"[{time.time() - t0:.0f}s] arc training pairs: {X.shape[0]:,} "
          f"({y.mean():.1%} positive)")
    arc = SGDClassifier(**ARC_PARAMS).fit(X, y)
    del X
    print(f"[{time.time() - t0:.0f}s] arc scorer trained")

    mats_va = score_matrices(S_va, arc)
    print("\n--- mismatch diagnostics (validation) ---")
    head_diagnostics(S_va, mats_va, gold_va)

    # ---- relation classifier with root rows excluded since root is decided by the tree ----
    keep = gold_tr > 0
    rel_model = SGDClassifier(**REL_PARAMS).fit(rel_features(S_tr, gold_tr)[keep], tr["rel"][keep])
    print(f"[{time.time() - t0:.0f}s] relation classifier trained ({len(rel_model.classes_)} classes)")
    params["rel"] = {k: v for k, v in REL_PARAMS.items() if k not in ("random_state", "n_jobs")}

   
    simple_rel = lambda heads: np.where(heads == 0, "root", cls_va)
    heads_e1 = decode(mats_va, "argmax")
    m1 = run_and_log("E1_lr_argmax", va, heads_e1, simple_rel(heads_e1), params,
                     notes="argmax heads (may not be trees), word-majority relations")
    heads_e2 = decode(mats_va, "eisner")
    m2 = run_and_log("E2_lr_eisner", va, heads_e2, simple_rel(heads_e2), params,
                     notes="Eisner heads, word-majority relations")
    print(f"\nE2 vs E1 (tree decoding): UAS {m1['UAS']:.4f} -> {m2['UAS']:.4f}, "
          f"valid trees {m1['valid_tree_pct']:.1f}% -> {m2['valid_tree_pct']:.1f}%")

    # E3: Run the relation classifier using the heads predicted by Eisner which is the pipeline
    m3 = run_and_log("E3_lr_eisner_relclf", va, heads_e2, predict_rels(S_va, heads_e2, rel_model),
                     params, notes="MAIN: relations predicted on predicted heads")
    print(f"\nE3 vs E2 (relation classifier): LAS {m2['LAS']:.4f} -> {m3['LAS']:.4f} "
          f"| vs rule baseline on val 0.3755")

    # E3-gold: Run the same relation classifier using the correct (gold) heads only for comparison.
    run_and_log("E3gold_relclf_on_gold_heads", va, gold_va, predict_rels(S_va, gold_va, rel_model),
                params, notes="DIAGNOSTIC ONLY: gold heads")

    # ---- save models and write the test submission
    MODELS.mkdir(exist_ok=True)
    joblib.dump({"arc": arc, "rel": rel_model}, MODELS / "graph_lr_eisner.joblib", compress=3)
    heads_te = decode(score_matrices(S_te, arc), "eisner")
    te_pred = te.assign(pred_head=heads_te, pred_rel=predict_rels(S_te, heads_te, rel_model))
    make_submission(te_pred, "graph_lr_eisner")
    print(f"[{time.time() - t0:.0f}s] done")