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
  E3-gold relation classifier on GOLD heads          -> For comparison: Uses the correct heads instead of predicted ones.    python src/parser.py                 # E4a (default, current best): sampled + top-5 hard negatives
    python src/parser.py --neg sampled   # E0-E3 (sampled negatives only)
    python src/parser.py --neg all       # E4b: every candidate head is a negative (partial_fit)
    python src/parser.py --tags tagger              # E5a: word class = tagger's relation
    python src/parser.py --tags tagger --dirfeats   # E5b: + tagger's head-direction features
(--tags tagger needs the tag files: run `python src/tagger.py` once first)
"""
import time
import argparse

import joblib
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.linear_model import SGDClassifier

from data import ROOT, SEED, train_val, load_test
from eisner import eisner, NEG
from evaluate import evaluate, print_report, log_result, save_val_preds
from features import Sentences, arc_features, rel_features, word_classes, word_classes_oof
from submit import make_submission


MODELS = ROOT / "models"
N_NEAR = 6        # nearest wrong heads used as negatives
N_RAND = 2        # extra random wrong heads (so far away heads are seen too)
N_HARD = 5        # E4a: highest-scoring wrong heads added per word
CHUNK = 300       # sentences per chunk when scoring/training on all candidates

ARC_PARAMS = dict(loss="log_loss", alpha=1e-6, max_iter=10, tol=None, average=True, random_state=SEED)
REL_PARAMS = dict(loss="log_loss", alpha=1e-6, max_iter=10, tol=None, average=True,
                  random_state=SEED, n_jobs=-1)
RUN_NAME = {"sampled": "E3_lr_eisner_relclf", "hard": "E4a_hard_neg", "all": "E4b_all_cand"}
SUB_NAME = {"sampled": "graph_lr_eisner", "hard": "E4a_hard_neg", "all": "E4b_all_cand"}

TAG_DIRS = {"tagger": ROOT / "tags",                # tagger.py (v1)
            "tagger2": ROOT / "tags_v2",            # tagger2.py
            "tagger2near": ROOT / "tags_v2near"}    # tagger2.py --neardist


def load_tags(df, split, tags_dir):
    """Tagger output for df (split = train_oof / val / test), aligned to df's rows by ID.
    Returns (rel_class, sdir) arrays: the explicit tag -> feature mapping written by the tagger."""
    path = tags_dir / f"tags_{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run the matching tagger script first")
    t = pd.read_csv(path, keep_default_na=False).set_index("ID")
    assert set(t.index) == set(df["ID"]), f"{path.name} does not match the {split} rows"
    t = t.loc[df["ID"]]                                   # same order as df
    return t["rel_class"].to_numpy(dtype=object), t["sdir"].to_numpy(dtype=object)


def run_names(mode, tags, dirfeats):
    # experiment / submission names for each option combination
    if tags == "majority":
        return RUN_NAME[mode], SUB_NAME[mode]
    if tags == "tagger":
        name = "E5b_tagger_dirfeats" if dirfeats else "E5a_tagger"
    elif tags == "tagger2":
        name = "E6a_tagger2" + ("_dirfeats" if dirfeats else "")
    else:
        name = "E6b_tagger2near" + ("_dirfeats" if dirfeats else "")
    if mode != "hard":
        name += f"_{mode}"                                # E5 runs are defined on top of E4a
    return name, name

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


def all_candidates(S, ks=None):
    """Every possible head 0..n for every word 1..n, except the word itself.
    Pairs come grouped by sentence, then by dependent (each word's candidates are contiguous)."""
    sent, hs, ds = [], [], []
    for k in (range(S.n_sent) if ks is None else ks):
        n = S.lens[k]
        h, d = np.meshgrid(np.arange(n + 1), np.arange(1, n + 1))   # rows = dependents
        keep = h != d                                               # no self-loops
        sent.append(np.full(keep.sum(), k)); hs.append(h[keep]); ds.append(d[keep])
    return np.concatenate(sent), np.concatenate(hs), np.concatenate(ds)

def pair_key(S, sent, h, d):
    # one integer per (dependent row, head) pair, to find duplicates
    return (S.row_start[sent] + d - 1).astype(np.int64) * 512 + h
 
 
def chunks(n_sent, order=None):
    # sentence indices in blocks of CHUNK
    order = np.arange(n_sent) if order is None else order
    for start in range(0, n_sent, CHUNK):
        yield order[start:start + CHUNK]

# ---------------------------------------------------------------- E4a: hard negatives
def hard_negatives(S, heads, model, sampled_keys):
    """For every training word: the N_HARD highest-scoring WRONG heads
    (gold head excluded, self excluded by all_candidates), minus pairs already sampled."""
    out_s, out_h, out_d = [], [], []
    for ks in chunks(S.n_sent):
        sent, h, d = all_candidates(S, ks)
        score = model.decision_function(arc_features(S, sent, h, d))
        row = S.row_start[sent] + d - 1                     # dependent's row = group id
        score = np.where(h == heads[row], -np.inf, score)   # never pick the gold head
        order = np.lexsort((-score, row))                   # by dependent, best score first
        r = row[order]
        first = np.r_[True, r[1:] != r[:-1]]                # first pair of each dependent
        start = np.maximum.accumulate(np.where(first, np.arange(len(r)), 0))
        rank = np.arange(len(r)) - start                    # 0 = best wrong head
        pick = order[(rank < N_HARD) & np.isfinite(score[order])]
        out_s.append(sent[pick]); out_h.append(h[pick]); out_d.append(d[pick])
    sent, h, d = np.concatenate(out_s), np.concatenate(out_h), np.concatenate(out_d)
    new = ~np.isin(pair_key(S, sent, h, d), sampled_keys)   # skip pairs already sampled
    return sent[new], h[new], d[new], len(new)
 
 
# ---------------------------------------------------------------- E4b: all candidates
def train_all_candidates(S, heads):
    """Every candidate is a training example. Features are rebuilt per chunk
    (memory stays small); chunk order and rows are reshuffled every epoch."""
    rng = np.random.default_rng(SEED + 1)                  # separate seeded generator
    model = SGDClassifier(**ARC_PARAMS)
    n_pairs = 0
    for epoch in range(ARC_PARAMS["max_iter"]):            # same 10 epochs as fit()
        for ks in chunks(S.n_sent, rng.permutation(S.n_sent)):
            sent, h, d = all_candidates(S, ks)
            y = (h == heads[S.row_start[sent] + d - 1]).astype(int)   # 1 = gold head
            X = arc_features(S, sent, h, d)
            p = rng.permutation(len(y))                    # shuffle rows inside the chunk
            model.partial_fit(X[p], y[p], classes=[0, 1])
            n_pairs += len(y) if epoch == 0 else 0
        print(f"  epoch {epoch + 1}/{ARC_PARAMS['max_iter']} done")
    return model, n_pairs

# ---------------------------------------------------------------- scoring + decoding
def score_matrices(S, model):
    """Score every arc, then log-softmax over each word's candidate heads, so
    every word's scores are on the same scale before decoding."""
    mats = []
    for ks in chunks(S.n_sent):                            # chunked to keep memory small
        sent, h, d = all_candidates(S, ks)
        raw = model.decision_function(arc_features(S, sent, h, d))
        pos = 0
        for k in ks:
            n = S.lens[k]
            m = np.full((n + 1, n + 1), NEG)              # m[h, d]; impossible arcs stay NEG
            cnt = n * n                                   # n dependents x n candidate heads each
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
    far_err = ((pred > 0) & (np.abs(pred - d) > N_NEAR))[wrong].mean()
    far_gold = (np.abs(g_all - d) > N_NEAR)[g_all > 0].mean()

    print(f"argmax errors whose predicted head is > {N_NEAR} words away: "
          f"{far_err:.1%} of errors (vs {far_gold:.1%} of gold heads that far)")
    return {"UAS_argmax": round(float(np.mean(rank == 1)), 4),
            "top3": round(float(np.mean(rank <= 3)), 4),
            "far_err_share": round(float(far_err), 4)}



# ---------------------------------------------------------------- relations
def predict_rels(S, heads, model):
    # best non root relation for each word
    # words attached to ROOT get "root"
    scores = model.decision_function(rel_features(S, heads))
    rel = model.classes_[scores.argmax(1)]
    return np.where(np.asarray(heads) == 0, "root", rel)


# ---------------------------------------------------------------- run
def run_and_log(name, df, heads, rels, params, vocab, notes="", extra=None):
    out = df.assign(pred_head=heads, pred_rel=rels)
    m, by_len, per_rel = evaluate(out, train_vocab=vocab)
    print(f"\n===== {name} =====")
    print_report(m, by_len, per_rel, top=10)
    log_result(name, m, params=params, notes=notes, extra=extra)
    save_val_preds(name, out)
    return m



if __name__ == "__main__":
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--neg", choices=["sampled", "hard", "all"], default="hard",
                    help="how the arc scorer's negative examples are chosen")
    ap.add_argument("--tags", choices=["majority", "tagger", "tagger2", "tagger2near"], default="majority",
                    help="word class feature: word-majority relation or a tagger's relation")
    ap.add_argument("--dirfeats", action="store_true",
                    help="add the tagger's sdir column (head direction [+ near/far]) as features")
    args = ap.parse_args()
    mode = args.neg
    if args.dirfeats and args.tags == "majority":
        ap.error("--dirfeats needs a tagger (--tags tagger / tagger2 / tagger2near)")
    run_name, sub_name = run_names(mode, args.tags, args.dirfeats)

    t0 = time.time()
    rng = np.random.default_rng(SEED)
    tr, va = train_val()
    te = load_test()
    vocab = set(tr["token"].str.lower())

    # Use out-of-fold predictions for training data and classes from the full training split for validation and test data.
    if args.tags == "majority":
        cls_tr = word_classes_oof(tr)
        cls_va = word_classes(tr, va)
        cls_te = word_classes(tr, te)
        dir_tr = dir_va = dir_te = None
    else:                                                 # tagger output (also out-of-fold for train)
        tags_dir = TAG_DIRS[args.tags]
        cls_tr, dir_tr = load_tags(tr, "train_oof", tags_dir)
        cls_va, dir_va = load_tags(va, "val", tags_dir)
        cls_te, dir_te = load_tags(te, "test", tags_dir)
        if not args.dirfeats:
            dir_tr = dir_va = dir_te = None               # E5a: relation part only
    S_tr, S_va, S_te = (Sentences(tr, cls_tr, dir_tr), Sentences(va, cls_va, dir_va),
                        Sentences(te, cls_te, dir_te))
    gold_tr, gold_va = tr["head"].to_numpy(), va["head"].to_numpy()
    print(f"[{time.time() - t0:.0f}s] features ready (--neg {mode} --tags {args.tags}"
          f"{' --dirfeats' if args.dirfeats else ''}) -> {run_name}")

    params = {"arc": {k: v for k, v in ARC_PARAMS.items() if k != "random_state"},
              "neg": mode, "n_near": N_NEAR, "n_rand": N_RAND}
    if args.tags != "majority":
        params.update(tags=args.tags, dirfeats=args.dirfeats)
    if mode == "hard":
        params["n_hard"] = N_HARD
    extra = {}


    # ---- E0 ----
    if mode == "sampled":
        gold_mats = []
        for k in range(S_va.n_sent):
            n = S_va.lens[k]
            m = np.zeros((n + 1, n + 1))
            m[gold_va[S_va.row_start[k]:S_va.row_start[k] + n], np.arange(1, n + 1)] = 1.0
            gold_mats.append(m)
        run_and_log("E0_eisner_gold_scores", va, decode(gold_mats, "eisner"), va["rel"],
                    {}, vocab, notes="decoder ceiling, not a model")

    # ---- arc scorer
    if mode in("sampled", "hard"):
        sent, h, d, y = train_candidates(S_tr, gold_tr, rng)
        X = arc_features(S_tr, sent, h, d)
        arc = SGDClassifier(**ARC_PARAMS).fit(X, y)
        print(f"[{time.time() - t0:.0f}s] arc scorer trained on {X.shape[0]:,} sampled pairs "
                f"({y.mean():.1%} positive)")

    
        del X
        if mode == "hard":
            hs_, hh_, hd_, n_found = hard_negatives(S_tr, gold_tr, arc, pair_key(S_tr, sent, h, d))
            print(f"[{time.time() - t0:.0f}s] hard negatives: {n_found:,} found, "
                  f"{len(hs_):,} new after removing pairs already sampled")
            sent, h, d = np.r_[sent, hs_], np.r_[h, hh_], np.r_[d, hd_]   # keep ALL sampled pairs
            y = np.r_[y, np.zeros(len(hs_), int)]
            X = arc_features(S_tr, sent, h, d)
            arc = SGDClassifier(**ARC_PARAMS).fit(X, y)                    # retrain from scratch
            print(f"[{time.time() - t0:.0f}s] arc scorer retrained on {X.shape[0]:,} pairs "
                  f"({y.mean():.1%} positive)")
            del X
        extra["arc_train_pairs"] = len(y)
    else:
        arc, n_pairs = train_all_candidates(S_tr, gold_tr)
        print(f"[{time.time() - t0:.0f}s] arc scorer trained on all {n_pairs:,} candidate pairs")
        extra["arc_train_pairs"] = n_pairs

    mats_va = score_matrices(S_va, arc)
    print("\n--- mismatch diagnostics (validation) ---")
    extra.update(head_diagnostics(S_va, mats_va, gold_va))
    

    # ---- relation classifier with root rows excluded since root is decided by the tree ----
    keep = gold_tr > 0
    rel_model = SGDClassifier(**REL_PARAMS).fit(rel_features(S_tr, gold_tr)[keep], tr["rel"][keep])
    print(f"[{time.time() - t0:.0f}s] relation classifier trained ({len(rel_model.classes_)} classes)")
    params["rel"] = {k: v for k, v in REL_PARAMS.items() if k not in ("random_state", "n_jobs")}

    heads_eis = decode(mats_va, "eisner")
    if mode == "sampled":
        simple_rel = lambda heads: np.where(heads == 0, "root", cls_va)
        heads_e1 = decode(mats_va, "argmax")
        m1 = run_and_log("E1_lr_argmax", va, heads_e1, simple_rel(heads_e1), params,vocab,
                        notes="argmax heads (may not be trees), word-majority relations")
        heads_e2 = decode(mats_va, "eisner")
        m2 = run_and_log("E2_lr_eisner", va, heads_e2, simple_rel(heads_e2), params,vocab,
                        notes="Eisner heads, word-majority relations")
        print(f"\nE2 vs E1 (tree decoding): UAS {m1['UAS']:.4f} -> {m2['UAS']:.4f}, "
            f"valid trees {m1['valid_tree_pct']:.1f}% -> {m2['valid_tree_pct']:.1f}%")

    # E3/ E4a / E4b: Run the relation classifier using the heads predicted by Eisner which is the pipeline
    m = run_and_log(run_name, va, heads_eis, predict_rels(S_va, heads_eis, rel_model),
                    params, vocab, notes="MAIN: relations predicted on predicted heads", extra=extra)
    print(f"\n{run_name}: LAS {m['LAS']:.4f} | E4a reference 0.7021 | E3 0.6972 | rule baseline 0.3755")

    

    if mode == "sampled":
    # E3-gold: Run the same relation classifier using the correct (gold) heads only for comparison.
        run_and_log("E3gold_relclf_on_gold_heads", va, gold_va, predict_rels(S_va, gold_va, rel_model),
                    params, vocab, notes="DIAGNOSTIC ONLY: gold heads")

    # m = run_and_log(run_name, va, heads_eis, predict_rels(S_va, heads_eis, rel_model),
    #                 params, vocab, notes="MAIN: relations predicted on predicted heads", extra=extra)
    # print(f"\n{run_name}: LAS {m['LAS']:.4f} | E4a reference 0.7021 | E3 0.6972 | rule baseline 0.3755")
    
    # ---- save models and write the test submission
    MODELS.mkdir(exist_ok=True)
    joblib.dump(arc, MODELS / f"arc_{sub_name}.joblib", compress=3)
    heads_te = decode(score_matrices(S_te, arc), "eisner")
    te_pred = te.assign(pred_head=heads_te, pred_rel=predict_rels(S_te, heads_te, rel_model))
    make_submission(te_pred, sub_name)
    print(f"[{time.time() - t0:.0f}s] done")