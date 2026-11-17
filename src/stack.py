"""
Step 9: stacked parsing = a level-2 parser that also sees a level-1 parser's tree.

Level 1 = the E9 ensemble (3 structured perceptrons on 3 tag sets + averaged relation
          classifiers), 5 epochs (validation UAS was flat from epoch 3 on).
          Training-part trees are OUT-OF-FOLD (trained on 4 folds, parse the 5th);
          validation/test trees come from level 1 trained on the whole training part.
Level 2 = the E9 ensemble again, with extra features from the level-1 tree:
          is (h, d) a level-1 arc, the reverse arc, h = level-1 grandparent, h and d
          level-1 siblings, level-1 relations, level-1 children of h.
          E10a: only the arc scorers use the level-1 tree
          E10b: the relation classifiers use it too

Run from the repo root (needs tags/, tags_v2/, tags_v2near/):
    python src/stack.py --level1    # once, ~30-45 min: writes level1/l1_{train_oof,val,test}.csv
    python src/stack.py --level2    # ~12-15 min: logs E10a + E10b, writes their submissions
"""
import argparse
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier

from data import ROOT, SEED, train_val, load_test
from evaluate import evaluate, print_report, log_result, save_val_preds
from features import Sentences, rel_features
from sp_parser import StructPerceptronArc
from submit import make_submission
import parser as P
import features as F
import sp_parser as SPP

MEMBERS = ["tagger", "tagger2", "tagger2near"]   # same 3 tag sets as E9
L1_EPOCHS = 5                                     # level-1 perceptron epochs
L1_DIR = ROOT / "level1"


# ---------------------------------------------------------------- helpers
def all_tags(tr, va, te):
    # {member: (train tags, val tags, test tags)}, each tags = (relation class, head direction)
    out = {}
    for m in MEMBERS:
        d = P.TAG_DIRS[m]
        out[m] = (P.load_tags(tr, "train_oof", d), P.load_tags(va, "val", d), P.load_tags(te, "test", d))
    return out


def subset(tags, mask):
    # the same tags restricted to some rows (always whole sentences)
    return {m: (c[mask], s[mask]) for m, (c, s) in tags.items()}


def average(mat_lists):
    # element-wise mean of each sentence's score matrix over the members
    return [sum(ms) / len(ms) for ms in zip(*mat_lists)]


def ensemble_heads(fit_df, fit_tags, apps, epochs, l1_fit=None, l1_apps=None):
    """Train one structured perceptron per tag set on fit_df, average their scores,
    decode with Eisner. apps = [(df, tags), ...]; l1_* = level-1 trees (or None).
    Returns a list of head arrays, one per app."""
    gold = fit_df["head"].to_numpy()
    l1_apps = l1_apps if l1_apps is not None else [None] * len(apps)
    mats = [[] for _ in apps]
    for m in MEMBERS:
        sp = StructPerceptronArc(epochs=epochs, seed=SEED).fit(Sentences(fit_df, *fit_tags[m], l1=l1_fit), gold)
        for j, (df, tags) in enumerate(apps):
            mats[j].append(P.score_matrices(Sentences(df, *tags[m], l1=l1_apps[j]), sp))
    return [P.decode(average(ms), "eisner") for ms in mats]


def ensemble_rels(fit_df, fit_tags, apps, heads, l1_fit=None, l1_apps=None):
    """One relation classifier per tag set (trained on GOLD heads of fit_df), probabilities
    averaged on the given predicted heads. Returns a list of relation arrays, one per app."""
    gold = fit_df["head"].to_numpy()
    keep = gold > 0
    l1_apps = l1_apps if l1_apps is not None else [None] * len(apps)
    probs, classes = [0.0] * len(apps), None
    for m in MEMBERS:
        S_fit = Sentences(fit_df, *fit_tags[m], l1=l1_fit)
        rel = SGDClassifier(**P.REL_PARAMS).fit(rel_features(S_fit, gold)[keep], fit_df["rel"].to_numpy()[keep])
        assert classes is None or (rel.classes_ == classes).all()       # same label order for all
        classes = rel.classes_
        for j, (df, tags) in enumerate(apps):
            S = Sentences(df, *tags[m], l1=l1_apps[j])
            probs[j] = probs[j] + rel.predict_proba(rel_features(S, heads[j])) / len(MEMBERS)
    return [np.where(h == 0, "root", classes[p.argmax(1)]) for h, p in zip(heads, probs)]


def save_l1(df, heads, rels, name):
    L1_DIR.mkdir(exist_ok=True)
    pd.DataFrame({"ID": df["ID"].to_numpy(), "l1_head": heads, "l1_rel": rels}) \
        .to_csv(L1_DIR / f"l1_{name}.csv", index=False)


def load_l1(df, name):
    # level-1 tree for df's rows (by ID): (heads, relations)
    path = L1_DIR / f"l1_{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run `python src/stack.py --level1` first")
    t = pd.read_csv(path, keep_default_na=False).set_index("ID")
    assert set(t.index) == set(df["ID"]), f"{path.name} does not match the rows"
    t = t.loc[df["ID"]]
    return t["l1_head"].to_numpy().astype(int), t["l1_rel"].to_numpy(dtype=object)


# ---------------------------------------------------------------- level 1
def level1(tr, va, te, tags, t0):
    tr_tags = {m: tags[m][0] for m in MEMBERS}
    va_tags = {m: tags[m][1] for m in MEMBERS}
    te_tags = {m: tags[m][2] for m in MEMBERS}

    # out-of-fold trees for the training part (sentence-level separation)
    h_oof = np.zeros(len(tr), dtype=int)
    r_oof = np.empty(len(tr), dtype=object)
    for k in sorted(tr["fold"].unique()):
        fm = (tr["fold"] != k).to_numpy()
        fit, app = tr[fm].reset_index(drop=True), tr[~fm].reset_index(drop=True)
        assert not set(fit["sentence_id"]) & set(app["sentence_id"])
        ft, at = subset(tr_tags, fm), subset(tr_tags, ~fm)
        [h] = ensemble_heads(fit, ft, [(app, at)], L1_EPOCHS)
        [r] = ensemble_rels(fit, ft, [(app, at)], [h])
        h_oof[~fm], r_oof[~fm] = h, r
        print(f"[{time.time() - t0:.0f}s] level-1 fold {k}: UAS {(h == app['head'].to_numpy()).mean():.4f}")

    # validation and test trees: level 1 trained on the whole training part
    apps = [(va, va_tags), (te, te_tags)]
    hv, ht = ensemble_heads(tr, tr_tags, apps, L1_EPOCHS)
    rv, rt = ensemble_rels(tr, tr_tags, apps, [hv, ht])
    save_l1(tr, h_oof, r_oof, "train_oof")
    save_l1(va, hv, rv, "val")
    save_l1(te, ht, rt, "test")

    g_tr, g_va = tr["head"].to_numpy(), va["head"].to_numpy()
    print(f"\nlevel 1, out-of-fold on the training part: UAS {(h_oof == g_tr).mean():.4f} "
          f"LAS {((h_oof == g_tr) & (r_oof == tr['rel'].to_numpy())).mean():.4f}")
    print(f"level 1 on validation: UAS {(hv == g_va).mean():.4f} "
          f"LAS {((hv == g_va) & (rv == va['rel'].to_numpy())).mean():.4f}  (E9 with 10 epochs: 0.8390 / 0.7993)")
    print(f"[{time.time() - t0:.0f}s] wrote level1/")


# ---------------------------------------------------------------- level 2
def level2(tr, va, te, tags, t0):
    tr_tags = {m: tags[m][0] for m in MEMBERS}
    va_tags = {m: tags[m][1] for m in MEMBERS}
    te_tags = {m: tags[m][2] for m in MEMBERS}
    l1_tr, l1_va, l1_te = load_l1(tr, "train_oof"), load_l1(va, "val"), load_l1(te, "test")
    apps = [(va, va_tags), (te, te_tags)]
    vocab = set(tr["token"].str.lower())

    hv, ht = ensemble_heads(tr, tr_tags, apps, StructPerceptronArc().epochs,
                            l1_fit=l1_tr, l1_apps=[l1_va, l1_te])
    print(f"[{time.time() - t0:.0f}s] level-2 heads: val UAS {(hv == va['head'].to_numpy()).mean():.4f} "
          f"(E9: 0.8390)")

    bits = int(np.log2(F.D_ARC))
    sfx = "" if bits == 22 else f"_bits{bits}"                  # E10 names stay unchanged at 2^22
    for name, use_l1 in [("E10a_stacked_arcs" + sfx, False), ("E10b_stacked" + sfx, True)]:
        rv, rt = ensemble_rels(tr, tr_tags, apps, [hv, ht],
                               l1_fit=l1_tr if use_l1 else None,
                               l1_apps=[l1_va, l1_te] if use_l1 else None)
        out = va.assign(pred_head=hv, pred_rel=rv)
        met, by_len, per_rel = evaluate(out, train_vocab=vocab)
        print(f"\n===== {name} =====")
        print_report(met, by_len, per_rel, top=8)
        log_result(name, met, params={"level1": f"E9 ensemble, {L1_EPOCHS} epochs, out-of-fold",
                                      "level2": "E9 ensemble + level-1 tree features",
                                      "relations_use_level1": use_l1, "arc_bits": bits},
                   notes="MAIN: stacked parser")
        save_val_preds(name, out)
        make_submission(te.assign(pred_head=ht, pred_rel=rt), name)
        print(f"{name}: LAS {met['LAS']:.4f} | E9 0.7993 | [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--level1", action="store_true", help="build the out-of-fold level-1 trees (once)")
    g.add_argument("--level2", action="store_true", help="train/evaluate the stacked level-2 parser")
    ap.add_argument("--arc-bits", type=int, default=22,
                    help="hashed arc-feature columns = 2^bits (22 = as before; 24 = fewer collisions)")
    args = ap.parse_args()
    F.D_ARC = SPP.D_ARC = 2 ** args.arc_bits             # used by arc_features and the perceptron weights
    t0 = time.time()
    tr, va = train_val()
    te = load_test()
    tags = all_tags(tr, va, te)
    if args.level1:
        level1(tr, va, te, tags, t0)
    else:
        level2(tr, va, te, tags, t0)