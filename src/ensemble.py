"""
Step 8: ensemble of three graph parsers that differ only in their tags.

Each member = the E4a recipe (sampled + hard negatives) with --dirfeats, on one tag set:
    member 1: tagger v1          (tags/)          = E5b
    member 2: tagger v2          (tags_v2/)       = E6a + dirfeats
    member 3: tagger v2 near/far (tags_v2near/)   = E6b
Heads:      average the members' log P(head | word) matrices, then ONE single-root Eisner.
Relations:  average the members' relation probabilities on those ensemble heads.

Needs all three tag folders: python src/tagger.py ; python src/tagger2.py ; python src/tagger2.py --neardist
Run from the repo root (about 8-10 minutes):
    python src/ensemble.py              # E8: logistic-regression arc scorers
    python src/ensemble.py --arc sp     # E9: structured-perceptron arc scorers (sp_parser.py)
Logs E8_ensemble3 / E9_ensemble3_sp to experiments/results.csv and writes the matching submission.
"""
import argparse
import time
import numpy as np
from sklearn.linear_model import SGDClassifier

from data import SEED, train_val, load_test
from evaluate import evaluate, print_report, log_result, save_val_preds
from features import Sentences, arc_features, rel_features
from submit import make_submission
from sp_parser import StructPerceptronArc
import parser as P

MEMBERS = ["tagger", "tagger2", "tagger2near"]   # one member per tag set, all with dirfeats
NAMES = {"lr": "E8_ensemble3", "sp": "E9_ensemble3_sp"}

def train_arc(S, gold):
    # exactly parser.py's default recipe: sampled negatives, then + top-5 hard negatives, retrain
    rng = np.random.default_rng(SEED)                       # same random draws as a parser.py run
    sent, h, d, y = P.train_candidates(S, gold, rng)
    arc = SGDClassifier(**P.ARC_PARAMS).fit(arc_features(S, sent, h, d), y)
    hs, hh, hd, _ = P.hard_negatives(S, gold, arc, P.pair_key(S, sent, h, d))
    sent, h, d = np.r_[sent, hs], np.r_[h, hh], np.r_[d, hd]
    y = np.r_[y, np.zeros(len(hs), int)]
    return SGDClassifier(**P.ARC_PARAMS).fit(arc_features(S, sent, h, d), y)


def average(mat_lists):
    # element-wise mean of each sentence's score matrix over the members
    return [sum(ms) / len(ms) for ms in zip(*mat_lists)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arc", choices=["lr", "sp"], default="lr",
                    help="arc scorer of every member: logistic regression (E8) or structured perceptron (E9)")
    arc_kind = ap.parse_args().arc
    NAME = NAMES[arc_kind]
    t0 = time.time()
    tr, va = train_val()
    te = load_test()
    vocab = set(tr["token"].str.lower())
    gold_tr, gold_va = tr["head"].to_numpy(), va["head"].to_numpy()

    # ---- every member's Sentences (its own tags: relation class + head direction)
    sents = {}
    for m in MEMBERS:
        d = P.TAG_DIRS[m]
        (c_tr, s_tr), (c_va, s_va), (c_te, s_te) = (P.load_tags(tr, "train_oof", d),
                                                    P.load_tags(va, "val", d), P.load_tags(te, "test", d))
        sents[m] = (Sentences(tr, c_tr, s_tr), Sentences(va, c_va, s_va), Sentences(te, c_te, s_te))

    # ---- phase 1: arc scorers -> averaged score matrices -> one Eisner
    mats_va, mats_te, member_uas = [], [], {}
    for m in MEMBERS:
        S_tr, S_va, S_te = sents[m]
        if arc_kind == "lr":
            arc = train_arc(S_tr, gold_tr)
        else:                                               # global training with Eisner in the loop
            arc = StructPerceptronArc(seed=SEED).fit(S_tr, gold_tr)
        mats_va.append(P.score_matrices(S_va, arc))
        mats_te.append(P.score_matrices(S_te, arc))
        member_uas[m] = round(float((P.decode(mats_va[-1], "eisner") == gold_va).mean()), 4)
        print(f"[{time.time() - t0:.0f}s] member {m}: arc scorer trained, alone UAS {member_uas[m]}")
    heads_va = P.decode(average(mats_va), "eisner")
    heads_te = P.decode(average(mats_te), "eisner")
    print(f"[{time.time() - t0:.0f}s] ensemble heads: UAS {(heads_va == gold_va).mean():.4f}")

    # ---- phase 2: relation classifiers (trained on GOLD heads), averaged on the ensemble heads
    keep = gold_tr > 0
    p_va = p_te = 0.0
    classes = None
    for m in MEMBERS:
        S_tr, S_va, S_te = sents[m]
        rel = SGDClassifier(**P.REL_PARAMS).fit(rel_features(S_tr, gold_tr)[keep], tr["rel"][keep])
        assert classes is None or (rel.classes_ == classes).all()     # same label order everywhere
        classes = rel.classes_
        p_va = p_va + rel.predict_proba(rel_features(S_va, heads_va)) / len(MEMBERS)
        p_te = p_te + rel.predict_proba(rel_features(S_te, heads_te)) / len(MEMBERS)
        print(f"[{time.time() - t0:.0f}s] member {m}: relation classifier done")
    rels_va = np.where(heads_va == 0, "root", classes[p_va.argmax(1)])
    rels_te = np.where(heads_te == 0, "root", classes[p_te.argmax(1)])

    # ---- evaluate, log, submit
    out = va.assign(pred_head=heads_va, pred_rel=rels_va)
    met, by_len, per_rel = evaluate(out, train_vocab=vocab)
    print(f"\n===== {NAME} =====")
    print_report(met, by_len, per_rel, top=10)
    arc_info = ({k: v for k, v in P.ARC_PARAMS.items() if k != "random_state"} if arc_kind == "lr"
                else {"structured_perceptron_epochs": StructPerceptronArc().epochs})
    log_result(NAME, met, params={"members": MEMBERS, "dirfeats": True, "arc_kind": arc_kind,
                                  "neg": "hard" if arc_kind == "lr" else "eisner-decoded trees",
                                  "arc": arc_info},
               notes="MAIN: average of 3 arc scorers + 3 relation classifiers (different tags)",
               extra={f"UAS_{m}": u for m, u in member_uas.items()})
    save_val_preds(NAME, out)
    make_submission(te.assign(pred_head=heads_te, pred_rel=rels_te), NAME)
    print(f"\n{NAME}: LAS {met['LAS']:.4f} | E6b 0.7763 | E5a 0.7706 | [{time.time() - t0:.0f}s] done")