"""
Step 6: tagger v2 = the v1 structured perceptron (imported from tagger.py) with
richer features and the number of epochs chosen on an inner dev fold.

Run from the repo root:
    python src/tagger2.py              # T2: writes tags_v2/      (supertag = rel/dir, as v1)
    python src/tagger2.py --neardist   #     writes tags_v2near/  (supertag = rel/dir/near-or-far)
    python src/tagger2.py --test       # self-tests only

Epochs: one tagger is trained on folds 1-4 and scored on fold 0 after every epoch;
the best of EPOCH_CHOICES is used for all final taggers (validation is never used).

Tag -> parser feature mapping (written explicitly into every tag file):
    plain:  "obj/L"    -> rel_class "obj", sdir "L"      ("root" -> "root", "ROOT")
    near:   "obj/L/n"  -> rel_class "obj", sdir "L/n"    (n = head adjacent, f = farther)
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

from data import ROOT, SEED, train_val, load_test
from baselines import shape
from tagger import Tagger, token_features, make_tag, split_tag, by_sentence, flat, BOS, EOS

EPOCH_CHOICES = [6, 10, 15]     # candidate numbers of epochs
TIE = 0.001                     # prefer fewer epochs unless more is better by > 0.1 point


# ---------------------------------------------------------------- richer features
def token_features2(tokens):
    """v1 features + wider context, affixes of neighbours and sentence-level cues."""
    base = token_features(tokens)                       # all v1 features, unchanged
    low = [t.lower() for t in tokens]
    shp = [shape(t) for t in tokens]
    n = len(tokens)
    pad = lambda i: BOS if i < 0 else EOS
    W = lambda i: low[i] if 0 <= i < n else pad(i)                    # word at i
    S = lambda i: shp[i] if 0 <= i < n else pad(i)                    # shape at i
    X = lambda i, k: low[i][-k:] if 0 <= i < n else pad(i)            # suffix-k at i
    P = lambda i, k: low[i][:k] if 0 <= i < n else pad(i)             # prefix-k at i
    slen = str(np.digitize(n, [2, 4, 8, 16, 31]))                     # sentence length bucket
    endp = str(int(shp[-1] == "punct"))                               # sentence ends in punctuation
    for i, t in enumerate(tokens):
        base[i] += [
            "x2-1=" + X(i - 1, 2), "x2+1=" + X(i + 1, 2),               # neighbour suffixes
            "x3-2=" + X(i - 2, 3), "x3+2=" + X(i + 2, 3),
            "p3-1=" + P(i - 1, 3), "p3+1=" + P(i + 1, 3),               # neighbour prefixes
            "sh-2=" + S(i - 2), "sh+2=" + S(i + 2),                     # shapes 2 away
            "capmid=%d" % (i > 0 and t[:1].isupper()),                  # capital, not first word
            "w-1|s3=" + W(i - 1) + "|" + low[i][-3:],                   # left word + own suffix
            "w+1|s3=" + W(i + 1) + "|" + low[i][-3:],                   # right word + own suffix
            "w-1|w+1=" + W(i - 1) + "|" + W(i + 1),                     # words around it
            "slen=" + slen, "toend=%d" % min(n - 1 - i, 5), "endp=" + endp,
        ]
    return base


class Tagger2(Tagger):
    featurize = staticmethod(token_features2)          # same perceptron, v2 features


# ---------------------------------------------------------------- near/far supertags
def make_tag_near(rel, head, token_id):
    # "obj/L/n": relation / head direction / n if the head is adjacent, else f
    if head == 0:
        return "root"
    return f"{make_tag(rel, head, token_id)}/{'n' if abs(head - token_id) == 1 else 'f'}"


def split_tag_near(tag):
    # explicit mapping: "obj/L/n" -> rel_class "obj", sdir "L/n"
    if tag == "root":
        return "root", "ROOT"
    rel, d, dist = tag.rsplit("/", 2)
    return rel, f"{d}/{dist}"


def gold_tags(df, near):
    mk = make_tag_near if near else make_tag
    return [mk(r, h, t) for r, h, t in zip(df["rel"], df["head"], df["token_id"])]


def rel_part(tags, near):
    sp = split_tag_near if near else split_tag
    return np.array([sp(t)[0] for t in tags])


def tag_frame(df, tags, near):
    rel, sdir = zip(*[(split_tag_near if near else split_tag)(t) for t in tags])
    return pd.DataFrame({"ID": df["ID"].to_numpy(), "tag": tags, "rel_class": rel, "sdir": sdir})


# ---------------------------------------------------------------- epoch choice on fold 0
def choose_epochs(tr, near):
    """Train on folds 1-4, score fold 0 (relation part) after every epoch."""
    fit_part, dev = tr[tr["fold"] != 0], tr[tr["fold"] == 0]
    dev_sents, dev_gold = by_sentence(dev, "token"), rel_part(gold_tags(dev, near), near)
    acc = {}

    def score(ep, tg):
        if ep in EPOCH_CHOICES:
            acc[ep] = float((rel_part(flat(tg.predict(dev_sents)), near) == dev_gold).mean())
            print(f"  -> fold-0 relation accuracy after {ep} epochs: {acc[ep]:.4f}")

    Tagger2().fit(by_sentence(fit_part, "token"), by_sentence(fit_part, "tag"),
                  epochs=max(EPOCH_CHOICES), on_epoch=score)
    best = EPOCH_CHOICES[0]
    for ep in EPOCH_CHOICES[1:]:                        # more epochs only if clearly better
        if acc[ep] > acc[best] + TIE:
            best = ep
    return best, acc


# ---------------------------------------------------------------- self-test
def self_test():
    toks = ["The", "dog", "barked", "."]
    f = token_features2(toks)
    assert len({len(x) for x in f}) == 1 and len(f[0]) > len(token_features(toks)[0])
    print(f"features: {len(f[0])} per word (v1: {len(token_features(toks)[0])}): passed")
    for rel, h, t in [("det", 2, 1), ("obj", 1, 3), ("nmod:poss", 5, 2), ("root", 0, 4)]:
        tag = make_tag_near(rel, h, t)
        r, sdir = split_tag_near(tag)
        assert r == rel and (sdir == "ROOT") == (h == 0)
    print("near/far tag mapping: passed")


# ---------------------------------------------------------------- run
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--neardist", action="store_true", help="supertags also mark near/far heads")
    ap.add_argument("--test", action="store_true", help="self-tests only")
    args = ap.parse_args()
    self_test()
    if args.test:
        raise SystemExit
    near = args.neardist
    out_dir = ROOT / ("tags_v2near" if near else "tags_v2")
    t0 = time.time()
    tr, va = train_val()
    te = load_test()
    tr = tr.assign(tag=gold_tags(tr, near))

    # ---- 1. choose the number of epochs on fold 0
    print(f"[{time.time() - t0:.0f}s] choosing epochs (train folds 1-4, score fold 0)")
    epochs, dev_acc = choose_epochs(tr, near)
    print(f"[{time.time() - t0:.0f}s] chosen: {epochs} epochs")

    # ---- 2. out-of-fold tags for the training part (sentence-level separation)
    oof = pd.Series(index=tr.index, dtype=object)
    for k in sorted(tr["fold"].unique()):
        fit_part, tag_part = tr[tr["fold"] != k], tr[tr["fold"] == k]
        assert not set(fit_part["sentence_id"]) & set(tag_part["sentence_id"])
        print(f"[{time.time() - t0:.0f}s] fold {k}")
        tg = Tagger2().fit(by_sentence(fit_part, "token"), by_sentence(fit_part, "tag"), epochs=epochs)
        oof[tag_part.index] = flat(tg.predict(by_sentence(tag_part, "token")))

    # ---- 3. validation and test tags from one tagger on the whole training part
    print(f"[{time.time() - t0:.0f}s] full training part")
    tg = Tagger2().fit(by_sentence(tr, "token"), by_sentence(tr, "tag"), epochs=epochs)
    va_tags = flat(tg.predict(by_sentence(va, "token")))
    te_tags = flat(tg.predict(by_sentence(te, "token")))

    out_dir.mkdir(exist_ok=True)
    tag_frame(tr, oof.tolist(), near).to_csv(out_dir / "tags_train_oof.csv", index=False)
    tag_frame(va, va_tags, near).to_csv(out_dir / "tags_val.csv", index=False)
    tag_frame(te, te_tags, near).to_csv(out_dir / "tags_test.csv", index=False)

    # ---- T2 metrics (relation part = what the parser uses as word class)
    va_gold = gold_tags(va, near)
    vrel, grel = rel_part(va_tags, near), rel_part(va_gold, near)
    unseen = ~va["token"].str.lower().isin(set(tr["token"].str.lower())).to_numpy()
    m = {
        "variant": "v2near" if near else "v2", "epochs": epochs, "fold0_acc_by_epoch": dev_acc,
        "n_tags": len(tg.tags), "n_features_per_word": len(token_features2(["a"])[0]),
        "val_tag_acc": float((np.array(va_tags) == np.array(va_gold)).mean()),
        "val_rel_acc": float((vrel == grel).mean()),
        "val_rel_acc_unseen": float((vrel == grel)[unseen].mean()),
        "oof_rel_acc_train": float((rel_part(oof.tolist(), near) == tr["rel"].to_numpy()).mean()),
        "seed": SEED,
    }
    (out_dir / "T2_metrics.json").write_text(json.dumps(m, indent=2))
    t1_path = ROOT / "tags" / "T1_metrics.json"
    t1 = json.loads(t1_path.read_text()) if t1_path.exists() else {}
    print(f"\n===== T2 ({m['variant']}, {epochs} epochs, {m['n_tags']} tags) on validation =====")
    print(f"full supertag accuracy {m['val_tag_acc']:.4f}")
    print(f"relation part {m['val_rel_acc']:.4f} (T1: {t1.get('val_rel_acc', float('nan')):.4f}) | "
          f"unseen words {m['val_rel_acc_unseen']:.4f} (T1: {t1.get('val_rel_acc_unseen', float('nan')):.4f})")
    print(f"out-of-fold relation accuracy on the training part: {m['oof_rel_acc_train']:.4f}")
    print(f"[{time.time() - t0:.0f}s] wrote {out_dir.name}/")
