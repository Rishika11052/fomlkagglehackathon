"""
Step 2b: evaluation and experiment logging.

Every model is scored the same way:
    from evaluate import evaluate, print_report, log_result
    metrics, by_len, per_rel = evaluate(val_df)   # val_df needs pred_head and pred_rel columns
    print_report(metrics, by_len, per_rel)
    log_result("my_experiment", metrics, params={"lr": 0.05})

Self-test (run from the repo root):
    python src/evaluate.py
"""
import json
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from data import ROOT, SEED, length_bucket, test_length_share

RESULTS_FILE = ROOT / "experiments" / "results.csv"   # one row per experiment
PREDS_DIR = ROOT / "experiments" / "preds"            # saved validation predictions


# ---------------------------------------------------------------- tree checks
def check_tree(heads):
    """heads[i] = head of word i+1, 0 = root. n = number of tokens.
    Returns a dict with bad_head, n_roots, cycle and unreachable.
    A valid tree has: every head in 0..n, no self-loops, exactly one root,
    no cycles, and every word reaching the root."""
    heads = [int(h) for h in heads]
    n = len(heads)
    # heads outside 0..n, or a word pointing to itself
    bad_head = [i + 1 for i, h in enumerate(heads) if not (0 <= h <= n) or h == i + 1]
    # words whose head is 0 (should be exactly one)
    n_roots = sum(h == 0 for h in heads)
    in_cycle, unreachable = set(), set()
    if not bad_head:                       # following heads is only safe if all are in range
        for start in range(1, n + 1):      # walk up from every word
            seen, a = [], start
            while a != 0 and a not in seen:   # stop at the root or at a repeated word
                seen.append(a)
                a = heads[a - 1]
            if a != 0:                     # stopped at a repeated word meaning cycle
                in_cycle.update(seen[seen.index(a):])
                unreachable.add(start)
    return {"bad_head": bad_head, "n_roots": n_roots,
            "cycle": sorted(in_cycle), "unreachable": sorted(unreachable)}


def is_valid_tree(heads):
    p = check_tree(heads)
    return not p["bad_head"] and p["n_roots"] == 1 and not p["cycle"] and not p["unreachable"]


# ---------------------------------------------------------------- scoring
def evaluate(df):
    """df: rows of validation tokens with gold 'head', 'rel' and predicted
    'pred_head', 'pred_rel'. Returns (metrics dict, by-length table, per-relation table).
    LAS = the Kaggle score (head AND relation correct)."""
    for col in ["sentence_id", "token_id", "head", "rel", "pred_head", "pred_rel"]:
        assert col in df.columns, f"missing column {col}"
    assert df["pred_head"].notna().all() and df["pred_rel"].notna().all(), "missing predictions"

    head_ok = df["pred_head"].astype(int).to_numpy() == df["head"].to_numpy()   # head right
    rel_ok = df["pred_rel"].astype(str).to_numpy() == df["rel"].to_numpy()       # relation right
    both = head_ok & rel_ok                                                      # Kaggle-correct

    m = {
        "tokens": len(df),
        "UAS": head_ok.mean(),                    # unlabeled attachment score
        "LA": rel_ok.mean(),                      # label accuracy
        "LAS": both.mean(),                       # labeled attachment score = Kaggle metric
        "head_ok_rel_wrong": (head_ok & ~rel_ok).mean(),   # where LAS is lost to the label
        "head_wrong": (~head_ok).mean(),                    # where LAS is lost to the head
    }

    # ---- by sentence length
    n_tok = df.groupby("sentence_id", sort=False)["token_id"].transform("size")   # length of each row's sentence
    tmp = pd.DataFrame({"bucket": length_bucket(n_tok), "head_ok": head_ok, "both": both})
    by_len = tmp.groupby("bucket", observed=False).agg(
        tokens=("both", "size"), UAS=("head_ok", "mean"), LAS=("both", "mean"))
    share = test_length_share()                          # test token share per bucket
    has = by_len["LAS"].notna()                          # buckets present in this val set
    by_len["test_token_share"] = share
    # LAS if validation had the same length mix as test (test has shorter sentences)
    m["LAS_testweighted"] = (by_len.loc[has, "LAS"] * share[has]).sum() / share[has].sum()

    # ---- per gold relation
    tmp = pd.DataFrame({"rel": df["rel"].to_numpy(), "pred_rel": df["pred_rel"].astype(str).to_numpy(),
                        "head_ok": head_ok, "rel_ok": rel_ok, "both": both})
    per_rel = tmp.groupby("rel").agg(
        count=("both", "size"), head_acc=("head_ok", "mean"),
        label_recall=("rel_ok", "mean"), LAS=("both", "mean"))
    # label precision: of the tokens predicted as this relation, how many really are
    per_rel["label_precision"] = tmp.groupby("pred_rel")["rel_ok"].mean()
    # LAS points lost on this relation, as a share of all validation tokens
    per_rel["lost_LAS_pts"] = (per_rel["count"] * (1 - per_rel["LAS"])) / len(df) * 100
    per_rel = per_rel.sort_values("lost_LAS_pts", ascending=False)

    # check if the predicted heads are valid trees?
    valid = [is_valid_tree(s.to_numpy()) for _, s in df.groupby("sentence_id", sort=False)["pred_head"]]
    m["valid_tree_pct"] = float(np.mean(valid) * 100)

    return m, by_len, per_rel


def print_report(m, by_len=None, per_rel=None, top=15):
    # headline numbers
    print(f"tokens {m['tokens']:,} | UAS {m['UAS']:.4f} | LA {m['LA']:.4f} | "
          f"LAS {m['LAS']:.4f} | LAS test-weighted {m['LAS_testweighted']:.4f} | "
          f"valid trees {m['valid_tree_pct']:.1f}%")
    print(f"LAS lost: wrong head {m['head_wrong']:.4f}, right head but wrong label {m['head_ok_rel_wrong']:.4f}")
    if by_len is not None:
        print("\nby sentence length:")
        print(by_len.round(4).to_string())
    if per_rel is not None:
        print(f"\nrelations losing the most LAS (top {top}):")
        print(per_rel.head(top).round(3).to_string())


# ---------------------------------------------------------------- experiment log
def git_version():
    # short commit hash, plus "+dirty" if there are uncommitted changes
    try:
        h = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                           capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                               capture_output=True, text=True).stdout.strip()
        return h + ("+dirty" if dirty else "")
    except Exception:
        return "no-git"


def log_result(name, m, params=None, notes=""):
    """Append one experiment to experiments/results.csv so every result is traceable:
    when it ran, which code version, which settings, which seed."""
    row = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "name": name,
        "git": git_version(),
        "seed": SEED,
        "LAS": round(m["LAS"], 4),
        "UAS": round(m["UAS"], 4),
        "LA": round(m["LA"], 4),
        "LAS_testweighted": round(m["LAS_testweighted"], 4),
        "valid_tree_pct": round(m["valid_tree_pct"], 1),
        "params": json.dumps(params or {}, sort_keys=True),
        "notes": notes,
    }
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    new = not RESULTS_FILE.exists()                          # write the header only once
    pd.DataFrame([row]).to_csv(RESULTS_FILE, mode="a", header=new, index=False)
    print(f"logged '{name}' -> {RESULTS_FILE.relative_to(ROOT)}")


def save_val_preds(name, df):
    # keep validation predictions for error analysis and ensembling later
    PREDS_DIR.mkdir(parents=True, exist_ok=True)
    df[["sentence_id", "token_id", "pred_head", "pred_rel"]].to_csv(PREDS_DIR / f"{name}.csv", index=False)


# ---------------------------------------------------------------- self-test
if __name__ == "__main__":
    from data import train_val

    # 1. tree checker on hand-built trees
    assert is_valid_tree([2, 0, 2])                       # 1 -> 2 <- 3
    assert check_tree([2, 1, 0])["cycle"] == [1, 2]       # 1 <-> 2 cycle
    assert check_tree([0, 0, 2])["n_roots"] == 2          # two roots
    assert check_tree([4, 0])["bad_head"] == [1]          # head 4 doesn't exist
    assert check_tree([1, 0])["bad_head"] == [1]          # self-loop
    print("tree checks passed")

    _, val = train_val()

    # 2. gold predictions must score exactly 1.0 everywhere
    gold = val.assign(pred_head=val["head"], pred_rel=val["rel"])
    m, _, _ = evaluate(gold)
    assert m["UAS"] == m["LA"] == m["LAS"] == 1.0 and m["valid_tree_pct"] == 100.0
    print("gold-as-prediction scores 1.0: passed")

    # 3. known numbers on a hand-made case: 4 tokens, 1 head wrong, 1 label wrong
    toy = pd.DataFrame({"sentence_id": ["s"] * 4, "token_id": [1, 2, 3, 4],
                        "head": [2, 0, 2, 3], "rel": ["det", "root", "obj", "case"],
                        "pred_head": [2, 0, 2, 2], "pred_rel": ["det", "root", "nsubj", "case"]})
    m, _, _ = evaluate(toy)
    assert (m["UAS"], m["LA"], m["LAS"]) == (0.75, 0.75, 0.5)
    print("hand-made case (UAS 0.75, LA 0.75, LAS 0.5): passed")

    # 4. a deliberately weak predictor, to see the full report format
    #    head = previous word (first word = root), relation = "punct" for symbols else "nsubj"
    weak = val.copy()
    weak["pred_head"] = weak["token_id"] - 1
    sym = weak["token"].map(lambda t: bool(t) and not any(c.isalnum() for c in t))
    weak["pred_rel"] = np.where(weak["pred_head"] == 0, "root", np.where(sym, "punct", "nsubj"))
    m, by_len, per_rel = evaluate(weak)
    print("\nexample report (head = previous word):")
    print_report(m, by_len, per_rel, top=8)