"""
Step 2a: loading the data and the fixed validation split.

Run ONCE from the repo root to create the split:
    python src/data.py
This writes splits/split.csv. Commit it: every experiment must use the same split.
Running it again does nothing unless you add --force (which would change the split).

Other files import from here:
    from data import load_train, load_test, train_val, length_bucket
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- settings
ROOT = Path(__file__).resolve().parent.parent   
DATA = ROOT / "data"                             
SPLIT_FILE = ROOT / "splits" / "split.csv"       
SEED = 42          # one seed used everywhere
VAL_FRAC = 0.10    # share of train sentences held out for validation
N_FOLDS = 5        # folds inside the training part

LEN_BINS = [0, 5, 10, 20, 30, 50, 10**6]                  # sentence-length bucket edges
LEN_LABELS = ["1-5", "6-10", "11-20", "21-30", "31-50", "51+"]  # bucket names


# ---------------------------------------------------------------- loading
def read_csv(name):
    # keep_default_na=False: tokens like "NA", "null", "nan" stay strings instead of NaN
    return pd.read_csv(DATA / name, keep_default_na=False)


def check_order(df):
    # models assume rows are grouped by sentence and token_id runs 1..n inside each one
    expected = df.groupby("sentence_id", sort=False).cumcount() + 1   # 1,2,3.. per sentence
    assert (df["token_id"].to_numpy() == expected.to_numpy()).all(), "rows are not in 1..n order"


def load_train():
    df = read_csv("train.csv")
    #split on the first "|" only
    parts = df["target"].str.split("|", n=1, expand=True)
    df["head"] = parts[0].astype(int)   # head token_id, 0 = root
    df["rel"] = parts[1]                 # dependency relation
    check_order(df)
    return df


def load_test():
    df = read_csv("test.csv")
    check_order(df)
    return df


def sentence_lengths(df):
    # number of tokens in each sentence in file order
    return df.groupby("sentence_id", sort=False).size()


def length_bucket(n_tokens):
    # turn sentence lengths into bucket names 
    return pd.cut(n_tokens, LEN_BINS, labels=LEN_LABELS)


def test_length_share():
    # share of TEST tokens in each length bucket
    # used to re-weight validation scores
    lens = sentence_lengths(load_test())
    return lens.groupby(length_bucket(lens), observed=False).sum() / lens.sum()


# ---------------------------------------------------------------- the fixed split
def make_split(force=False):
    """Hold out VAL_FRAC of train sentences, stratified by length bucket so the
    validation set has the same length mix as train. The remaining sentences
    get fold numbers 0..N_FOLDS-1. fold = -1 means validation."""
    if SPLIT_FILE.exists() and not force:
        print(f"{SPLIT_FILE} already exists, keeping it (use --force to overwrite)")
        return pd.read_csv(SPLIT_FILE)

    lens = sentence_lengths(load_train())            # tokens per sentence
    buckets = length_bucket(lens)                     # bucket of each sentence
    rng = np.random.default_rng(SEED)                 # seeded random generator
    fold = pd.Series(-1, index=lens.index)            # start with everything = validation

    for b in LEN_LABELS:                              # split each length bucket separately
        ids = rng.permutation(lens.index[buckets == b].to_numpy())  # shuffled sentence ids
        n_val = int(round(len(ids) * VAL_FRAC))       # how many go to validation
        rest = ids[n_val:]                            # the first n_val stay -1 (validation)
        fold[rest] = np.arange(len(rest)) % N_FOLDS   # round-robin folds 0..4 for the rest

    split = pd.DataFrame({"sentence_id": lens.index,
                          "n_tokens": lens.to_numpy(),
                          "fold": fold.to_numpy()})
    SPLIT_FILE.parent.mkdir(parents=True, exist_ok=True)   # create splits/ if missing
    split.to_csv(SPLIT_FILE, index=False)
    return split


def load_split():
    # read the fixed split
    # error if it hasn't been created yet
    if not SPLIT_FILE.exists():
        raise FileNotFoundError("run `python src/data.py` once to create splits/split.csv")
    return pd.read_csv(SPLIT_FILE)


def train_val():
    """Returns (train_part, val_part): train.csv divided by the fixed split.
    Both keep a 'fold' column (0..4 in train_part, -1 in val_part)."""
    train = load_train()
    split = load_split()[["sentence_id", "fold"]]
    train = train.merge(split, on="sentence_id", how="left")   # left merge keeps row order
    assert train["fold"].notna().all(), "some sentences are missing from the split file"
    train_part = train[train["fold"] >= 0].reset_index(drop=True)
    val_part = train[train["fold"] == -1].reset_index(drop=True)
    return train_part, val_part


# ---------------------------------------------------------------- run once
if __name__ == "__main__":
    split = make_split(force="--force" in sys.argv)
    tr, va = train_val()

    # summary to confirm the split looks right
    print(f"train part: {tr.sentence_id.nunique():,} sentences, {len(tr):,} tokens")
    print(f"val part:   {va.sentence_id.nunique():,} sentences, {len(va):,} tokens")
    print("sentences per fold:", split["fold"].value_counts().sort_index().to_dict())

    # length mix should be (almost) identical in every part
    mix = pd.crosstab(split["fold"], length_bucket(split["n_tokens"]), normalize="index") * 100
    print("\n% of sentences per length bucket, by fold (-1 = validation):")
    print(mix.round(1).to_string())