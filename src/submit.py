"""
Run from the repo root:
    python src/submit.py
"""
import re
from pathlib import Path

import pandas as pd

from data import ROOT, load_test, read_csv, sentence_lengths

SUB_DIR = ROOT / "submissions"
LABEL_RE = re.compile(r"^\d+\|[a-z]+(:[a-z]+)?$")   


def make_submission(test_pred, name):
    """test_pred: the test DataFrame (from load_test) plus pred_head and pred_rel.
    Writes submissions/<name>.csv after checking it. Returns the path."""
    sample = read_csv("sample.csv")
    test = load_test()

    # every test row must be present once with a prediction
    assert len(test_pred) == len(test), "test_pred must have one row per test token"
    assert set(test_pred["ID"]) == set(test["ID"]), "test_pred IDs differ from test.csv"
    assert test_pred["pred_head"].notna().all() and test_pred["pred_rel"].notna().all(), \
        "some test rows have no prediction"

    # build "HEAD|DEPREL" for each test ID
    labels = test_pred["pred_head"].astype(int).astype(str) + "|" + test_pred["pred_rel"].astype(str)
    id_to_label = dict(zip(test_pred["ID"], labels))

    # start from sample.csv so columns, rows and order are same
    sub = sample.copy()
    # rows that belong to test.csv
    is_test = sub["ID"].isin(id_to_label.keys())         
    sub.loc[is_test, "Label"] = sub.loc[is_test, "ID"].map(id_to_label)

    check_submission(sub, sample, test_pred)

    SUB_DIR.mkdir(parents=True, exist_ok=True)
    path = SUB_DIR / f"{name}.csv"
    sub.to_csv(path, index=False)
    print(f"wrote {path.relative_to(ROOT)}  ({is_test.sum():,} test rows filled, {len(sub):,} rows total)")
    return path


def check_submission(sub, sample, test_pred):
    """Error if the file is bad."""
    
    assert list(sub.columns) == list(sample.columns) == ["ID", "Label"], f"columns are {list(sub.columns)}"
    assert len(sub) == len(sample), "row count differs from sample.csv"
    assert (sub["ID"].to_numpy() == sample["ID"].to_numpy()).all(), "row order differs from sample.csv"
    assert sub["ID"].is_unique, "duplicate IDs"

    # check if test labels look like HEAD|DEPREL
    test_labels = sub.loc[sub["ID"].isin(test_pred["ID"]), "Label"]
    bad = test_labels[~test_labels.map(lambda s: bool(LABEL_RE.match(s)))]
    assert bad.empty, f"badly formatted labels, e.g. {bad.head().tolist()}"

    # predicted heads must point inside their sentence (0..n)
    n = test_pred["sentence_id"].map(sentence_lengths(test_pred))
    assert ((test_pred["pred_head"] >= 0) & (test_pred["pred_head"] <= n)).all(), "head outside its sentence"

    # warning for relations that never occur in train
    known = set(read_csv("train.csv")["target"].str.split("|", n=1).str[1])
    unknown = set(test_pred["pred_rel"]) - known
    if unknown:
        print(f"WARNING: relations not seen in train: {sorted(unknown)}")


