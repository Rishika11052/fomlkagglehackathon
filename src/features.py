"""
Feature extraction for the graph based parser.

    from features import word_classes_oof, word_classes, Sentences, arc_features, rel_features
    sents = Sentences(df, classes)             # Token attributes, with ROOT and EOS added
    X = arc_features(sents, sent, h, d)         # one row per (head h, dependent d) pair
    X = rel_features(sents, heads)              # one row per token given a tree

Every feature is a hashed combination of simple attributes. Hashing uses fixed
integer arithmetic so the same input always gives the same feature columns.(not python's hash())

Run from the repo root:
    python src/features.py
"""
import zlib

import numpy as np
import pandas as pd
import scipy.sparse as sp

from baselines import fit_stats, predict_rel, shape

D_ARC = 2 ** 22        # hashed feature columns for the arc scorer
D_REL = 2 ** 20        # hashed feature columns for the relation classifier
ROOT, EOS = "<ROOT>", "<EOS>"   # pseudo-tokens added at the start and end of every sentence
VERBISH = {"root", "ccomp", "xcomp", "advcl", "advcl:relcl", "acl", "acl:relcl", "parataxis", "csubj"}
DIST_EDGES = [2, 3, 4, 6, 11]   # distance buckets: 1, 2, 3, 4-5, 6-10, 11+


# ---------------------------------------------------------------- word classes
def word_classes(fit_df, apply_df):
    # majority relation of each word, learned on fit_df, looked up for apply_df
    return predict_rel(apply_df["token"], fit_stats(fit_df)).to_numpy()


def word_classes_oof(train_part):
    """"Generate classes for the training data using 5 folds. Each fold's classes come from
    the other 4 folds, so no token ever sees its own label."""
    out = np.empty(len(train_part), dtype=object)
    for k in sorted(train_part["fold"].unique()):
        mask = (train_part["fold"] == k).to_numpy()
        out[mask] = word_classes(train_part[~mask], train_part[mask])
    return out


# ---------------------------------------------------------------- stable hashing
def str_ids(values):
    # string -> fixed 32-bit number 
    table = {v: zlib.crc32(v.encode("utf-8")) for v in pd.unique(np.asarray(values, dtype=object))}
    return np.array([table[v] for v in values], dtype=np.uint64)


def mix(x):
    # splitmix64 finaliser: scrambles bits so similar inputs land in unrelated columns
    x = x ^ (x >> np.uint64(30)); x = x * np.uint64(0xBF58476D1CE4E5B9)
    x = x ^ (x >> np.uint64(27)); x = x * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def combine(template_id, cols, n):
    # one hashed feature: template number + the attribute values it combines
    x = mix(np.full(n, template_id + 1, dtype=np.uint64))
    for c in cols:
        x = mix(x ^ c)
    return x


# ---------------------------------------------------------------- per-token attributes
class Sentences:
    """Store each sentence's token attributes in the order [ROOT, w1, ..., wn, EOS] so a
    word's neighbours are just position -1 / +1. Rows keep the order of df."""

    def __init__(self, df, classes):
        lens = df.groupby("sentence_id", sort=False).size().to_numpy()   # tokens per sentence
        self.lens = lens
        self.n_sent = len(lens)
        self.pad_start = np.r_[0, np.cumsum(lens + 2)[:-1]]              # ROOT position of each sentence
        self.row_start = np.r_[0, np.cumsum(lens)[:-1]]                  # first df row of each sentence
        self.sent_of_row = np.repeat(np.arange(self.n_sent), lens)       # sentence index of each row
        self.local = df["token_id"].to_numpy()                           # 1..n inside the sentence
        self.row_pos = self.pad_start[self.sent_of_row] + self.local     # padded position of each row

        P = int((lens + 2).sum())                                        # total padded positions
        root_pos = self.pad_start
        eos_pos = self.pad_start + lens + 1
        tok = np.full(P, EOS, dtype=object); tok[root_pos] = ROOT        # pseudo tokens
        tok[self.row_pos] = df["token"].to_numpy()                       # real tokens
        cls = np.full(P, EOS, dtype=object); cls[root_pos] = ROOT
        cls[self.row_pos] = classes

        special = np.zeros(P, bool); special[root_pos] = special[eos_pos] = True
        lower = np.array([t if s else t.lower() for t, s in zip(tok, special)], dtype=object)
        suf3 = np.array([t if s else "s:" + t[-3:] for t, s in zip(lower, special)], dtype=object)
        shp = np.array([t if s else shape(t) for t, s in zip(tok, special)], dtype=object)

        self.W = str_ids(lower)        # word (lowercased)
        self.S3 = str_ids(suf3)        # 3-letter suffix
        self.SHP = str_ids(shp)        # word shape
        self.C = str_ids(cls)          # word class (majority relation)

        self.prev = np.arange(P) - 1; self.prev[root_pos] = root_pos     # left neighbour (ROOT stays ROOT)
        self.next = np.arange(P) + 1; self.next[eos_pos] = eos_pos       # right neighbour (EOS stays EOS)
        real = ~special
        self.cum_punct = np.cumsum(real & (shp == "punct"))              # running counts, for
        self.cum_comma = np.cumsum(real & (tok == ","))                  # "how many between" features
        self.cum_verb = np.cumsum(real & np.isin(cls, list(VERBISH)))


def dist_bucket(dist):
    return np.digitize(dist, DIST_EDGES).astype(np.uint64)   


def between(cum, a, b):
    # count of marked positions strictly between padded positions a and b
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    return cum[hi - 1] - cum[lo]


def to_csr(cols_list, n, dim, extra_rows=None, extra_cols=None):
    # stack hashed feature columns into a sparse 0/1 matrix (n rows x dim columns)
    T = len(cols_list)
    idx = np.empty((n, T), dtype=np.int32)                      # 32-bit column ids: half the memory
    for j, x in enumerate(cols_list):
        idx[:, j] = (x % np.uint64(dim)).astype(np.int32)
    if extra_rows is None:                                      # fixed number of features per row
        ptr_type = np.int32 if n * T < 2 ** 31 - 1 else np.int64
        indptr = np.arange(0, n * T + 1, T, dtype=ptr_type)
        return sp.csr_matrix((np.ones(n * T), idx.ravel(), indptr), shape=(n, dim))
    rows = np.r_[np.repeat(np.arange(n), T), extra_rows]        # plus a variable number per row
    cols = np.r_[idx.ravel(), (extra_cols % np.uint64(dim)).astype(np.int32)]
    return sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, dim))


# ---------------------------------------------------------------- arc features
ARC_TEMPLATES = [
    ("hw",), ("hc",), ("hs",), ("hp",), ("hw", "hc"),                 # head alone
    ("dw",), ("dc",), ("ds",), ("dp",), ("dw", "dc"),                 # dependent alone
    ("hw", "dw"), ("hc", "dc"), ("hw", "dc"), ("hc", "dw"),           # head + dependent
    ("hs", "ds"), ("hs", "dc"), ("hc", "ds"), ("hp", "dp"),
    ("hc", "hcR", "dcL", "dc"), ("hcL", "hc", "dcL", "dc"),           # neighbours (MSTParser style)
    ("hc", "hcR", "dc", "dcR"), ("hcL", "hc", "dc", "dcR"),
    ("hc", "dc", "npunct"), ("hc", "dc", "comma"), ("hc", "dc", "nverb"),  # in between
]


def arc_features(S, sent, h, d):
    """S: Sentences; sent: sentence index per pair; h: head (0 = ROOT); d: dependent (1..n).
    Each template appears twice: alone, and combined with direction + distance."""
    sent, h, d = np.asarray(sent), np.asarray(h), np.asarray(d)
    n = len(h)
    gh = S.pad_start[sent] + h                         # padded position of the head
    gd = S.pad_start[sent] + d                         # padded position of the dependent
    direction = np.where(h == 0, 2, np.where(h < d, 0, 1)).astype(np.uint64)  # 0 head left, 1 right, 2 ROOT
    dist = np.where(h == 0, d, np.abs(h - d))          # for ROOT: position of the word instead
    dd = direction * np.uint64(8) + dist_bucket(dist)  # direction + distance bucket in one value

    a = {
        "hw": S.W[gh], "hc": S.C[gh], "hs": S.S3[gh], "hp": S.SHP[gh],
        "dw": S.W[gd], "dc": S.C[gd], "ds": S.S3[gd], "dp": S.SHP[gd],
        "hcL": S.C[S.prev[gh]], "hcR": S.C[S.next[gh]],
        "dcL": S.C[S.prev[gd]], "dcR": S.C[S.next[gd]],
        "npunct": np.minimum(between(S.cum_punct, gh, gd), 3).astype(np.uint64),
        "comma": (between(S.cum_comma, gh, gd) > 0).astype(np.uint64),
        "nverb": np.minimum(between(S.cum_verb, gh, gd), 2).astype(np.uint64),
    }
    cols = [combine(999, [dd], n)]                     # bias by direction + distance
    for t, names in enumerate(ARC_TEMPLATES):
        vals = [a[k] for k in names]
        cols.append(combine(t, vals, n))               # template alone
        cols.append(combine(t + 500, vals + [dd], n))  # template x direction/distance
    return to_csr(cols, n, D_ARC)


# ---------------------------------------------------------------- relation features
REL_TEMPLATES = [
    ("dw",), ("ds",), ("dp",), ("dc",), ("hw",), ("hc",), ("hs",),
    ("dw", "dir"), ("dc", "dir"), ("ds", "dir"), ("dc", "hc", "dir"), ("dw", "hc", "dir"),
    ("dc", "hw", "dir"), ("dw", "hw"), ("ds", "hc", "dir"), ("dc", "dir", "bk"), ("dw", "dir", "bk"),
    ("dcL", "dc", "dcR"), ("dwL",), ("dwR",), ("dc", "dwL"), ("dc", "dwR"), ("dw", "dcL"), ("dw", "dcR"),
    ("dc", "hc", "gc"), ("hc", "gc", "dir"),            # grandparent (head of the head)
    ("dc", "nch"), ("dw", "nch"), ("dc", "first"), ("dc", "last"), (),
]


def rel_features(S, heads):
    """One row per token, in the same order as df. heads: head of every token- gold for training,
    predicted for evaluation. Uses the word, its head, the head's head and the
    word's own children in that tree."""
    heads = np.asarray(heads)
    n = len(heads)
    sent = S.sent_of_row
    d = S.local
    gd = S.row_pos
    gh = S.pad_start[sent] + heads                     # padded position of the head    
    head_row = S.row_start[sent] + np.maximum(heads, 1) - 1
    g = np.where(heads > 0, heads[head_row], 0)
    gg = S.pad_start[sent] + g
    gc = np.where(heads > 0, S.C[gg], S.C[S.pad_start[sent]])   # ROOT class if no grandparent
    direction = np.where(heads == 0, 2, np.where(heads < d, 0, 1)).astype(np.uint64)
    dist = np.where(heads == 0, d, np.abs(heads - d))
    nch = np.bincount(np.where(heads > 0, head_row, n), minlength=n + 1)[:n]  # children per token

    a = {
        "dw": S.W[gd], "ds": S.S3[gd], "dp": S.SHP[gd], "dc": S.C[gd],
        "hw": S.W[gh], "hc": S.C[gh], "hs": S.S3[gh], "gc": gc,
        "dir": direction, "bk": dist_bucket(dist),
        "dwL": S.W[S.prev[gd]], "dwR": S.W[S.next[gd]],
        "dcL": S.C[S.prev[gd]], "dcR": S.C[S.next[gd]],
        "nch": np.minimum(nch, 4).astype(np.uint64),
        "first": (d == 1).astype(np.uint64),
        "last": (d == S.lens[sent]).astype(np.uint64),
    }
    cols = [combine(1000 + t, [a[k] for k in names], n) for t, names in enumerate(REL_TEMPLATES)]

    # For each child add two features to its head's row:
    # The child's class and its position (left or right).
    # The head's class, the child's class and its position (left or right).
    child = np.flatnonzero(heads > 0)                  # rows that have a word as head
    parent = head_row[child]                           # their head's row
    side = (S.local[child] < S.local[parent]).astype(np.uint64)   # child left (1) or right (0) of head
    cc = S.C[S.row_pos[child]]
    f1 = combine(1100, [cc, side], len(child))
    f2 = combine(1101, [S.C[S.row_pos[parent]], cc, side], len(child))
    return to_csr(cols, n, D_REL, np.r_[parent, parent], np.r_[f1, f2])



