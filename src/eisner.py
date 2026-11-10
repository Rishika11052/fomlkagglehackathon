"""
Implements a single root projective Eisner decoder.

    from eisner import eisner
    heads = eisner(S)     # S[h, d] = score for attaching word d to head h 
                          # The matrix has shape (n+1, n+1), with index 0 representing root.
    # heads[d-1] = head of word d (0 = root); always exactly one root, no cycles and no self loops

Run from the repo root:
    python src/eisner.py
"""
import itertools
import time

import numpy as np

NEG = -1e18   # "impossible" score 


def eisner(S):
    """Best projective tree with exactly one word attached to root.
    first build the best tree using words 1 to n then choose a word r to attach to ROOT
        Score = left subtree score + right subtree score + S[0, r].."""
    S = np.asarray(S, dtype=np.float64)
    n = S.shape[0] - 1                       # number of words
    if n == 1:
        return [0]                           # one word then root
    A = S[1:, 1:].copy()                     # scores for attaching word d to word h(0 indexed)
    np.fill_diagonal(A, NEG)                 # no self loops

    # spans over words 0 to n-1, where [s, t] represents the span from s to t.
    Cr = np.full((n, n), NEG); Cl = np.full((n, n), NEG)   # head at s (Cr) or at t (Cl) then complete
    Ir = np.full((n, n), NEG); Il = np.full((n, n), NEG)   # arc s->t (Ir) or t->s (Il) then incomplete
    bCr = np.zeros((n, n), int); bCl = np.zeros((n, n), int)  #Store split points using back pointers.
    bI = np.zeros((n, n), int)                               # shared split point for Ir and Il
    idx = np.arange(n)
    Cr[idx, idx] = 0.0                       # a single word is a complete span of score 0
    Cl[idx, idx] = 0.0

    for k in range(1, n):                    # span length
        s = np.arange(n - k)                 
        t = s + k                            
        r = s[:, None] + np.arange(k)[None, :]  

        #for incomplete join right facing [s, r] and left facing [r+1, t] then add the arc
        m = Cr[s[:, None], r] + Cl[r + 1, t[:, None]]
        best = m.argmax(1)
        bI[s, t] = r[np.arange(len(s)), best]
        base = m[np.arange(len(s)), best]
        Ir[s, t] = base + A[s, t]            # arc s -> t
        Il[s, t] = base + A[t, s]            # arc t -> s

        #for complete head at t: join left-facing [s, r] with incomplete (t -> r) on [r, t], r = s+1 to t
        m = Cl[s[:, None], r] + Il[r, t[:, None]]
        best = m.argmax(1)
        bCl[s, t] = r[np.arange(len(s)), best]
        Cl[s, t] = m[np.arange(len(s)), best]

        # for complete head at s: join incomplete (s -> r) on [s, r] with right-facing [r, t], r = s+1 to t
        r2 = r + 1
        m = Ir[s[:, None], r2] + Cr[r2, t[:, None]]
        best = m.argmax(1)
        bCr[s, t] = r2[np.arange(len(s)), best]
        Cr[s, t] = m[np.arange(len(s)), best]

    # attach exactly one word r to ROOT
    total = Cl[0, :] + Cr[:, n - 1] + S[0, 1:]
    root = int(total.argmax())

    heads = np.full(n, -1)
    heads[root] = -1                          # made to 0 at the end
    stack = [("Cl", 0, root), ("Cr", root, n - 1)]  
    while stack:
        kind, s, t = stack.pop()
        if s == t:
            continue                          # Single word spans dont need further splitting.
        if kind == "Cl":                      # head at t then unpack left part [s, r] and arc t -> r
            r = bCl[s, t]
            stack += [("Cl", s, r), ("Il", r, t)]
        elif kind == "Cr":                    # head at s then unpack arc s -> r and right part [r, t]
            r = bCr[s, t]
            stack += [("Ir", s, r), ("Cr", r, t)]
        elif kind == "Il":                    # arc t -> s
            heads[s] = t
            r = bI[s, t]
            stack += [("Cr", s, r), ("Cl", r + 1, t)]
        else:                                 # arc s -> t
            heads[t] = s
            r = bI[s, t]
            stack += [("Cr", s, r), ("Cl", r + 1, t)]

    out = heads + 1                           # comvert word indices back to 1 indexed word ids
    out[root] = 0                             # attach the selected root word to ROOT.
    return out.tolist()


def tree_score(S, heads):
    # total score of a tree = sum of its arc scores
    return sum(S[h, d] for d, h in enumerate(heads, start=1))


