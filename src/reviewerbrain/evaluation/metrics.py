"""Evaluation helpers for ReviewerBrain held-out retrieval evaluation."""

import os


def rankdata(a):
    import numpy as np
    return np.argsort(np.argsort(a)).astype(np.float64)


def spearman(a, b):
    """Spearman rank correlation between two score vectors."""
    import numpy as np
    if len(a) < 2:
        return float("nan")
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def is_boilerplate(hunk):
    """True if the hunk head looks like license/boilerplate text."""
    head = (hunk or "")[:600].lower()
    return ("licensed under" in head) or ("copyright 2" in head)
