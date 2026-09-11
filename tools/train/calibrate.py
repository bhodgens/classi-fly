"""Per-class route-threshold calibration from TRAIN-side quantiles (leaf 04 Task 3).

Never use a fixed absolute threshold tuned on another model family - it does
not transfer. Everything here is derived from the training scores themselves.

For each class, candidate thresholds are the quantiles (0.00 .. 1.00) of that
class's own train-side scores. The chosen threshold is the LOWEST candidate
whose lane precision (true-class rows routed / all rows routed) meets
``target_precision`` - the lowest qualifying quantile maximizes coverage at
the required precision. If no candidate qualifies, the class's maximum
observed score is used (route only on near-certainty).
"""

import math


def _quantile(sorted_vals, q):
    """Linear-interpolation quantile (numpy 'linear' method), q in [0,1]."""
    n = len(sorted_vals)
    if n == 0:
        raise ValueError("no scores for class")
    if n == 1:
        return float(sorted_vals[0])
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return float(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)


def _lane_precision(scores, labels, cls, t):
    """Precision of 'score_c >= t routes to cls' over the whole train set."""
    tp = 0
    fp = 0
    for row, lab in zip(scores, labels):
        crossed = row.get(cls, 0.0) >= t
        if lab == cls:
            if crossed:
                tp += 1
        elif crossed:
            fp += 1
    denom = tp + fp
    if denom == 0:
        return 1.0
    return tp / denom


def choose_thresholds(scores, labels, target_precision=0.97):
    """Per-class route threshold via quantile search on train scores.

    scores: list of {class: score} rows (e.g. softmax probabilities per
    lane) aligned with labels. Returns {class: threshold}. Deterministic:
    sorted per-class scores, ascending candidate scan, no RNG.
    """
    classes = []
    seen = set()
    for lab in labels:
        if lab not in seen:
            seen.add(lab)
            classes.append(lab)
    if not classes:
        raise ValueError("no labels")

    by_class = {c: [] for c in classes}
    for row, lab in zip(scores, labels):
        if lab in by_class:
            by_class[lab].append(float(row.get(lab, 0.0)))

    out = {}
    for cls in classes:
        own = sorted(by_class[cls])
        chosen = None
        for i in range(41):  # quantiles 0.00 .. 1.00 step 0.025, ascending
            t = _quantile(own, i / 40.0)
            if _lane_precision(scores, labels, cls, t) >= target_precision:
                chosen = t
                break
        if chosen is None:
            chosen = own[-1]
        out[cls] = chosen
    return out
