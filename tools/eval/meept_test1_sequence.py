"""meept test 1 (mapping 8.1 in docs/DIRECTIONS.md): can session/turn CONTEXT
improve classification of the state-dependent intents (quickplan vs code),
where per-message classification plateaus?

WHY THIS EXISTS
docs/DIRECTIONS.md 8.1 records that "implement tasks 7 and 8" is code text but
may mean "execute the approved plan", and the discriminator is whether a plan is
active. Per-message accuracy on that class plateaus near 84%. The cheap
prerequisite test called for there: concatenate the last three turns' features
and fit a plain logistic regression; if that beats the per-message ceiling,
session context is usable.

WHAT THIS SCRIPT ACTUALLY MEASURES (and its honest limitation)
The adjudicated gold corpus (data/e1_inputs.json, 361 cases) has NO session or
turn linkage: every case is an independent message. The live dispatch log
(~/.meept/metrics.db) DOES have session_id/turn_no but no gold intent labels and
no populated outcome/corrected_agent. So the literal 3-turn-concatenation test
is not runnable on labelled data. What IS runnable is the deployment-honest
reduced test:

  (a) per-message: ridge on the frozen 1024-dim embedding
  (b) same + context features available WITHOUT message text:
        - quickplan orchestration-cue match (the regex in
          meept/internal/agent/quickplan_cue.go)
        - message length (log1p chars)
        - plan-active indicator (unavailable for the gold corpus: no session
          linkage; see the audit block in the results JSON - recorded as a
          constant-zero column so its absence is explicit, not hidden)

Also reported: a cue-only rule and a length-only rule, as the signal strength of
each context feature alone.

Protocol: 5-fold stratified CV, seed 42, shuffle each class's indices with
np.random.default_rng(42), fold = position % 5 (identical to
tools/eval/axis_readout_heads.py). Ridge penalty picked per outer fold by a
deterministic inner 4-fold CV on the train side only. No text is ever written
out; only numbers, counts and feature names.

Run:
  python3 tools/eval/meept_test1_sequence.py \
      --out tools/eval/meept_test1_sequence_results.json
"""

import argparse
import json
import math
import os
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
GOLD = REPO / "data" / "e1_inputs.json"
SEED = 42
FOLDS = 5
LAMBDA_GRID = (1e-3, 1e-2, 1e-1, 1.0, 3.0, 10.0, 30.0, 100.0)
TARGET_CLASSES = ("quickplan", "code")

# The meept orchestration-cue regex, transcribed verbatim from
# meept/internal/agent/quickplan_cue.go (QuickPlanCuePattern). Go's (?i) and \b
# translate directly; the alternation order is preserved.
CUE_PATTERN = re.compile(
    r"(?i)\b(subagents?|tasks? \d|"
    r"task list|waves?|leaves?|leaf \d|plan\.md|handoff|checklist|in order|"
    r"one at a time|sealed plan|tracking table|as you (find|go)|, then\b|"
    r"and correct them|and fix them|without (asking|stopping)|no check-?ins?|"
    r"just (do|make|apply)|make it happen|to completion|finish the remaining|"
    r"carry on with the plan|execute (the|what)|implement (the|all) plan|"
    r"implement tasks?|work (through|items)|knock out|carry out|"
    r"complete the outstanding)\b"
)


# ------------------------------------------------------------------ folds
def strat_folds(y, classes, folds=FOLDS, seed=SEED):
    n = len(y)
    fold = np.zeros(n, dtype=int)
    rng = np.random.default_rng(seed)
    arr = np.asarray(y)
    for cls in classes:
        idx = np.where(arr == cls)[0]
        rng.shuffle(idx)
        for j, i in enumerate(idx):
            fold[i] = j % folds
    return fold


# ------------------------------------------------------------------ ridge
def ridge_fit_predict(Xtr, Ytr, Xte, lam):
    d = Xtr.shape[1]
    A = Xtr.T @ Xtr + lam * np.eye(d)
    W = np.linalg.solve(A, Xtr.T @ Ytr)
    return Xte @ W


def one_hot(y, classes):
    idx = {c: i for i, c in enumerate(classes)}
    Y = np.zeros((len(y), len(classes)))
    for i, c in enumerate(y):
        Y[i, idx[c]] = 1.0
    return Y


def standardize(Xtr, Xte, post_scale=None):
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0)
    sd[sd < 1e-9] = 1.0
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    if post_scale is not None:
        Xtr = Xtr * post_scale
        Xte = Xte * post_scale
    return Xtr, Xte


def inner_lambda(X, y, classes, seed=SEED, post_scale=None):
    """Deterministic inner 4-fold CV on the train side; smallest tie wins."""
    inner = strat_folds(y, classes, folds=4, seed=seed)
    y = np.asarray(y)
    best_lam, best_err = LAMBDA_GRID[0], float("inf")
    for lam in LAMBDA_GRID:
        err, tot = 0, 0
        for f in range(4):
            tr = inner != f
            te = inner == f
            Xtr, Xte = standardize(X[tr], X[te], post_scale)
            Ytr = one_hot(y[tr], classes)
            pred = ridge_fit_predict(Xtr, Ytr, Xte, lam).argmax(axis=1)
            truth = np.array([classes.index(c) for c in y[te]])
            err += int((pred != truth).sum())
            tot += int(te.sum())
        acc = 1.0 - err / max(tot, 1)
        if acc < best_err - 1e-12:
            best_err = err / max(tot, 1)
            best_lam = lam
    return best_lam


def cv_predict(X, y, classes, fold, post_scale=None):
    oof = np.empty(len(y), dtype=object)
    lams = []
    for f in range(FOLDS):
        tr = fold != f
        te = fold == f
        lam = inner_lambda(X[tr], list(np.asarray(y)[tr]), classes,
                           post_scale=post_scale)
        lams.append(lam)
        Xtr, Xte = standardize(X[tr], X[te], post_scale)
        Ytr = one_hot(list(np.asarray(y)[tr]), classes)
        pred = ridge_fit_predict(Xtr, Ytr, Xte, lam).argmax(axis=1)
        for k, i in enumerate(np.where(te)[0]):
            oof[i] = classes[pred[k]]
    return list(oof), lams


def prf(y_true, y_pred, cls):
    tp = sum(1 for a, b in zip(y_true, y_pred) if a == cls and b == cls)
    fp = sum(1 for a, b in zip(y_true, y_pred) if a != cls and b == cls)
    fn = sum(1 for a, b in zip(y_true, y_pred) if a == cls and b != cls)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r,
            "support": sum(1 for a in y_true if a == cls)}


def acc(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


# ------------------------------------------------------------------ audit
def audit():
    out = {}
    out["audited_at_utc"] = __import__("datetime").datetime.utcnow().isoformat(timespec="seconds") + "Z"
    m = os.path.expanduser("~/.meept/metrics.db")
    if not os.path.exists(m):
        return {"error": "metrics.db missing"}
    c = sqlite3.connect(m)
    cur = c.cursor()
    out["dispatch_rows"] = cur.execute("select count(*) from dispatch_log").fetchone()[0]
    out["dispatch_sessions"] = cur.execute(
        "select count(distinct session_id) from dispatch_log").fetchone()[0]
    sizes = Counter(r[0] for r in cur.execute(
        "select count(*) n from dispatch_log group by session_id"))
    out["session_size_histogram"] = {str(k): sizes[k] for k in sorted(sizes)}
    out["sessions_gt1"] = sum(v for k, v in sizes.items() if k > 1)
    out["sessions_ge3"] = sum(v for k, v in sizes.items() if k >= 3)
    out["outcome_counts"] = dict(cur.execute(
        "select outcome,count(*) from dispatch_log group by outcome"))
    out["corrected_agent_nonempty"] = cur.execute(
        "select count(*) from dispatch_log where corrected_agent is not null "
        "and corrected_agent!=''").fetchone()[0]
    out["rows_with_error"] = cur.execute(
        "select count(*) from dispatch_log where error is not null and error!=''"
    ).fetchone()[0]
    out["intent_type_counts"] = dict(cur.execute(
        "select intent_type,count(*) from dispatch_log group by intent_type"))
    # Can plan-active state be resolved per turn? plans.db records plans with a
    # source_session and a state; check whether that join is usable.
    p = os.path.expanduser("~/.meept/plans.db")
    out["plans_db"] = {}
    if os.path.exists(p):
        cp = sqlite3.connect(p)
        out["plans_db"]["plan_states"] = dict(cp.execute(
            "select state,count(*) from plans group by state"))
        out["plans_db"]["plans_with_source_session"] = cp.execute(
            "select count(*) from plans where source_session is not null "
            "and source_session!=''").fetchone()[0]
        out["plans_db"]["plans_confirmed"] = cp.execute(
            "select count(*) from plans where confirmed_at is not null "
            "and confirmed_at!=''").fetchone()[0]
        out["plans_db"]["plan_sessions_rows"] = cp.execute(
            "select count(*) from plan_sessions").fetchone()[0]
        out["plans_db"]["plan_signoffs_rows"] = cp.execute(
            "select count(*) from plan_signoffs").fetchone()[0]
        disp_sessions = set(r[0] for r in cur.execute(
            "select distinct session_id from dispatch_log"))
        plan_src = set(r[0] for r in cp.execute(
            "select distinct source_session from plans where source_session "
            "is not null and source_session!=''"))
        out["plans_db"]["plan_source_sessions_matching_dispatch_sessions"] = len(
            plan_src.intersection(disp_sessions))
    # Gold corpus: any session/turn linkage at all?
    g = json.loads(GOLD.read_text())
    gold = g["gold"]
    out["gold_cases"] = len(gold)
    out["gold_has_session_field"] = any(
        any(k in x for k in ("session_id", "turn_no", "turn", "session"))
        for x in gold)
    return out


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "tools/eval/meept_test1_sequence_results.json"))
    args = ap.parse_args()

    data = json.loads(GOLD.read_text())
    gold, vecs = data["gold"], np.asarray(data["vecs"], dtype=np.float64)
    assert len(vecs) >= len(gold), "vecs shorter than gold"

    cases = [(i, x["text"], x["intent"]) for i, x in enumerate(gold)
             if x["intent"] in TARGET_CLASSES]
    idx = np.array([i for i, _, _ in cases])
    texts = [t for _, t, _ in cases]
    y = [lbl for _, _, lbl in cases]
    classes = list(TARGET_CLASSES)
    classes.sort()

    Xe = vecs[idx]                                    # 1024-dim frozen embedding
    cue = np.array([1.0 if CUE_PATTERN.search(t) else 0.0 for t in texts])
    ln = np.array([math.log1p(len(t)) for t in texts])
    # Plan-active indicator: NOT derivable for this corpus (no session linkage).
    # Kept as an explicit zero column so the test reports the features it has.
    plan_active = np.zeros(len(texts))

    Xctx = np.column_stack([cue, ln, plan_active])
    Xa = Xe
    Xb = np.column_stack([Xe, Xctx])
    # Block-matched variant: under standard scaling each column has unit
    # variance, so 3 context columns carry ~3/1024 of the model's variance and
    # are drowned by the embedding block. Re-weight the standardized context
    # columns so the block's total variance matches the embedding block's - the
    # fair shot at finding any usable context signal at all. The scale is
    # applied AFTER standardization (a raw pre-scale is cancelled by it).
    ctx_scale = math.sqrt(Xe.shape[1] / Xctx.shape[1])
    Xb_matched = np.column_stack([Xe, Xctx])
    matched_scale = np.concatenate(
        [np.ones(Xe.shape[1]), np.full(Xctx.shape[1], ctx_scale)])

    fold = strat_folds(y, classes)

    res = {"audit": audit()}
    res["reduced_test"] = {
        "target_classes": classes,
        "subset_size": len(texts),
        "subset_counts": dict(Counter(y)),
        "cue_positive_counts": {
            c: int(cue[np.asarray(y) == c].sum()) for c in classes},
        "cue_positive_rate": {
            c: float(cue[np.asarray(y) == c].mean()) for c in classes},
        "context_features": ["quickplan_cue_match", "log1p_msg_len",
                             "plan_active(unavailable=0)"],
        "context_block_scale_for_matched_variant": "sqrt(1024/3)",
        "cv": "5-fold stratified, seed 42, shuffle per class, fold = position % 5",
        "ridge_lambda_grid": list(LAMBDA_GRID),
    }

    preds = {}
    variants = (
        ("per_message_embedding", Xa, None),
        ("embedding_plus_context", Xb, None),
        ("embedding_plus_context_block_matched", Xb_matched, matched_scale),
    )
    for name, X, post_scale in variants:
        pred, lams = cv_predict(X, y, classes, fold, post_scale=post_scale)
        pred = [str(p) for p in pred]
        preds[name] = pred
        res["reduced_test"][name] = {
            "accuracy": acc(y, pred),
            "n_correct": int(sum(1 for a, b in zip(y, pred) if a == b)),
            "n_total": len(y),
            "per_class": {c: prf(y, pred, c) for c in classes},
            "lambdas_per_fold": lams,
        }

    # Signal strength of each context feature alone.
    cue_rule = ["quickplan" if c > 0 else "code" for c in cue]
    len_rule = ["quickplan" if l > np.median(ln) else "code" for l in ln]
    res["context_feature_alone"] = {
        "cue_only_rule": {
            "accuracy": acc(y, cue_rule),
            "n_correct": int(sum(1 for a, b in zip(y, cue_rule) if a == b)),
            "per_class": {c: prf(y, cue_rule, c) for c in classes},
        },
        "length_only_rule": {
            "accuracy": acc(y, len_rule),
            "n_correct": int(sum(1 for a, b in zip(y, len_rule) if a == b)),
            "per_class": {c: prf(y, len_rule, c) for c in classes},
        },
        "majority_rule": max(Counter(y).values()) / len(y),
    }

    a = res["reduced_test"]["per_message_embedding"]
    b = res["reduced_test"]["embedding_plus_context"]
    bm = res["reduced_test"]["embedding_plus_context_block_matched"]

    # Does the cue carry signal the embedding lacks? Count what the cue-only
    # rule fixes and breaks relative to the per-message model, out-of-fold.
    pm = preds["per_message_embedding"]
    fixed = sum(1 for t, p, q in zip(y, pm, cue_rule) if q == t and p != t)
    broke = sum(1 for t, p, q in zip(y, pm, cue_rule) if q != t and p == t)
    res["cue_vs_embedding_signal"] = {
        "cue_fixes_embedding_errors": fixed,
        "cue_breaks_embedding_correct": broke,
        "net": fixed - broke,
    }

    res["verdict"] = {
        "delta_correct_cases": b["n_correct"] - a["n_correct"],
        "delta_accuracy": b["accuracy"] - a["accuracy"],
        "delta_correct_cases_block_matched": bm["n_correct"] - a["n_correct"],
        "delta_accuracy_block_matched": bm["accuracy"] - a["accuracy"],
        "context_beats_per_message": bm["n_correct"] > a["n_correct"],
        "session_sequence_model_trainable_today": False,
        "session_context_measurable_today": False,
        "data_needed": (
            "labelled multi-turn transcripts: dispatch_sessions with >=3 turns "
            "is 20 (>=2 is 42) and NONE carry a gold intent label "
            "(outcome is 'pending' for all but a handful, corrected_agent is "
            "empty in every row, no table records plan-active per turn). "
            "A sequence model that folds 3 turns needs at least a few hundred "
            "labelled multi-turn sessions in the quickplan/code pair to detect "
            "an effect; today that number is zero."
        ),
    }

    Path(args.out).write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
