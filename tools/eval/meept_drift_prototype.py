#!/usr/bin/env python3
"""Session-drift anomaly detector prototype on real meept dispatch data.

Lane 3 (docs/DIRECTIONS.md sec 8.1, docs/LANE3-AND-SIZE.md) applied to
~/.meept/metrics.db dispatch_log.

Method:
  - Per-turn feature vector: [confidence] + intent one-hot. Intent is one-hot
    encoded against the 13 e1 intent classes; DB intents outside that vocab
    (fallback/agent-side labels) get an extra "out_of_vocab" dimension, so an
    unusual intent label itself registers as novelty.
  - has_parts and margin are ALL-ZERO / ALL-NULL in the current DB (verified),
    so they are dropped rather than added as constant columns.
  - turn_no is deliberately EXCLUDED from novelty features: every multi-turn
    session has turn_no>0 by construction, so it would flag every session
    trivially. Reported separately as a confound note.
  - Reference population = per-turn vectors from single-turn sessions
    (stable by construction) + IsolationForest trained on the same.
  - Threshold calibrated at the 95th percentile of reference cosine distances.
  - Scored: all 42 multi-turn sessions, turn by turn.
  - Baseline rule: "intent changed between consecutive turns".

PRIVACY: input_summary text never leaves the DB into output files. Only
input_hash, counts, metrics, scores appear in results. No git operations.
"""

import json
import os
import sqlite3
import sys

import numpy as np
from sklearn.ensemble import IsolationForest

DB_PATH = os.path.expanduser("~/.meept/metrics.db")
E1_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "e1_inputs.json")
OUT_PATH = os.path.join(os.path.dirname(__file__), "meept_drift_prototype_results.json")


def load_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    rows = cur.execute(
        "SELECT session_id, turn_no, intent_type, confidence, margin, has_parts, "
        "input_hash, classifier_method FROM dispatch_log ORDER BY session_id, turn_no, id"
    ).fetchall()
    con.close()
    sessions = {}
    for sid, turn_no, intent, conf, margin, has_parts, h, method in rows:
        sessions.setdefault(sid, []).append(
            {"turn_no": turn_no, "intent": intent, "confidence": conf,
             "margin": margin, "has_parts": has_parts, "input_hash": h,
             "method": method}
        )
    return sessions


def main():
    with open(E1_PATH) as f:
        e1 = json.load(f)
    classes = e1["classes"]  # 13 intent classes
    class_idx = {c: i for i, c in enumerate(classes)}
    n_class_dims = len(classes) + 1  # +1 = out_of_vocab

    sessions = load_db()
    n_rows = sum(len(v) for v in sessions.values())
    single = {s: t for s, t in sessions.items() if len(t) == 1}
    multi = {s: t for s, t in sessions.items() if len(t) > 1}

    def featurize(turn):
        v = np.zeros(1 + n_class_dims)
        v[0] = turn["confidence"] if turn["confidence"] is not None else 0.0
        if turn["intent"] in class_idx:
            v[1 + class_idx[turn["intent"]]] = 1.0
        else:
            v[n_class_dims] = 1.0  # out-of-vocab intent
        return v

    # ---- reference population: single-turn sessions (stable by construction)
    ref = np.array([featurize(t[0]) for t in single.values()])
    ref_norm = ref / np.maximum(np.linalg.norm(ref, axis=1, keepdims=True), 1e-9)
    centroid = ref_norm.mean(axis=0)
    centroid /= max(np.linalg.norm(centroid), 1e-9)
    ref_cos = 1.0 - ref_norm @ centroid
    thr = float(np.percentile(ref_cos, 95))

    iso = IsolationForest(n_estimators=200, random_state=0, contamination="auto")
    iso.fit(ref)

    # ---- score multi-turn sessions
    results = []
    drift_cos = drift_iso = rule_hits = 0
    overlap_cos = overlap_iso = 0
    intent_change_sessions = 0
    for sid in sorted(multi):
        turns = multi[sid]
        vecs = np.array([featurize(t) for t in turns])
        vecs_n = vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-9)
        cos_d = 1.0 - vecs_n @ centroid
        iso_pred = iso.predict(vecs)  # -1 = outlier
        iso_flag = [p == -1 for p in iso_pred]
        rule_flag = [
            turns[i]["intent"] != turns[i - 1]["intent"]
            for i in range(1, len(turns))
        ]
        intents = [t["intent"] for t in turns]
        any_rule = any(rule_flag)
        any_cos = bool((cos_d > thr).any())
        any_iso = any(iso_flag)
        drift_cos += any_cos
        drift_iso += any_iso
        rule_hits += any_rule
        overlap_cos += any_cos and any_rule
        overlap_iso += any_iso and any_rule
        intent_change_sessions += any_rule
        results.append({
            "session_id": sid,
            "n_turns": len(turns),
            "intents": intents,  # labels only; no input text
            "cosine_dist": [round(float(d), 4) for d in cos_d],
            "cosine_flag": [bool(d > thr) for d in cos_d],
            "iso_flag": [bool(b) for b in iso_flag],
            "rule_flag_intent_change": [bool(b) for b in rule_flag],
            "any_drift_cosine": any_cos,
            "any_drift_iso": any_iso,
            "any_rule": any_rule,
        })

    n_multi = len(multi)
    n_rule_only_cos = drift_cos - overlap_cos
    n_cos_only_rule = rule_hits - overlap_cos
    precision_cos = overlap_cos / drift_cos if drift_cos else 0.0
    recall_cos = overlap_cos / rule_hits if rule_hits else 0.0

    # turn_no confound note: how many multi-turn sessions have any turn_no>0
    n_turnno_gt0 = sum(1 for t in multi.values() if any(x["turn_no"] > 0 for x in t))

    # --- sharper discrimination: session-level rarity vs within-session drift
    # (a) turn-level: does the cos flag coincide with the turn where intent changed?
    change_turns_flagged = change_turns_total = 0
    same_turns_flagged = same_turns_total = 0
    # (b) per-session novelty DELTA (max-min cos dist): within-session drift
    #     signal, independent of absolute class rarity
    deltas_change = []   # sessions where intent changed somewhere
    deltas_stable = []   # sessions with constant intent
    # (c) intent composition of flagged sessions: how much of the novelty signal
    #     is just "contains out-of-vocab intent (e.g. chat)"?
    flagged_with_oov = 0
    flagged_total = 0
    for sid in sorted(multi):
        turns = multi[sid]
        vecs = np.array([featurize(t) for t in turns])
        vecs_n = vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-9)
        cos_d = 1.0 - vecs_n @ centroid
        flags = [bool(d > thr) for d in cos_d]
        intents = [t["intent"] for t in turns]
        any_rule = any(intents[i] != intents[i - 1] for i in range(1, len(turns)))
        for i in range(1, len(turns)):
            if intents[i] != intents[i - 1]:
                change_turns_total += 1
                change_turns_flagged += flags[i]
            else:
                same_turns_total += 1
                same_turns_flagged += flags[i]
        delta = float(cos_d.max() - cos_d.min())
        (deltas_change if any_rule else deltas_stable).append(delta)
        if any(flags):
            flagged_total += 1
            if any(x not in class_idx for x in intents):
                flagged_with_oov += 1

    out = {
        "meta": {
            "db_rows": n_rows,
            "db_sessions": len(sessions),
            "single_turn_sessions": len(single),
            "multi_turn_sessions": n_multi,
            "multi_turn_sessions_ge3": sum(1 for t in multi.values() if len(t) >= 3),
            "e1_reference_classes": classes,
            "feature_dims": 1 + n_class_dims,
            "features_used": ["confidence", "intent_one_hot_vs_e1_13_plus_out_of_vocab"],
            "features_dropped": {
                "margin": "all NULL in dispatch_log (verified)",
                "has_parts": "all 0 in dispatch_log (verified)",
                "turn_no": "excluded from novelty features - constant-signal confound; "
                           f"{n_turnno_gt0}/{n_multi} multi-turn sessions have turn_no>0 by construction",
            },
            "reference_population_size": len(single),
            "cosine_threshold_p95": round(thr, 4),
            "reference_cos_distance": {
                "mean": round(float(ref_cos.mean()), 4),
                "p50": round(float(np.percentile(ref_cos, 50)), 4),
                "p95": round(float(thr), 4),
                "max": round(float(ref_cos.max()), 4),
            },
        },
        "summary": {
            "sessions_with_cosine_drift": drift_cos,
            "sessions_with_isoforest_drift": drift_iso,
            "sessions_with_intent_change_rule": rule_hits,
            "overlap_cosine_and_rule": overlap_cos,
            "overlap_isoforest_and_rule": overlap_iso,
            "cosine_precision_vs_rule": round(precision_cos, 3),
            "cosine_recall_vs_rule": round(recall_cos, 3),
            "cosine_only_sessions": n_cos_only_rule,
            "rule_only_cosine_sessions": n_rule_only_cos,
        },
        "drift_vs_rarity": {
            "note": "does the novelty signal track WITHIN-session change, or just "
                    "session-level intent rarity?",
            "flagged_change_turns": f"{change_turns_flagged}/{change_turns_total}",
            "flagged_same_intent_turns": f"{same_turns_flagged}/{same_turns_total}",
            "novelty_delta_maxmin_mean_change_sessions": round(float(np.mean(deltas_change)), 4),
            "novelty_delta_maxmin_mean_stable_sessions": round(float(np.mean(deltas_stable)), 4),
            "novelty_delta_maxmin_median_change": round(float(np.median(deltas_change)), 4),
            "novelty_delta_maxmin_median_stable": round(float(np.median(deltas_stable)), 4),
            "flagged_sessions_containing_oov_intent": f"{flagged_with_oov}/{flagged_total}",
        },
        "sessions": results,
    }

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)

    s = out["summary"]
    print(f"DB: {n_rows} rows / {len(sessions)} sessions "
          f"({len(single)} single-turn, {n_multi} multi-turn)")
    print(f"threshold(cos p95 over reference): {thr:.4f}")
    print(f"cosine drift: {s['sessions_with_cosine_drift']}/{n_multi} sessions")
    print(f"isoforest drift: {s['sessions_with_isoforest_drift']}/{n_multi} sessions")
    print(f"intent-change rule: {s['sessions_with_intent_change_rule']}/{n_multi} sessions")
    print(f"overlap cosine&rule: {overlap_cos}, cosine-only: {n_cos_only_rule}, "
          f"rule-only: {n_rule_only_cos}")
    print(f"precision(cos vs rule)={precision_cos:.2f} recall={recall_cos:.2f}")
    # sample of cosine-only sessions for inspection
    for r in results:
        if r["any_drift_cosine"] and not r["any_rule"]:
            print("cosine-only:", r["session_id"], r["intents"],
                  r["cosine_dist"], r["cosine_flag"])
    dvr = out["drift_vs_rarity"]
    print(f"change turns flagged: {dvr['flagged_change_turns']} | "
          f"same-intent turns flagged: {dvr['flagged_same_intent_turns']}")
    print(f"novelty delta (max-min): change={dvr['novelty_delta_maxmin_mean_change_sessions']} "
          f"stable={dvr['novelty_delta_maxmin_mean_stable_sessions']} (means)")
    print(f"flagged sessions containing out-of-vocab intent: "
          f"{dvr['flagged_sessions_containing_oov_intent']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
