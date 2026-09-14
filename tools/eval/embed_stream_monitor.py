"""Embedding-stream health check: does the reservoir's TEMPORAL INTEGRATION
make it a better DISTRIBUTION-SHIFT detector than per-item embedding novelty?

Context. docs/DIRECTIONS.md 9.3 proposed the reservoir as a meept embedding
CLASSIFIER; that was retracted (per-item novelty AUC 0.586, frontier_ood).
The lane-3 control result (152/200 flagged, 5-step latency, 0 false/100 -
docs/LANE3-AND-SIZE.md) suggests the reservoir is instead good as an ANOMALY
DETECTOR: its internal state integrates the recent input history, so a
distribution shift should move the state distribution, not just individual
items. This script tests that on meept's real embedding stream.

Design.
  1. Standard reservoir recipe (tools/eval/meept_test3_degraded.py):
     larval connectome CSR, s = tanh(decay*s + W@s + win@x), steps=4,
     decay=0.8, win = rng(42).integers(-6,7,(1024,2952)) * 0.02,
     weight_scale = 1/127. The state is NOT reset between items.
  2. Feed a stream: gold warmup -> injected OOD segment -> return to gold.
  3. Score at each step:
     - STATE-NOVELTY: cosine distance between the current state and the
       rolling window of the last W states (W in {10, 25, 50}).
     - EMBED-NOVELTY (the AUC-0.586 baseline): per-item cosine distance to
       the nearest item in the same rolling window (streaming version) and
       to the fixed gold calibration set (the offline baseline).
  4. Thresholds: per detector, the 99th percentile of scores on a
     gold-only stream (no shift) - so the nominal false-alarm budget is 1%.
  5. Metrics over trials: detection latency (OOD items until first crossing),
     false alarms / 100 gold steps on the post-return gold segment.

Run:  python3 tools/eval/embed_stream_monitor.py
Out:  tools/eval/embed_stream_monitor_results.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

REPO = Path(__file__).resolve().parents[2]
ADJ = REPO / "data" / "larval_adjacency.json"
GOLD = REPO / "data" / "e1_inputs.json"
SYN = REPO / "data" / "ood_synthetic_embedded.json"
OUT = REPO / "tools" / "eval" / "embed_stream_monitor_results.json"

SEED = 42
STEPS = 4
DECAY = 0.8
IN_SCALE = 0.02
WEIGHT_SCALE = 1.0 / 127.0
WINDOWS = (10, 25, 50)
WARMUP = 100          # gold items before the shift
SHIFT_LEN = 30        # OOD items injected (latency measured within these)
RETURN_LEN = 150      # gold items after the shift (false alarms measured here)
TRIALS = 12
THR_PCTL = 0.99       # gold-only score percentile -> threshold


def norm(m):
    return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12)


class StreamReservoir:
    """Standard recipe; state persists across items (that is the point)."""

    def __init__(self, adj):
        n = int(adj["neurons"])
        indptr = np.asarray(adj["indptr"], dtype=np.int64)
        self.n = n
        self.vals = np.asarray(adj["weights"], dtype=np.float64) * WEIGHT_SCALE
        self.W = sp.csr_matrix(
            (self.vals, np.asarray(adj["indices"], dtype=np.int64), indptr),
            shape=(n, n))
        proj = np.random.default_rng(SEED).integers(-6, 7, size=(1024, n)).astype(np.int8)
        self.win = proj.astype(np.float64) * IN_SCALE
        self.s = np.zeros(n)

    def reset(self):
        self.s = np.zeros(self.n)

    def step(self, x):
        drive = x @ self.win
        for _ in range(STEPS):
            self.s = np.tanh(DECAY * self.s + (self.W @ self.s) + drive)
        return self.s


def rolling_cos_scores(current, window):
    """Mean cosine distance from `current` to each row of `window`."""
    if len(window) == 0:
        return None
    wn = norm(window)
    c = current / max(np.linalg.norm(current), 1e-12)
    return float(np.mean(1.0 - wn @ c))


def make_stream(rng, gold, ood):
    """Indices: warmup gold, shift OOD, return gold. Returns (vec_list, seg_list)."""
    g = rng.choice(len(gold), size=WARMUP + RETURN_LEN, replace=True)
    o = rng.choice(len(ood), size=SHIFT_LEN, replace=True)
    vecs = np.vstack([gold[g[:WARMUP]], ood[o], gold[g[WARMUP:]]])
    segs = (["warmup"] * WARMUP + ["shift"] * SHIFT_LEN + ["return"] * RETURN_LEN)
    return vecs, segs


def run_stream(res, vecs, state_keep, embed_keep, gold_cal_norm):
    """Yield per-step scores for the four detectors."""
    state_win, embed_win = [], []
    scores = {"state": [], "embed_win": [], "embed_cal": []}
    for x in vecs:
        s = res.step(x)
        scores["state"].append(rolling_cos_scores(s, state_win))
        xn = x / max(np.linalg.norm(x), 1e-12)
        scores["embed_win"].append(rolling_cos_scores(xn, embed_win))
        scores["embed_cal"].append(float((1.0 - gold_cal_norm @ xn).min()))
        # keep-after-score so the current item is never in its own window
        if state_keep is None or len(state_win) < state_keep:
            state_win.append(s)
        else:
            state_win.pop(0); state_win.append(s)
        if embed_keep is None or len(embed_win) < embed_keep:
            embed_win.append(xn)
        else:
            embed_win.pop(0); embed_win.append(xn)
    return {k: np.array([np.inf if v is None else v for v in vals])
            for k, vals in scores.items()}


def main() -> int:
    d = json.loads(GOLD.read_text())
    gold_all = norm(np.asarray(d["vecs"], dtype=np.float64)[:len(d["gold"])])
    ood_gold = norm(np.asarray(d["vecs"], dtype=np.float64)[len(d["gold"]):])
    syn = norm(np.asarray(json.loads(SYN.read_text())["vecs"], dtype=np.float64))
    ood = np.vstack([ood_gold, syn])
    adj = json.loads(ADJ.read_text())
    res = StreamReservoir(adj)
    rng = np.random.default_rng(SEED)

    # Fixed gold calibration set for the offline embed-novelty baseline:
    # disjoint from the streams' sampled gold items is impossible with
    # replacement sampling, so we use the full gold set minus a held-out
    # tenth, matching the leave-some-out protocol of frontier_ood.novelty.
    cal_idx = rng.choice(len(gold_all), size=len(gold_all) // 10 * 9, replace=False)
    gold_cal = norm(gold_all[cal_idx])

    # ---- pass 1: gold-only streams to set thresholds (nominal 1% FA budget)
    thr = {w: {} for w in WINDOWS}
    gold_scores = {w: {"state": [], "embed_win": [], "embed_cal": []} for w in WINDOWS}
    for t in range(TRIALS):
        g = rng.choice(len(gold_all), size=WARMUP + SHIFT_LEN + RETURN_LEN, replace=True)
        for w in WINDOWS:
            res.reset()
            sc = run_stream(res, gold_all[g], w, w, gold_cal)
            for k in sc:
                gold_scores[w][k].extend(sc[k][w:].tolist())  # burn-in = window
    for w in WINDOWS:
        for k in gold_scores[w]:
            thr[w][k] = float(np.quantile(gold_scores[w][k], THR_PCTL))

    # ---- pass 2: shift streams
    results = {w: {k: {"latencies": [], "fa_per_100": []}
                   for k in ("state", "embed_win", "embed_cal")}
               for w in WINDOWS}
    for t in range(TRIALS):
        vecs, segs = make_stream(rng, gold_all, ood)
        segs = np.array(segs)
        for w in WINDOWS:
            res.reset()
            sc = run_stream(res, vecs, w, w, gold_cal)
            shift_mask = segs == "shift"
            return_mask = segs == "return"
            for k in sc:
                # detection latency: first crossing inside the shift segment
                cross = np.where(shift_mask & (sc[k] > thr[w][k]))[0]
                lat = int(cross[0] - WARMUP + 1) if len(cross) else None
                # false alarms on the post-return gold segment (post burn-in)
                ret_idx = np.where(return_mask)[0][w:]
                fa = float((sc[k][ret_idx] > thr[w][k]).mean() * 100.0)
                results[w][k]["latencies"].append(lat)
                results[w][k]["fa_per_100"].append(fa)

    def summarize(det):
        lat = [l for l in det["latencies"] if l is not None]
        return {
            "detected_trials": f"{len(lat)}/{len(det['latencies'])}",
            "median_latency": (float(np.median(lat)) if lat else None),
            "worst_latency": (int(np.max(lat)) if lat else None),
            "fa_per_100_gold_mean": round(float(np.mean(det["fa_per_100"])), 2),
        }

    summary = {
        str(w): {k: summarize(results[w][k]) for k in results[w]}
        for w in WINDOWS
    }

    # ---- verdict: state-novelty vs the best embed baseline, per window and overall
    def rank(w):
        s = summary[str(w)]
        def key(k):
            d_ = s[k]
            detected = int(d_["detected_trials"].split("/")[0])
            lat = d_["median_latency"] if d_["median_latency"] is not None else 999
            return (detected, -d_["fa_per_100_gold_mean"], -lat)
        return key

    best_state_w = max(WINDOWS, key=lambda w: rank(w)("state"))
    best_embed_k = max(("embed_win", "embed_cal"),
                       key=lambda k: rank(best_state_w)(k))
    st, em = summary[str(best_state_w)]["state"], summary[str(best_state_w)][best_embed_k]
    s_lat = st["median_latency"] if st["median_latency"] is not None else 999
    e_lat = em["median_latency"] if em["median_latency"] is not None else 999
    if (st["detected_trials"] > em["detected_trials"]
            or (st["detected_trials"] == em["detected_trials"]
                and s_lat < e_lat - 1)):
        verdict = "STATE-NOVELTY BETTER"
    elif abs(s_lat - e_lat) <= 1 and st["detected_trials"] == em["detected_trials"]:
        verdict = "PARITY"
    else:
        verdict = "STATE-NOVELTY WORSE"

    out = {
        "protocol": {
            "warmup_gold": WARMUP, "shift_ood": SHIFT_LEN, "return_gold": RETURN_LEN,
            "trials": TRIALS, "threshold_pctile": THR_PCTL, "seed": SEED,
            "reservoir": "larval 2952, steps=4, decay=0.8, win rng(42)*0.02, ws=1/127",
            "state_not_reset_between_items": True,
            "pool_sizes": {"gold": len(gold_all), "ood_gold": len(ood_gold),
                           "ood_synthetic": len(syn)},
        },
        "thresholds": {str(w): thr[w] for w in WINDOWS},
        "results": summary,
        "verdict": {
            "best_state_window": best_state_w,
            "best_embed_baseline": best_embed_k,
            "verdict": verdict,
        },
    }
    OUT.write_text(json.dumps(out, indent=2))

    print(f"streams: {TRIALS} trials x ({WARMUP} gold -> {SHIFT_LEN} OOD -> {RETURN_LEN} gold)")
    print(f"OOD pool: {len(ood)} (28 gold-OOD + {len(syn)} synthetic)\n")
    print(f"{'window':>6s} | {'detector':>10s} | detected | median lat | worst lat | FA/100 gold")
    for w in WINDOWS:
        for k, v in summary[str(w)].items():
            print(f"{w:>6d} | {k:>10s} | {v['detected_trials']:>8s} | "
                  f"{str(v['median_latency']):>10s} | {str(v['worst_latency']):>9s} | "
                  f"{v['fa_per_100_gold_mean']:>10.2f}")
    print(f"\nbest state window: {best_state_w}; best embed baseline: {best_embed_k}")
    print(f"VERDICT: {verdict}")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
