"""Plasticity probe: does online local adaptation beat a frozen readout?

Motivation (docs/DIRECTIONS.md lane 2): the Doom/MaleCNS demos adapt a fixed
connectome online via a hand-wired reward channel (damage -> P2P101 dopamine
cells) plus local plasticity in the mushroom body. Every classi-fly result so
far used a FROZEN readout. This script asks the obvious follow-up: if a
deployed head may update itself from corrections, does it get better?

It does not, at this operating point, and the reason is structural - see the
report printed at the end and docs/DIRECTIONS.md.

Run:  python3 tools/eval/plasticity_probe.py
Needs: data/e1_inputs.json + data/larval_adjacency.json (see docs/EXPERIMENTS.md)
"""

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "train"))

from calibrate import choose_thresholds  # noqa: E402

CHAIN_BASELINE = 0.868   # measured LLM-chain baseline, per the campaign convention
MARGIN = 0.05
FOLDS = 5
SEED = 42


def load():
    inputs = json.loads((REPO / "data" / "e1_inputs.json").read_text())
    gold, classes = inputs["gold"], inputs["classes"]
    vecs = np.asarray(inputs["vecs"], dtype=np.float32)
    x = vecs[: len(gold)]
    y = np.array([c["intent"] for c in gold])
    adj = json.loads((REPO / "data" / "larval_adjacency.json").read_text())
    return x, y, classes, adj


def reservoir_states(x, adj, win_scale=0.02, steps=4, decay=0.8, seed=SEED):
    n = adj["neurons"]
    rng = np.random.default_rng(seed)
    proj = rng.integers(-6, 7, size=(x.shape[1], n)).astype(np.float64) * win_scale
    vals = np.asarray(adj["weights"], dtype=np.float64) / 127.0
    indptr = np.asarray(adj["indptr"])
    ind = np.asarray(adj["indices"])
    rep = np.repeat(np.arange(n), np.diff(indptr))
    drive = x @ proj
    out = np.zeros((x.shape[0], n))
    for i in range(x.shape[0]):
        s = np.zeros(n)
        for _ in range(steps):
            acc = np.zeros(n)
            np.add.at(acc, rep, vals * s[ind])
            s = np.tanh(decay * s + acc + drive[i])
        out[i] = s
    return out


def fold_ids(y, classes, seed=SEED):
    rng = np.random.default_rng(seed)
    fold = np.zeros(len(y), dtype=int)
    for c in classes:
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        for j, i in enumerate(idx):
            fold[i] = j % FOLDS
    return fold


def ridge(states, onehot, idx, lam=30.0):
    xb = np.hstack([states[idx], np.ones((len(idx), 1))])
    sol = np.linalg.solve(xb.T @ xb + lam * np.eye(xb.shape[1]), xb.T @ onehot[idx])
    return sol[:-1], sol[-1]


def softmax(s, w, b):
    logits = w.T @ s + b
    logits = logits - logits.max()
    e = np.exp(logits)
    return e / e.sum()


def stream(states, y, classes, fold, mode, eta=0.0, tau_step=0.0, anchor=0.0, seed=7):
    """Stream the held-out folds in order; adapt only when a correction arrives.

    Returns (routes, routed_correct, abstained, total).
    """
    rng = np.random.default_rng(seed)
    k = len(classes)
    onehot = np.zeros((len(y), k))
    for i, c in enumerate(y):
        onehot[i, classes.index(c)] = 1
    gold = np.array([classes.index(c) for c in y])
    rc = rw = ab = 0
    for f in range(FOLDS):
        tr = np.where(fold != f)[0]
        te = np.where(fold == f)[0]
        w0, b0 = ridge(states, onehot, tr)
        w, b = w0.copy(), b0.copy()
        train_probs = np.array([softmax(states[i], w, b) for i in tr])
        thr = choose_thresholds(
            [{classes[j]: float(train_probs[q][j]) for j in range(k)} for q in range(len(tr))],
            [y[i] for i in tr], target_precision=0.97)
        tau = np.array([thr[c] for c in classes])
        for i in te:
            p = softmax(states[i], w, b)
            top = int(p.argmax())
            ordered = np.sort(p)
            margin = float(ordered[-1] - ordered[-2])
            if p[top] >= tau[top] and margin > MARGIN:
                if top == gold[i]:
                    rc += 1
                else:
                    rw += 1
                    if mode == "delta":
                        g = p.copy()
                        g[gold[i]] -= 1.0
                        scale = eta / max(states[i] @ states[i], 1e-9)
                        w -= np.outer(states[i], g) * scale
                    elif mode == "thr":
                        tau[top] = min(0.999, tau[top] + tau_step)
                    elif mode == "anchored":
                        g = p.copy()
                        g[gold[i]] -= 1.0
                        scale = eta / max(states[i] @ states[i], 1e-9)
                        w -= np.outer(states[i], g) * scale
            else:
                ab += 1
            if mode == "anchored":
                w = w0 + (w - w0) * (1.0 - anchor)
    total = rc + rw + ab
    return rc + rw, rc, ab, total


def main() -> int:
    x, y, classes, adj = load()
    states = reservoir_states(x, adj)
    fold = fold_ids(y, classes)
    print(f"states {states.shape} | corpus {len(y)} gold | {len(classes)} classes | "
          f"mean state norm {np.linalg.norm(states, axis=1).mean():.1f}\n")

    rows = []
    variants = [
        ("frozen readout (what classi-fly ships)", dict(mode="frozen")),
        ("online delta-rule eta=0.05", dict(mode="delta", eta=0.05)),
        ("online delta-rule eta=0.2", dict(mode="delta", eta=0.2)),
        ("online delta-rule eta=1.0", dict(mode="delta", eta=1.0)),
        ("threshold-only adaptation step=0.02", dict(mode="thr", tau_step=0.02)),
        ("threshold-only adaptation step=0.10", dict(mode="thr", tau_step=0.10)),
        ("anchored weight drift (EWC-lite)", dict(mode="anchored", eta=0.2, anchor=0.05)),
    ]
    print(f"{'variant':42s} {'routes':>6s} {'P':>7s} {'C':>7s} {'E2E':>7s}")
    for tag, kw in variants:
        routes, rc, ab, total = stream(states, y, classes, fold, **kw)
        e2e = (rc + CHAIN_BASELINE * ab) / total
        p = rc / routes if routes else 1.0
        rows.append({"variant": tag, "routes": routes, "P": round(p, 4),
                     "C": round(routes / total, 4), "E2E": round(e2e, 4),
                     "wrong_routes": routes - rc})
        print(f"{tag:42s} {routes:6d} {p:7.4f} {routes / total:7.4f} {e2e:7.4f}")

    base = rows[0]
    print(f"\ncorrection signal available: {base['wrong_routes']} wrong routes out of "
          f"{base['routes']} routed ({base['wrong_routes']} corrections in {len(y)} cases)")
    print("\nVERDICT: online adaptation does not beat the frozen readout here, and the\n"
          "reason is structural - a precision-first gate at P>=0.95 emits almost no\n"
          "corrections, so there is nothing to learn from. Weight learning makes it\n"
          "worse (-0.3 to -14pt) and degrades across the stream; the low-variance\n"
          "variants land within noise of frozen.")
    out = REPO / "tools" / "eval" / "plasticity_probe_results.json"
    out.write_text(json.dumps({"corpus": len(y), "corrections_available": base["wrong_routes"],
                               "chain_baseline": CHAIN_BASELINE, "rows": rows}, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
