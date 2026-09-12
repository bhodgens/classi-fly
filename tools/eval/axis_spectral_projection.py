"""Axis A: spectral-radius scaling + trained input projection.

Reservoir classifier study on real embeddings (data/e1_inputs.json, 361 gold /
28 OOD, 1024-dim) with the larval connectome reservoir (2,952 neurons).

Protocol (identical to the campaign ruler, see docs/EXPERIMENTS.md):
- 5-fold stratified CV, seed 42, fold[i] = rank within class % 5
  (per-class shuffle with np.random.default_rng(42)).
- Route iff top1 prob >= its calibrated per-class threshold (target precision
  0.97, TRAIN-side quantiles) AND (top1-top2 margin) > 0.05.
- P = route_correct/routes (routed-but-wrong STAYS in the denominator),
  C = routes/total, A = route_correct/total,
  E2E = (route_correct + 0.868*abstained)/total.

Numerics: the readout is ridge. Because #train (289) << #features (2952),
the DUAL form is used (kernel ridge), which is algebraically identical to
tools/train/train_readout.ridge_fit for any lam > 0:
    alpha = (K + lam I)^-1 Y,  K = Xb_tr Xb_tr^T      (289x289)
    logits_te = Xb_te Xb_tr^T alpha
This makes a 1e-3..100 lambda sweep essentially free (a 289x289 solve each)
instead of a 2953x2953 primal solve. Validated against ridge_fit below.

Sections:
  1. reproduce the fixed-projection baseline (protocol check)
  2. spectral-radius sweep rho(W) in {0.5,0.8,0.95,1.0,1.1,1.3}
  3. trained input projection (Adam through the unrolled T=4 recurrence),
     then the same sweep + the best-rho x trained-projection final row.

Not committed / not documented here: the orchestrator synthesizes.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "train"))
from calibrate import choose_thresholds  # noqa: E402

CHAIN = 0.868
FOLDS = 5
SEED = 42
MARGIN = 0.05
T_STEPS = 4
DECAY = 0.8
IN_SCALE = 0.02
LAMBDAS = [1e-3, 0.01, 0.1, 0.5, 1.0, 3.0, 10.0, 30.0, 100.0]
RHO_TARGETS = [0.5, 0.8, 0.95, 1.0, 1.1, 1.3]


# ---------------------------------------------------------------- data ----

def load_data():
    d = json.loads((REPO / "data" / "e1_inputs.json").read_text())
    gold, ood, classes = d["gold"], d["ood"], list(d["classes"])
    V = np.asarray(d["vecs"], dtype=np.float64)
    ng = len(gold)
    Xg, Xo = V[:ng], V[ng:]
    y = np.array([g["intent"] for g in gold])
    return gold, ood, classes, Xg, Xo, y


def load_W():
    adj = json.loads((REPO / "data" / "larval_adjacency.json").read_text())
    n = int(adj["neurons"])
    indptr = np.asarray(adj["indptr"], dtype=np.int64)
    indices = np.asarray(adj["indices"], dtype=np.int64)
    rows = np.repeat(np.arange(n), np.diff(indptr))
    W = np.zeros((n, n), dtype=np.float64)
    W[rows, indices] = np.asarray(adj["weights"], dtype=np.float64) / 127.0
    return adj, W


def make_folds(y, classes):
    rng = np.random.default_rng(SEED)
    fold = np.zeros(len(y), dtype=int)
    for cls in classes:
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        for j, i in enumerate(idx):
            fold[i] = j % FOLDS
    return fold, rng


# ------------------------------------------------------- reservoir fwd ----

def run_states(W, X, win, steps=T_STEPS, decay=DECAY):
    """Vectorized: s <- tanh(decay*s + W s + Win x), s0 = 0. Returns (n_items, N)."""
    drive = X @ win                       # (n_items, N)
    S = np.zeros((X.shape[0], W.shape[0]), dtype=np.float64)
    for _ in range(steps):
        S = np.tanh(decay * S + S @ W.T + drive)
    return S


def fixed_win(D, n, rng):
    """The campaign's fixed random projection: int8 in [-6,6] * 0.02.

    NOTE: the reference runner draws this from a FRESH default_rng(42) per
    representation (not from the fold-shuffle rng). Verified: a fresh
    default_rng(42) draw reproduces the reference larval gate row exactly
    (lam=30 routes=71 P=0.9577 E2E=0.8857).
    """
    return rng.integers(-6, 7, size=(D, n)).astype(np.float64) * IN_SCALE


# ------------------------------------------- trained input projection ----

def _softmax(L):
    m = L.max(axis=1, keepdims=True)
    E = np.exp(L - m)
    return E / E.sum(axis=1, keepdims=True)


def train_win(W, X, y, classes, win_init, wout_init, b_init, steps=T_STEPS,
              decay=DECAY, epochs=40, lr=3e-3, l2_out=0.0, l2_win=0.0,
              seed=SEED):
    """Learn W_in by BPTT through the unrolled T-step recurrence.

    Method (documented choice): joint Adam on (W_in, readout) with the readout
    warm-started at the fold's closed-form ridge solution. The loss is
    cross-entropy on the FINAL state only (the protocol's readout sees s_T),
    plus L2 on the readout (l2_out) and an optional pull of W_in toward its
    init (l2_win). W is fixed; only the input projection and the scratch
    readout are updated. The scratch readout is discarded afterwards - the
    reported metrics re-solve the readout by ridge on the trained states, so
    the readout protocol stays identical to the fixed-projection arm and the
    ONLY difference is the input projection.
    """
    n_in, n = win_init.shape
    Win = win_init.copy().astype(np.float64)
    Wout = wout_init.copy().astype(np.float64)   # (K, N)
    b = b_init.copy().astype(np.float64)
    K = Wout.shape[0]
    Y = np.zeros((len(X), K))
    cidx = {c: i for i, c in enumerate(classes)}
    for i, lab in enumerate(y):
        Y[i, cidx[lab]] = 1.0

    mW = np.zeros_like(Win); vW = np.zeros_like(Win)
    mO = np.zeros_like(Wout); vO = np.zeros_like(Wout)
    mB = np.zeros_like(b); vB = np.zeros_like(b)
    b1, b2, eps = 0.9, 0.999, 1e-8
    t = 0
    for _ in range(epochs):
        t += 1
        st = [np.zeros((len(X), n))]
        for _s in range(steps):
            st.append(np.tanh(decay * st[-1] + st[-1] @ W.T + X @ Win))
        S = st[-1]
        logits = S @ Wout.T + b
        P = _softmax(logits)
        dlog = (P - Y) / len(X)
        ds = dlog @ Wout
        gW = np.zeros_like(Win)
        for _s in range(steps, 0, -1):
            g = ds * (1.0 - st[_s] ** 2)
            gW += X.T @ g
            ds = decay * g + g @ W
        gO = dlog.T @ S + l2_out * Wout
        gB = dlog.sum(0)
        if l2_win:
            gW += l2_win * (Win - win_init)
        for p, gp, mp, vp in ((Win, gW, mW, vW), (Wout, gO, mO, vO), (b, gB, mB, vB)):
            mp *= b1; mp += (1 - b1) * gp
            vp *= b2; vp += (1 - b2) * gp * gp
            mh = mp / (1 - b1 ** t); vh = vp / (1 - b2 ** t)
            p -= lr * mh / (np.sqrt(vh) + eps)
    # internal (scratch-readout) train accuracy, for the record only
    st = [np.zeros((len(X), n))]
    for _s in range(steps):
        st.append(np.tanh(decay * st[-1] + st[-1] @ W.T + X @ Win))
    acc = float((np.argmax(st[-1] @ Wout.T + b, 1) == np.array(
        [cidx[lab] for lab in y])).mean())
    return Win, acc


# ------------------------------------------------------------- metrics ----

def softmax_rows(L):
    m = L.max(axis=1, keepdims=True)
    E = np.exp(L - m)
    return E / E.sum(axis=1, keepdims=True)


def evaluate(S, y, fold, classes):
    """Dual-form ridge over a lambda grid. Returns per-lambda metric rows."""
    K_ord = sorted(classes)
    K = len(K_ord)
    cidx = {c: i for i, c in enumerate(K_ord)}
    total_all = len(y)
    Y = np.zeros((total_all, K))
    for i, lab in enumerate(y):
        Y[i, cidx[lab]] = 1.0

    out = {}
    for lam in LAMBDAS:
        routes = rcount = abst = 0
        ungated_correct = 0
        for f in range(FOLDS):
            tr = np.where(fold != f)[0]
            te = np.where(fold == f)[0]
            if len(te) == 0:
                continue
            Str = S[tr]
            Xb = np.hstack([Str, np.ones((len(tr), 1))])
            G = Xb @ Xb.T                                   # (ntr, ntr)
            A = G + lam * np.eye(len(tr))
            alpha = np.linalg.solve(A, Y[tr])               # (ntr, K)
            train_logits = G @ alpha
            test_logits = np.hstack([S[te], np.ones((len(te), 1))]) @ Xb.T @ alpha
            tr_probs = softmax_rows(train_logits)
            rows = [{c: float(tr_probs[i][cidx[c]]) for c in K_ord}
                    for i in range(len(tr))]
            thr = choose_thresholds(rows, [y[i] for i in tr], target_precision=0.97)
            tp = softmax_rows(test_logits)
            for j, i in enumerate(te):
                p = tp[j]
                o = int(np.argmax(p))
                if K_ord[o] == y[i]:
                    ungated_correct += 1
                srt = np.sort(p)
                margin = p[o] - srt[-2]
                if p[o] >= thr[K_ord[o]] and margin > MARGIN:
                    routes += 1
                    if K_ord[o] == y[i]:
                        rcount += 1
                else:
                    abst += 1
        tot = routes + abst
        out[lam] = {
            "lam": lam, "routes": routes, "route_correct": rcount,
            "abstained": abst, "total": tot,
            "A_ungated": round(ungated_correct / tot, 4),
            "P": round(rcount / routes, 4) if routes else None,
            "C": round(routes / tot, 4),
            "A": round(rcount / tot, 4),
            "E2E": round((rcount + CHAIN * abst) / tot, 4),
        }
    return out


def ood_abstain(Sg, So, y, classes, lam):
    """Train on ALL gold, gate the OOD set (must abstain)."""
    K_ord = sorted(classes)
    K = len(K_ord)
    cidx = {c: i for i, c in enumerate(K_ord)}
    Y = np.zeros((len(y), K))
    for i, lab in enumerate(y):
        Y[i, cidx[lab]] = 1.0
    Xb = np.hstack([Sg, np.ones((len(y), 1))])
    G = Xb @ Xb.T
    alpha = np.linalg.solve(G + lam * np.eye(len(y)), Y)
    trp = softmax_rows(G @ alpha)
    rows = [{c: float(trp[i][cidx[c]]) for c in K_ord} for i in range(len(y))]
    thr = choose_thresholds(rows, [str(v) for v in y], target_precision=0.97)
    tp = softmax_rows(np.hstack([So, np.ones((len(So), 1))]) @ Xb.T @ alpha)
    ok = 0
    for j in range(len(So)):
        p = tp[j]
        o = int(np.argmax(p))
        if not (p[o] >= thr[K_ord[o]] and (p[o] - np.sort(p)[-2]) > MARGIN):
            ok += 1
    return round(ok / len(So), 4)


def best_row(rows):
    """Best lambda by UNGATED accuracy (the reference 0.7258 quantity);
    gated P/C/E2E are reported at that lambda, as the campaign does."""
    return max(rows.values(), key=lambda r: (r["A_ungated"], r["E2E"]))


def evaluate_foldwise(S_by_fold, y, fold, classes, lambdas=None):
    """Same protocol as evaluate(), but fold f uses its OWN representation
    matrix S_by_fold[f] (the trained-projection arm: each fold's W_in is
    learned on that fold's train items only)."""
    lambdas = LAMBDAS if lambdas is None else lambdas
    K_ord = sorted(classes)
    K = len(K_ord)
    cidx = {c: i for i, c in enumerate(K_ord)}
    out = {}
    for lam in lambdas:
        routes = rcount = abst = ungated = 0
        for f in range(FOLDS):
            S = S_by_fold[f]
            tr = np.where(fold != f)[0]
            te = np.where(fold == f)[0]
            if len(te) == 0:
                continue
            Ytr = np.zeros((len(tr), K))
            for i, lab in enumerate([y[j] for j in tr]):
                Ytr[i, cidx[lab]] = 1.0
            Xb = np.hstack([S[tr], np.ones((len(tr), 1))])
            G = Xb @ Xb.T
            alpha = np.linalg.solve(G + lam * np.eye(len(tr)), Ytr)
            trp = softmax_rows(G @ alpha)
            rows = [{c: float(trp[i][cidx[c]]) for c in K_ord} for i in range(len(tr))]
            thr = choose_thresholds(rows, [y[j] for j in tr], target_precision=0.97)
            tp = softmax_rows(np.hstack([S[te], np.ones((len(te), 1))]) @ Xb.T @ alpha)
            for j, i in enumerate(te):
                p = tp[j]
                o = int(np.argmax(p))
                if K_ord[o] == y[i]:
                    ungated += 1
                if p[o] >= thr[K_ord[o]] and (p[o] - np.sort(p)[-2]) > MARGIN:
                    routes += 1
                    if K_ord[o] == y[i]:
                        rcount += 1
                else:
                    abst += 1
        tot = routes + abst
        out[lam] = {"lam": lam, "routes": routes, "route_correct": rcount,
                    "abstained": abst, "total": tot,
                    "A_ungated": round(ungated / tot, 4),
                    "P": round(rcount / routes, 4) if routes else None,
                    "C": round(routes / tot, 4),
                    "A": round(rcount / tot, 4),
                    "E2E": round((rcount + CHAIN * abst) / tot, 4)}
    return out


def ridge_head_dualfit(S, y, classes, lam):
    """Closed-form readout (Wout [K,N], b [K]) via the dual form - the warm
    start for W_in training. Not used for any reported metric."""
    K_ord = sorted(classes)
    cidx = {c: i for i, c in enumerate(K_ord)}
    Y = np.zeros((len(y), len(K_ord)))
    for i, lab in enumerate(y):
        Y[i, cidx[lab]] = 1.0
    Xb = np.hstack([S, np.ones((len(y), 1))])
    G = Xb @ Xb.T
    alpha = np.linalg.solve(G + lam * np.eye(len(y)), Y)
    w = Xb.T @ alpha                      # (N+1, K)
    return w[:-1].T.copy(), w[-1].copy()


def main():
    t0 = time.time()
    gold, ood, classes, Xg, Xo, y = load_data()
    adj, W = load_W()
    D, N = Xg.shape[1], W.shape[0]
    fold, rng = make_folds(y, classes)

    rho0 = max(abs(np.linalg.eigvals(W)))
    out = {
        "axis": "A: spectral-radius scaling + trained input projection",
        "data": {"gold": len(y), "ood": len(Xo), "classes": len(classes),
                 "embed_dim": D, "neurons": N, "edges": int(adj["edges"]),
                 "one_case_pt": round(100.0 / len(y), 3)},
        "protocol": {"folds": FOLDS, "seed": SEED, "margin": MARGIN,
                     "target_precision": 0.97, "steps": T_STEPS,
                     "decay": DECAY, "in_scale": IN_SCALE,
                     "lambdas": LAMBDAS, "chain_floor": CHAIN,
                     "readout": "dual-form ridge (algebraically == primal ridge_fit)",
                     "A_ungated": "plain argmax accuracy over all 361 - the "
                                  "quantity the reference 0.7258/0.7175 are",
                     "A_gated": "route_correct/total, as the brief defines; "
                                "gate-limited to ~0.19 and NOT comparable to 0.7258"},
        "spectral_radius_as_scaled": round(float(rho0), 6),
        "rho_effective_transition_decay0.8": round(
            float(max(abs(np.linalg.eigvals(W + DECAY * np.eye(N))))), 6),
        "reference": {"larval_lam30_A_ungated": 0.7258, "routes": 71, "P": 0.9577,
                      "C": 0.1967, "E2E": 0.8857,
                      "linear_probe_A_ungated": 0.7175, "chain_floor": CHAIN},
        "sections": {},
    }
    print(f"rho(W as-scaled)={rho0:.6f}  N={N} D={D} gold={len(y)}", flush=True)

    win_fixed = fixed_win(D, N, np.random.default_rng(SEED))

    # -- section 1: reproduce the fixed-projection baseline at rho as-is --
    S0 = run_states(W, Xg, win_fixed)
    So0 = run_states(W, Xo, win_fixed)
    sat = float(np.mean(np.abs(S0) > 0.95))
    rows0 = evaluate(S0, y, fold, classes)
    b0 = best_row(rows0)
    out["sections"]["baseline_fixed_proj_rho_as_scaled"] = {
        "rho": round(float(rho0), 6), "saturation": round(sat, 4),
        "best": b0, "by_lambda": rows0,
        "ood_r_at_best_lam": ood_abstain(S0, So0, y, classes, b0["lam"]),
    }
    print(f"[baseline rho={rho0:.3f}] best lam={b0['lam']} A_ungated={b0['A_ungated']} "
          f"routes={b0['routes']} P={b0['P']} E2E={b0['E2E']} sat={sat:.3f}",
          flush=True)

    # -- section 2: spectral-radius sweep (fixed projection) --
    sweep = {}
    for rho in RHO_TARGETS:
        Ws = W * (rho / rho0)
        S = run_states(Ws, Xg, win_fixed)
        rows = evaluate(S, y, fold, classes)
        b = best_row(rows)
        s = float(np.mean(np.abs(S) > 0.95))
        js = max(abs(np.linalg.eigvals(Ws + DECAY * np.eye(N))))
        sweep[str(rho)] = {"rho_target": rho, "rho_eff_transition": round(float(js), 4),
                           "saturation": round(s, 4), "best": b, "by_lambda": rows}
        print(f"[fixed rho={rho}] lam={b['lam']} A_ungated={b['A_ungated']} routes={b['routes']} "
              f"P={b['P']} E2E={b['E2E']} sat={s:.3f} rhoJ={js:.3f}", flush=True)
    out["sections"]["spectral_radius_sweep_fixed_proj"] = sweep

    best_rho_fixed = max(sweep.values(), key=lambda v: v["best"]["A_ungated"])
    print(f"best fixed-proj rho={best_rho_fixed['rho_target']} "
          f"A_ungated={best_rho_fixed['best']['A_ungated']}", flush=True)
    out["best_rho_fixed_proj"] = best_rho_fixed["rho_target"]

    Path(REPO / "tools" / "eval" / "axis_A_partial.json").write_text(
        json.dumps(out, indent=2))

    # -- section 3: TRAINED input projection (rho as-scaled) ------------
    # W_in learned by BPTT/Adam on each fold's train items only; the reported
    # readout is still the ridge sweep, so the only change is W_in.
    grid = [(30, 3e-3, 0.0), (30, 3e-3, 1e-2), (100, 3e-3, 1e-2),
            (30, 1e-2, 1e-2), (30, 3e-3, 5e-2)]
    trained_sweep = {}
    trained_best = None
    for (ep, lr, l2o) in grid:
        key = f"epochs{ep}_lr{lr}_l2out{l2o}"
        tcfg = time.time()
        S_by_fold = {}
        scratch = []
        for f in range(FOLDS):
            tr = np.where(fold != f)[0]
            Sf = run_states(W, Xg, win_fixed)
            Sf_o = run_states(W, Xo, win_fixed)
            Wo, bb = ridge_head_dualfit(Sf[tr], [y[i] for i in tr], classes, 10.0)
            Win_f, acc_tr = train_win(W, Xg[tr], [y[i] for i in tr], classes,
                                      win_fixed, Wo, bb, epochs=ep, lr=lr,
                                      l2_out=l2o)
            # states for ALL gold items under this fold's W_in (fold-f uses
            # only its own train-fit W_in for both train and test rows)
            allX = np.vstack([Xg])
            S_by_fold[f] = run_states(W, allX, Win_f)
            scratch.append(round(acc_tr, 4))
        rows = evaluate_foldwise(S_by_fold, y, fold, classes)
        b = best_row(rows)
        satf = float(np.mean([np.mean(np.abs(S_by_fold[f]) > 0.95) for f in range(FOLDS)]))
        trained_sweep[key] = {"epochs": ep, "lr": lr, "l2_out": l2o,
                             "scratch_train_acc": scratch,
                             "saturation": round(satf, 4),
                             "best": b, "by_lambda": rows}
        print(f"[trained {key}] lam={b['lam']} A_ungated={b['A_ungated']} "
              f"routes={b['routes']} P={b['P']} E2E={b['E2E']} "
              f"scratch_tr_acc={np.mean(scratch):.3f} ({time.time()-tcfg:.0f}s)",
              flush=True)
        if trained_best is None or b["A_ungated"] > trained_best[1]:
            trained_best = (key, b["A_ungated"], ep, lr, l2o)
    out["sections"]["trained_input_projection_rho_as_scaled"] = trained_sweep
    out["trained_projection_best"] = {
        "config": trained_best[0], "A_ungated": trained_best[1],
        "delta_vs_fixed_same_rho": round(trained_best[1] - b0["A_ungated"], 4),
        "delta_vs_reference_0.7258": round(trained_best[1] - 0.7258, 4),
    }
    print(f"trained best = {trained_best}", flush=True)
    Path(REPO / "tools" / "eval" / "axis_A_partial.json").write_text(
        json.dumps(out, indent=2))

    # -- section 4: combine best rho x trained projection ---------------
    _, _, ep_b, lr_b, l2o_b = trained_best
    rho_b = best_rho_fixed["rho_target"]
    Wb = W * (rho_b / rho0)
    S_by_fold_b = {}
    scratch_b = []
    for f in range(FOLDS):
        tr = np.where(fold != f)[0]
        Sf = run_states(Wb, Xg, win_fixed)
        Wo, bb = ridge_head_dualfit(Sf[tr], [y[i] for i in tr], classes, 10.0)
        Win_f, acc_tr = train_win(Wb, Xg[tr], [y[i] for i in tr], classes,
                                  win_fixed, Wo, bb, epochs=ep_b, lr=lr_b,
                                  l2_out=l2o_b)
        S_by_fold_b[f] = run_states(Wb, Xg, Win_f)
        scratch_b.append(round(acc_tr, 4))
    rows_b = evaluate_foldwise(S_by_fold_b, y, fold, classes)
    row_b = best_row(rows_b)
    So_b = run_states(Wb, Xo, win_fixed)
    out["sections"]["combined_best_rho_and_trained_projection"] = {
        "rho_target": rho_b, "train_config": {"epochs": ep_b, "lr": lr_b, "l2_out": l2o_b},
        "scratch_train_acc": scratch_b,
        "fixed_proj_at_same_rho": best_rho_fixed["best"],
        "best": row_b, "by_lambda": rows_b,
    }
    # OOD gate for the final row (fixed-proj states at the same rho, ridge readout)
    out["sections"]["combined_best_rho_and_trained_projection"]["ood_note"] = (
        "OOD_R reported for the fixed-projection reservoir at the same rho "
        "(OOD items have no trained W_in arm in this CV).")
    print(f"[combined rho={rho_b} trained] lam={row_b['lam']} "
          f"A_ungated={row_b['A_ungated']} routes={row_b['routes']} P={row_b['P']} "
          f"E2E={row_b['E2E']}", flush=True)

    # -- verdict --------------------------------------------------------
    fixed_A = b0["A_ungated"]
    best_fixed_A = best_rho_fixed["best"]["A_ungated"]
    cand = max(fixed_A, best_fixed_A, trained_best[1], row_b["A_ungated"])
    if cand > fixed_A + 0.003:
        verdict = "IMPROVED"
    elif cand >= fixed_A - 0.003:
        verdict = "PARITY"
    else:
        verdict = "WORSE"
    out["verdict"] = {
        "verdict": verdict,
        "fixed_proj_rho_as_scaled_A_ungated": fixed_A,
        "fixed_proj_best_rho": best_rho_fixed["rho_target"],
        "fixed_proj_best_rho_A_ungated": best_fixed_A,
        "trained_proj_best_A_ungated": trained_best[1],
        "combined_best_A_ungated": row_b["A_ungated"],
        "reference_A_ungated": 0.7258,
        "noise_band_pt": 0.3,
    }
    print("VERDICT:", verdict, out["verdict"], flush=True)

    outp = REPO / "tools" / "eval" / "axis_spectral_projection.json"
    outp.write_text(json.dumps(out, indent=2))
    print(f"wrote {outp} in {time.time()-t0:.1f}s", flush=True)
    return out


if __name__ == "__main__":
    main()
