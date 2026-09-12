"""Axis B: mushroom-body (MB) subcircuit as the reservoir, vs the whole larval brain.

Protocol is identical to the committed E1/E3 reference (see
docs/EXPERIMENTS.md + tools/eval/e1_real.py + data/e1_corrected_results.json):

  5-fold stratified CV, seed 42 (per-class rng.shuffle, fold = pos % 5)
  reservoir: s = tanh(0.8*s + W@s + win@x), 4 steps, in_scale 0.02,
             proj = rng.integers(-6, 7, (1024, n)) on the SAME rng stream that
             the reference used (drawn after the fold shuffle)
  readout  : closed-form ridge with bias, penalty SWEPT over 1e-3 .. 100
  gate     : top-1 softmax prob >= per-class threshold calibrated on TRAIN-side
             scores via tools/train/calibrate.choose_thresholds(target=0.97)
             AND top1-top2 margin > 0.05; else abstain
  P  = route_correct / routes        (routed-but-wrong STAYS in the denominator)
  C  = routes / total
  A  = ungated argmax accuracy over all cases  (the reference's "A (all cases)")
  A_task = route_correct / total     (the task brief's A formula; both reported)
  E2E = (route_correct + 0.868 * abstained) / total

Subcircuit identification: cell types from Winding et al. 2023 supplementary
table S2 (data/science.add9330_data_s2.csv), joined to the adjacency's 2,952
row IDs through the larval CSV header.  Primary MB set = celltype in
{KC, MBON, MBIN, MB-FBN, MB-FFN}; alternates KC-only and MB-core (KC+MBON+MBIN).
Induced subgraph, re-indexed to a compact 0..n-1 CSR.
Size-matched random control = same n neurons drawn uniformly from the 2,952.

Writes tools/eval/axis_mb_subcircuit.json.  Does not touch docs/ or git.
"""

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "train"))

from calibrate import choose_thresholds  # noqa: E402
import states as st  # noqa: E402
from train_readout import ridge_fit, _one_hot, _class_scores  # noqa: E402

FOLDS = 5
SEED = 42
MARGIN_FLOOR = 0.05
STEPS = 4
DECAY = 0.8
IN_SCALE = 0.02
CHAIN_BASELINE = 0.868
LAMS = [1e-3, 1e-2, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0]
N_RANDOM = 5
MB_CELLTYPES = {"KC", "MBON", "MBIN", "MB-FBN", "MB-FFN"}
OUT = REPO / "tools" / "eval" / "axis_mb_subcircuit.json"


def softmax(logits):
    m = max(logits)
    exps = [np.exp(v - m) for v in logits]
    z = sum(exps)
    return [e / z for e in exps]


def logits_for(W, bias, s):
    return [sum(W[i][j] * v for i, v in enumerate(s)) + bias[j] for j in range(len(bias))]


def induced_subgraph(adj, keep_rows):
    """Induced subgraph on keep_rows, re-indexed to a compact 0..k-1 CSR."""
    keep = sorted(set(int(i) for i in keep_rows))
    old2new = {old: new for new, old in enumerate(keep)}
    indptr = [0]
    indices = []
    weights = []
    n = adj["neurons"]
    ip = adj["indptr"]
    ix = adj["indices"]
    wt = adj["weights"]
    for old in keep:
        lo, hi = ip[old], ip[old + 1]
        for e in range(lo, hi):
            tgt = ix[e]
            if tgt in old2new:
                indices.append(old2new[tgt])
                weights.append(int(wt[e]))
        indptr.append(len(indices))
    k = len(keep)
    assert len(indptr) == k + 1
    src = {"name": adj["name"], "neurons": k, "edges": len(indices), "indptr": indptr,
           "indices": indices, "weights": weights}
    return src, keep


def make_cfg(src, D, proj, classes):
    n = src["neurons"]
    w = np.asarray(src["weights"], dtype=np.float64) / 127.0
    return {
        "name": src["name"], "neurons": n, "edges": src["edges"],
        "embed_dim": D, "steps": STEPS, "decay": DECAY, "classes": classes,
        "indptr": src["indptr"], "indices": src["indices"],
        "weights": w.tolist(), "weight_scale": 1.0 / 127.0,
        "win": (proj.astype(np.float64) * IN_SCALE).reshape(-1).tolist(),
    }


def eval_lam(S, So, y, fold, classes, lam):
    K = len(classes)
    n_total = len(y)
    correct = abstained = routes = 0
    ungated = 0
    for f in range(FOLDS):
        tr = np.where(fold != f)[0]
        te = np.where(fold == f)[0]
        if len(te) == 0:
            continue
        S_tr = [S[i] for i in tr]
        y_tr = [y[i] for i in tr]
        W, bias = ridge_fit(S_tr, _one_hot(y_tr, classes), lam=lam)
        thr = choose_thresholds(_class_scores(W, bias, S_tr, classes), y_tr,
                                target_precision=0.97)
        for i in te:
            probs = softmax(logits_for(W, bias, S[i]))
            top = int(np.argmax(probs))
            if classes[top] == y[i]:
                ungated += 1
            margin = probs[top] - sorted(probs)[-2]
            if probs[top] >= thr[classes[top]] and margin > MARGIN_FLOOR:
                routes += 1
                if classes[top] == y[i]:
                    correct += 1
            else:
                abstained += 1
    total = correct + abstained
    return {
        "lam": lam, "routes": routes, "route_correct": correct, "abstained": abstained,
        "total": total,
        "P": round(correct / routes, 4) if routes else 1.0,
        "C": round(routes / total, 4) if total else 0.0,
        "A": round(ungated / total, 4) if total else 0.0,
        "A_task": round(correct / total, 4) if total else 0.0,
        "E2E": round((correct + CHAIN_BASELINE * abstained) / total, 4) if total else 0.0,
    }


def ood_rate(S_all, So, y, classes, lam):
    if len(So) == 0:
        return None
    W, bias = ridge_fit([S_all[i] for i in range(len(y))], _one_hot(list(y), classes), lam=lam)
    thr = choose_thresholds(_class_scores(W, bias, [S_all[i] for i in range(len(y))], classes),
                            list(y), target_precision=0.97)
    abst = 0
    for j in range(len(So)):
        probs = softmax(logits_for(W, bias, So[j]))
        top = int(np.argmax(probs))
        margin = probs[top] - sorted(probs)[-2]
        if not (probs[top] >= thr[classes[top]] and margin > MARGIN_FLOOR):
            abst += 1
    return round(abst / len(So), 4)


def run_one(src, D, proj, classes, Xg, Xo, y, fold, label):
    cfg = make_cfg(src, D, proj, classes)
    S = np.asarray([st.states(cfg, x.tolist()) for x in Xg])
    So = (np.asarray([st.states(cfg, x.tolist()) for x in Xo])
          if len(Xo) else np.zeros((0, src["neurons"])))
    sat = float(np.mean(np.abs(S) > 0.95))
    rows = [eval_lam(S, So, y, fold, classes, lam) for lam in LAMS]
    best = max(rows, key=lambda r: (r["A"], r["E2E"]))
    out = {
        "label": label, "neurons": src["neurons"], "edges": src["edges"],
        "saturation": round(sat, 4), "sweep": rows,
        "best_lam": best["lam"], "A": best["A"], "A_task": best["A_task"],
        "routes": best["routes"], "route_correct": best["route_correct"],
        "abstained": best["abstained"], "total": best["total"],
        "P": best["P"], "C": best["C"], "E2E": best["E2E"],
        "OOD_R": ood_rate(S, So, y, classes, best["lam"]),
        "route_count_disclosure": f"{best['routes']}/{best['total']} direct routes",
    }
    print(f"[{label}] n={out['neurons']} e={out['edges']} sat={out['saturation']} "
          f"best_lam={out['best_lam']} A={out['A']} routes={out['routes']}/361 "
          f"P={out['P']} C={out['C']} A_task={out['A_task']} E2E={out['E2E']} "
          f"OOD_R={out['OOD_R']}", flush=True)
    return out


def main():
    d = json.loads((REPO / "data" / "e1_inputs.json").read_text())
    gold, ood, classes, vecs = d["gold"], d["ood"], d["classes"], d["vecs"]
    E = np.asarray(vecs, dtype=np.float64)
    D = E.shape[1]
    n_gold = len(gold)
    Xg, Xo = E[:n_gold], E[n_gold:]
    y = np.array([c["intent"] for c in gold])
    print(f"gold={n_gold} ood={len(ood)} classes={len(classes)} D={D}", flush=True)

    adj = json.loads((REPO / "data" / "larval_adjacency.json").read_text())
    whole_n = adj["neurons"]

    # ---- annotation join: CSV header id -> adjacency row -------------------
    with open(REPO / "data" / "larval_signed_connectivity.csv") as f:
        hdr = f.readline().rstrip("\n").split(",")
    id2row = {int(x): i for i, x in enumerate(hdr[1:])}
    id2ct = {}
    with open(REPO / "data" / "science.add9330_data_s2.csv") as f:
        for r in csv.DictReader(f):
            ct = r["celltype"].strip()
            for c in ("left_id", "right_id"):
                v = r[c].strip()
                if v != "no pair":
                    id2ct[int(v)] = ct
    mapped = sum(1 for nid in id2row if nid in id2ct)
    mb_rows = sorted(i for nid, i in id2row.items() if id2ct.get(nid) in MB_CELLTYPES)
    kc_rows = sorted(i for nid, i in id2row.items() if id2ct.get(nid) == "KC")
    core_rows = sorted(i for nid, i in id2row.items() if id2ct.get(nid) in {"KC", "MBON", "MBIN"})
    ct_hist = Counter(id2ct[nid] for nid in id2row if nid in id2ct)
    # densest same-size control: top-382 neurons by total degree (in + out)
    outdeg = np.diff(np.asarray(adj["indptr"], dtype=np.int64))
    indeg = np.bincount(np.asarray(adj["indices"], dtype=np.int64), minlength=whole_n)
    totdeg = outdeg + indeg
    hub_rows = sorted(int(i) for i in np.argsort(-totdeg, kind="stable")[:len(mb_rows)])
    print(f"annotated rows {mapped}/{whole_n}; MB={len(mb_rows)} KC={len(kc_rows)} "
          f"core={len(core_rows)}", flush=True)

    # ---- folds: SAME rng stream as the reference --------------------------
    rng = np.random.default_rng(SEED)
    fold = np.zeros(n_gold, dtype=int)
    for cls in classes:
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        for j, i in enumerate(idx):
            fold[i] = j % FOLDS
    fold_sizes = [int((fold == f).sum()) for f in range(FOLDS)]

    # rng draws a fresh projection per reservoir, exactly like e1_real.py
    def draw_proj(n):
        return rng.integers(-6, 7, size=(D, n)).astype(np.int8)

    results = {
        "experiment": "axis-b-mushroom-body-subcircuit",
        "protocol": {
            "folds": FOLDS, "seed": SEED, "margin_floor": MARGIN_FLOOR,
            "steps": STEPS, "decay": DECAY, "in_scale": IN_SCALE,
            "target_precision": 0.97, "lam_sweep": LAMS,
            "chain_baseline": CHAIN_BASELINE,
            "fold_sizes": fold_sizes,
        },
        "subcircuit_identification": {
            "source": "Winding et al. 2023 Science 379:eadd9330 supplementary table S2 "
                      "(data/science.add9330_data_s2.csv), celltype column",
            "join": "annotation neuron IDs joined to the 2,952 adjacency row IDs via the "
                    "larval_signed_connectivity.csv numeric header",
            "annotated_rows": mapped, "total_rows": whole_n,
            "mb_celltypes": sorted(MB_CELLTYPES),
            "celltype_histogram_joined": dict(sorted(ct_hist.items(), key=lambda kv: -kv[1])),
            "extraction": "induced subgraph (both endpoints in the set), re-indexed to a "
                          "compact 0..n-1 CSR; weights kept at the source int scale /127",
        },
        "reference": {
            "whole_larval_best_lam": 30, "A": 0.7258, "routes": 71, "P": 0.9577,
            "C": 0.1967, "E2E": 0.8857,
            "note": "from tools/eval/e1_corrected_results.json (whole 2,952-neuron larval "
                    "reservoir, same protocol, margin floor 0.05)",
        },
        "reservoirs": {},
    }

    # 1) whole brain, drawn first so its projection matches the reference stream
    proj = draw_proj(whole_n)
    results["reservoirs"]["whole_brain"] = run_one(
        adj, D, proj, classes, Xg, Xo, y, fold, "whole brain (2952)")
    OUT.write_text(json.dumps(results, indent=2))

    # 2) MB subcircuit + alternates
    for name, rows in (("mb_full", mb_rows), ("mb_core_kc_mbon_mbin", core_rows),
                       ("kc_only", kc_rows)):
        src, keep = induced_subgraph(adj, rows)
        src["name"] = f"{name}-v1"
        p = draw_proj(src["neurons"])
        results["reservoirs"][name] = run_one(src, D, p, classes, Xg, Xo, y, fold, name)
        results["reservoirs"][name]["kept_row_ids_sha_len"] = len(keep)
        OUT.write_text(json.dumps(results, indent=2))

    # 3) controls.  A size-matched random subgraph is NOT density-matched: the
    #    larval brain is ~0.7% dense, so 382 random neurons carry ~1k edges
    #    while the MB's 382 neurons carry 5.6k.  The hub control (top-382 by
    #    total degree) is the opposite extreme - the densest induced subgraph
    #    of the same size - so the MB can be bracketed between the two.
    n_mb = len(mb_rows)
    src_hub, _ = induced_subgraph(adj, hub_rows)
    src_hub["name"] = f"hub-{n_mb}-v1"
    p = draw_proj(src_hub["neurons"])
    results["reservoirs"]["hub_top_degree"] = run_one(
        src_hub, D, p, classes, Xg, Xo, y, fold, f"hub top-degree ({n_mb})")
    OUT.write_text(json.dumps(results, indent=2))

    srng = np.random.default_rng(1234)
    controls = []
    for r in range(N_RANDOM):
        pick = srng.choice(whole_n, size=n_mb, replace=False)
        src, _ = induced_subgraph(adj, pick.tolist())
        src["name"] = f"random-{n_mb}-{r}"
        p = draw_proj(src["neurons"])
        controls.append(run_one(src, D, p, classes, Xg, Xo, y, fold,
                                f"random size-matched #{r} ({n_mb})"))
        OUT.write_text(json.dumps(results, indent=2))
    agg = {k: round(float(np.mean([c[k] for c in controls])), 4)
           for k in ("A", "A_task", "P", "C", "E2E")}
    agg_routes = round(float(np.mean([c["routes"] for c in controls])), 1)
    results["random_size_matched_control"] = {
        "n_neurons": n_mb, "n_draws": N_RANDOM, "runs": controls,
        "mean_edges": round(float(np.mean([c["edges"] for c in controls])), 1),
        "mean_A": agg["A"], "mean_A_task": agg["A_task"], "mean_P": agg["P"],
        "mean_C": agg["C"], "mean_E2E": agg["E2E"], "mean_routes": agg_routes,
        "min_A": min(c["A"] for c in controls), "max_A": max(c["A"] for c in controls),
        "density_note": "size-matched but NOT density-matched: random 382-neuron induced "
                        "subgraphs carry ~1.0k edges vs the MB's 5.6k (the brain is ~0.7% "
                        "dense while the MB is internally dense); hub_top_degree is the "
                        "densest same-size control bound.",
    }

    # ---- compact comparison table ----------------------------------------
    def row(label, key):
        r = results["reservoirs"][key]
        return {"representation": label, "neurons": r["neurons"], "edges": r["edges"],
                "best_lam": r["best_lam"], "A_ungated_all_cases": r["A"],
                "A_task_route_correct_over_total": r["A_task"], "routes": r["routes"],
                "total": r["total"], "P": r["P"], "C": r["C"], "E2E": r["E2E"],
                "OOD_R": r["OOD_R"]}
    table = [row("whole brain (larval, 2952)", "whole_brain"),
             row("MB full (KC+MBON+MBIN+MB-FBN+MB-FFN)", "mb_full"),
             row("MB core (KC+MBON+MBIN)", "mb_core_kc_mbon_mbin"),
             row("KC only", "kc_only"),
             row("degree-hub top-382 (densest same-size control)", "hub_top_degree"),
             {"representation": f"random size-matched 382 (mean of {N_RANDOM})",
              "neurons": n_mb, "edges": results["random_size_matched_control"]["mean_edges"],
              "best_lam": None, "A_ungated_all_cases": agg["A"],
              "A_task_route_correct_over_total": agg["A_task"],
              "routes": agg_routes, "total": 361, "P": agg["P"], "C": agg["C"],
              "E2E": agg["E2E"], "OOD_R": None},
             {"representation": "committed reference: whole brain lam=30",
              "neurons": 2952, "edges": 63545, "best_lam": 30, "A_ungated_all_cases": 0.7258,
              "A_task_route_correct_over_total": None, "routes": 71, "total": 361,
              "P": 0.9577, "C": 0.1967, "E2E": 0.8857, "OOD_R": 0.9286}]
    results["table"] = table
    OUT.write_text(json.dumps(results, indent=2))
    print("wrote", OUT, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
