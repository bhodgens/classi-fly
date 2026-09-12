"""E1/E3: the real experiment. Larval-connectome reservoir vs baselines on a
real, adjudicated intent corpus, with REAL embeddings from a local
OpenAI-compatible embedding server.

Corpus: the meept classifier corpora (base + adversarial, JSON5), which are the
campaign's adjudicated ruler. Text is read from the meept checkout and never
written into this repo (only metrics are).

Protocol (mirrors plans/classi-fly/master.md and the campaign rules):
- 5-fold stratified CV, seed 42; the readout (and the baselines) see only the
  train folds; per-class route thresholds are calibrated on TRAIN-side scores.
- Route only when top-1 probability clears its calibrated per-class threshold
  AND the top1-top2 margin clears the floor; otherwise abstain.
- E2E = (routes_correct + CHAIN_BASELINE * abstained) / total, deterministic.
- Route counts are always disclosed. Silver-labelled cases are excluded from
  the headline. OOD cases are scored as "must abstain", not as a class.

Run:
  python3 tools/eval/e1_real.py --embed-url http://127.0.0.1:8090/v1 \
      --out tools/eval/e1_real_results.json
"""

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "eval"))
sys.path.insert(0, str(REPO / "tools" / "train"))
sys.path.insert(0, str(REPO / "tools" / "ingest"))

from embeddings import embed_texts  # noqa: E402
from baselines import centroid_margin, knn_unanimity  # noqa: E402
from train_readout import ridge_fit, _one_hot, _class_scores  # noqa: E402
from calibrate import choose_thresholds  # noqa: E402
import states as st  # noqa: E402

CHAIN_BASELINE = 0.868
FOLDS = 5
SEED = 42
MARGIN_FLOOR = 0.005
STEPS = 4
DECAY = 0.8


def normalise_json5(text: str) -> str:
    """JSON5 -> JSON for this corpus dialect: strip // comments, quote bare
    keys, drop trailing commas."""
    text = re.sub(r"//[^\n]*", "", text)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)", r'\1"\2"\3', text)
    text = re.sub(r",(\s*[}\]])", r"\1", text)
    return text


def load_meept_corpus(base_path: Path, adv_path: Path):
    """Flatten the meept corpora into {text, intent|None, ood, silver}."""
    out = []
    if base_path.exists():
        d = json.loads(normalise_json5(base_path.read_text()))
        for cat, cases in d.get("categories", {}).items():
            for c in cases:
                out.append({
                    "text": c["input"],
                    "intent": c.get("expected_intent") or cat,
                    "ood": False,
                    "silver": False,
                    "source": "base",
                })
    if adv_path.exists():
        d = json.loads(normalise_json5(adv_path.read_text()))
        for c in d.get("cases", []):
            out.append({
                "text": c["input"],
                "intent": c.get("expected_intent"),
                "ood": bool(c.get("ood")),
                "silver": bool(c.get("silver")),
                "source": "adversarial",
            })
    return out


def build_reservoir(adj: dict, embed_dim: int, proj: np.ndarray, in_scale: float,
                    classes: list) -> dict:
    n = adj["neurons"]
    return {
        "name": adj["name"], "neurons": n, "edges": adj["edges"],
        "embed_dim": embed_dim, "steps": STEPS, "decay": DECAY, "classes": classes,
        "indptr": adj["indptr"], "indices": adj["indices"],
        "weights": (np.asarray(adj["weights"], dtype=np.float64) / 127.0).tolist(),
        "weight_scale": 1.0 / 127.0,
        "win": (proj.astype(np.float64) * in_scale).reshape(-1).tolist(),
    }


def softmax(logits):
    m = max(logits)
    exps = [np.exp(v - m) for v in logits]
    z = sum(exps)
    return [e / z for e in exps]


def logits_for(W, bias, s):
    return [sum(W[i][j] * v for i, v in enumerate(s)) + bias[j] for j in range(len(bias))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed-url", required=True)
    ap.add_argument("--model", default="qwen3-emb")
    ap.add_argument("--meept", default="/Users/caimlas/git/meept")
    ap.add_argument("--larval", default=str(REPO / "data" / "larval_adjacency.json"))
    ap.add_argument("--synthetic-neurons", type=int, default=2048)
    ap.add_argument("--out", default=str(REPO / "tools" / "eval" / "e1_real_results.json"))
    args = ap.parse_args()

    meept = Path(args.meept)
    corpus = load_meept_corpus(
        meept / "testdata/eval/classifier-test-corpus.json5",
        meept / "testdata/eval/classifier-adversarial-corpus.json5",
    )
    gold = [c for c in corpus if not c["ood"] and not c["silver"] and c["intent"]]
    ood = [c for c in corpus if c["ood"]]
    silver = [c for c in corpus if c["silver"]]
    classes = sorted({c["intent"] for c in gold})
    print(f"corpus: {len(corpus)} total | gold {len(gold)} | ood {len(ood)} | "
          f"silver {len(silver)} | classes {len(classes)}")
    print("intent histogram:", dict(sorted(Counter(c['intent'] for c in gold).items(),
                                           key=lambda x: -x[1])))

    # ---- embed once (disk-cached) -------------------------------------
    t0 = time.time()
    all_texts = [c["text"] for c in gold] + [c["text"] for c in ood]
    vecs = embed_texts(all_texts, url=args.embed_url, model_id=args.model)
    E = np.asarray(vecs, dtype=np.float64)
    D = E.shape[1]
    print(f"embedded {E.shape[0]} texts, dim {D}, {time.time()-t0:.1f}s")

    Xg, Xo = E[:len(gold)], E[len(gold):]

    results = {"experiment": "E1/E3-real", "embedding": f"real:{args.model}",
               "embed_dim": D, "items": {"gold": len(gold), "ood": len(ood),
                                          "silver_excluded": len(silver)},
               "classes": classes, "folds": FOLDS, "seed": SEED,
               "chain_baseline": CHAIN_BASELINE}

    # ---- folds (stratified, shared by reservoir and baselines) --------
    rng = np.random.default_rng(SEED)
    y = np.array([c["intent"] for c in gold])
    fold = np.zeros(len(gold), dtype=int)
    for cls in classes:
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        for j, i in enumerate(idx):
            fold[i] = j % FOLDS

    adjacencies = {"larval": json.loads(Path(args.larval).read_text())}
    # E3: synthetic reservoir at comparable scale
    sys.path.insert(0, str(REPO / "tools" / "ingest"))
    from synthetic import generate  # noqa: E402
    adjacencies["synthetic"] = generate(seed=SEED, n=args.synthetic_neurons,
                                        fan_in=8, inhib_frac=0.2)
    for k in adjacencies:
        adjacencies[k]["name"] = f"{k}-v1"

    for label, adj in adjacencies.items():
        n = adj["neurons"]
        proj = rng.integers(-6, 7, size=(D, n)).astype(np.int8)
        # drive scale chosen so states stay out of tanh saturation
        in_scale = 0.02
        cfg = build_reservoir(adj, D, proj, in_scale, classes)
        S = np.asarray([st.states(cfg, x.tolist()) for x in Xg])
        So = np.asarray([st.states(cfg, x.tolist()) for x in Xo]) if len(Xo) else np.zeros((0, n))
        sat = float(np.mean(np.abs(S) > 0.95))
        print(f"[{label}] neurons={n} edges={adj['edges']} states={S.shape} saturation={sat:.3f}")

        res = run_folds(S, y, fold, classes, Xg, S, So, [c["text"] for c in gold])
        res.update({"neurons": n, "edges": adj["edges"], "license": adj.get("license", ""),
                    "saturation": round(sat, 4)})
        results.setdefault("reservoirs", {})[label] = res
        print_metrics(label, res)

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")
    return 0


def run_folds(S, y, fold, classes, Xg, S_all, So, TEXTS) -> dict:
    """5-fold: reservoir head, judge mode, and baselines on the same folds."""
    K = len(classes)
    cls_idx = {c: i for i, c in enumerate(classes)}
    n_total = len(y)
    correct = abstained = routes = 0
    head_correct = 0
    base_stats = {name: {"correct": 0, "routes": 0, "abstained": 0}
                  for name in ("centroid", "knn", "tfidf")}
    ood_abstain = 0
    per_fold = []
    for f in range(FOLDS):
        tr = np.where(fold != f)[0]
        te = np.where(fold == f)[0]
        if len(te) == 0:
            continue
        S_tr = [S[i] for i in tr]
        y_tr = [y[i] for i in tr]
        W, bias = ridge_fit(S_tr, _one_hot(y_tr, classes))
        thr = choose_thresholds(_class_scores(W, bias, S_tr, classes), y_tr,
                                target_precision=0.97)
        f_routes = f_ok = f_abs = 0
        for i in te:
            probs = softmax(logits_for(W, bias, S[i]))
            top = int(np.argmax(probs))
            margin = probs[top] - sorted(probs)[-2]
            if probs[top] >= thr[classes[top]] and margin > MARGIN_FLOOR:
                f_routes += 1
                if classes[top] == y[i]:
                    f_ok += 1
            else:
                f_abs += 1
        routes += f_routes
        correct += f_ok
        abstained += f_abs
        per_fold.append({"fold": int(f), "train": int(len(tr)), "test": int(len(te)),
                         "routes": f_routes, "route_correct": f_ok, "abstained": f_abs})

        # baselines on the same test fold
        for name, head in (("centroid", centroid_margin()), ("knn", knn_unanimity())):
            head.fit([Xg[i] for i in tr], [y[i] for i in tr])
            for i in te:
                lab, conf, mg = head.predict(Xg[i])
                if lab == y[i]:
                    base_stats[name]["correct"] += 1
                base_stats[name]["routes"] += 1
        # tfidf logistic on RAW TEXT (no embeddings) - the cheap text-only head
        try:
            from baselines import tfidf_logistic
            head = tfidf_logistic()
            head.fit([TEXTS[i] for i in tr], [y[i] for i in tr])
            for i in te:
                lab, conf, mg = head.predict(TEXTS[i])
                if lab == y[i]:
                    base_stats["tfidf"]["correct"] += 1
                base_stats["tfidf"]["routes"] += 1
        except Exception as e:  # noqa: BLE001
            base_stats["tfidf"]["error"] = str(e)

    # OOD: trained on ALL gold, route thresholds from all gold, must abstain
    Wf, biasf = ridge_fit([S[i] for i in range(n_total)], _one_hot(list(y), classes))
    thrf = choose_thresholds(_class_scores(Wf, biasf, [S[i] for i in range(n_total)], classes),
                             list(y), target_precision=0.97)
    for j in range(len(So)):
        probs = softmax(logits_for(Wf, biasf, So[j]))
        top = int(np.argmax(probs))
        margin = probs[top] - sorted(probs)[-2]
        if not (probs[top] >= thrf[classes[top]] and margin > MARGIN_FLOOR):
            ood_abstain += 1

    total = correct + abstained
    out = {
        "routes": routes, "route_correct": correct, "abstained": abstained,
        "total": total,
        "P": round(correct / routes, 4) if routes else 1.0,
        "C": round(routes / total, 4) if total else 0.0,
        "A": round(correct / total, 4) if total else 0.0,
        "E2E": round((correct + CHAIN_BASELINE * abstained) / total, 4) if total else 0.0,
        "OOD_R": round(ood_abstain / len(So), 4) if len(So) else None,
        "baselines": {k: {**v, "acc": round(v["correct"] / max(v["routes"], 1), 4)}
                      for k, v in base_stats.items()},
        "per_fold": per_fold,
        "route_count_disclosure": f"{routes}/{total} direct routes",
    }
    return out


def print_metrics(label, r):
    bl = " ".join(f"{k}={v.get('acc')}" for k, v in r["baselines"].items() if "acc" in v)
    print(f"[{label}] routes={r['routes']}/{r['total']} P={r['P']} C={r['C']} "
          f"A={r['A']} E2E={r['E2E']} OOD_R={r['OOD_R']} | baselines: {bl}")


if __name__ == "__main__":
    raise SystemExit(main())
