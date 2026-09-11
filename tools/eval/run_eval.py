"""Evaluation harness: head mode, judge mode, baselines, and scoring.

Protocol (RESEARCH.md s1/s7, precision-first):
- P   = precision of direct routes = routed_correct / routes
- C   = coverage = routes / total
- E2E = (gate_correct + CHAIN_BASELINE * abstained) / total
  CHAIN_BASELINE is the MEASURED LLM-chain baseline (86.8%) from the meept
  campaign: every abstention is credited with the chain's measured accuracy.
  The formula is DETERMINISTIC - never a random draw.
- OOD_R = OOD abstain rate. OOD cases are SCORED (their abstentions feed
  OOD_R and their routes count toward P/C), never skipped.
- Route-count disclosure is mandatory: printed table AND results.json both
  carry routes/total.

5-fold cross-evaluation with fixed, stratified fold assignment (seed 42).
Fold assignment only decides which items are held out; the `.fly` artifact
already carries the readout and is never refit here. The primary head
(default: centroid-margin) and the other baselines ARE refit per fold on the
training split, like any honest baseline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Optional

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import baselines  # noqa: E402
import corpus as corpus_mod  # noqa: E402
import embeddings as emb  # noqa: E402
import flyio  # noqa: E402

# The measured LLM-chain baseline from the meept campaign (86.8%). Named
# constant on purpose: the E2E chain-credit formula below must stay
# deterministic and auditable, never a sampled draw.
CHAIN_BASELINE = 0.868

DEFAULT_FOLDS = 5
DEFAULT_SEED = 42


def fold_indices(labels, folds: int, seed: int = DEFAULT_SEED):
    """Stratified fold assignment: each index appears in exactly one fold."""
    n = len(labels)
    used = int(folds)
    note: Optional[str] = None
    if n < used:
        # Documented fallback: leave-one-out when the corpus is too small.
        used = n
        note = "leave-one-out (corpus smaller than the requested fold count)"
    rng = random.Random(seed)
    by_label: dict = {}
    for i, lbl in enumerate(labels):
        by_label.setdefault(str(lbl), []).append(i)
    assignment = [0] * n
    for lbl in sorted(by_label):
        idxs = list(by_label[lbl])
        rng.shuffle(idxs)
        for pos, i in enumerate(idxs):
            assignment[i] = pos % used
    return assignment, used, note


def _gate_row(vec, res: "flyio.Reservoir", primary, mode: str) -> dict:
    """One case's gate decision under one mode."""
    r = res.Classify(vec)
    head_pred, head_conf, head_margin = primary.predict(vec)
    if mode == "head":
        route = not r.Abstained
    else:  # judge: route only when the reservoir and the primary head agree
        route = (not r.Abstained) and r.Class == head_pred
    return {
        "pred": r.Class,
        "conf": r.Confidence,
        "margin": r.Margin,
        "abstained": not route,
        "head_pred": head_pred,
        "head_conf": head_conf,
        "head_margin": head_margin,
    }


def _score_mode(res, X, texts, labels, ood_flags, assignment, folds_used,
                primary, mode: str) -> dict:
    """Score one gate mode over the fixed folds. OOD cases are scored too."""
    total = len(labels)
    routes = 0
    routed_correct = 0
    abstained = 0
    ood_total = 0
    ood_abstained = 0
    predictions: list = []
    for f in range(folds_used):
        train = [i for i in range(total) if assignment[i] != f]
        test = [i for i in range(total) if assignment[i] == f]
        if not train or not test:
            continue
        primary.fit([X[i] for i in train], [labels[i] for i in train])
        for i in test:
            row = _gate_row(X[i], res, primary, mode)
            row["text_sha256"] = hashlib.sha256(texts[i].encode("utf-8")).hexdigest()
            row["gold"] = labels[i]
            row["ood"] = bool(ood_flags[i])
            row["fold"] = f
            predictions.append(row)
            if row["abstained"]:
                abstained += 1
                if ood_flags[i]:
                    ood_abstained += 1
            else:
                routes += 1
                if row["pred"] == row["gold"]:
                    routed_correct += 1
            if ood_flags[i]:
                ood_total += 1
    # DETERMINISTIC chain-credit scoring (no randomness, ever):
    #   E2E = (gate_correct + CHAIN_BASELINE * abstained) / total
    gate_correct = routed_correct
    return {
        "mode": mode,
        "P": (routed_correct / routes) if routes else 0.0,
        "C": routes / total if total else 0.0,
        "E2E": (gate_correct + CHAIN_BASELINE * abstained) / total if total else 0.0,
        "OOD_R": (ood_abstained / ood_total) if ood_total else 0.0,
        "routes": routes,
        "total": total,
        "gate_correct": gate_correct,
        "abstained": abstained,
        "ood_total": ood_total,
        "ood_abstained": ood_abstained,
        "predictions": predictions,
    }


def run_eval(fly_path, corpus_items, *, mode: str = "all", embed_url=None,
             cache_dir=None, client=None, embed_model: str = emb.DEFAULT_EMBED_MODEL,
             folds: int = DEFAULT_FOLDS, seed: int = DEFAULT_SEED,
             primary_head=None) -> dict:
    """Run the harness. mode: head | judge | all. Returns the metrics dict."""
    if mode not in ("head", "judge", "all"):
        raise ValueError("run_eval: mode must be head|judge|all")
    texts = [it["text"] for it in corpus_items]
    labels = [it["intent"] for it in corpus_items]
    ood_flags = [it.get("ood", False) for it in corpus_items]
    if not texts:
        raise ValueError("run_eval: empty corpus")
    X = emb.embed_texts(texts, model_id=embed_model, url=embed_url,
                        cache_dir=cache_dir, client=client)
    res = flyio.Reservoir.Load(fly_path)
    if res._m.d != len(X[0]):
        raise ValueError(
            "run_eval: embedding dim %d != artifact embed_dim %d"
            % (len(X[0]), res._m.d))
    assignment, folds_used, fold_note = fold_indices(labels, folds, seed)
    if primary_head is None:
        primary_head = baselines.centroid_margin()

    if mode == "all":
        out: dict = {
            "modes": {
                m: _score_mode(res, X, texts, labels, ood_flags, assignment,
                               folds_used, primary_head, m)
                for m in ("head", "judge")
            },
            "chain_baseline": CHAIN_BASELINE,
            "folds_used": folds_used,
            "folds_note": fold_note,
            "total": len(texts),
        }
        return out
    metrics = _score_mode(res, X, texts, labels, ood_flags, assignment,
                          folds_used, primary_head, mode)
    metrics["chain_baseline"] = CHAIN_BASELINE
    metrics["folds_used"] = folds_used
    metrics["folds_note"] = fold_note
    return metrics


def _mode_result(results: dict, m: str):
    if "modes" in results:
        return results["modes"].get(m)
    return results if results.get("mode") == m else None


def format_table(results: dict) -> str:
    """Printable table + headline. ALWAYS discloses the route count."""
    lines = []
    cb = float(results.get("chain_baseline", CHAIN_BASELINE))
    lines.append("classi-fly eval  (chain baseline %.1f%%)" % (cb * 100))
    lines.append("%-8s %8s %8s %8s %8s %14s" %
                 ("mode", "P", "C", "E2E", "OOD_R", "routes/total"))
    for m in ("head", "judge"):
        r = _mode_result(results, m)
        if r is None:
            continue
        lines.append("%-8s %8.3f %8.3f %8.3f %8.3f %7d/%-6d" %
                     (m, r["P"], r["C"], r["E2E"], r["OOD_R"],
                      r["routes"], r["total"]))
    # Headline names the route count, per the route-count disclosure rule.
    for m in ("head", "judge"):
        r = _mode_result(results, m)
        if r is None:
            continue
        if r["E2E"] > cb:
            lines.append("%s: beat the %.1f%% floor on %d routes"
                         % (m, cb * 100, r["routes"]))
        else:
            lines.append("%s: did NOT beat the %.1f%% floor (%d routes)"
                         % (m, cb * 100, r["routes"]))
    if results.get("folds_note"):
        lines.append("fold note: " + str(results["folds_note"]))
    return "\n".join(lines)


def write_results(out_path, results: dict) -> None:
    """Write results.json + report.md. Per-case rows carry hashes, not text."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    report = p.with_name("report.md")
    report.write_text(
        "# classi-fly eval report\n\n```\n" + format_table(results) + "\n```\n",
        encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="classi-fly eval harness")
    ap.add_argument("--fly", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--embed-url", default=None,
                    help="OpenAI-compatible embeddings base URL (required unless cached)")
    ap.add_argument("--embed-model", default=emb.DEFAULT_EMBED_MODEL)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--mode", choices=["head", "judge", "all"], default="all")
    ap.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args(argv)
    items = corpus_mod.load_corpus(args.corpus)
    results = run_eval(args.fly, items, mode=args.mode, embed_url=args.embed_url,
                       cache_dir=args.cache_dir, embed_model=args.embed_model,
                       folds=args.folds, seed=args.seed)
    write_results(args.out, results)
    print(format_table(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
