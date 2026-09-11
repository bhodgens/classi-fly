"""Recompute the headline metrics from per-case data - independently.

Adversarial rule (leaf 08, Review Checklist): this module must NEVER call the
harness's own summary/aggregation code. It takes a ``results.json`` produced by
``tools/eval/run_eval.py``, walks the per-case ``predictions`` rows, and
recomputes P / C / E2E / OOD_R from the raw row flags with its own formulas.
A disagreement between the harness's reported number and the recomputed one is
a FINDING, never silently hidden.

Formulas (restated here from RESEARCH.md s1/s7, independent of the harness
source):
- P     = routed_correct / routes               (precision of direct routes)
- C     = routes / total                        (coverage)
- E2E   = (routed_correct + chain_baseline * abstained) / total
          DETERMINISTIC chain credit: every abstention is credited with the
          chain's measured accuracy. Never a sampled draw.
- OOD_R = ood_abstained / ood_total             (OOD abstain rate)

Route-count discipline: the recomputed ``routes`` count is always part of the
verdict line. A sub-one-case margin at n=48 routes is noise and is labeled as
such by ``verdict_lines``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

# The measured LLM-chain baseline the harness credits abstentions with.
# Read from results.json's chain_baseline when present; this constant is only
# the fallback for fixtures that omit it.
DEFAULT_CHAIN_BASELINE = 0.868

# RESEARCH.md s7 success bars, for the judge-mode verdict.
FLOOR_CHAIN = 0.868          # chain-only floor (meept campaign measurement)
BAR_EXISTING_VETO = 0.8735   # the existing veto's measured E2E
MIN_PRECISION = 0.97
MIN_OOD_RECALL = 0.95


def _mode_dict(results: dict):
    """Yield (mode_name, mode_block) for both single-mode and all-mode files."""
    if "modes" in results:
        for name, block in results["modes"].items():
            yield str(name), block
        return
    yield str(results.get("mode", "unknown")), results


def _row_flags(row: dict) -> tuple:
    """Normalize one prediction row to (abstained, pred_ok, ood).

    The harness writes rows as
    ``{"pred", "gold", "abstained", "ood", ...}``. We recompute
    ``pred_ok = (pred == gold)`` ourselves from the stored strings instead of
    trusting any per-row "correct" field.
    """
    abstained = bool(row.get("abstained", False))
    pred_ok = (str(row.get("pred")) == str(row.get("gold"))) and not abstained
    ood = bool(row.get("ood", False))
    return abstained, pred_ok, ood


def recompute(rows, *, total=None, chain_baseline: float = DEFAULT_CHAIN_BASELINE) -> dict:
    """Recompute P/C/E2E/OOD_R from per-case prediction rows.

    ``rows``: iterable of prediction dicts (abstained/pred/gold/ood flags).
    ``total``: the denominator; defaults to len(rows). It must equal the
    harness's ``total`` - a mismatch is a finding, so ``recompute_results``
    checks it explicitly.
    ``chain_baseline``: the deterministic credit per abstention.
    """
    routes = 0
    routed_correct = 0
    abstained = 0
    ood_total = 0
    ood_abstained = 0
    n = 0
    for row in rows:
        n += 1
        is_abs, ok, is_ood = _row_flags(row)
        if is_abs:
            abstained += 1
            if is_ood:
                ood_abstained += 1
        else:
            routes += 1
            if ok:
                routed_correct += 1
        if is_ood:
            ood_total += 1
    if total is None:
        total = n
    return {
        "P": (routed_correct / routes) if routes else 0.0,
        "C": (routes / total) if total else 0.0,
        "E2E": ((routed_correct + chain_baseline * abstained) / total) if total else 0.0,
        "OOD_R": (ood_abstained / ood_total) if ood_total else 0.0,
        "routes": routes,
        "total": total,
        "rows_seen": n,
        "gate_correct": routed_correct,
        "abstained": abstained,
        "ood_total": ood_total,
        "ood_abstained": ood_abstained,
    }


def _approx_eq(a: float, b: float, tol: float = 1e-9) -> bool:
    return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)


def recompute_results(results: dict, *, chain_baseline=None) -> list:
    """Recompute every mode in a parsed results.json; returns finding rows.

    Each row: {mode, metric, harness, recomputed, agree, routes, total, ...}.
    Differences (beyond float-printing noise) are findings with ``agree=False``.
    Structural problems (missing predictions, wrong total) are findings too.
    """
    cb = results.get("chain_baseline") if chain_baseline is None else chain_baseline
    cb = DEFAULT_CHAIN_BASELINE if cb is None else float(cb)
    findings = []
    for mode, block in _mode_dict(results):
        preds = block.get("predictions")
        if not isinstance(preds, list):
            findings.append({
                "mode": mode, "metric": "predictions", "harness": None,
                "recomputed": None, "agree": False,
                "note": "no per-case predictions in results; headline is unreproducible",
                "routes": None, "total": block.get("total"),
            })
            continue
        total = block.get("total")
        rec = recompute(preds, total=total, chain_baseline=cb)
        # The route count MUST be disclosed and MUST match the recomputation.
        if rec["routes"] != block.get("routes"):
            findings.append({
                "mode": mode, "metric": "routes", "harness": block.get("routes"),
                "recomputed": rec["routes"], "agree": False,
                "note": "reported route count disagrees with the per-case data",
                "routes": rec["routes"], "total": rec["total"],
            })
        if rec["total"] != block.get("total"):
            findings.append({
                "mode": mode, "metric": "total", "harness": block.get("total"),
                "recomputed": rec["total"], "agree": False,
                "note": "reported total disagrees with the number of prediction rows",
                "routes": rec["routes"], "total": rec["total"],
            })
        for metric in ("P", "C", "E2E", "OOD_R"):
            if metric not in block:
                findings.append({
                    "mode": mode, "metric": metric, "harness": None,
                    "recomputed": rec[metric], "agree": False,
                    "note": "harness did not report this metric",
                    "routes": rec["routes"], "total": rec["total"],
                })
                continue
            agree = _approx_eq(block[metric], rec[metric])
            findings.append({
                "mode": mode, "metric": metric,
                "harness": block[metric], "recomputed": rec[metric],
                "agree": agree, "note": None if agree else "RECOMPUTED VALUE DIFFERS",
                "routes": rec["routes"], "total": rec["total"],
            })
        # Expose the recomputed route context on every metric row of the mode.
        for f in findings:
            if f["mode"] == mode and f["metric"] in ("P", "C", "E2E", "OOD_R"):
                f.setdefault("routes", rec["routes"])
                f.setdefault("total", rec["total"])
    return findings


def margin_label(routes: int, e2e: float, floor: float = FLOOR_CHAIN) -> str:
    """Route-count discipline: label sub-one-case margins as noise."""
    margin_cases = (e2e - floor) * routes
    if e2e <= floor:
        return "below the %.2f%% floor" % (floor * 100)
    if margin_cases < 1.0:
        return ("beats the floor by %.2f cases on %d routes - "
                "SUB-ONE-CASE MARGIN, noise at this n" % (margin_cases, routes))
    return "beats the floor by %.2f cases on %d routes" % (margin_cases, routes)


def verdict_lines(results: dict) -> list:
    """The headline verdicts: does judge mode clear every success bar?"""
    cb = float(results.get("chain_baseline", DEFAULT_CHAIN_BASELINE))
    lines = []
    for mode, block in _mode_dict(results):
        preds = block.get("predictions") or []
        rec = recompute(preds, total=block.get("total"), chain_baseline=cb)
        checks = [
            ("E2E > %.2f%% (chain floor)" % (FLOOR_CHAIN * 100),
             rec["E2E"] > FLOOR_CHAIN),
            ("E2E > %.2f%% (existing veto)" % (BAR_EXISTING_VETO * 100),
             rec["E2E"] > BAR_EXISTING_VETO),
            ("P >= %d%%" % (MIN_PRECISION * 100), rec["P"] >= MIN_PRECISION),
            ("OOD_R >= %d%%" % (MIN_OOD_RECALL * 100), rec["OOD_R"] >= MIN_OOD_RECALL),
        ]
        for label, ok in checks:
            lines.append("%s %s: %s" % (
                mode, label, "PASS" if ok else "FAIL"))
        lines.append("%s routes/total: %d/%d - %s" % (
            mode, rec["routes"], rec["total"],
            margin_label(rec["routes"], rec["E2E"])))
    return lines


def format_table(findings: list) -> str:
    """Printable harness-vs-recomputed comparison; any diff is visible."""
    lines = ["%-6s %-6s %12s %12s %8s %10s" %
             ("mode", "metric", "harness", "recomputed", "agree", "routes")]
    for f in findings:
        if f["metric"] not in ("P", "C", "E2E", "OOD_R"):
            continue
        h = "n/a" if f["harness"] is None else "%.6f" % f["harness"]
        r = "n/a" if f["recomputed"] is None else "%.6f" % f["recomputed"]
        lines.append("%-6s %-6s %12s %12s %8s %6d/%-4d" %
                     (f["mode"], f["metric"], h, r,
                      "yes" if f["agree"] else "NO",
                      f.get("routes") or 0, f.get("total") or 0))
    bad = [f for f in findings if not f["agree"]]
    if bad:
        lines.append("")
        lines.append("DISAGREEMENTS (%d):" % len(bad))
        for f in bad:
            lines.append("  %s/%s: harness=%r recomputed=%r (%s)" %
                         (f["mode"], f["metric"], f["harness"],
                          f["recomputed"], f["note"]))
    else:
        lines.append("all metrics agree to 1e-9")
    return "\n".join(lines)


def recompute_file(path, *, chain_baseline=None) -> list:
    """Load a results.json from disk and recompute it. Returns findings."""
    results = json.loads(Path(path).read_text(encoding="utf-8"))
    return recompute_results(results, chain_baseline=chain_baseline)


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(
        description="Recompute P/C/E2E/OOD_R from a results.json (independent formulas)")
    p.add_argument("results", help="path to results.json")
    p.add_argument("--chain-baseline", type=float, default=None,
                   help="override the chain baseline credit (default: the file's own value)")
    args = p.parse_args(argv)
    findings = recompute_file(args.results, chain_baseline=args.chain_baseline)
    print(format_table(findings))
    print()
    results = json.loads(Path(args.results).read_text(encoding="utf-8"))
    for line in verdict_lines(results):
        print(line)
    return 0 if all(f["agree"] for f in findings) else 1


if __name__ == "__main__":
    raise SystemExit(main())
