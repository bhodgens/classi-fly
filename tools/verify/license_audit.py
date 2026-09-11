"""License provenance audit: can any non-commercial data reach a shippable artifact?

Walks every `.fly` artifact under ``testdata/`` trees and ``tools/`` (plus any
extra directories passed in), reads the CONTAINER HEADER (the authoritative
provenance: ``license`` and ``source``/``attribution`` fields per Contract 1),
and classifies each artifact:

- SHIPPABLE       - license is a known commercial-OK license (synthetic/none,
                    CC0, CC-BY family) AND the NOTICE covers its source.
- BENCHMARK-ONLY  - license is a known non-commercial license (CC BY-NC).
- UNKNOWN         - license string not in the policy table: NOT shippable by
                    default (fail-closed).

Then it cross-checks the distinct sources/attributions found in artifacts
against the repo NOTICE: every distinct non-synthetic source found must be
listed in NOTICE. A missing NOTICE entry is a finding.

This module is independent of tools/ingest/registry.py: the license policy is
restated here (a check that imports the producer's own policy table could be
silently relaxed by the producer).
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

MAGIC = b"FLYRES01"
ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"

# License policy, restated independently from NOTICE + docs/INTEGRATION.md:
#   synthetic/none/CC0/CC-BY*  -> commercial OK (attribution required for CC-BY)
#   CC BY-NC (any variant)     -> benchmark only, never ship
# Anything else -> UNKNOWN -> fail-closed (not shippable) until policy decides.
COMMERCIAL_OK = {
    "none", "", "cc0-1.0", "cc-by-4.0", "cc-by", "cc-by-3.0", "cc-by-sa-4.0",
    "mit", "apache-2.0", "public-domain", "synthetic",
}
NON_COMMERCIAL = {
    "cc-by-nc-4.0", "cc-by-nc", "cc-by-nc-sa-4.0", "cc-by-nc-nd-4.0",
    "research-only", "non-commercial",
}

# NOTICE substring keys -> the source families NOTICE must cover.
NOTICE_COVERAGE = {
    "hemibrain": "Hemibrain connectome",
    "larval": "Larval connectome",
    "flywire": "FlyWire-derived data",
    "synthetic": "Synthetic reservoirs",
}


def read_fly_header(path) -> dict:
    """Read ONLY the header of a .fly file (zstd frames decompressed if possible)."""
    raw = Path(path).read_bytes()
    if raw[:4] == ZSTD_MAGIC:
        try:
            from compression import zstd as _z  # Python 3.14+ stdlib
            raw = _z.decompress(raw)
        except ImportError:
            try:
                import zstandard
                raw = zstandard.ZstdDecompressor().decompress(raw)
            except ImportError:
                raise RuntimeError(
                    "license_audit: %s is zstd-compressed but no zstd codec is "
                    "importable; cannot read header" % path)
    if raw[:8] != MAGIC:
        raise ValueError("license_audit: %s is not a FLYRES01 artifact" % path)
    (header_len,) = struct.unpack_from("<I", raw, 8)
    header = json.loads(raw[12:12 + header_len].decode("utf-8"))
    if not isinstance(header, dict):
        raise ValueError("license_audit: %s header is not an object" % path)
    return header


def classify_license(license_str: str) -> str:
    """SHIPPABLE | BENCHMARK-ONLY | UNKNOWN (fail-closed)."""
    lic = str(license_str or "").strip().lower()
    if lic in NON_COMMERCIAL:
        return "BENCHMARK-ONLY"
    if lic in COMMERCIAL_OK:
        return "SHIPPABLE"
    # CC-BY variants not in the table: CC-BY family is attribution-licensed.
    if lic.startswith("cc-by") and "nc" not in lic and "nd" not in lic:
        return "SHIPPABLE"
    return "UNKNOWN"


def _notice_key(source: str) -> str:
    """Map an artifact's source string to the NOTICE source family it belongs to."""
    for key in NOTICE_COVERAGE:
        if key in source:
            return key
    return source


def _artifact_source(header: dict) -> str:
    src = str(header.get("source", "") or "").lower()
    if src:
        return src
    name = str(header.get("name", "") or "").lower()
    for key in NOTICE_COVERAGE:
        if key in name:
            return key
    return "unknown"


def find_fly_files(extra_dirs=()) -> list:
    """Every .fly under the repo's testdata trees and tools/ (+ extra dirs)."""
    here = Path(__file__).resolve().parent          # tools/verify
    roots = [
        here.parent / "eval" / "testdata",
        here.parent / "train" / "testdata",
        Path.cwd() / "testdata",
        here.parent,
    ]
    roots.extend(Path(d) for d in extra_dirs)
    seen = set()
    out = []
    for root in roots:
        if not root.exists():
            continue
        for p in sorted(root.rglob("*.fly")):
            rp = p.resolve()
            if rp not in seen and "__pycache__" not in str(rp):
                seen.add(rp)
                out.append(rp)
    return out


def audit(extra_dirs=(), notice_path=None) -> dict:
    """Walk the artifacts, classify, cross-check NOTICE. Returns the report."""
    here = Path(__file__).resolve().parent
    notice_path = Path(notice_path) if notice_path else here.parent.parent / "NOTICE"
    notice_text = notice_path.read_text(encoding="utf-8") if notice_path.exists() else ""
    if not notice_text:
        raise FileNotFoundError("license_audit: NOTICE not found at %s" % notice_path)

    artifacts = []
    for path in find_fly_files(extra_dirs):
        header = read_fly_header(path)
        lic = header.get("license", "")
        verdict = classify_license(lic)
        source = _artifact_source(header)
        notice_key = _notice_key(source)
        notice_hit = NOTICE_COVERAGE[notice_key] in notice_text if notice_key in NOTICE_COVERAGE else False
        attribution = header.get("attribution", "")
        if verdict == "SHIPPABLE" and str(lic).lower().startswith("cc-by"):
            # CC-BY shipping requires the attribution to be carried in the header.
            if not str(attribution or "").strip():
                verdict = "UNKNOWN"  # CC-BY without attribution is not shippable
        artifacts.append({
            "path": str(path),
            "name": header.get("name", ""),
            "license": lic,
            "source": source,
            "attribution": attribution,
            "verdict": verdict,
            "notice_covers_source": notice_hit,
        })

    distinct_sources = sorted({a["source"] for a in artifacts})
    notice_findings = []
    for src in distinct_sources:
        covered = [a for a in artifacts
                   if _notice_key(a["source"]) == _notice_key(src)
                   and a["notice_covers_source"]]
        if _notice_key(src) in NOTICE_COVERAGE and not covered:
            notice_findings.append(
                "NOTICE does not mention source %r (required by artifacts above)" % src)

    nonshippable = [a for a in artifacts if a["verdict"] != "SHIPPABLE"]
    return {
        "artifacts": artifacts,
        "distinct_sources": distinct_sources,
        "notice_findings": notice_findings,
        "shippable_count": len(artifacts) - len(nonshippable),
        "nonshippable_count": len(nonshippable),
        "ok": all(a["verdict"] == "SHIPPABLE" for a in artifacts) and not notice_findings,
    }


def format_report(rep: dict) -> str:
    lines = ["license audit: %d artifact(s)" % len(rep["artifacts"]),
             "%-52s %-16s %-12s %s" % ("path", "license", "verdict", "notice")]
    for a in rep["artifacts"]:
        p = a["path"]
        p = p if len(p) <= 50 else "..." + p[-47:]
        lines.append("%-52s %-16s %-12s %s" % (
            p, a["license"] or "(none)", a["verdict"],
            "yes" if a["notice_covers_source"] else "MISSING"))
    lines.append("distinct sources: %s" % ", ".join(rep["distinct_sources"]))
    for f in rep["notice_findings"]:
        lines.append("FINDING: " + f)
    lines.append("VERDICT: %s" % (
        "PASS - no non-commercial artifact is shippable and NOTICE covers every source"
        if not rep["nonshippable_count"] and not rep["notice_findings"]
        else ("FAIL - %d artifact(s) not shippable" % rep["nonshippable_count"]
              if rep["nonshippable_count"] else "FAIL - NOTICE gaps")))
    return "\n".join(lines)


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="License-provenance audit of .fly artifacts")
    p.add_argument("--dir", action="append", default=[],
                   help="extra directory to walk (repeatable)")
    p.add_argument("--notice", default=None, help="NOTICE path override")
    args = p.parse_args(argv)
    rep = audit(extra_dirs=args.dir, notice_path=args.notice)
    print(format_report(rep))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
