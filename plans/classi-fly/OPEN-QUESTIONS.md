# classi-fly - Open Questions

Owner decisions. Q1, Q2, Q4 DECIDED 2026-09-11 by the owner; the rest default
to the recommendations below and are treated as accepted.

Summary: 6 questions, 3 were decisions - ALL RESOLVED. Tree is cleared to
execute.

---

**Q1 - Consumer interface: Go import, sidecar, or both? - DECIDED: BOTH.**
- Owner: 2026-09-11. Rationale: the sidecar is needed for testing in this repo.
- Consequence: `07-integration.md` builds both faces as planned; the exported
  Go API (Contract 2) stays public. No tree changes.

**Q2 - Is the offline tooling allowed to be Python? - DECIDED: YES.**
- Owner: 2026-09-11. Runtime stays Go; ingest/train/eval are Python 3.12.
- Consequence: leaves 02/03/04/05 proceed as authored (Python tooling). No tree
  changes.

**Q3 - Which connectome first? - DEFAULTED: larval.**
- Rec accepted by default: larval (CC BY 4.0, 3,016 neurons, one file,
  published working core). Hemibrain later. See RESEARCH.md §4.
- Requires owner approval of the ~35 MB download when Tier 1 starts.

**Q4 - Is the synthetic reservoir the shipped default? - DECIDED: YES.**
- Owner: 2026-09-11. Ship synthetic by default; switch to a connectome only if
  the harness shows it beats synthetic (experiment E3).
- Consequence: `08-verification.md` keeps the license audit as a check, not a
  CI gate; attribution stays optional-to-release.

**Q5 - Where do the plans live? - DEFAULTED: plans/classi-fly/ in this repo.**

**Q6 - Should the meept adapter be built now or deferred? - DEFAULTED: DEFER.**
- classi-fly ships independently; `07-integration.md` documents the generic
  judge pattern with meept as an illustrative instance.
