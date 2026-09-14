"""Robustness check for the size-sweep conclusion: syn512 at blink3 across
3 graph seeds, plus N=1536 and N=2048 to locate the size threshold."""
import json
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "control"))
import lane1_hardening as lh  # noqa: E402
import reservoir_imitation as ri  # noqa: E402

out = []
for n, seed in [(512, 123), (512, 124), (512, 125), (1536, 123), (2048, 123)]:
    W, proj = lh.synthetic_reservoir(n, seed=seed)
    rows = lh.train_and_eval(W, proj, 3, f"syn{n}_s{seed}")
    out.extend(rows)
p = REPO / "tools" / "control" / "lane1_size_robustness.json"
p.write_text(json.dumps(out, indent=2) + "\n")
print("wrote", p)
