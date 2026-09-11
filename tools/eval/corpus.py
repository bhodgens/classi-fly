"""Corpus loading (JSON / JSON5 subset) with the OOD flag preserved.

Corpus format: a list of
``{"text": "...", "intent": "...", "ood": false, "source": "..."}``.
``ood`` defaults to False. OOD items are NEVER dropped: the harness scores
them (their abstentions feed OOD_R).
"""
from __future__ import annotations

import json
from pathlib import Path


def _strip_json5(text: str) -> str:
    """Remove // and /* */ comments and trailing commas (JSON5 subset)."""
    out = []
    i, n = 0, len(text)
    in_str = False
    while i < n:
        ch = text[i]
        if in_str:
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 2
                continue
            if ch == '"':
                in_str = False
            i += 1
            continue
        if ch == '"':
            in_str = True
            out.append(ch)
            i += 1
            continue
        if ch == "/" and i + 1 < n and text[i + 1] == "/":
            while i < n and text[i] != "\n":
                i += 1
            continue
        if ch == "/" and i + 1 < n and text[i + 1] == "*":
            i += 2
            while i + 1 < n and not (text[i] == "*" and text[i + 1] == "/"):
                i += 1
            i += 2
            continue
        out.append(ch)
        i += 1
    s = "".join(out)
    keep = []
    for i, ch in enumerate(s):
        if ch == ",":
            j = i + 1
            while j < len(s) and s[j] in " \t\r\n":
                j += 1
            if j < len(s) and s[j] in "}]":
                continue
        keep.append(ch)
    return "".join(keep)


def loads_lenient(text: str):
    """Parse strict JSON, falling back to a JSON5 subset (comments/trailing commas)."""
    try:
        return json.loads(text)
    except ValueError:
        return json.loads(_strip_json5(text))


def load_corpus(path) -> list:
    """Load corpus items, normalizing types; keeps the ood flag (default False)."""
    items = loads_lenient(Path(path).read_text(encoding="utf-8"))
    if not isinstance(items, list):
        raise ValueError("corpus: expected a list of items in %s" % path)
    norm = []
    for idx, it in enumerate(items):
        if not isinstance(it, dict) or "text" not in it or "intent" not in it:
            raise ValueError("corpus: item %d missing text/intent" % idx)
        norm.append({
            "text": str(it["text"]),
            "intent": str(it["intent"]),
            "ood": bool(it.get("ood", False)),
            "source": str(it.get("source", "")),
        })
    return norm
