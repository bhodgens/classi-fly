"""The three baseline heads.

Shared shape: ``fit(states, labels)`` / ``predict(state) -> (label, conf, margin)``.

- ``centroid_margin``: nearest class centroid by cosine; margin = top1 - top2
  cosine (non-negative by construction - the abstain gate uses thresholds,
  not the margin sign); conf = softmax over cosines at temperature 0.1.
  States are embedding vectors.
- ``knn_unanimity``: cosine kNN with k=5; conf = majority vote share
  (1.0 = unanimity); margin = (majority - runner-up) / k. States are
  embedding vectors.
- ``tfidf_logistic``: char 2-4 gram TF-IDF + logistic regression (sklearn).
  States are raw text.
"""
from __future__ import annotations

import numpy as np

# Cosine temperature: well-separated clusters -> conf near 1.
_SOFTMAX_T = 0.1


def _cos(mat: np.ndarray, v: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64).ravel()
    mn = np.maximum(np.linalg.norm(mat, axis=1, keepdims=True), 1e-12)
    vn = max(float(np.linalg.norm(v)), 1e-12)
    return (mat / mn) @ (v / vn)


class centroid_margin:
    def fit(self, states, labels):
        states = np.asarray(states, dtype=np.float64)
        self._labels = [str(l) for l in labels]
        self._classes = sorted(set(self._labels))
        rows = []
        for c in self._classes:
            idx = [i for i, l in enumerate(self._labels) if l == c]
            rows.append(states[idx].mean(axis=0))
        self._cent = np.vstack(rows)
        return self

    def predict(self, state):
        if not hasattr(self, "_cent"):
            raise RuntimeError("centroid_margin: predict before fit")
        sims = _cos(self._cent, state)
        order = np.argsort(-sims)
        top = int(order[0])
        if len(order) > 1:
            margin = float(sims[top] - sims[int(order[1])])
        else:
            margin = 1.0
        e = np.exp((sims - sims.max()) / _SOFTMAX_T)
        conf = float(e[top] / e.sum())
        return self._classes[top], conf, margin


class knn_unanimity:
    def __init__(self, k: int = 5):
        self.k = k

    def fit(self, states, labels):
        self._st = np.asarray(states, dtype=np.float64)
        self._labels = [str(l) for l in labels]
        return self

    def predict(self, state):
        if not hasattr(self, "_st"):
            raise RuntimeError("knn_unanimity: predict before fit")
        sims = _cos(self._st, state)
        kk = min(self.k, len(self._labels))
        top = np.argsort(-sims)[:kk]
        counts = {}
        for i in top:
            lbl = self._labels[int(i)]
            counts[lbl] = counts.get(lbl, 0) + 1
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        label, maj = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else 0
        return label, maj / kk, (maj - second) / kk


class tfidf_logistic:
    def fit(self, states, labels):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        self._vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 4),
                                    lowercase=True, min_df=1)
        x = self._vec.fit_transform([str(s) for s in states])
        self._clf = LogisticRegression(max_iter=1000, random_state=0).fit(x, [str(l) for l in labels])
        return self

    def predict(self, state):
        if not hasattr(self, "_clf"):
            raise RuntimeError("tfidf_logistic: predict before fit")
        x = self._vec.transform([str(state)])
        proba = self._clf.predict_proba(x)[0]
        order = np.argsort(-proba)
        classes = [str(c) for c in self._clf.classes_]
        margin = (float(proba[order[0]] - proba[order[1]])
                  if len(order) > 1 else float(proba[order[0]]))
        return classes[int(order[0])], float(proba[order[0]]), margin
