from __future__ import annotations

import logging
import math
import pickle
import re
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

_TOKENIZE_RE = re.compile(r"[^\w\s]", re.UNICODE)


def tokenize(text: str) -> list[str]:
    text = _TOKENIZE_RE.sub(" ", text.lower())
    return [t for t in text.split() if len(t) > 1]


class BM25Index:
    """
    BM25 Okapi implementation — no external deps.
    k1=1.5, b=0.75 by default (Sarvam-style for mixed-language docs).
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._ids: list[str] = []
        self._tokenized: list[list[str]] = []
        self._df: dict[str, int] = defaultdict(int)
        self._avgdl: float = 0.0
        self._n: int = 0

    def build(self, texts: list[str], ids: list[str]) -> None:
        assert len(texts) == len(ids)
        self._ids = ids
        self._tokenized = [tokenize(t) for t in texts]
        self._n = len(texts)
        self._avgdl = sum(len(t) for t in self._tokenized) / max(1, self._n)
        self._df = defaultdict(int)
        for tokens in self._tokenized:
            for term in set(tokens):
                self._df[term] += 1
        logger.info("BM25 index built: %d docs, vocab=%d", self._n, len(self._df))

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        query_terms = tokenize(query)
        if not query_terms or self._n == 0:
            return []

        scores: list[float] = []
        for doc_tokens in self._tokenized:
            dl = len(doc_tokens)
            tf_map: dict[str, int] = defaultdict(int)
            for t in doc_tokens:
                tf_map[t] += 1

            score = 0.0
            for term in query_terms:
                if term not in self._df:
                    continue
                tf = tf_map.get(term, 0)
                idf = math.log((self._n - self._df[term] + 0.5) / (self._df[term] + 0.5) + 1)
                norm_tf = (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * dl / self._avgdl)
                )
                score += idf * norm_tf
            scores.append(score)

        top_indices = sorted(range(self._n), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self._ids[i], scores[i]) for i in top_indices if scores[i] > 0]

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump(self, f)
        logger.info("BM25 index saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "BM25Index":
        with open(path, "rb") as f:
            idx = pickle.load(f)
        logger.info("BM25 index loaded: %d docs", idx._n)
        return idx
