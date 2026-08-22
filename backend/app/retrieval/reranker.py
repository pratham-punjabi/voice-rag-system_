from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Lightweight cross-encoder reranker for precision.
    Uses ms-marco-MiniLM-L-6-v2 by default (~12ms per batch of 20 on CPU).
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        max_length: int = 512,
    ) -> None:
        self._model_name = model_name
        self._max_length = max_length
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._model_name, max_length=self._max_length)
            logger.info("CrossEncoder loaded: %s", self._model_name)
        except Exception as exc:
            logger.warning("CrossEncoder unavailable: %s. Falling back to score order.", exc)

    async def warmup(self) -> None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load)
        if self._model:
            await self.rerank("warmup query", [("dummy", "warmup text", 0.5)])

    async def rerank(
        self,
        query: str,
        candidates: list[tuple[str, str, float]],
    ) -> list[tuple[str, float]]:
        """
        Args:
            query: user query string
            candidates: [(chunk_id, text, initial_score)]
        Returns:
            [(chunk_id, rerank_score)] sorted descending
        """
        if not candidates:
            return []
        if self._model is None:
            self._load()
        if self._model is None:
            # Fallback: return original order
            return [(cid, score) for cid, _, score in candidates]

        pairs = [(query, text[:self._max_length]) for _, text, _ in candidates]
        loop = asyncio.get_event_loop()
        scores = await loop.run_in_executor(
            None,
            lambda: self._model.predict(pairs, show_progress_bar=False),
        )
        ranked = sorted(
            zip([cid for cid, _, _ in candidates], scores.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked

    @property
    def available(self) -> bool:
        return self._model is not None
