from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class ChromaVectorStore:
    """
    Production ChromaDB vector store with persistent storage.
    Supports upsert, search by embedding, and collection management.
    """

    def __init__(
        self,
        persist_dir: str = "data/indexes/chromadb",
        collection_name: str = "rag_documents",
        dimension: int = 768,
    ) -> None:
        self._persist_dir = persist_dir
        self._collection_name = collection_name
        self._dimension = dimension
        self._client = None
        self._collection = None

    def _get_client(self):
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
            if self._client is None:
                Path(self._persist_dir).mkdir(parents=True, exist_ok=True)
                self._client = chromadb.PersistentClient(
                    path=self._persist_dir,
                    settings=ChromaSettings(anonymized_telemetry=False),
                )
            return self._client
        except ImportError as exc:
            raise RuntimeError("chromadb not installed. Run: pip install chromadb") from exc

    def _get_collection(self):
        if self._collection is None:
            client = self._get_client()
            self._collection = client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def upsert(
        self,
        ids: list[str],
        embeddings: np.ndarray,
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Upsert chunks into ChromaDB."""
        collection = self._get_collection()
        emb_list = embeddings.tolist() if hasattr(embeddings, "tolist") else list(embeddings)

        # ChromaDB requires metadata values to be str/int/float/bool only
        safe_metadatas = []
        for m in metadatas:
            safe = {
                k: (str(v) if not isinstance(v, (str, int, float, bool)) else v)
                for k, v in m.items()
                if v is not None
            }
            safe_metadatas.append(safe)

        # Batch upsert in chunks of 1000
        batch_size = 1000
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i : i + batch_size]
            batch_embs = emb_list[i : i + batch_size]
            batch_docs = documents[i : i + batch_size]
            batch_metas = safe_metadatas[i : i + batch_size]
            collection.upsert(
                ids=batch_ids,
                embeddings=batch_embs,
                documents=batch_docs,
                metadatas=batch_metas,
            )
        logger.info("ChromaDB upserted %d chunks into '%s'", len(ids), self._collection_name)

    def search(
        self, query_vec: np.ndarray, top_k: int = 20
    ) -> list[tuple[str, float, str, dict]]:
        """
        Search ChromaDB by embedding similarity.
        Returns [(chunk_id, score, document_text, metadata)].
        """
        collection = self._get_collection()
        if collection.count() == 0:
            return []

        query_list = query_vec.tolist() if hasattr(query_vec, "tolist") else list(query_vec)
        results = collection.query(
            query_embeddings=[query_list],
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        output = []
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        for cid, dist, doc, meta in zip(ids, distances, documents, metadatas):
            # Chroma cosine distance: 0=identical, 2=opposite → convert to similarity
            score = 1.0 - (dist / 2.0)
            output.append((cid, score, doc or "", meta or {}))

        return sorted(output, key=lambda x: x[1], reverse=True)

    def count(self) -> int:
        try:
            return self._get_collection().count()
        except Exception:
            return 0

    def delete_collection(self) -> None:
        client = self._get_client()
        try:
            client.delete_collection(self._collection_name)
            self._collection = None
            logger.info("ChromaDB collection '%s' deleted", self._collection_name)
        except Exception as exc:
            logger.warning("Could not delete collection: %s", exc)

    @property
    def is_loaded(self) -> bool:
        try:
            return self._get_collection().count() > 0
        except Exception:
            return False
