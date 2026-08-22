from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from pydantic import BaseModel, Field

from backend.app.core.config import get_settings

router = APIRouter(prefix="/api/ingest", tags=["ingestion"])
logger = logging.getLogger(__name__)


class DocumentItem(BaseModel):
    id: str | None = None
    text: str = Field(..., min_length=10)
    title: str = ""
    metadata: dict = {}


class IngestRequest(BaseModel):
    documents: list[DocumentItem]
    replace: bool = Field(False, description="If true, delete existing collection before ingesting")


class IngestResponse(BaseModel):
    status: str
    total_documents: int
    total_chunks: int
    chroma_collection: str
    message: str = ""


@router.post("/documents", response_model=IngestResponse, summary="Ingest documents into ChromaDB")
async def ingest_documents(body: IngestRequest):
    """
    Ingest custom documents into ChromaDB vector store.
    These documents will be retrievable by the RAG pipeline in 'dataset' or 'hybrid' mode.
    """
    from backend.app.ingestion.indexer import ingest_documents as _ingest

    settings = get_settings()
    docs = [
        {
            "id": d.id or f"doc_{i}",
            "text": d.text,
            "title": d.title,
            "metadata": d.metadata,
        }
        for i, d in enumerate(body.documents)
    ]

    try:
        result = await _ingest(docs, settings, replace=body.replace)
        return IngestResponse(
            status=result["status"],
            total_documents=result["total_documents"],
            total_chunks=result["total_chunks"],
            chroma_collection=result["chroma_collection"],
            message=f"Successfully ingested {result['total_chunks']} chunks into ChromaDB",
        )
    except Exception as exc:
        logger.exception("Ingestion failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(exc)}")


@router.post("/dataset", summary="Run dataset ingestion from HuggingFace")
async def ingest_dataset():
    """
    Trigger full dataset ingestion from HuggingFace (configured by DATASET_NAME env).
    Long-running operation — runs in background.
    """
    from backend.app.ingestion.indexer import run_ingestion
    settings = get_settings()
    try:
        stats = await run_ingestion(settings)
        return {
            "status": "success",
            "report": stats.report(),
        }
    except Exception as exc:
        logger.exception("Dataset ingestion failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Dataset ingestion failed: {str(exc)}")


@router.get("/status", summary="ChromaDB collection status")
async def ingest_status():
    """Check how many documents are currently indexed in ChromaDB."""
    from backend.app.retrieval.chroma_store import ChromaVectorStore
    settings = get_settings()
    try:
        store = ChromaVectorStore(
            persist_dir=settings.chroma_persist_dir,
            collection_name=settings.chroma_collection_name,
        )
        count = store.count()
        return {
            "chroma_collection": settings.chroma_collection_name,
            "chroma_persist_dir": settings.chroma_persist_dir,
            "total_chunks": count,
            "has_data": count > 0,
            "data_source_mode": settings.data_source_mode,
        }
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}


@router.delete("/collection", summary="Clear ChromaDB collection")
async def clear_collection():
    """Delete all documents from the ChromaDB collection."""
    from backend.app.retrieval.chroma_store import ChromaVectorStore
    settings = get_settings()
    try:
        store = ChromaVectorStore(
            persist_dir=settings.chroma_persist_dir,
            collection_name=settings.chroma_collection_name,
        )
        store.delete_collection()
        return {"status": "deleted", "collection": settings.chroma_collection_name}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
