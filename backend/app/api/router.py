from fastapi import APIRouter

from backend.app.api.endpoints_health import router as health_router
from backend.app.api.endpoints_query import router as query_router
from backend.app.api.endpoints_voice import router as voice_router
from backend.app.api.endpoints_ingest import router as ingest_router
from backend.app.api.endpoints_metrics import router as metrics_router

api_router = APIRouter()
api_router.include_router(query_router)
api_router.include_router(voice_router)
api_router.include_router(health_router)
api_router.include_router(ingest_router)
api_router.include_router(metrics_router)
