from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from backend.app.agents.orchestrator import Orchestrator
from backend.app.retrieval.cache import QueryCache


def get_orchestrator(request: Request) -> Orchestrator:
    return request.app.state.orchestrator


def get_cache(request: Request) -> QueryCache:
    return request.app.state.query_cache


OrchestratorDep = Annotated[Orchestrator, Depends(get_orchestrator)]
CacheDep = Annotated[QueryCache, Depends(get_cache)]
