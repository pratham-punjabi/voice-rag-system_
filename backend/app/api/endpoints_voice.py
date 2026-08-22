from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from backend.app.api.dependencies import get_orchestrator

router = APIRouter(prefix="/api/voice", tags=["voice"])
logger = logging.getLogger(__name__)


@router.post("/query", summary="Voice query with audio file")
async def voice_query(
    audio: UploadFile = File(..., description="WAV/WebM audio file"),
    sample_rate: int = Form(default=16000),
    orchestrator=Depends(get_orchestrator),
):
    """
    Submit a voice query. The audio is sent to Sarvam STT for transcription,
    then processed through the RAG pipeline.
    """
    audio_bytes = await audio.read()
    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty audio file")

    result = await orchestrator.process_voice(audio_bytes, sample_rate=sample_rate)
    return result


@router.post("/transcript", summary="Submit pre-transcribed text from browser STT")
async def voice_transcript(
    transcript: str = Form(..., description="Text transcribed by browser Web Speech API"),
    orchestrator=Depends(get_orchestrator),
):
    """
    Submit text already transcribed by the browser's Web Speech API.
    This bypasses server-side STT (Sarvam) and runs the query directly.
    Used when no Sarvam API key is configured.
    """
    if not transcript or not transcript.strip():
        raise HTTPException(status_code=400, detail="Empty transcript")

    result = await orchestrator.process_text(transcript.strip())
    result["source"] = "browser_stt"
    return result
