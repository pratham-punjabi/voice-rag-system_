from __future__ import annotations


class RAGBaseError(Exception):
    """Base for all application errors."""
    def __init__(self, message: str, code: str = "INTERNAL_ERROR") -> None:
        super().__init__(message)
        self.message = message
        self.code = code


# ── STT ───────────────────────────────────────────────────────────────────────
class STTError(RAGBaseError):
    pass

class STTConnectionError(STTError):
    def __init__(self, msg: str = "STT service unavailable") -> None:
        super().__init__(msg, "STT_CONNECTION_ERROR")

class STTTimeoutError(STTError):
    def __init__(self, msg: str = "STT request timed out") -> None:
        super().__init__(msg, "STT_TIMEOUT")

class STTEmptyAudioError(STTError):
    def __init__(self, msg: str = "No audio detected") -> None:
        super().__init__(msg, "STT_EMPTY_AUDIO")

class STTEmptyTranscriptError(STTError):
    def __init__(self, msg: str = "Transcription produced empty result") -> None:
        super().__init__(msg, "STT_EMPTY_TRANSCRIPT")

class STTInvalidAPIKeyError(STTError):
    def __init__(self, msg: str = "Invalid Sarvam API key") -> None:
        super().__init__(msg, "STT_INVALID_API_KEY")


# ── Query ─────────────────────────────────────────────────────────────────────
class QueryError(RAGBaseError):
    pass

class InvalidQueryError(QueryError):
    def __init__(self, msg: str = "Query is invalid or empty") -> None:
        super().__init__(msg, "INVALID_QUERY")

class UnsafeQueryError(QueryError):
    def __init__(self, msg: str = "Query flagged as unsafe") -> None:
        super().__init__(msg, "UNSAFE_QUERY")

class PromptInjectionError(QueryError):
    def __init__(self, msg: str = "Possible prompt injection detected") -> None:
        super().__init__(msg, "PROMPT_INJECTION")


# ── Retrieval ─────────────────────────────────────────────────────────────────
class RetrievalError(RAGBaseError):
    pass

class VectorDBUnavailableError(RetrievalError):
    def __init__(self, msg: str = "Vector database unavailable") -> None:
        super().__init__(msg, "VECTOR_DB_UNAVAILABLE")

class IndexNotFoundError(RetrievalError):
    def __init__(self, msg: str = "Index not found. Run ingestion first.") -> None:
        super().__init__(msg, "INDEX_NOT_FOUND")

class LowConfidenceError(RetrievalError):
    def __init__(self, msg: str = "Retrieved documents have insufficient confidence") -> None:
        super().__init__(msg, "LOW_CONFIDENCE")

class NoDocumentsRetrievedError(RetrievalError):
    def __init__(self, msg: str = "No relevant documents found") -> None:
        super().__init__(msg, "NO_DOCUMENTS")


# ── Generation ────────────────────────────────────────────────────────────────
class GenerationError(RAGBaseError):
    pass

class LLMUnavailableError(GenerationError):
    def __init__(self, msg: str = "LLM provider unavailable") -> None:
        super().__init__(msg, "LLM_UNAVAILABLE")

class LLMTimeoutError(GenerationError):
    def __init__(self, msg: str = "LLM generation timed out") -> None:
        super().__init__(msg, "LLM_TIMEOUT")

class LLMMalformedResponseError(GenerationError):
    def __init__(self, msg: str = "LLM returned malformed response") -> None:
        super().__init__(msg, "LLM_MALFORMED_RESPONSE")

class HallucinationDetectedError(GenerationError):
    def __init__(self, msg: str = "Hallucination detected in generated answer") -> None:
        super().__init__(msg, "HALLUCINATION_DETECTED")


# ── Dataset / Ingestion ───────────────────────────────────────────────────────
class DatasetError(RAGBaseError):
    pass

class DatasetDownloadError(DatasetError):
    def __init__(self, msg: str = "Failed to download dataset") -> None:
        super().__init__(msg, "DATASET_DOWNLOAD_ERROR")

class InvalidDatasetSchemaError(DatasetError):
    def __init__(self, msg: str = "Dataset schema does not match expected format") -> None:
        super().__init__(msg, "INVALID_DATASET_SCHEMA")


# ── Rate Limit ────────────────────────────────────────────────────────────────
class RateLimitError(RAGBaseError):
    def __init__(self, msg: str = "Rate limit exceeded") -> None:
        super().__init__(msg, "RATE_LIMIT_EXCEEDED")
