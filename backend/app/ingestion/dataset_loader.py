from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

logger = logging.getLogger(__name__)


@dataclass
class DatasetSchema:
    text_field: str
    id_field: str
    title_field: str | None
    language_field: str | None
    query_field: str | None
    relevance_field: str | None
    all_fields: list[str]
    total_rows: int
    sample_languages: list[str]


def inspect_schema(dataset) -> DatasetSchema:
    """Detect dataset schema without loading the full dataset into RAM."""

    # Streaming datasets expose column information through features/info
    features = getattr(dataset, "features", None)

    if features is None:
        raise RuntimeError(
            "Unable to detect dataset schema from streaming dataset."
        )

    all_fields = list(features.keys())

    logger.info("Dataset fields detected: %s", all_fields)

    # Text field
    text_candidates = [
        "passage",
        "text",
        "body",
        "content",
        "answer",
        "document",
        "context",
        "passage_text",
        "para",
    ]

    text_field = next(
        (f for f in text_candidates if f in all_fields),
        next(
            (
                f
                for f in all_fields
                if "text" in f.lower() or "passage" in f.lower()
            ),
            all_fields[0],
        ),
    )

    # ID field
    id_candidates = [
        "id",
        "passage_id",
        "doc_id",
        "pid",
        "_id",
    ]

    id_field = next(
        (f for f in id_candidates if f in all_fields),
        all_fields[0],
    )

    # Title
    title_candidates = [
        "title",
        "heading",
        "topic",
        "subject",
    ]

    title_field = next(
        (f for f in title_candidates if f in all_fields),
        None,
    )

    # Language
    lang_candidates = [
        "language",
        "lang",
        "locale",
        "language_code",
    ]

    language_field = next(
        (f for f in lang_candidates if f in all_fields),
        None,
    )

    # Query
    query_candidates = [
        "query",
        "question",
        "input",
        "q",
    ]

    query_field = next(
        (f for f in query_candidates if f in all_fields),
        None,
    )

    # Relevance
    rel_candidates = [
        "relevance",
        "label",
        "is_relevant",
        "score",
        "qrels",
    ]

    relevance_field = next(
        (f for f in rel_candidates if f in all_fields),
        None,
    )

    schema = DatasetSchema(
        text_field=text_field,
        id_field=id_field,
        title_field=title_field,
        language_field=language_field,
        query_field=query_field,
        relevance_field=relevance_field,
        all_fields=all_fields,
        total_rows=-1,
        sample_languages=[],
    )

    logger.info(
        "Detected schema: text=%s id=%s title=%s lang=%s",
        schema.text_field,
        schema.id_field,
        schema.title_field,
        schema.language_field,
    )

    return schema


def row_to_document(
    row: dict[str, Any],
    schema: DatasetSchema,
) -> dict[str, Any]:
    """Normalize a raw dataset row into a document."""

    text = str(
        row.get(schema.text_field, "")
    ).strip()

    doc_id = str(
        row.get(schema.id_field, "")
    )

    title = (
        str(row.get(schema.title_field, ""))
        if schema.title_field
        else ""
    )

    language = (
        str(row.get(schema.language_field, "en"))
        if schema.language_field
        else "en"
    )

    excluded_fields = {
        schema.text_field,
        schema.id_field,
    }

    if schema.title_field:
        excluded_fields.add(schema.title_field)

    if schema.language_field:
        excluded_fields.add(schema.language_field)

    metadata = {
        k: v
        for k, v in row.items()
        if k not in excluded_fields
    }

    return {
        "doc_id": doc_id,
        "text": text,
        "title": title,
        "language": language,
        "metadata": metadata,
    }


def load_dataset_streaming(
    dataset_name,
    split="train",
    language_filter=None,
):
    """
    Load a large Hugging Face dataset using streaming.

    IMPORTANT:
    Never switch to streaming=False because the dataset
    contains millions of records.
    """

    logger.info(
        "Loading dataset in streaming mode: %s",
        dataset_name,
    )

    logger.info(
        "Dataset split: %s",
        split,
    )

    ds = load_dataset(
        dataset_name,
        split=split,
        streaming=True,
    )

    logger.info(
        "Streaming dataset initialized successfully."
    )

    schema = inspect_schema(ds)

    return ds, schema