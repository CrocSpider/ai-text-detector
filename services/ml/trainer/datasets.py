from __future__ import annotations

import csv
import json
import logging
import random
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from trainer.config import DatasetConfig
from trainer.text_utils import TextChunk, chunk_text, normalize_text, text_length_bucket


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class DocumentRecord:
    document_id: str
    text: str
    label: int
    metadata: dict[str, str] = field(default_factory=dict)
    segment_labels: list[int] | None = None
    """Optional per-segment labels.  When provided (e.g. from mixed-origin
    datasets), each segment gets its own label instead of inheriting the
    document-level label.  Length must match the number of chunks produced
    by ``chunk_text``."""


@dataclass(slots=True)
class SegmentRecord:
    segment_id: str
    document_id: str
    text: str
    label: int
    segment_index: int
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class PreparedSplit:
    name: str
    documents: list[DocumentRecord]
    segments: list[SegmentRecord]


@dataclass(slots=True)
class SourceSpec:
    uri: str
    text_key: str
    label_key: str
    document_id_key: str | None
    language_key: str | None
    domain_key: str | None
    writer_profile_key: str | None
    attack_type_key: str | None
    source_dataset_key: str | None
    extraction_quality_key: str | None
    segment_labels_key: str | None = None
    label_strategy: str = "default"
    label_map: dict[str, int] = field(default_factory=dict)
    metadata_defaults: dict[str, str] = field(default_factory=dict)
    include: dict[str, list[str]] = field(default_factory=dict)
    exclude: dict[str, list[str]] = field(default_factory=dict)


def prepare_datasets(dataset_config: DatasetConfig, seed: int) -> dict[str, PreparedSplit]:
    return {
        "train": _prepare_split(
            name="train",
            sources=dataset_config.train_sources,
            dataset_config=dataset_config,
            seed=seed,
            max_examples=dataset_config.max_train_examples,
        ),
        "validation": _prepare_split(
            name="validation",
            sources=dataset_config.validation_sources,
            dataset_config=dataset_config,
            seed=seed + 1,
            max_examples=dataset_config.max_eval_examples,
        ),
        "test": _prepare_split(
            name="test",
            sources=dataset_config.test_sources,
            dataset_config=dataset_config,
            seed=seed + 2,
            max_examples=dataset_config.max_eval_examples,
        ),
    }


def _prepare_split(
    *,
    name: str,
    sources: list[Any],
    dataset_config: DatasetConfig,
    seed: int,
    max_examples: int | None,
) -> PreparedSplit:
    documents = load_documents(sources=sources, dataset_config=dataset_config)
    rng = random.Random(seed)
    if max_examples is not None and len(documents) > max_examples:
        documents = rng.sample(documents, max_examples)

    segments: list[SegmentRecord] = []
    for document in documents:
        chunks = chunk_text(
            document.text,
            target_chars=dataset_config.target_chunk_chars,
            max_chars=dataset_config.max_chunk_chars,
        )
        if not chunks:
            continue

        resolved_chunk_labels = _resolve_chunk_labels(document, chunks)

        for index, chunk in enumerate(chunks):
            if resolved_chunk_labels is not None:
                segment_label = resolved_chunk_labels[index]
                label_origin = "segment"
            else:
                segment_label = document.label
                label_origin = "document"

            seg_metadata = dict(document.metadata)
            seg_metadata["label_origin"] = label_origin

            segments.append(
                SegmentRecord(
                    segment_id=f"{document.document_id}::seg::{index}",
                    document_id=document.document_id,
                    text=chunk.text,
                    label=segment_label,
                    segment_index=index,
                    metadata=seg_metadata,
                )
            )

    logger.info("Prepared %s split with %s documents and %s segments", name, len(documents), len(segments))
    return PreparedSplit(name=name, documents=documents, segments=segments)


def load_documents(sources: list[Any], dataset_config: DatasetConfig) -> list[DocumentRecord]:
    documents: list[DocumentRecord] = []
    source_specs = normalize_source_specs(sources, dataset_config)
    for source_spec in source_specs:
        raw_records = _load_source_records(source_spec.uri)
        for index, record in enumerate(raw_records):
            normalized = _normalize_record(
                record,
                index=index,
                source=source_spec,
                min_text_chars=dataset_config.min_text_chars,
            )
            if normalized is not None:
                documents.append(normalized)
    return documents


def normalize_source_specs(sources: list[Any], dataset_config: DatasetConfig) -> list[SourceSpec]:
    source_specs: list[SourceSpec] = []
    for source in sources:
        if isinstance(source, str):
            source_specs.append(
                SourceSpec(
                    uri=source,
                    text_key=dataset_config.text_key,
                    label_key=dataset_config.label_key,
                    document_id_key=dataset_config.document_id_key,
                    language_key=dataset_config.language_key,
                    domain_key=dataset_config.domain_key,
                    writer_profile_key=dataset_config.writer_profile_key,
                    attack_type_key=dataset_config.attack_type_key,
                    source_dataset_key=dataset_config.source_dataset_key,
                    extraction_quality_key=dataset_config.extraction_quality_key,
                    segment_labels_key=dataset_config.segment_labels_key,
                )
            )
            continue

        if not isinstance(source, dict):
            raise ValueError("Each dataset source must be a string or mapping")

        uri = str(source.get("uri") or source.get("source") or "").strip()
        if not uri:
            raise ValueError("Dataset source mappings must include 'uri'")

        source_specs.append(
            SourceSpec(
                uri=uri,
                text_key=str(source.get("text_key", dataset_config.text_key)),
                label_key=str(source.get("label_key", dataset_config.label_key)),
                document_id_key=_optional_string(source.get("document_id_key", dataset_config.document_id_key)),
                language_key=_optional_string(source.get("language_key", dataset_config.language_key)),
                domain_key=_optional_string(source.get("domain_key", dataset_config.domain_key)),
                writer_profile_key=_optional_string(source.get("writer_profile_key", dataset_config.writer_profile_key)),
                attack_type_key=_optional_string(source.get("attack_type_key", dataset_config.attack_type_key)),
                source_dataset_key=_optional_string(source.get("source_dataset_key", dataset_config.source_dataset_key)),
                extraction_quality_key=_optional_string(
                    source.get("extraction_quality_key", dataset_config.extraction_quality_key)
                ),
                segment_labels_key=_optional_string(
                    source.get("segment_labels_key", dataset_config.segment_labels_key)
                ),
                label_strategy=str(source.get("label_strategy", "default")),
                label_map=_normalize_label_map(source.get("label_map", {})),
                metadata_defaults=_normalize_string_mapping(source.get("metadata_defaults", {})),
                include=_normalize_filters(source.get("include", source.get("filters", {}))),
                exclude=_normalize_filters(source.get("exclude", {})),
            )
        )

    return source_specs


def _load_source_records(source: str) -> list[dict[str, Any]]:
    if source.startswith("hf://"):
        return _load_huggingface_source(source)

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Dataset source does not exist: {source}")

    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        records = []
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
        return records
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            if isinstance(payload.get("records"), list):
                return payload["records"]
            if isinstance(payload.get("data"), list):
                return payload["data"]
        raise ValueError(f"Unsupported JSON structure in {source}")
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    raise ValueError(f"Unsupported dataset source format: {source}")


def _load_huggingface_source(source: str) -> list[dict[str, Any]]:
    parsed = urlparse(source)
    dataset_name = parsed.netloc + parsed.path
    dataset_name = dataset_name.lstrip("/")
    params = parse_qs(parsed.query)
    split = params.get("split", ["train"])[0]
    subset = params.get("subset", [None])[0]

    try:
        load_dataset = getattr(import_module("datasets"), "load_dataset")
    except ImportError as exc:
        raise RuntimeError("The datasets package is required for hf:// sources") from exc

    dataset = load_dataset(dataset_name, subset, split=split)
    return [dict(row) for row in dataset]


def _normalize_record(
    record: dict[str, Any],
    *,
    index: int,
    source: SourceSpec,
    min_text_chars: int,
) -> DocumentRecord | None:
    if not isinstance(record, dict):
        return None

    if source.include and not _matches_filters(record, source.include):
        return None
    if source.exclude and _matches_filters(record, source.exclude):
        return None

    text = str(record.get(source.text_key, "") or "")
    text = normalize_text(text)
    if len(text) < min_text_chars:
        return None

    label_value = record.get(source.label_key)
    label = _coerce_label(label_value, strategy=source.label_strategy, label_map=source.label_map)
    if label is None:
        return None

    document_id = _document_id_for(record=record, source=source, index=index)
    metadata = dict(source.metadata_defaults)
    metadata.update(
        {
            "language": _metadata_value(record, source.language_key, metadata.get("language", "unknown")),
            "domain": _metadata_value(record, source.domain_key, metadata.get("domain", "unknown")),
            "writer_profile": _metadata_value(
                record,
                source.writer_profile_key,
                metadata.get("writer_profile", "unknown"),
            ),
            "attack_type": _metadata_value(record, source.attack_type_key, metadata.get("attack_type", "none")),
            "source_dataset": _metadata_value(
                record,
                source.source_dataset_key,
                metadata.get("source_dataset", _source_stem(source.uri)),
            ),
            "extraction_quality": _metadata_value(
                record,
                source.extraction_quality_key,
                metadata.get("extraction_quality", "good"),
            ),
            "text_length_bucket": text_length_bucket(text),
        }
    )

    # Parse optional segment-level labels for mixed-origin documents.
    segment_labels = _parse_segment_labels(record, source)

    return DocumentRecord(
        document_id=document_id, text=text, label=label, metadata=metadata,
        segment_labels=segment_labels,
    )


def _document_id_for(record: dict[str, Any], source: SourceSpec, index: int) -> str:
    if source.document_id_key:
        value = record.get(source.document_id_key)
        if value not in {None, ""}:
            return str(value)
    return f"{_source_stem(source.uri)}-{index}"


def _parse_segment_labels(record: dict[str, Any], source: SourceSpec) -> list[int] | None:
    """Extract optional per-segment labels from a raw record.

    Expected format: a JSON array of ints (one per paragraph/chunk), e.g.
    ``"segment_labels": [0, 0, 1, 1, 0]``.

    Returns ``None`` when the key is absent or the value is not a valid list.
    """
    if not source.segment_labels_key:
        return None
    raw = record.get(source.segment_labels_key)
    if raw is None:
        return None
    if isinstance(raw, str):
        try:
            import json as _json
            raw = _json.loads(raw)
        except (ValueError, TypeError):
            return None
    if not isinstance(raw, list):
        return None
    coerced: list[int] = []
    for item in raw:
        label = _coerce_label(item, strategy=source.label_strategy, label_map=source.label_map)
        if label is None:
            return None  # all-or-nothing: any unresolvable label invalidates the list
        coerced.append(label)
    return coerced if coerced else None


def _resolve_chunk_labels(
    document: DocumentRecord,
    chunks: list[TextChunk],
) -> list[int] | None:
    """Map per-segment labels to per-chunk labels.

    Handles three cases:
    1. ``segment_labels`` is ``None`` → returns ``None`` (use document label).
    2. ``len(segment_labels) == len(chunks)`` → labels already 1-to-1.
    3. ``len(segment_labels) == n_paragraphs`` → aggregate paragraph labels
       to the chunk level via majority vote (preferring label 1 on tie).

    Falls back to ``None`` with a warning when neither match applies.
    """
    seg_labels = document.segment_labels
    if seg_labels is None:
        return None

    if len(seg_labels) == len(chunks):
        return seg_labels

    # Paragraph count is derivable from the last chunk's end_paragraph index.
    n_paragraphs = chunks[-1].end_paragraph + 1 if chunks else 0

    if len(seg_labels) == n_paragraphs:
        return [
            _majority_label(seg_labels[chunk.start_paragraph : chunk.end_paragraph + 1])
            for chunk in chunks
        ]

    logger.warning(
        "Document %s has %d segment labels but %d chunks (%d paragraphs) — "
        "falling back to document-level label",
        document.document_id,
        len(seg_labels),
        len(chunks),
        n_paragraphs,
    )
    return None


def _majority_label(labels: list[int]) -> int:
    """Return majority label, preferring 1 (AI-generated) on tie."""
    if not labels:
        return 0
    ones = sum(labels)
    return 1 if ones * 2 >= len(labels) else 0


def _metadata_value(record: dict[str, Any], key: str | None, default: str) -> str:
    if key:
        value = record.get(key)
        if value not in {None, ""}:
            return _string_or_default(value, default)
    return default


def _coerce_label(value: Any, *, strategy: str = "default", label_map: dict[str, int] | None = None) -> int | None:
    normalized_map = label_map or {}
    normalized = _normalize_scalar(value)
    if normalized_map and normalized in normalized_map:
        return normalized_map[normalized]

    if strategy == "non_human_is_ai":
        if value is None:
            return None
        if normalized in {"", "null", "none"}:
            return None
        if normalized in {"0", "false", "human", "negative"}:
            return 0
        return 1

    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(float(value) >= 0.5)
    if value is None:
        return None

    if normalized in {"1", "true", "ai", "machine", "generated", "positive"}:
        return 1
    if normalized in {"0", "false", "human", "negative"}:
        return 0
    return None


def _matches_filters(record: dict[str, Any], filters: dict[str, list[str]]) -> bool:
    for key, allowed_values in filters.items():
        value = record.get(key)
        if isinstance(value, list):
            normalized_values = {_normalize_scalar(item) for item in value}
            if not normalized_values.intersection(allowed_values):
                return False
            continue

        if _normalize_scalar(value) not in allowed_values:
            return False
    return True


def _normalize_filters(raw_filters: Any) -> dict[str, list[str]]:
    if raw_filters is None or raw_filters == {}:
        return {}
    if not isinstance(raw_filters, dict):
        raise ValueError("Dataset source filters must be a mapping")

    normalized: dict[str, list[str]] = {}
    for key, value in raw_filters.items():
        if isinstance(value, list):
            normalized[str(key)] = [_normalize_scalar(item) for item in value]
        else:
            normalized[str(key)] = [_normalize_scalar(value)]
    return normalized


def _normalize_label_map(raw_label_map: Any) -> dict[str, int]:
    if raw_label_map is None or raw_label_map == {}:
        return {}
    if not isinstance(raw_label_map, dict):
        raise ValueError("Dataset source label_map must be a mapping")

    normalized: dict[str, int] = {}
    for key, value in raw_label_map.items():
        normalized[_normalize_scalar(key)] = int(value)
    return normalized


def _normalize_string_mapping(raw_mapping: Any) -> dict[str, str]:
    if raw_mapping is None or raw_mapping == {}:
        return {}
    if not isinstance(raw_mapping, dict):
        raise ValueError("Dataset source metadata_defaults must be a mapping")
    return {str(key): str(value) for key, value in raw_mapping.items()}


def _normalize_scalar(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    stripped = str(value).strip()
    return stripped or None


def _source_stem(source: str) -> str:
    if source.startswith("hf://"):
        parsed = urlparse(source)
        dataset_name = (parsed.netloc + parsed.path).strip("/")
        return dataset_name.replace("/", "-")
    return Path(source).stem


def _string_or_default(value: Any, default: str) -> str:
    if value is None:
        return default
    stripped = str(value).strip()
    return stripped or default
