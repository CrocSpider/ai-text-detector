from __future__ import annotations

from trainer.datasets import DocumentRecord, PreparedSplit, SegmentRecord, _resolve_chunk_labels
from trainer.features import extract_text_features
from trainer.meta import build_document_feature_rows
from trainer.text_utils import TextChunk


def test_resolve_chunk_labels_aggregates_paragraph_labels_with_majority_tie_break() -> None:
    document = DocumentRecord(
        document_id="doc-1",
        text="Paragraph one.\n\nParagraph two.\n\nParagraph three.",
        label=0,
        metadata={},
        segment_labels=[0, 1, 1],
    )
    chunks = [
        TextChunk(text="Paragraph one.\n\nParagraph two.", start_paragraph=0, end_paragraph=1),
        TextChunk(text="Paragraph three.", start_paragraph=2, end_paragraph=2),
    ]

    assert _resolve_chunk_labels(document, chunks) == [1, 1]


def test_build_document_feature_rows_uses_majority_segment_label_for_mixed_documents() -> None:
    document = DocumentRecord(
        document_id="doc-2",
        text="First paragraph.\n\nSecond paragraph.\n\nThird paragraph.",
        label=0,
        metadata={"language": "en", "extraction_quality": "good"},
    )
    segments = [
        SegmentRecord(
            segment_id="doc-2::seg::0",
            document_id="doc-2",
            text="First paragraph.",
            label=1,
            segment_index=0,
            metadata={"label_origin": "segment"},
        ),
        SegmentRecord(
            segment_id="doc-2::seg::1",
            document_id="doc-2",
            text="Second paragraph.",
            label=0,
            segment_index=1,
            metadata={"label_origin": "segment"},
        ),
        SegmentRecord(
            segment_id="doc-2::seg::2",
            document_id="doc-2",
            text="Third paragraph.",
            label=1,
            segment_index=2,
            metadata={"label_origin": "segment"},
        ),
    ]
    split = PreparedSplit(name="train", documents=[document], segments=segments)
    feature_rows = {
        segment.segment_id: extract_text_features(segment.text)
        for segment in segments
    }

    rows = build_document_feature_rows(
        split=split,
        feature_rows=feature_rows,
        classifier_probabilities={segment.segment_id: 0.7 for segment in segments},
        stylometry_probabilities={segment.segment_id: 0.6 for segment in segments},
    )

    assert rows[0]["label"] == 1
