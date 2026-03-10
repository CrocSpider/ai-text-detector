from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


ConfidenceLevel = Literal["low", "medium", "high"]
RiskBand = Literal["low", "moderate", "elevated", "high"]
SourceType = Literal["text", "file"]
ExtractionQuality = Literal["good", "fair", "poor"]


class TextAnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=200000)
    title: str | None = Field(default=None, max_length=200)
    source_name: str | None = Field(default=None, max_length=200)


class SegmentSignals(BaseModel):
    classifier_probability: float
    stylometric_anomaly_score: float
    surprisal_signal: float
    consistency_signal: float


class SegmentResult(BaseModel):
    segment_id: str
    paragraph_index: int
    start_paragraph: int
    end_paragraph: int
    label: str
    excerpt: str
    risk_score: int
    confidence_level: ConfidenceLevel
    key_signals: list[str]
    signals: SegmentSignals


class ModelSignalSummary(BaseModel):
    classifier_probability: float
    stylometric_anomaly_score: float
    surprisal_signal: float
    cross_segment_consistency: float
    quality_penalty: float
    uncertainty_penalty: float
    model_agreement: float


class AnalysisResult(BaseModel):
    document_id: str
    created_at: datetime
    source_type: SourceType
    source_name: str
    language: str
    language_supported: bool
    extraction_quality: ExtractionQuality
    extraction_warnings: list[str]
    character_count: int
    token_count: int
    overall_risk_score: int
    confidence_level: ConfidenceLevel
    risk_band: RiskBand
    short_summary: str
    recommendation: str
    key_signals: list[str]
    model_agreement_summary: str
    calibration_disclaimer: str
    limitations: list[str]
    warnings: list[str]
    signals: ModelSignalSummary
    segments: list[SegmentResult]
    model_version: str
    calibration_version: str


class BatchAnalysisResponse(BaseModel):
    count: int
    results: list[AnalysisResult]


class EvaluationMetric(BaseModel):
    name: str
    value: float
    target: float | None = None


class EvaluationSlice(BaseModel):
    label: str
    auroc: float
    false_positive_rate: float
    ece: float
    notes: str


class EvaluationSummary(BaseModel):
    model_version: str
    calibration_version: str
    last_run: datetime
    document_metrics: list[EvaluationMetric]
    segment_metrics: list[EvaluationMetric]
    slices: list[EvaluationSlice]
    threshold_guidance: list[str]
