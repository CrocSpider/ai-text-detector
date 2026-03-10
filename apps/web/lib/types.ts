export type ConfidenceLevel = "low" | "medium" | "high";
export type RiskBand = "low" | "moderate" | "elevated" | "high";
export type ExtractionQuality = "good" | "fair" | "poor";

export interface SegmentSignals {
  classifier_probability: number;
  stylometric_anomaly_score: number;
  surprisal_signal: number;
  consistency_signal: number;
}

export interface SegmentResult {
  segment_id: string;
  paragraph_index: number;
  start_paragraph: number;
  end_paragraph: number;
  label: string;
  excerpt: string;
  risk_score: number;
  confidence_level: ConfidenceLevel;
  key_signals: string[];
  signals: SegmentSignals;
}

export interface ModelSignalSummary {
  classifier_probability: number;
  stylometric_anomaly_score: number;
  surprisal_signal: number;
  cross_segment_consistency: number;
  quality_penalty: number;
  uncertainty_penalty: number;
  model_agreement: number;
}

export interface AnalysisResult {
  document_id: string;
  created_at: string;
  source_type: "text" | "file";
  source_name: string;
  language: string;
  language_supported: boolean;
  extraction_quality: ExtractionQuality;
  extraction_warnings: string[];
  character_count: number;
  token_count: number;
  overall_risk_score: number;
  confidence_level: ConfidenceLevel;
  risk_band: RiskBand;
  short_summary: string;
  recommendation: string;
  key_signals: string[];
  model_agreement_summary: string;
  calibration_disclaimer: string;
  limitations: string[];
  warnings: string[];
  signals: ModelSignalSummary;
  segments: SegmentResult[];
  model_version: string;
  calibration_version: string;
}

export interface BatchAnalysisResponse {
  count: number;
  results: AnalysisResult[];
}

export interface EvaluationMetric {
  name: string;
  value: number;
  target?: number | null;
}

export interface EvaluationSlice {
  label: string;
  auroc: number;
  false_positive_rate: number;
  ece: number;
  notes: string;
}

export interface EvaluationSummary {
  model_version: string;
  calibration_version: string;
  last_run: string;
  document_metrics: EvaluationMetric[];
  segment_metrics: EvaluationMetric[];
  slices: EvaluationSlice[];
  threshold_guidance: string[];
}
