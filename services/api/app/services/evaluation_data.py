from __future__ import annotations

from datetime import datetime, timezone

from app.schemas.analysis import EvaluationMetric, EvaluationSlice, EvaluationSummary


def sample_evaluation_summary() -> EvaluationSummary:
    return EvaluationSummary(
        model_version="heuristic-ensemble-v0",
        calibration_version="temperature-lite-v0",
        last_run=datetime.now(timezone.utc),
        document_metrics=[
            EvaluationMetric(name="AUROC", value=0.86, target=0.90),
            EvaluationMetric(name="AUPRC", value=0.84, target=0.88),
            EvaluationMetric(name="F1", value=0.77, target=0.80),
            EvaluationMetric(name="ECE", value=0.08, target=0.05),
        ],
        segment_metrics=[
            EvaluationMetric(name="AUROC", value=0.81, target=0.86),
            EvaluationMetric(name="FPR", value=0.09, target=0.05),
            EvaluationMetric(name="FNR", value=0.18, target=0.12),
        ],
        slices=[
            EvaluationSlice(
                label="Short text (<150 tokens)",
                auroc=0.71,
                false_positive_rate=0.14,
                ece=0.12,
                notes="Confidence is deliberately capped for short inputs.",
            ),
            EvaluationSlice(
                label="Non-native English essays",
                auroc=0.76,
                false_positive_rate=0.12,
                ece=0.11,
                notes="Needs dedicated fairness calibration before strong claims are appropriate.",
            ),
            EvaluationSlice(
                label="Human-edited AI drafts",
                auroc=0.68,
                false_positive_rate=0.08,
                ece=0.15,
                notes="Mixed provenance remains a key failure mode.",
            ),
        ],
        threshold_guidance=[
            "Keep 0-24 as low risk, but still show explanations and limitations.",
            "Treat 25-49 as a human-review band rather than an actionable signal.",
            "Require medium or high confidence before surfacing elevated-risk guidance.",
        ],
    )
