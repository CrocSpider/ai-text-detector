import Link from "next/link";

import type { AnalysisResult } from "@/lib/types";
import { SegmentHeatmap } from "@/components/SegmentHeatmap";

interface ResultCardProps {
  result: AnalysisResult;
  showLink?: boolean;
}

function bandClass(score: number) {
  if (score <= 24) return "badge badge--low";
  if (score <= 49) return "badge badge--moderate";
  if (score <= 74) return "badge badge--elevated";
  return "badge badge--high";
}

export function ResultCard({ result, showLink = true }: ResultCardProps) {
  return (
    <article className="result-card">
      <div className="result-card__header">
        <div>
          <p className="eyebrow">{result.source_name}</p>
          <h3>{result.recommendation}</h3>
          <p className="result-summary">{result.short_summary}</p>
        </div>
        <div className="score-panel">
          <div className="score-ring">{result.overall_risk_score}</div>
          <span className={bandClass(result.overall_risk_score)}>{result.risk_band} risk</span>
          <span className="meta-chip">Confidence: {result.confidence_level}</span>
        </div>
      </div>

      <div className="signal-grid">
        <div className="signal-grid__item">
          <span>Language</span>
          <strong>{result.language_supported ? result.language : `${result.language} (limited support)`}</strong>
        </div>
        <div className="signal-grid__item">
          <span>Extraction quality</span>
          <strong>{result.extraction_quality}</strong>
        </div>
        <div className="signal-grid__item">
          <span>Tokens</span>
          <strong>{result.token_count}</strong>
        </div>
        <div className="signal-grid__item">
          <span>Model agreement</span>
          <strong>{Math.round(result.signals.model_agreement * 100)}%</strong>
        </div>
      </div>

      <div className="tag-list">
        {result.key_signals.map((signal) => (
          <span className="tag" key={signal}>
            {signal}
          </span>
        ))}
      </div>

      <details className="details-panel" open>
        <summary>Why this result</summary>
        <p>{result.model_agreement_summary}</p>
        <p>{result.calibration_disclaimer}</p>
        <SegmentHeatmap segments={result.segments} />
      </details>

      <details className="details-panel">
        <summary>Warnings and limitations</summary>
        <ul className="stack-list">
          {result.warnings.map((warning) => (
            <li key={warning}>{warning}</li>
          ))}
          {result.limitations.map((limitation) => (
            <li key={limitation}>{limitation}</li>
          ))}
        </ul>
      </details>

      {showLink ? (
        <div className="result-card__footer">
          <Link href={`/results/${result.document_id}`}>Open details</Link>
        </div>
      ) : null}
    </article>
  );
}
