import type { SegmentResult } from "@/lib/types";

interface SegmentHeatmapProps {
  segments: SegmentResult[];
}

function riskClass(score: number) {
  if (score <= 24) return "segment segment--low";
  if (score <= 49) return "segment segment--moderate";
  if (score <= 74) return "segment segment--elevated";
  return "segment segment--high";
}

export function SegmentHeatmap({ segments }: SegmentHeatmapProps) {
  return (
    <div className="segment-list">
      {segments.map((segment) => (
        <article key={segment.segment_id} className={riskClass(segment.risk_score)}>
          <div className="segment__header">
            <div>
              <h4>{segment.label}</h4>
              <span>
                Paragraphs {segment.start_paragraph + 1}-{segment.end_paragraph + 1}
              </span>
            </div>
            <div className="score-pill">{segment.risk_score}</div>
          </div>
          <p>{segment.excerpt}</p>
          <ul className="inline-list">
            {segment.key_signals.map((signal) => (
              <li key={signal}>{signal}</li>
            ))}
          </ul>
        </article>
      ))}
    </div>
  );
}
