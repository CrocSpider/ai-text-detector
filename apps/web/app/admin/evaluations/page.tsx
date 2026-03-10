import Link from "next/link";

import { getEvaluationSummary } from "@/lib/api";

export const dynamic = "force-dynamic";

export default async function EvaluationPage() {
  const summary = await getEvaluationSummary();

  return (
    <main className="page-shell page-shell--narrow">
      <div className="page-backlink">
        <Link href="/">Back to analyzer</Link>
      </div>

      <section className="hero-card hero-card--compact">
        <p className="eyebrow">Admin evaluation</p>
        <h1>Benchmark snapshot</h1>
        <p className="lead">
          Model: {summary.model_version} - Calibration: {summary.calibration_version}
        </p>
      </section>

      <section className="result-card">
        <h2>Document-level metrics</h2>
        <div className="metric-grid">
          {summary.document_metrics.map((metric) => (
            <div key={metric.name} className="signal-grid__item">
              <span>{metric.name}</span>
              <strong>{metric.value.toFixed(2)}</strong>
            </div>
          ))}
        </div>
      </section>

      <section className="result-card">
        <h2>Fairness and robustness slices</h2>
        <div className="stacked-cards">
          {summary.slices.map((slice) => (
            <article key={slice.label} className="segment segment--neutral">
              <div className="segment__header">
                <div>
                  <h4>{slice.label}</h4>
                  <span>{slice.notes}</span>
                </div>
              </div>
              <div className="metric-grid">
                <div className="signal-grid__item">
                  <span>AUROC</span>
                  <strong>{slice.auroc.toFixed(2)}</strong>
                </div>
                <div className="signal-grid__item">
                  <span>FPR</span>
                  <strong>{slice.false_positive_rate.toFixed(2)}</strong>
                </div>
                <div className="signal-grid__item">
                  <span>ECE</span>
                  <strong>{slice.ece.toFixed(2)}</strong>
                </div>
              </div>
            </article>
          ))}
        </div>
      </section>

      <section className="result-card">
        <h2>Threshold guidance</h2>
        <ul className="stack-list">
          {summary.threshold_guidance.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
      </section>
    </main>
  );
}
