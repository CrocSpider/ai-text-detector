"use client";

import { useMemo, useState } from "react";

import { Disclaimer } from "@/components/Disclaimer";
import { ResultCard } from "@/components/ResultCard";
import { TextInput } from "@/components/TextInput";
import { UploadPanel } from "@/components/UploadPanel";
import { analyzeFiles, analyzeText } from "@/lib/api";
import type { AnalysisResult } from "@/lib/types";

export function AnalyzerClient() {
  const [text, setText] = useState("");
  const [files, setFiles] = useState<File[]>([]);
  const [results, setResults] = useState<AnalysisResult[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const buttonLabel = useMemo(() => {
    if (loading) return "Analyzing...";
    if (files.length > 1 && text.trim()) return "Analyze text and files";
    if (files.length > 1) return "Analyze batch";
    return "Analyze";
  }, [files.length, loading, text]);

  async function handleAnalyze() {
    setError(null);
    if (!text.trim() && files.length === 0) {
      setError("Paste text or upload at least one file to analyze.");
      return;
    }

    setLoading(true);
    try {
      const tasks: Promise<AnalysisResult[]>[] = [];
      if (text.trim()) {
        tasks.push(analyzeText(text.trim(), "Pasted text").then((result) => [result]));
      }
      if (files.length > 0) {
        tasks.push(analyzeFiles(files));
      }

      const settled = await Promise.all(tasks);
      setResults(settled.flat());
    } catch (analysisError) {
      setError(analysisError instanceof Error ? analysisError.message : "The analysis request failed.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="page-shell">
      <section className="hero-card">
        <p className="eyebrow">Advisory detector</p>
        <h1>AI Text Origin Risk Analyzer</h1>
        <p className="lead">
          Estimate whether a document shows signals consistent with machine-generated or heavily
          machine-edited writing. The system highlights uncertainty, suspicious sections, and reasons.
        </p>
      </section>

      <section className="workspace-grid">
        <UploadPanel files={files} onFilesChange={setFiles} />
        <TextInput value={text} onChange={setText} />
      </section>

      <section className="action-row">
        <button type="button" className="primary-button" onClick={handleAnalyze} disabled={loading}>
          {buttonLabel}
        </button>
        <button
          type="button"
          className="secondary-button"
          onClick={() => {
            setText("");
            setFiles([]);
            setResults([]);
            setError(null);
          }}
          disabled={loading}
        >
          Reset
        </button>
      </section>

      {error ? <div className="notice-card notice-card--error">{error}</div> : null}

      <Disclaimer />

      {results.length > 0 ? (
        <section className="results-stack">
          <div className="section-heading">
            <h2>Results</h2>
            <span>{results.length} document(s)</span>
          </div>
          {results.map((result) => (
            <ResultCard key={result.document_id} result={result} />
          ))}
        </section>
      ) : (
        <section className="placeholder-card">
          <h2>What you will get</h2>
          <ul className="stack-list">
            <li>A calibrated 0-100 risk score with separate confidence.</li>
            <li>Paragraph-level highlights instead of a single opaque verdict.</li>
            <li>Warnings for short text, poor extraction, and unsupported languages.</li>
          </ul>
        </section>
      )}
    </main>
  );
}
