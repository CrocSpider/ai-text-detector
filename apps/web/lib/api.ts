import type { AnalysisResult, BatchAnalysisResponse, EvaluationSummary } from "@/lib/types";

function getApiBaseUrl() {
  if (typeof window === "undefined") {
    // Server-side (SSR / API routes): prefer the internal k8s service URL.
    return process.env.API_INTERNAL_BASE_URL ?? process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";
  }
  // Browser-side: prefer an explicit override, otherwise use the rewrite proxy
  // so no URL needs to be baked into the Docker image at build time.
  return process.env.NEXT_PUBLIC_API_BASE_URL ?? "/api-proxy";
}

async function parseResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const fallback = `Request failed with status ${response.status}`;
    let detail: string | undefined;
    try {
      const payload = (await response.json()) as { detail?: string };
      detail = payload.detail;
    } catch {
      detail = undefined;
    }
    throw new Error(detail ?? fallback);
  }
  return (await response.json()) as T;
}

export async function analyzeText(text: string, sourceName?: string): Promise<AnalysisResult> {
  const response = await fetch(`${getApiBaseUrl()}/v1/analyze/text`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, source_name: sourceName }),
  });

  return parseResponse<AnalysisResult>(response);
}

export async function analyzeFiles(files: File[]): Promise<AnalysisResult[]> {
  const formData = new FormData();
  for (const file of files) {
    formData.append("files", file);
  }

  const response = await fetch(`${getApiBaseUrl()}/v1/analyze/batch`, {
    method: "POST",
    body: formData,
  });

  const payload = await parseResponse<BatchAnalysisResponse>(response);
  return payload.results;
}

export async function getResult(documentId: string): Promise<AnalysisResult> {
  const response = await fetch(`${getApiBaseUrl()}/v1/results/${documentId}`, { cache: "no-store" });
  return parseResponse<AnalysisResult>(response);
}

export async function getEvaluationSummary(): Promise<EvaluationSummary> {
  const response = await fetch(`${getApiBaseUrl()}/v1/admin/evaluations/summary`, { cache: "no-store" });
  return parseResponse<EvaluationSummary>(response);
}
