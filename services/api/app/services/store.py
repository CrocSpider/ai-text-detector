from __future__ import annotations

from threading import Lock

from app.schemas.analysis import AnalysisResult


class InMemoryResultStore:
    def __init__(self) -> None:
        self._data: dict[str, AnalysisResult] = {}
        self._lock = Lock()

    def save(self, result: AnalysisResult) -> None:
        with self._lock:
            self._data[result.document_id] = result

    def get(self, document_id: str) -> AnalysisResult | None:
        with self._lock:
            return self._data.get(document_id)


result_store = InMemoryResultStore()
