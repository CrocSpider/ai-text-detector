from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_upload_txt_file() -> None:
    payload = (
        "The project update covers two issues. The first issue concerns a release blocker that required a manual rollback.\n\n"
        "The second issue explains how the team documented the fix and why additional verification is still needed."
    )
    response = client.post(
        "/v1/analyze/file",
        files={"file": ("sample.txt", payload.encode("utf-8"), "text/plain")},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["source_type"] == "file"
    assert data["source_name"] == "sample.txt"
    assert data["segments"]


def test_batch_limit_returns_multiple_results() -> None:
    files = [
        ("files", ("one.txt", b"Paragraph one.\n\nParagraph two with more detail.", "text/plain")),
        ("files", ("two.txt", b"Another document with two paragraphs.\n\nFinal paragraph.", "text/plain")),
    ]
    response = client.post("/v1/analyze/batch", files=files)

    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 2
    assert len(data["results"]) == 2
