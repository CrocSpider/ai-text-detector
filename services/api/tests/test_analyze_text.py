from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_analyze_text_returns_expected_shape() -> None:
    response = client.post(
        "/v1/analyze/text",
        json={
            "text": (
                "This is a short business memo. It uses some structure, but it also varies in length and detail.\n\n"
                "The second paragraph introduces a different pace and gives a small concrete example from customer support."
            )
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert 0 <= data["overall_risk_score"] <= 100
    assert data["confidence_level"] in {"low", "medium", "high"}
    assert data["recommendation"]
    assert data["segments"]
    assert data["signals"]["model_agreement"] >= 0


def test_analyze_text_rejects_empty_input() -> None:
    response = client.post("/v1/analyze/text", json={"text": ""})
    assert response.status_code == 422
