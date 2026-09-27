from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


def test_http_boundaries_and_missing_index(tmp_path):
    settings = Settings(
        index_path=tmp_path / "index",
        data_path=tmp_path / "policies",
        session_path=tmp_path / "session.sqlite",
    )
    with TestClient(create_app(settings)) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert "HttpOnly" in page.headers["set-cookie"]
        assert "script-src 'self'" in page.headers["content-security-policy"]
        assert "<textarea" in page.text
        health = client.get("/health").json()
        assert health["status"] == "ok" and health["index_available"] is False
        assert client.post("/api/chat", json={"message": "Canada 2025"}).status_code == 503
        assert client.post("/api/chat", json={"message": " "}).status_code == 422
        assert client.post("/api/chat", json={"message": "a" * 2001}).status_code == 422
        assert client.post("/api/chat", content=b"a" * 9000).status_code == 413
        assert (
            client.post(
                "/api/chat", json={"message": "test"}, headers={"Origin": "https://example.com"}
            ).status_code
            == 403
        )
        assert client.get("/health", headers={"host": "evil.example"}).status_code == 400
        assert client.get("/documents/missing").status_code == 404
        assert client.get("/documents/canada-2025").status_code == 404
        assert client.get("/documents/..%2f..%2fetc%2fpasswd").status_code == 404
