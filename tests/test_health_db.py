import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apps.api.dependencies import get_db_client
from apps.api.main import app


class _OkClient:
    def ping(self) -> None:
        return None


class _FailClient:
    def ping(self) -> None:
        raise RuntimeError("connection refused")


def test_health_db_ok() -> None:
    app.dependency_overrides[get_db_client] = lambda: _OkClient()
    client = TestClient(app)

    response = client.get("/health/db")

    assert response.status_code == 200
    assert response.json()["db"] == "ok"
    app.dependency_overrides.clear()


def test_health_db_failure() -> None:
    app.dependency_overrides[get_db_client] = lambda: _FailClient()
    client = TestClient(app)

    response = client.get("/health/db")

    assert response.status_code == 503
    assert response.json()["detail"] == "Database health check failed: connection refused"
    app.dependency_overrides.clear()
