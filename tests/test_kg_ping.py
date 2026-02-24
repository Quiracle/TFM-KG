import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apps.api.dependencies import get_fuseki_client
from apps.api.main import app


class _FakeFusekiClient:
    def __init__(self, results_count: int):
        self._results_count = results_count

    def ping(self) -> int:
        return self._results_count


def test_kg_ping_returns_ok_with_results_count() -> None:
    app.dependency_overrides[get_fuseki_client] = lambda: _FakeFusekiClient(0)
    client = TestClient(app)

    response = client.get("/kg/ping")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "results_count": 0}
    app.dependency_overrides.clear()
