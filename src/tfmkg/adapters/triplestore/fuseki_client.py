from __future__ import annotations

import json
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class FusekiClient:
    def __init__(self, base_url: str, dataset: str):
        self._query_url = f"{base_url.rstrip('/')}/{dataset}/query"

    def ping(self) -> int:
        query = "SELECT * WHERE { ?s ?p ?o } LIMIT 1"
        body = urlencode({"query": query}).encode("utf-8")
        request = Request(
            self._query_url,
            data=body,
            headers={
                "Accept": "application/sparql-results+json",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            method="POST",
        )

        with urlopen(request, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))

        bindings = payload.get("results", {}).get("bindings", [])
        return len(bindings)
