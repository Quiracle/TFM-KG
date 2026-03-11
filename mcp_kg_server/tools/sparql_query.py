from __future__ import annotations

import json
import re
import socket
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

from ..settings import settings

_FORBIDDEN_UPDATE_KEYWORDS = re.compile(
    r"\b(INSERT|DELETE|LOAD|CLEAR|DROP|CREATE|MOVE|COPY|ADD)\b",
    re.IGNORECASE,
)
_LIMIT_PATTERN = re.compile(r"\bLIMIT\s+\d+\b", re.IGNORECASE)


class SparqlQueryOutput(BaseModel):
    row_count: int
    head: dict[str, Any]
    results: dict[str, Any]
    truncated: bool
    latency_ms: int


def _validate_query_is_read_only(query: str) -> str:
    normalized = query.strip()
    if not normalized:
        raise ValueError("Query must not be empty.")

    forbidden_match = _FORBIDDEN_UPDATE_KEYWORDS.search(normalized)
    if forbidden_match:
        keyword = forbidden_match.group(1).upper()
        raise ValueError(
            f"Read-only query required. Forbidden SPARQL UPDATE keyword detected: {keyword}."
        )

    if ";" in normalized.rstrip(";"):
        raise ValueError("Multiple statements are not allowed. Submit exactly one query statement.")

    return normalized


def _apply_limit_if_missing(query: str, max_rows: int) -> tuple[str, bool]:
    if _LIMIT_PATTERN.search(query):
        return query.rstrip(";"), False
    return f"{query.rstrip(';')}\nLIMIT {max_rows}", True


def _execute_sparql_query(query: str, timeout_ms: int) -> dict[str, Any]:
    body = urlencode({"query": query}).encode("utf-8")
    request = Request(
        settings.fuseki_query_url,
        data=body,
        headers={
            "Accept": "application/sparql-results+json",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        method="POST",
    )

    timeout_seconds = timeout_ms / 1000.0
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            raw_payload = response.read().decode("utf-8")
            return json.loads(raw_payload)
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Fuseki HTTP error {exc.code}: {detail}") from exc
    except socket.timeout as exc:
        raise RuntimeError(f"Fuseki query timed out after {timeout_ms}ms.") from exc
    except URLError as exc:
        raise RuntimeError(f"Fuseki query failed: {exc.reason}") from exc


def register_sparql_query_tool(server: FastMCP) -> None:
    @server.tool()
    def sparql_query(
        query: str,
        timeout_ms: int | None = None,
        max_rows: int | None = None,
    ) -> SparqlQueryOutput:
        """
        Execute a safe, read-only SPARQL query against Fuseki.
        """
        effective_timeout_ms = settings.mcp_kg_timeout_ms if timeout_ms is None else timeout_ms
        effective_max_rows = settings.mcp_kg_max_rows if max_rows is None else max_rows

        if effective_timeout_ms <= 0:
            raise ValueError("timeout_ms must be a positive integer.")
        if effective_max_rows <= 0:
            raise ValueError("max_rows must be a positive integer.")

        validated_query = _validate_query_is_read_only(query)
        query_with_limit, limit_was_applied = _apply_limit_if_missing(validated_query, effective_max_rows)

        started_at = time.perf_counter()
        payload = _execute_sparql_query(query_with_limit, effective_timeout_ms)
        latency_ms = int((time.perf_counter() - started_at) * 1000)

        head = payload.get("head", {})
        results = payload.get("results", {})
        bindings = results.get("bindings", [])
        row_count = len(bindings)
        truncated = bool(limit_was_applied and row_count >= effective_max_rows)

        return SparqlQueryOutput(
            row_count=row_count,
            head=head,
            results=results,
            truncated=truncated,
            latency_ms=latency_ms,
        )
