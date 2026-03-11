from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .settings import settings


def _extract_structured_result(result: Any) -> dict[str, Any] | None:
    if isinstance(result, tuple) and len(result) >= 2 and isinstance(result[1], dict):
        return result[1]
    if isinstance(result, dict):
        return result
    return None


def log_mcp_tool_call(
    tool_name: str,
    request: dict[str, Any],
    latency_ms: int,
    result: Any = None,
    error: dict[str, str] | None = None,
) -> None:
    structured = _extract_structured_result(result) or {}
    row_count = structured.get("row_count")
    truncated = structured.get("truncated")

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool_name": tool_name,
        "request": request,
        "row_count": row_count if isinstance(row_count, int) else None,
        "truncated": truncated if isinstance(truncated, bool) else None,
        "latency_ms": latency_ms,
        "errors": error,
    }

    try:
        log_path = Path(settings.mcp_tool_log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=True) + "\n")
    except Exception:
        # Telemetry must never break tool execution.
        return
