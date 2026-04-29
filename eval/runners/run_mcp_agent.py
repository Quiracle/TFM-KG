from __future__ import annotations

import argparse
import asyncio
import json
import time
from typing import Any

from common import (
    ABSTAIN,
    append_jsonl_row,
    build_error_row,
    build_raw_row,
    ensure_output_dirs,
    extract_structured_tool_result,
    init_jsonl_output,
    load_jsonl,
    load_text_file,
    log_runner_error,
    question_id_from_row,
    resolve_repo_path,
    select_dataset_rows,
    to_jsonable,
    truncate_text,
)
from mcp_kg_server.server import create_server
from src.tfmkg.core.config import settings


def _is_rate_limit_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code == 429:
        return True

    response = getattr(exc, "response", None)
    response_status = getattr(response, "status_code", None)
    if response_status == 429:
        return True

    message = str(exc).lower()
    return "429" in message or "rate_limit_error" in message or "too many requests" in message


def _extract_retry_after_seconds(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None

    for header_name in (
        "retry-after",
        "anthropic-ratelimit-requests-reset",
        "anthropic-ratelimit-tokens-reset",
    ):
        raw_value = headers.get(header_name)
        if raw_value is None:
            continue
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return None


def _create_anthropic_message_with_retry(
    *,
    client: Any,
    request: dict[str, Any],
    question_id: str,
    iteration: int,
    max_retries: int,
    base_backoff_s: float,
    max_backoff_s: float,
    meta: dict[str, Any],
) -> Any:
    retry_count = 0

    while True:
        try:
            return client.messages.create(**request)
        except Exception as exc:  # noqa: BLE001
            if not _is_rate_limit_error(exc) or retry_count >= max_retries:
                raise

            retry_count += 1
            retry_after = _extract_retry_after_seconds(exc)
            backoff = min(max_backoff_s, base_backoff_s * (2 ** (retry_count - 1)))
            sleep_s = retry_after if retry_after is not None else backoff
            sleep_s = max(0.0, min(max_backoff_s, float(sleep_s)))

            meta["rate_limit_retries"] = int(meta.get("rate_limit_retries", 0)) + 1
            print(
                f"[run_mcp_agent] question_id={question_id} iteration={iteration} "
                f"rate_limit_retry={retry_count}/{max_retries} sleep_s={sleep_s:.2f}"
            )
            time.sleep(sleep_s)


def _anthropic_block_to_dict(block: Any) -> dict[str, Any]:
    if isinstance(block, dict):
        return block
    if hasattr(block, "model_dump"):
        dumped = block.model_dump()
        if isinstance(dumped, dict):
            return dumped

    block_type = getattr(block, "type", "")
    if block_type == "text":
        return {"type": "text", "text": getattr(block, "text", "")}
    if block_type == "tool_use":
        return {
            "type": "tool_use",
            "id": getattr(block, "id", ""),
            "name": getattr(block, "name", ""),
            "input": getattr(block, "input", {}),
        }
    return {"type": str(block_type), "value": str(block)}


def _extract_text_and_tool_uses(content_blocks: list[Any]) -> tuple[str, list[dict[str, Any]]]:
    text_parts: list[str] = []
    tool_uses: list[dict[str, Any]] = []
    for block in content_blocks:
        block_type = getattr(block, "type", "")
        if block_type == "text":
            text = getattr(block, "text", "")
            if isinstance(text, str) and text.strip():
                text_parts.append(text.strip())
        if block_type == "tool_use":
            tool_uses.append(
                {
                    "id": getattr(block, "id", ""),
                    "name": getattr(block, "name", ""),
                    "input": getattr(block, "input", {}),
                }
            )
    return " ".join(text_parts).strip(), tool_uses


def _tool_result_contexts(tool_name: str, structured_result: Any) -> list[str]:
    contexts: list[str] = []
    payload = structured_result
    if isinstance(payload, dict):
        if "results" in payload and isinstance(payload.get("results"), dict):
            bindings = payload["results"].get("bindings")
            if isinstance(bindings, list):
                for row in bindings[:3]:
                    contexts.append(truncate_text(f"{tool_name}: {to_jsonable(row)}"))
        if "triples" in payload and isinstance(payload.get("triples"), list):
            for row in payload["triples"][:3]:
                contexts.append(truncate_text(f"{tool_name}: {to_jsonable(row)}"))
        if "results" in payload and isinstance(payload.get("results"), list):
            for row in payload["results"][:3]:
                contexts.append(truncate_text(f"{tool_name}: {to_jsonable(row)}"))
    if not contexts:
        contexts.append(truncate_text(f"{tool_name}: {to_jsonable(payload)}"))
    return contexts


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MCP agent evaluation with Claude tool use.")
    parser.add_argument(
        "--dataset",
        default="eval/datasets/core_eval_v1.jsonl",
        help="Path to the evaluation dataset JSONL file.",
    )
    parser.add_argument(
        "--output",
        default="eval/outputs/raw/mcp_agent_raw.jsonl",
        help="Path to write raw runner output JSONL.",
    )
    parser.add_argument(
        "--max-iterations",
        default=6,
        type=int,
        help="Tool loop iteration cap (used in later milestones).",
    )
    parser.add_argument(
        "--max-tool-calls-per-turn",
        default=4,
        type=int,
        help="Maximum tool calls to execute from one Claude response.",
    )
    parser.add_argument(
        "--prompt",
        default="eval/prompts/mcp_agent_v1.txt",
        help="Prompt file for MCP agent answering.",
    )
    parser.add_argument(
        "--model",
        default=settings.anthropic_llm_model,
        help="Anthropic model for MCP agent answering.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=400,
        help="Maximum response tokens per Claude call.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max number of dataset rows to process (0 = all).",
    )
    parser.add_argument(
        "--rate-limit-retries",
        type=int,
        default=8,
        help="Retries for Anthropic 429 responses per messages.create call.",
    )
    parser.add_argument(
        "--rate-limit-backoff-s",
        type=float,
        default=3.0,
        help="Base backoff seconds for Anthropic 429 retries.",
    )
    parser.add_argument(
        "--rate-limit-max-backoff-s",
        type=float,
        default=90.0,
        help="Max sleep seconds between Anthropic 429 retries.",
    )
    parser.add_argument(
        "--inter-request-delay-s",
        type=float,
        default=0.0,
        help="Optional delay before each Anthropic request to reduce burst rate.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dataset_path = resolve_repo_path(args.dataset)
    output_path = resolve_repo_path(args.output)
    prompt_path = resolve_repo_path(args.prompt)

    if not dataset_path.exists():
        raise SystemExit(f"Dataset not found: {dataset_path}")
    if not prompt_path.exists():
        raise SystemExit(f"Prompt not found: {prompt_path}")
    if args.max_iterations <= 0:
        raise SystemExit("--max-iterations must be > 0")
    if args.max_tool_calls_per_turn <= 0:
        raise SystemExit("--max-tool-calls-per-turn must be > 0")
    if args.rate_limit_retries < 0:
        raise SystemExit("--rate-limit-retries must be >= 0")
    if args.rate_limit_backoff_s < 0:
        raise SystemExit("--rate-limit-backoff-s must be >= 0")
    if args.rate_limit_max_backoff_s <= 0:
        raise SystemExit("--rate-limit-max-backoff-s must be > 0")
    if args.inter_request_delay_s < 0:
        raise SystemExit("--inter-request-delay-s must be >= 0")

    ensure_output_dirs()
    rows = select_dataset_rows(load_jsonl(dataset_path), args.limit)
    prompt_text = load_text_file(prompt_path)
    init_jsonl_output(output_path)

    model = args.model
    mcp_server = create_server()
    tools = asyncio.run(mcp_server.list_tools())
    anthropic_tools: list[dict[str, Any]] = []
    for tool in tools:
        anthropic_tools.append(
            {
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": tool.inputSchema or {"type": "object", "properties": {}},
            }
        )

    anthropic_client: Any = None
    client_error: Exception | None = None
    try:
        from anthropic import Anthropic

        if not settings.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required for run_mcp_agent.")
        anthropic_client = Anthropic(
            api_key=settings.anthropic_api_key,
            timeout=settings.ollama_timeout_s,
            max_retries=0,
        )
    except Exception as exc:  # noqa: BLE001
        client_error = exc

    print(
        f"[run_mcp_agent] Starting run. "
        f"Loaded {len(rows)} rows from {dataset_path}."
    )
    print(f"[run_mcp_agent] Prompt: {prompt_path}")
    print(
        f"[run_mcp_agent] Discovered tools={len(anthropic_tools)} "
        f"names={[tool['name'] for tool in anthropic_tools]}"
    )
    print(f"[run_mcp_agent] Output: {output_path}")

    failed = 0
    for idx, row in enumerate(rows, start=1):
        question = str(row.get("question", "")).strip()
        qid = question_id_from_row(row, idx)
        if not question:
            failed += 1
            err = RuntimeError("Dataset row has empty question.")
            log_runner_error("mcp_agent", qid, err)
            append_jsonl_row(
                output_path,
                build_error_row(
                    question_id=qid,
                    system="mcp_agent",
                    model=model,
                    question=question,
                    error=err,
                    meta={"dataset_row": row},
                ),
            )
            continue

        started = time.perf_counter()
        tool_trace: list[dict[str, Any]] = []
        retrieved_contexts: list[str] = []
        final_answer = ABSTAIN
        meta: dict[str, Any] = {
            "prompt_version": prompt_path.name,
            "max_iterations": args.max_iterations,
            "max_tool_calls_per_turn": args.max_tool_calls_per_turn,
            "tool_schemas": [tool["name"] for tool in anthropic_tools],
            "rate_limit_retries_max": args.rate_limit_retries,
            "rate_limit_backoff_s": args.rate_limit_backoff_s,
            "rate_limit_max_backoff_s": args.rate_limit_max_backoff_s,
            "inter_request_delay_s": args.inter_request_delay_s,
        }

        if anthropic_client is None:
            failed += 1
            assert client_error is not None
            log_runner_error("mcp_agent", qid, client_error)
            append_jsonl_row(
                output_path,
                build_error_row(
                    question_id=qid,
                    system="mcp_agent",
                    model=model,
                    question=question,
                    error=client_error,
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    meta=meta,
                ),
            )
            continue

        messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
        try:
            for iteration in range(1, args.max_iterations + 1):
                if args.inter_request_delay_s > 0:
                    time.sleep(args.inter_request_delay_s)

                response = _create_anthropic_message_with_retry(
                    client=anthropic_client,
                    request={
                        "model": model,
                        "system": prompt_text,
                        "messages": messages,
                        "tools": anthropic_tools,
                        "temperature": args.temperature,
                        "max_tokens": args.max_output_tokens,
                    },
                    question_id=qid,
                    iteration=iteration,
                    max_retries=args.rate_limit_retries,
                    base_backoff_s=args.rate_limit_backoff_s,
                    max_backoff_s=args.rate_limit_max_backoff_s,
                    meta=meta,
                )

                assistant_content = [_anthropic_block_to_dict(block) for block in response.content]
                text_answer, tool_uses = _extract_text_and_tool_uses(response.content)
                messages.append({"role": "assistant", "content": assistant_content})

                if not tool_uses:
                    final_answer = text_answer or ABSTAIN
                    meta["stop_reason"] = getattr(response, "stop_reason", None)
                    break

                if len(tool_uses) > args.max_tool_calls_per_turn:
                    meta.setdefault("tool_calls_truncated", 0)
                    meta["tool_calls_truncated"] += len(tool_uses) - args.max_tool_calls_per_turn
                selected_tool_uses = tool_uses[: args.max_tool_calls_per_turn]

                tool_results_blocks: list[dict[str, Any]] = []
                for tool_call in selected_tool_uses:
                    tool_name = str(tool_call.get("name", ""))
                    tool_input = tool_call.get("input", {})
                    if not isinstance(tool_input, dict):
                        tool_input = {}
                    tool_use_id = str(tool_call.get("id", ""))

                    tool_started = time.perf_counter()
                    try:
                        raw_result = asyncio.run(mcp_server.call_tool(tool_name, tool_input))
                        structured_result = extract_structured_tool_result(raw_result)
                        serialized_result = to_jsonable(structured_result)
                        tool_error = None
                        success = True
                    except Exception as exc:  # noqa: BLE001
                        serialized_result = {
                            "error": {
                                "type": type(exc).__name__,
                                "message": str(exc),
                            }
                        }
                        tool_error = str(exc)
                        success = False

                    tool_latency_ms = int((time.perf_counter() - tool_started) * 1000)
                    tool_trace.append(
                        {
                            "iteration": iteration,
                            "tool_name": tool_name,
                            "tool_use_id": tool_use_id,
                            "input": to_jsonable(tool_input),
                            "success": success,
                            "error": tool_error,
                            "latency_ms": tool_latency_ms,
                            "result": serialized_result,
                        }
                    )
                    retrieved_contexts.extend(_tool_result_contexts(tool_name, serialized_result))
                    tool_results_blocks.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": json.dumps(serialized_result, ensure_ascii=True),
                        }
                    )

                messages.append({"role": "user", "content": tool_results_blocks})
            else:
                meta["stop_reason"] = "max_iterations_exhausted"
                final_answer = ABSTAIN

            mcp_row = build_raw_row(
                question_id=qid,
                system="mcp_agent",
                model=model,
                question=question,
                answer=final_answer or ABSTAIN,
                latency_ms=int((time.perf_counter() - started) * 1000),
                retrieved_contexts=retrieved_contexts[:50],
                citations=[],
                tool_trace=tool_trace,
                meta={
                    **meta,
                    "tool_calls": len(tool_trace),
                },
            )
        except Exception as exc:  # noqa: BLE001
            failed += 1
            log_runner_error("mcp_agent", qid, exc)
            mcp_row = build_error_row(
                question_id=qid,
                system="mcp_agent",
                model=model,
                question=question,
                error=exc,
                latency_ms=int((time.perf_counter() - started) * 1000),
                meta={
                    **meta,
                    "tool_trace": tool_trace,
                },
            )

        append_jsonl_row(output_path, mcp_row)

    print(
        f"[run_mcp_agent] Completed rows={len(rows)} failed={failed} "
        f"output={output_path}"
    )


if __name__ == "__main__":
    main()
