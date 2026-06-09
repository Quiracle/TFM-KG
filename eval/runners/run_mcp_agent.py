from __future__ import annotations

import argparse
import asyncio
import json
import time
from typing import Any

import httpx

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

try:
    from google import genai
    from google.genai import errors as gemini_errors
    from google.genai import types as gemini_types
except ImportError:
    genai = None
    gemini_errors = None
    gemini_types = None


def _default_model_for_provider(provider: str) -> str:
    if provider == "anthropic":
        return settings.anthropic_llm_model
    if provider == "gemini":
        return settings.gemini_llm_model
    raise SystemExit("run_mcp_agent supports LLM_PROVIDER=anthropic or LLM_PROVIDER=gemini.")


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


def _clean_gemini_schema(value: Any) -> Any:
    if isinstance(value, list):
        return [_clean_gemini_schema(item) for item in value]
    if not isinstance(value, dict):
        return value

    if "anyOf" in value:
        non_null = [
            item for item in value["anyOf"]
            if not (isinstance(item, dict) and item.get("type") == "null")
        ]
        if len(non_null) == 1:
            return _clean_gemini_schema(non_null[0])

    cleaned: dict[str, Any] = {}
    for key, item in value.items():
        if key in {"title", "default"}:
            continue
        cleaned[key] = _clean_gemini_schema(item)
    return cleaned


def _build_gemini_tool(tools: list[Any]) -> Any:
    if gemini_types is None:
        raise RuntimeError("google-genai is required when LLM_PROVIDER=gemini.")

    declarations = []
    for tool in tools:
        schema = _clean_gemini_schema(tool.inputSchema or {"type": "object", "properties": {}})
        declarations.append(
            gemini_types.FunctionDeclaration(
                name=tool.name,
                description=tool.description or "",
                parameters_json_schema=schema,
            )
        )
    return gemini_types.Tool(function_declarations=declarations)


def _create_gemini_client() -> Any:
    if genai is None or gemini_types is None:
        raise RuntimeError("google-genai is required when LLM_PROVIDER=gemini.")
    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is required when LLM_PROVIDER=gemini.")

    retry_options = gemini_types.HttpRetryOptions(attempts=1)
    return genai.Client(
        api_key=settings.gemini_api_key,
        http_options=gemini_types.HttpOptions(
            timeout=settings.ollama_timeout_s * 1000,
            retry_options=retry_options,
        ),
    )


def _create_gemini_content_with_retry(
    *,
    client: Any,
    model: str,
    contents: list[Any],
    config: Any,
) -> Any:
    api_errors = () if gemini_errors is None else (gemini_errors.APIError,)
    retries = 3
    for attempt in range(1, retries + 1):
        try:
            return client.models.generate_content(model=model, contents=contents, config=config)
        except api_errors + (httpx.TimeoutException, httpx.NetworkError) as exc:
            is_last = attempt == retries
            status_code = getattr(exc, "code", None)
            retryable = isinstance(exc, (httpx.TimeoutException, httpx.NetworkError))
            retryable = retryable or status_code is None or status_code == 429 or status_code >= 500
            if is_last or not retryable:
                raise RuntimeError(f"Gemini MCP agent request failed: {exc}") from None
            time.sleep(0.3 * attempt)
    raise RuntimeError("Gemini MCP agent request failed after retries.")


def _gemini_response_content(response: Any) -> Any | None:
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return None
    return getattr(candidates[0], "content", None)


def _extract_gemini_text_and_function_calls(response: Any) -> tuple[str, list[Any]]:
    content = _gemini_response_content(response)
    text_parts: list[str] = []
    for part in getattr(content, "parts", None) or []:
        text = getattr(part, "text", None)
        if isinstance(text, str) and text.strip():
            text_parts.append(text.strip())
    function_calls = getattr(response, "function_calls", None) or []
    return " ".join(text_parts).strip(), list(function_calls)


def _gemini_finish_reason(response: Any) -> str | None:
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return None
    finish_reason = getattr(candidates[0], "finish_reason", None)
    return None if finish_reason is None else str(finish_reason)


def _run_gemini_mcp_answer(
    *,
    client: Any,
    model: str,
    prompt_text: str,
    gemini_tool: Any,
    mcp_server: Any,
    args: argparse.Namespace,
    question_id: str,
    question: str,
    started: float,
    meta: dict[str, Any],
) -> dict[str, Any]:
    if gemini_types is None:
        raise RuntimeError("google-genai is required when LLM_PROVIDER=gemini.")

    contents = [
        gemini_types.Content(
            role="user",
            parts=[gemini_types.Part(text=question)],
        )
    ]
    config = gemini_types.GenerateContentConfig(
        system_instruction=prompt_text,
        tools=[gemini_tool],
        tool_config=gemini_types.ToolConfig(
            function_calling_config=gemini_types.FunctionCallingConfig(mode="AUTO")
        ),
        automatic_function_calling=gemini_types.AutomaticFunctionCallingConfig(disable=True),
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
    )

    tool_trace: list[dict[str, Any]] = []
    retrieved_contexts: list[str] = []
    final_answer = ABSTAIN

    for iteration in range(1, args.max_iterations + 1):
        if args.inter_request_delay_s > 0:
            time.sleep(args.inter_request_delay_s)

        response = _create_gemini_content_with_retry(
            client=client,
            model=model,
            contents=contents,
            config=config,
        )
        response_content = _gemini_response_content(response)
        if response_content is not None:
            contents.append(response_content)

        text_answer, function_calls = _extract_gemini_text_and_function_calls(response)
        if not function_calls:
            final_answer = text_answer or ABSTAIN
            meta["stop_reason"] = _gemini_finish_reason(response)
            break

        if len(function_calls) > args.max_tool_calls_per_turn:
            meta.setdefault("tool_calls_truncated", 0)
            meta["tool_calls_truncated"] += len(function_calls) - args.max_tool_calls_per_turn

        response_parts: list[Any] = []
        for function_call in function_calls[: args.max_tool_calls_per_turn]:
            tool_name = str(getattr(function_call, "name", "") or "")
            raw_args = getattr(function_call, "args", {}) or {}
            tool_input = dict(raw_args) if isinstance(raw_args, dict) else {}
            tool_use_id = str(getattr(function_call, "id", "") or f"{iteration}:{len(tool_trace) + 1}")

            tool_started = time.perf_counter()
            try:
                raw_result = asyncio.run(mcp_server.call_tool(tool_name, tool_input))
                structured_result = extract_structured_tool_result(raw_result)
                serialized_result = to_jsonable(structured_result)
                response_payload = {"output": serialized_result}
                tool_error = None
                success = True
            except Exception as exc:  # noqa: BLE001
                serialized_result = {
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
                }
                response_payload = serialized_result
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

            response_kwargs: dict[str, Any] = {"name": tool_name, "response": response_payload}
            if getattr(function_call, "id", None):
                response_kwargs["id"] = getattr(function_call, "id")
            response_parts.append(
                gemini_types.Part(
                    function_response=gemini_types.FunctionResponse(**response_kwargs)
                )
            )

        contents.append(gemini_types.Content(role="user", parts=response_parts))
    else:
        meta["stop_reason"] = "max_iterations_exhausted"
        final_answer = ABSTAIN

    return build_raw_row(
        question_id=question_id,
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MCP agent evaluation with tool use.")
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
        "--provider",
        default=None,
        help="LLM provider override. Defaults to LLM_PROVIDER from .env.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="LLM model override. Defaults to the configured model for the selected provider.",
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

    provider = str(args.provider or settings.llm_provider).strip().lower()
    model = str(args.model or _default_model_for_provider(provider)).strip()
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
    gemini_tool = _build_gemini_tool(tools) if provider == "gemini" else None

    anthropic_client: Any = None
    gemini_client: Any = None
    client_error: Exception | None = None
    try:
        if provider == "anthropic":
            from anthropic import Anthropic

            if not settings.anthropic_api_key:
                raise RuntimeError("ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic.")
            anthropic_client = Anthropic(
                api_key=settings.anthropic_api_key,
                timeout=settings.ollama_timeout_s,
                max_retries=0,
            )
        elif provider == "gemini":
            gemini_client = _create_gemini_client()
        else:
            raise RuntimeError("run_mcp_agent supports LLM_PROVIDER=anthropic or LLM_PROVIDER=gemini.")
    except Exception as exc:  # noqa: BLE001
        client_error = exc

    print(
        f"[run_mcp_agent] Starting run. "
        f"Loaded {len(rows)} rows from {dataset_path}."
    )
    print(f"[run_mcp_agent] Prompt: {prompt_path}")
    print(f"[run_mcp_agent] Provider: {provider}")
    print(f"[run_mcp_agent] Model: {model}")
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
            "llm_provider": provider,
            "rate_limit_retries_max": args.rate_limit_retries,
            "rate_limit_backoff_s": args.rate_limit_backoff_s,
            "rate_limit_max_backoff_s": args.rate_limit_max_backoff_s,
            "inter_request_delay_s": args.inter_request_delay_s,
        }

        if (provider == "anthropic" and anthropic_client is None) or (provider == "gemini" and gemini_client is None):
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
            if provider == "gemini":
                assert gemini_client is not None
                assert gemini_tool is not None
                mcp_row = _run_gemini_mcp_answer(
                    client=gemini_client,
                    model=model,
                    prompt_text=prompt_text,
                    gemini_tool=gemini_tool,
                    mcp_server=mcp_server,
                    args=args,
                    question_id=qid,
                    question=question,
                    started=started,
                    meta=meta,
                )
            else:
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
