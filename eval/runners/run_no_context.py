from __future__ import annotations

import argparse
import time
from typing import Any

from common import (
    ABSTAIN,
    append_jsonl_row,
    build_error_row,
    build_raw_row,
    ensure_output_dirs,
    init_jsonl_output,
    load_jsonl,
    load_text_file,
    log_runner_error,
    question_id_from_row,
    resolve_repo_path,
    select_dataset_rows,
)
from src.tfmkg.core.config import settings
from src.tfmkg.adapters.llm import AnthropicMessagesClient, GeminiGenerateAdapter, OllamaChatClient, OpenAIResponsesClient
from src.tfmkg.domain.ports.llm import LLMClientPort, LLMMessage


def _default_model_for_provider(provider: str) -> str:
    if provider == "openai":
        return settings.openai_llm_model
    if provider == "anthropic":
        return settings.anthropic_llm_model
    if provider == "gemini":
        return settings.gemini_llm_model
    if provider == "ollama":
        return settings.ollama_llm_model
    raise SystemExit(f"Unsupported LLM_PROVIDER={provider!r}. Use openai, anthropic, gemini, or ollama.")


def _build_llm_client(provider: str, model: str) -> LLMClientPort:
    if provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        return OpenAIResponsesClient(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            model=model,
            timeout_s=settings.ollama_timeout_s,
        )
    if provider == "anthropic":
        if not settings.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic")
        return AnthropicMessagesClient(
            api_key=settings.anthropic_api_key,
            model=model,
            timeout_s=settings.ollama_timeout_s,
        )
    if provider == "gemini":
        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is required when LLM_PROVIDER=gemini")
        return GeminiGenerateAdapter(
            api_key=settings.gemini_api_key,
            model=model,
            timeout_s=settings.ollama_timeout_s,
        )
    if provider == "ollama":
        return OllamaChatClient(
            base_url=settings.ollama_base_url,
            model=model,
            timeout_s=settings.ollama_timeout_s,
            stream=settings.ollama_stream,
        )
    raise SystemExit(f"Unsupported LLM_PROVIDER={provider!r}. Use openai, anthropic, gemini, or ollama.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the no-context baseline.")
    parser.add_argument(
        "--dataset",
        default="eval/datasets/core_eval_v1.jsonl",
        help="Path to the evaluation dataset JSONL file.",
    )
    parser.add_argument(
        "--output",
        default="eval/outputs/raw/no_context_raw.jsonl",
        help="Path to write raw runner output JSONL.",
    )
    parser.add_argument(
        "--prompt",
        default="eval/prompts/answer_no_context_v1.txt",
        help="Prompt file for no-context answering.",
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
        default=300,
        help="Maximum response tokens.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max number of dataset rows to process (0 = all).",
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

    ensure_output_dirs()
    rows = select_dataset_rows(load_jsonl(dataset_path), args.limit)
    prompt_text = load_text_file(prompt_path)
    init_jsonl_output(output_path)

    provider = str(args.provider or settings.llm_provider).strip().lower()
    model = str(args.model or _default_model_for_provider(provider)).strip()
    client: LLMClientPort | None = None
    client_error: Exception | None = None
    try:
        client = _build_llm_client(provider, model)
    except Exception as exc:  # noqa: BLE001
        client_error = exc

    print(
        f"[run_no_context] Starting run. "
        f"Loaded {len(rows)} rows from {dataset_path}."
    )
    print(f"[run_no_context] Prompt: {prompt_path}")
    print(f"[run_no_context] Provider: {provider}")
    print(f"[run_no_context] Model: {model}")
    print(f"[run_no_context] Output: {output_path}")

    failed = 0
    for idx, row in enumerate(rows, start=1):
        question = str(row.get("question", "")).strip()
        qid = question_id_from_row(row, idx)
        if not question:
            failed += 1
            err = RuntimeError("Dataset row has empty question.")
            log_runner_error("no_context", qid, err)
            append_jsonl_row(
                output_path,
                build_error_row(
                    question_id=qid,
                    system="no_context",
                    model=model,
                    question=question,
                    error=err,
                    meta={"dataset_row": row},
                ),
            )
            continue

        started = time.perf_counter()
        if client is None:
            failed += 1
            assert client_error is not None
            log_runner_error("no_context", qid, client_error)
            append_jsonl_row(
                output_path,
                build_error_row(
                    question_id=qid,
                    system="no_context",
                    model=model,
                    question=question,
                    error=client_error,
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    meta={"missing_client": True},
                ),
            )
            continue

        try:
            llm_result = client.generate(
                messages=[
                    LLMMessage(role="system", content=prompt_text),
                    LLMMessage(role="user", content=question),
                ],
                temperature=args.temperature,
                max_output_tokens=args.max_output_tokens,
            )
            answer = llm_result.text.strip() if llm_result.text else ""
            if not answer:
                answer = ABSTAIN
            raw_row = build_raw_row(
                question_id=qid,
                system="no_context",
                model=model,
                question=question,
                answer=answer,
                latency_ms=int((time.perf_counter() - started) * 1000),
                retrieved_contexts=[],
                citations=[],
                tool_trace=[],
                meta={
                    "prompt_version": prompt_path.name,
                    "llm_provider": provider,
                    "prompt_tokens": llm_result.prompt_tokens,
                    "completion_tokens": llm_result.completion_tokens,
                },
            )
        except Exception as exc:  # noqa: BLE001
            failed += 1
            log_runner_error("no_context", qid, exc)
            raw_row = build_error_row(
                question_id=qid,
                system="no_context",
                model=model,
                question=question,
                error=exc,
                latency_ms=int((time.perf_counter() - started) * 1000),
                meta={"prompt_version": prompt_path.name},
            )

        append_jsonl_row(output_path, raw_row)

    print(
        f"[run_no_context] Completed rows={len(rows)} failed={failed} "
        f"output={output_path}"
    )


if __name__ == "__main__":
    main()
