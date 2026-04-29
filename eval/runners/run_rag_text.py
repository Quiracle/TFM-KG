from __future__ import annotations

import argparse
import time

import httpx

from common import (
    ABSTAIN,
    append_jsonl_row,
    build_error_row,
    build_raw_row,
    ensure_output_dirs,
    extract_retrieved_contexts_from_debug,
    init_jsonl_output,
    load_jsonl,
    load_text_file,
    log_runner_error,
    question_id_from_row,
    resolve_repo_path,
    select_dataset_rows,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the RAG baseline through /query.")
    parser.add_argument(
        "--dataset",
        default="eval/datasets/core_eval_v1.jsonl",
        help="Path to the evaluation dataset JSONL file.",
    )
    parser.add_argument(
        "--output",
        default="eval/outputs/raw/rag_text_raw.jsonl",
        help="Path to write raw runner output JSONL.",
    )
    parser.add_argument(
        "--api-base-url",
        default="http://localhost:8000",
        help="API base URL.",
    )
    parser.add_argument(
        "--prompt",
        default="eval/prompts/answer_with_context_v1.txt",
        help="System prompt file for context-based answering.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="top_k passed to /query.",
    )
    parser.add_argument(
        "--dataset-version",
        default="dev",
        help="dataset_version passed to /query.",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=45.0,
        help="HTTP timeout in seconds for /query.",
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

    print(
        f"[run_rag_text] Starting run. "
        f"Loaded {len(rows)} rows from {dataset_path}."
    )
    print(f"[run_rag_text] Prompt: {prompt_path} ({len(prompt_text.splitlines())} lines)")
    print(f"[run_rag_text] API base URL: {args.api_base_url}")
    print("[run_rag_text] API mode: hybrid")
    print(f"[run_rag_text] Output: {output_path}")

    failed = 0
    endpoint = args.api_base_url.rstrip("/") + "/query"
    with httpx.Client(timeout=args.timeout_s) as client:
        for idx, row in enumerate(rows, start=1):
            question = str(row.get("question", "")).strip()
            qid = question_id_from_row(row, idx)
            if not question:
                failed += 1
                err = RuntimeError("Dataset row has empty question.")
                log_runner_error("rag_text", qid, err)
                append_jsonl_row(
                    output_path,
                    build_error_row(
                        question_id=qid,
                        system="rag_text",
                        model="api:/query",
                        question=question,
                        error=err,
                        meta={"dataset_row": row},
                    ),
                )
                continue

            started = time.perf_counter()
            try:
                response = client.post(
                    endpoint,
                    json={
                        "question": question,
                        "mode": "hybrid",
                        "top_k": args.top_k,
                        "dataset_version": args.dataset_version,
                        "debug": True,
                        "system_prompt": prompt_text,
                    },
                )
                response.raise_for_status()
                payload = response.json()

                answer = str(payload.get("answer", "")).strip() or ABSTAIN
                citations = payload.get("citations", [])
                if not isinstance(citations, list):
                    citations = []
                debug = payload.get("debug", {})
                if not isinstance(debug, dict):
                    debug = {}

                rag_row = build_raw_row(
                    question_id=qid,
                    system="rag_text",
                    model=str(debug.get("prompt_meta", {}).get("llm_model", "api:/query")),
                    question=question,
                    answer=answer,
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    retrieved_contexts=extract_retrieved_contexts_from_debug(debug),
                    citations=citations,
                    tool_trace=[],
                    forced_abstained=bool(payload.get("abstained", False)),
                    meta={
                        "prompt_version": prompt_path.name,
                        "api_mode_requested": "hybrid",
                        "api_mode": payload.get("mode"),
                        "api_top_k": payload.get("top_k"),
                        "api_abstained": payload.get("abstained"),
                        "retrieved_chunk_ids": debug.get("retrieved_chunk_ids"),
                        "abstain_reason": debug.get("abstain_reason"),
                        "providers": debug.get("providers"),
                    },
                )
            except Exception as exc:  # noqa: BLE001
                failed += 1
                log_runner_error("rag_text", qid, exc)
                rag_row = build_error_row(
                    question_id=qid,
                    system="rag_text",
                    model="api:/query",
                    question=question,
                    error=exc,
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    meta={"prompt_version": prompt_path.name, "endpoint": endpoint},
                )

            append_jsonl_row(output_path, rag_row)

    print(
        f"[run_rag_text] Completed rows={len(rows)} failed={failed} "
        f"output={output_path}"
    )


if __name__ == "__main__":
    main()
