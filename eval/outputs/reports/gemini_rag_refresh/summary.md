# Evaluation Summary

- Source rows: 72
- Used RAGAS scored file: False
- Used judge scored file: True

## System Comparison

| system | total_rows | scored_rows | correctness_rate | abstain_accuracy | hallucination_rate | mean_latency_ms |
| --- | --- | --- | --- | --- | --- | --- |
| rag_text | 72 | 72 | 0.361 | 1.000 | 0.014 | 1798.250 |

## Retrieval Diagnostics

| system | rows_with_context | rows_with_ragas_scores | faithfulness | context_precision | context_recall | tool_calls_per_answer |
| --- | --- | --- | --- | --- | --- | --- |
| rag_text | 72 | 0 | NA | NA | NA | 0.000 |
| mcp_agent | 0 | 0 | NA | NA | NA | NA |

## Error Slices

| slice | system | total_cases | judged_cases | incorrect_cases | error_rate |
| --- | --- | --- | --- | --- | --- |
| single_hop | all | 20 | 20 | 7 | 0.350 |
| single_hop | rag_text | 20 | 20 | 7 | 0.350 |
| multi_hop | all | 11 | 11 | 11 | 1.000 |
| multi_hop | rag_text | 11 | 11 | 11 | 1.000 |
| trap_questions | all | 8 | 8 | 0 | 0.000 |
| trap_questions | rag_text | 8 | 8 | 0 | 0.000 |
| abstain_cases | all | 46 | 46 | 38 | 0.826 |
| abstain_cases | rag_text | 46 | 46 | 38 | 0.826 |
