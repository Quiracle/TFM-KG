# Evaluation Summary

- Source rows: 15
- Used RAGAS scored file: True
- Used judge scored file: True

## System Comparison

| system | total_rows | scored_rows | correctness_rate | abstain_accuracy | hallucination_rate | mean_latency_ms |
| --- | --- | --- | --- | --- | --- | --- |
| mcp_agent | 5 | 5 | 0.400 | NA | 0.000 | 28198.400 |
| no_context | 5 | 5 | 0.000 | NA | 0.800 | 2813.400 |
| rag_text | 5 | 5 | 0.000 | NA | 0.000 | 139.200 |

## Retrieval Diagnostics

| system | rows_with_context | rows_with_ragas_scores | faithfulness | context_precision | context_recall | tool_calls_per_answer |
| --- | --- | --- | --- | --- | --- | --- |
| rag_text | 0 | 0 | NA | NA | NA | 0.000 |
| mcp_agent | 5 | 0 | NA | NA | NA | 6.200 |

## Error Slices

| slice | system | total_cases | judged_cases | incorrect_cases | error_rate |
| --- | --- | --- | --- | --- | --- |
| single_hop | all | 15 | 15 | 13 | 0.867 |
| single_hop | mcp_agent | 5 | 5 | 3 | 0.600 |
| single_hop | no_context | 5 | 5 | 5 | 1.000 |
| single_hop | rag_text | 5 | 5 | 5 | 1.000 |
| multi_hop | all | 0 | 0 | 0 | NA |
| multi_hop | mcp_agent | 0 | 0 | 0 | NA |
| multi_hop | no_context | 0 | 0 | 0 | NA |
| multi_hop | rag_text | 0 | 0 | 0 | NA |
| trap_questions | all | 0 | 0 | 0 | NA |
| trap_questions | mcp_agent | 0 | 0 | 0 | NA |
| trap_questions | no_context | 0 | 0 | 0 | NA |
| trap_questions | rag_text | 0 | 0 | 0 | NA |
| abstain_cases | all | 9 | 9 | 9 | 1.000 |
| abstain_cases | mcp_agent | 3 | 3 | 3 | 1.000 |
| abstain_cases | no_context | 1 | 1 | 1 | 1.000 |
| abstain_cases | rag_text | 5 | 5 | 5 | 1.000 |
