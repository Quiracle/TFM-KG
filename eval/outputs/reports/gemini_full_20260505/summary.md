# Evaluation Summary

- Source rows: 216
- Used RAGAS scored file: True
- Used judge scored file: True

## System Comparison

| system | total_rows | scored_rows | correctness_rate | abstain_accuracy | hallucination_rate | mean_latency_ms |
| --- | --- | --- | --- | --- | --- | --- |
| mcp_agent | 72 | 72 | 0.125 | 1.000 | 0.028 | 6418.167 |
| no_context | 72 | 72 | 0.139 | 0.625 | 0.125 | 2062.139 |
| rag_text | 72 | 72 | 0.361 | 1.000 | 0.028 | 2274.819 |

## Retrieval Diagnostics

| system | rows_with_context | rows_with_ragas_scores | faithfulness | context_precision | context_recall | tool_calls_per_answer |
| --- | --- | --- | --- | --- | --- | --- |
| rag_text | 72 | 72 | 1.000 | 0.297 | 0.396 | 0.000 |
| mcp_agent | 59 | 59 | 0.143 | 0.130 | 0.035 | 2.250 |

## Error Slices

| slice | system | total_cases | judged_cases | incorrect_cases | error_rate |
| --- | --- | --- | --- | --- | --- |
| single_hop | all | 60 | 60 | 46 | 0.767 |
| single_hop | mcp_agent | 20 | 20 | 20 | 1.000 |
| single_hop | no_context | 20 | 20 | 19 | 0.950 |
| single_hop | rag_text | 20 | 20 | 7 | 0.350 |
| multi_hop | all | 33 | 33 | 33 | 1.000 |
| multi_hop | mcp_agent | 11 | 11 | 11 | 1.000 |
| multi_hop | no_context | 11 | 11 | 11 | 1.000 |
| multi_hop | rag_text | 11 | 11 | 11 | 1.000 |
| trap_questions | all | 24 | 24 | 2 | 0.083 |
| trap_questions | mcp_agent | 8 | 8 | 0 | 0.000 |
| trap_questions | no_context | 8 | 8 | 2 | 0.250 |
| trap_questions | rag_text | 8 | 8 | 0 | 0.000 |
| abstain_cases | all | 143 | 143 | 122 | 0.853 |
| abstain_cases | mcp_agent | 68 | 68 | 60 | 0.882 |
| abstain_cases | no_context | 29 | 29 | 24 | 0.828 |
| abstain_cases | rag_text | 46 | 46 | 38 | 0.826 |
