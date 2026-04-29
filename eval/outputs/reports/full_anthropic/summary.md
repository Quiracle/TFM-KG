# Evaluation Summary

- Source rows: 216
- Used RAGAS scored file: True
- Used judge scored file: True

## System Comparison

| system | total_rows | scored_rows | correctness_rate | abstain_accuracy | hallucination_rate | mean_latency_ms |
| --- | --- | --- | --- | --- | --- | --- |
| mcp_agent | 72 | 72 | 0.542 | 0.250 | 0.125 | 27940.292 |
| no_context | 72 | 72 | 0.167 | 1.000 | 0.292 | 2609.056 |
| rag_text | 72 | 72 | 0.111 | 1.000 | 0.000 | 169.194 |

## Retrieval Diagnostics

| system | rows_with_context | rows_with_ragas_scores | faithfulness | context_precision | context_recall | tool_calls_per_answer |
| --- | --- | --- | --- | --- | --- | --- |
| rag_text | 0 | 0 | NA | NA | NA | 0.000 |
| mcp_agent | 65 | 56 | 0.161 | 0.154 | 0.420 | 4.681 |

## Error Slices

| slice | system | total_cases | judged_cases | incorrect_cases | error_rate |
| --- | --- | --- | --- | --- | --- |
| single_hop | all | 60 | 60 | 46 | 0.767 |
| single_hop | mcp_agent | 20 | 20 | 7 | 0.350 |
| single_hop | no_context | 20 | 20 | 19 | 0.950 |
| single_hop | rag_text | 20 | 20 | 20 | 1.000 |
| multi_hop | all | 33 | 33 | 27 | 0.818 |
| multi_hop | mcp_agent | 11 | 11 | 5 | 0.455 |
| multi_hop | no_context | 11 | 11 | 11 | 1.000 |
| multi_hop | rag_text | 11 | 11 | 11 | 1.000 |
| trap_questions | all | 24 | 24 | 0 | 0.000 |
| trap_questions | mcp_agent | 8 | 8 | 0 | 0.000 |
| trap_questions | no_context | 8 | 8 | 0 | 0.000 |
| trap_questions | rag_text | 8 | 8 | 0 | 0.000 |
| abstain_cases | all | 121 | 121 | 103 | 0.851 |
| abstain_cases | mcp_agent | 32 | 32 | 30 | 0.938 |
| abstain_cases | no_context | 17 | 17 | 9 | 0.529 |
| abstain_cases | rag_text | 72 | 72 | 64 | 0.889 |
