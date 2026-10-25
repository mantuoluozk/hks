# Diagnostic benchmark (not an official acceptance verdict)

See config.txt, env.sh and both raw JSONL/log files for the exact workload.

| random_input_len | random_output_len | max_concurrency | completed | duration | output_throughput | total_throughput | mean_ttft_ms | p99_ttft_ms | mean_tpot_ms | mean_e2e_latency_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4096 | 1024 | 1 | 8 | 99.97 | 81.95 | 432.18 | 225.53 | 288.71 | 11.99 | 12492.26 |
| 4096 | 1024 | 8 | 16 | 40.43 | 405.25 | 2136.59 | 1688.97 | 3642.41 | 16.80 | 18878.86 |
| 4096 | 1024 | 64 | 128 | 180.59 | 725.79 | 3820.68 | 21451.09 | 70602.40 | 53.08 | 75749.13 |
| 16384 | 1024 | 1 | 8 | 108.05 | 75.81 | 1358.12 | 630.37 | 888.47 | 12.58 | 13502.02 |
| 16384 | 1024 | 8 | 16 | 68.94 | 237.65 | 4254.29 | 2551.70 | 5564.03 | 27.31 | 30493.16 |
| 16384 | 1024 | 64 | 128 | 568.76 | 230.45 | 4124.47 | 164826.46 | 297119.57 | 54.98 | 221073.75 |
