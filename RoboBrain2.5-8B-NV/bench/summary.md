# SCNet BW benchmark (self-test; not official acceptance)

See config.txt, env.sh and both raw JSONL/log files for the exact workload.

| random_input_len | random_output_len | max_concurrency | completed | duration | output_throughput | total_throughput | mean_ttft_ms | p99_ttft_ms | mean_tpot_ms | mean_e2e_latency_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4096 | 1024 | 1 | 8 | 324.36 | 25.26 | 182.28 | 3285.35 | 3315.43 | 36.42 | 40538.37 |
| 4096 | 1024 | 8 | 16 | 129.03 | 126.98 | 916.97 | 15763.24 | 24508.58 | 47.62 | 64475.85 |
| 4096 | 1024 | 64 | 128 | 714.57 | 183.43 | 1323.68 | 181088.22 | 333445.14 | 111.36 | 295005.24 |
| 16384 | 1024 | 1 | 8 | 380.92 | 21.51 | 427.51 | 6989.56 | 7036.10 | 39.71 | 47608.71 |
| 16384 | 1024 | 8 | 16 | 208.09 | 78.73 | 1563.83 | 30050.46 | 50258.33 | 72.29 | 104006.37 |
| 16384 | 1024 | 64 | 128 | 1542.93 | 84.95 | 1685.76 | 516300.77 | 715605.26 | 88.87 | 607212.44 |
