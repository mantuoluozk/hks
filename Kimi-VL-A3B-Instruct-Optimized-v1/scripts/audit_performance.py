"""Audit deviations of throughput, latency, TTFT and TPOT from saved runs."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
prefixes = [ROOT/"results/scnet"/name/"performance" for name in
            ["native-vllm-v3-20261028", "fresh-vllm-20261028"]]
summaries = []
fields = ["output_tokens_per_second", "mean_latency_seconds", "mean_ttft_seconds", "mean_tpot_seconds"]
for prefix in prefixes:
    summary = json.loads((prefix/"summary.json").read_text())
    for cell in summary["workloads"]:
        rows = [json.loads(line) for line in
                (prefix/f"c{cell['concurrency']}-o{cell['max_tokens']}/requests.jsonl").read_text().splitlines()]
        assert len(rows) == 8
        totals = {"mean_latency_seconds": sum(row["seconds"] for row in rows)/8,
                  "mean_ttft_seconds": sum(row["ttft_seconds"] for row in rows)/8,
                  "mean_tpot_seconds": sum((row["seconds"]-row["ttft_seconds"])/(row["usage"]["completion_tokens"]-1) for row in rows)/8,
                  "output_tokens_per_second": sum(row["usage"]["completion_tokens"] for row in rows)/cell["wall_seconds"]}
        assert all(math.isclose(totals[key], cell[key], rel_tol=1e-8) for key in fields)
    summaries.append(summary)
comparisons = []
for first, second in zip(summaries[0]["workloads"], summaries[1]["workloads"]):
    assert (first["concurrency"], first["max_tokens"]) == (second["concurrency"], second["max_tokens"])
    for key in fields:
        comparisons.append({"concurrency": first["concurrency"], "max_tokens": first["max_tokens"],
                            "metric": key, "first": first[key], "repeat": second[key],
                            "deviation_percent": (second[key]/first[key]-1)*100})
maximum = max(abs(row["deviation_percent"]) for row in comparisons)
result = {"comparisons": comparisons, "metrics": fields,
          "maximum_absolute_deviation_percent": maximum, "all_metrics_within_20_percent": maximum <= 20,
          "scope": "Four workload cells and four reported metrics; all independently recalculated from request records",
          "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in [prefix/"summary.json" for prefix in prefixes]}}
(ROOT/"results/scnet/reproduction-metric-audit-20261028.json").write_text(json.dumps(result, indent=2)+"\n")
print(json.dumps({k: v for k, v in result.items() if k != "comparisons"}), flush=True)
