"""Fixed demo-image HTTP performance matrix; no concurrent accuracy traffic."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import statistics
import time

import requests
from smoke_http import ROOT, request

QUESTION = "Describe this picture in detailed English sentences, including the buildings, their shapes, and the surroundings."


def call(image, budget):
    with requests.Session() as session:
        session.trust_env = False
        return request(session, QUESTION, [image], stream=True, max_tokens=budget)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke-summary", type=Path, required=True)
    parser.add_argument("--requests-per-cell", type=int, default=8)
    args = parser.parse_args()
    assert args.requests_per_cell >= 8 and args.requests_per_cell % 2 == 0
    assert json.loads(args.smoke_summary.read_text())["all_passed"]
    args.output.mkdir(parents=True, exist_ok=False)
    image = (ROOT/"data/examples/demo.png").read_bytes()
    warmups = []
    # Warm all tested shapes before the measured matrix in both fresh processes.
    for concurrency in [1, 2]:
        for budget in [32, 128]:
            with concurrent.futures.ThreadPoolExecutor(concurrency) as pool:
                warm = list(pool.map(lambda _: call(image, budget), range(2)))
            warmups.extend(warm)
            (args.output/"warmup.json").write_text(json.dumps(warmups))
            assert all(row["output"] and row["usage"]["completion_tokens"] > 0 for row in warm)
    workloads = []
    for concurrency in [1, 2]:
        for budget in [32, 128]:
            target = args.output/f"c{concurrency}-o{budget}"
            target.mkdir()
            # Identical per-cell warmup in each independent server run.
            with concurrent.futures.ThreadPoolExecutor(concurrency) as pool:
                warmups.extend(list(pool.map(lambda _: call(image, budget), range(2))))
                (args.output/"warmup.json").write_text(json.dumps(warmups))
            started = time.monotonic()
            with concurrent.futures.ThreadPoolExecutor(concurrency) as pool:
                rows = list(pool.map(lambda _: call(image, budget), range(args.requests_per_cell)))
            wall = time.monotonic()-started
            for i, row in enumerate(rows):
                assert row["output"] and row["done_marker"] and row["finish_reason"] in {"stop", "length"}
                assert row["usage"]["completion_tokens"] > 0
                row["request_index"] = i
                tokens = row["usage"]["completion_tokens"]
                row["tpot_seconds"] = ((row["seconds"]-row["ttft_seconds"])/(tokens-1)
                                       if tokens > 1 else None)
            tokens = sum(row["usage"]["completion_tokens"] for row in rows)
            summary = {"concurrency": concurrency, "max_tokens": budget,
                       "requests": len(rows), "wall_seconds": wall,
                       "completion_tokens": tokens, "output_tokens_per_second": tokens/wall,
                       "mean_latency_seconds": statistics.mean(row["seconds"] for row in rows),
                       "mean_ttft_seconds": statistics.mean(row["ttft_seconds"] for row in rows),
                       "mean_tpot_seconds": statistics.mean(row["tpot_seconds"] for row in rows
                                                             if row["tpot_seconds"] is not None),
                       "prompt_tokens": [row["usage"]["prompt_tokens"] for row in rows],
                       "stopped_at_limit": sum(row["finish_reason"] == "length" for row in rows)}
            (target/"requests.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows))
            (target/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
            workloads.append(summary)
            print(json.dumps(summary), flush=True)
    manifest = {"workloads": workloads, "counted_requests": sum(row["requests"] for row in workloads),
                "warmup_requests": 16, "image_sha256": hashlib.sha256(image).hexdigest(),
                "question": QUESTION, "timing_scope": "Client HTTP streaming: encode/send, server prefill/decode, receive; initialization excluded",
                "tpot_definition": "(client latency - first non-empty content latency) / (completion_tokens - 1)",
                "generation": {"temperature": 0, "top_p": 1, "seed": 42},
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "http_client_sha256": hashlib.sha256((ROOT/"scripts/smoke_http.py").read_bytes()).hexdigest(),
                "official_performance_protocol": False}
    (args.output/"summary.json").write_text(json.dumps(manifest, indent=2)+"\n")


if __name__ == "__main__":
    main()
