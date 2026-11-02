"""Verify saved evidence without executing a model or downloading data."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent


def read(relative):
    return json.loads((ROOT/relative).read_text(encoding="utf-8"))


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-8, abs_tol=1e-8), (a, b)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    manifest = read("results/evidence-manifest.json")
    for rel, expected in manifest["sha256"].items():
        path = (ROOT/rel).resolve()
        assert path.is_relative_to(ROOT.resolve())
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, rel
    weights = read("results/weights-download-20261028.json")
    assert weights["state"] == "complete_verified"
    assert weights["verified_files"] == weights["total_files"] == 20
    assert weights["downloaded_bytes"] == weights["expected_bytes"] == 32819646059
    publisher = read("sources/publisher-files.json")
    for file in publisher["siblings"]:
        if file["rfilename"] in weights["publisher_hash_checks"]:
            proof = weights["publisher_hash_checks"][file["rfilename"]]
            assert proof["bytes"] == file["size"]
            assert proof["hash"] == (file["lfs"]["sha256"] if file.get("lfs") else file["blobId"])
    for prefix in manifest["functional_runs"]:
        summary = read(prefix+"/summary.json")
        assert summary["all_passed"] and summary["functional_passed"] == summary["functional_total"] == 8
        rows = {key: read(prefix+"/"+key+".json") for key in
                ["text", "red", "blue-zh", "ocr", "demo", "two-images", "repeat", "stream"]}
        assert all(row["passed"] and not row.get("error") for row in rows.values())
        assert rows["text"]["output"].strip() == "4"
        assert "red" in rows["red"]["output"].lower()
        assert "蓝" in rows["blue-zh"]["output"] or "blue" in rows["blue-zh"]["output"].lower()
        assert "HKS2026" in re.sub(r"\s+", "", rows["ocr"]["output"])
        assert re.search(r"red.*blue", rows["two-images"]["output"].lower(), re.S)
        assert rows["red"]["output"] == rows["repeat"]["output"] == rows["stream"]["output"]
        stream = rows["stream"]
        assert stream["done_marker"] and stream["finish_reason"] == "stop"
        assert "".join(choice.get("delta", {}).get("content") or ""
                       for chunk in stream["chunks"] for choice in chunk.get("choices", [])) == stream["output"]
    accuracy = read("results/scnet/ocrbench-full-20261028/summary.json")
    scored = [json.loads(line) for line in (ROOT/"results/scnet/ocrbench-full-20261028/scored.jsonl").read_text(encoding="utf-8").splitlines()]
    original = [json.loads(line) for line in (ROOT/"results/scnet/ocrbench-full-20261028/predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    protocol = read("results/scnet/ocrbench-full-20261028/manifest.json")
    assert len(original) == len(scored) == accuracy["samples"] == 1000 and accuracy["full_dataset"]
    assert len({row["id"] for row in original}) == 1000
    assert [row["id"] for row in original] == protocol["ids"]
    assert all({k: v for k, v in a.items() if k != "correct"} == b for a, b in zip(scored, original))
    assert sum(row["correct"] for row in scored) == accuracy["correct"]
    assert sum(bool(row["error"]) for row in scored) == accuracy["errors"] == 0
    assert sum(row.get("finish_reason") == "length" for row in scored) == accuracy["truncated"]
    close(accuracy["accuracy"], accuracy["correct"]/1000)
    audit = read("results/scnet/ocrbench-full-20261028/upstream-scoring-audit.json")
    assert audit["all_matched"] and not audit["mismatched_ids"]
    assert audit["upstream_correct"] == audit["recorded_correct"] == accuracy["correct"]
    assert audit["predictions_sha256"] == hashlib.sha256((ROOT/"results/scnet/ocrbench-full-20261028/predictions.jsonl").read_bytes()).hexdigest()
    perf_runs = []
    for prefix in manifest["performance_runs"]:
        summary = read(prefix+"/summary.json")
        assert summary["counted_requests"] == 32 and len(summary["workloads"]) == 4
        for cell in summary["workloads"]:
            folder = prefix+f"/c{cell['concurrency']}-o{cell['max_tokens']}"
            rows = [json.loads(line) for line in (ROOT/(folder+"/requests.jsonl")).read_text(encoding="utf-8").splitlines()]
            assert len(rows) == cell["requests"] == 8
            assert all(row["done_marker"] and row["output"] for row in rows)
            tokens = sum(row["usage"]["completion_tokens"] for row in rows)
            assert tokens == cell["completion_tokens"]
            close(cell["output_tokens_per_second"], tokens/cell["wall_seconds"])
            close(cell["mean_latency_seconds"], sum(row["seconds"] for row in rows)/8)
            close(cell["mean_ttft_seconds"], sum(row["ttft_seconds"] for row in rows)/8)
        perf_runs.append(summary)
    a, b = perf_runs
    assert a["script_sha256"] == b["script_sha256"]
    assert a["question"] == b["question"] and a["image_sha256"] == b["image_sha256"]
    reproduction = read("results/scnet/reproduction-summary-20261028.json")
    deviations = []
    for first, second, saved in zip(a["workloads"], b["workloads"], reproduction["comparisons"]):
        assert (first["concurrency"], first["max_tokens"]) == (second["concurrency"], second["max_tokens"])
        delta = (second["output_tokens_per_second"]/first["output_tokens_per_second"]-1)*100
        close(delta, saved["deviation_percent"])
        deviations.append(abs(delta))
    close(max(deviations), reproduction["maximum_absolute_deviation_percent"])
    assert reproduction["performance_reproduction_passed"] == (max(deviations) <= 20)
    metric_audit = read("results/scnet/reproduction-metric-audit-20261028.json")
    full_deltas = []
    for first, second in zip(a["workloads"], b["workloads"]):
        for key in metric_audit["metrics"]:
            full_deltas.append(abs((second[key]/first[key]-1)*100))
    close(max(full_deltas), metric_audit["maximum_absolute_deviation_percent"])
    assert metric_audit["all_metrics_within_20_percent"] == (max(full_deltas) <= 20)
    optimized = read("results/optimization-20261102T020940Z/verified-optimization.json")
    assert optimized["measured_repeats"] == 1 and len(optimized["comparison"]) == 4
    assert all(row["same_text"] and row["same_output_lengths"] and row["same_input_lengths"] for row in optimized["comparison"])
    formal = read("results/formal-20261102T021857Z/independent-audit.json")
    assert formal["requests"] == 608 and formal["verified_cells"] == 12
    assert formal["backend_counters"][0]["generation_tokens"] == 623008
    entry = read("results/entry-20261102T032847Z/summary.json")
    assert entry["passed"] and entry["owned_service_stopped"] and entry["functional_passed"] == 1
    assert len(entry["weights_preflight"]["publisher_files"]) == 20
    result = {"scope": "Optimized-v1 and retained historical eager evidence",
              "optimized_official_requests": formal["requests"],
              "optimized_short_min_gain_percent": min(row["gain_pct"] for row in optimized["comparison"]),
              "standard_entry_passed": True,
              "evidence_files_verified": len(manifest["sha256"]), "historical_eager_functional_runs_passed": 2,
              "ocrbench_correct": accuracy["correct"], "ocrbench_total": 1000,
              "historical_eager_performance_reproduction_passed": reproduction["performance_reproduction_passed"],
              "historical_eager_all_metrics_within_20_percent": metric_audit["all_metrics_within_20_percent"],
              "official_accuracy_pass": None}
    print(json.dumps(result, indent=2))
    if not reproduction["performance_reproduction_passed"] or not metric_audit["all_metrics_within_20_percent"]:
        raise SystemExit(2)
    if not args.selfcheck:
        raise SystemExit(2)  # Competition acceptance is not established by local checks.


if __name__ == "__main__":
    main()
