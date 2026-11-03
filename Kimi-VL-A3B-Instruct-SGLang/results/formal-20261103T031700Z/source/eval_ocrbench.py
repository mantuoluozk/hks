"""Pinned full OCRBench via local API; preserve errors/truncations in denominator."""
import argparse
import base64
import concurrent.futures
import hashlib
import io
import json
from pathlib import Path
import time

from PIL import Image
import pyarrow.parquet as pq
import requests

ROOT = Path(__file__).resolve().parents[1]
MODEL = "kimi-vl-a3b-instruct"
URL = "http://127.0.0.1:30004"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(8*1024*1024), b""):
            h.update(data)
    return h.hexdigest()


def grade(row, prediction):
    if prediction.get("error"):
        return False
    def normalize(text):
        text = text.strip().replace("\n", " ")
        return text.replace(" ", "") if row["dataset"] == "HME100k" else text.lower()
    answers = row["answer"] if isinstance(row["answer"], list) else [row["answer"]]
    return any(normalize(answer) in normalize(prediction["output"]) for answer in answers)


def infer(row, max_tokens):
    data = row["image"]["bytes"]
    with Image.open(io.BytesIO(data)) as image:
        mime, size = Image.MIME[image.format], image.size
    content = [{"type": "image_url", "image_url": {"url": f"data:{mime};base64,"+base64.b64encode(data).decode()}},
               {"type": "text", "text": row["question"]}]
    payload = {"model": MODEL, "messages": [{"role": "user", "content": content}],
               "temperature": 0, "top_p": 1, "seed": 42, "max_tokens": max_tokens, "stream": False}
    result = {"id": row["_id"], "category": row["question_type"], "dataset": row["dataset"],
              "question": row["question"], "image_sha256": hashlib.sha256(data).hexdigest(),
              "image_size": size, "error": None}
    start = time.monotonic()
    try:
        with requests.Session() as session:
            session.trust_env = False
            response = session.post(URL+"/v1/chat/completions", json=payload, timeout=(10, 600))
            response.raise_for_status()
            raw = response.json()
        choice = raw["choices"][0]
        result.update(output=choice["message"].get("content") or "", finish_reason=choice["finish_reason"],
                      usage=raw.get("usage"), raw_response=raw)
        if not result["output"].strip():
            result["error"] = "empty output"
    except Exception as error:
        result.update(output="", error=str(error))
    result["seconds"] = time.monotonic()-start
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, default=ROOT/"data/ocrbench/test.parquet")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--smoke-summary", type=Path, required=True)
    args = parser.parse_args()
    assert json.loads(args.smoke_summary.read_text())["all_passed"], "Functional gate has not passed"
    meta = json.loads((ROOT/"sources/ocrbench-dataset.json").read_text())
    source = meta["files"][0]
    assert args.data_path.stat().st_size == source["size"]
    assert digest(args.data_path) == source["lfs"]["oid"]
    samples = pq.read_table(args.data_path).to_pylist()
    assert len(samples) == 1000
    for i, row in enumerate(samples):
        row["_id"] = str(row.get("id", f"ocrbench_test_{i:04d}"))
    assert len({row["_id"] for row in samples}) == 1000
    if args.limit:
        samples = samples[:args.limit]
    manifest = {"model": MODEL, "revision": "398eede0903cd983a2bfa0cc634e9ac1d843f375",
                "dataset": meta, "ids": [r["_id"] for r in samples], "workers": args.workers,
                "generation": {"temperature": 0, "top_p": 1, "seed": 42, "max_tokens": args.max_tokens},
                "prompt": "Original OCRBench question and original image; no benchmark-specific resizing override",
                "evaluator_sha256": digest(Path(__file__)),
                "source_scorer_sha256": digest(ROOT/"sources/ocrbench_example.py"),
                "launch_script_sha256": digest(ROOT/"serve-scnet.sh"),
                "environment_script_sha256": digest(ROOT/"run-scnet.sh"),
                "public_baseline": 867, "official_accuracy_pass": None}
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/"manifest.json").exists():
        assert json.loads((args.output/"manifest.json").read_text()) == manifest, "Resume protocol mismatch"
    else:
        (args.output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    raw_path = args.output/"predictions.jsonl"
    records = {}
    if raw_path.exists():
        for line in raw_path.read_text().splitlines():
            record = json.loads(line)
            assert record["id"] not in records
            records[record["id"]] = record
    assert set(records).issubset({r["_id"] for r in samples})
    pending = [row for row in samples if row["_id"] not in records]
    with raw_path.open("a") as stream, concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        for record in pool.map(lambda row: infer(row, args.max_tokens), pending):
            stream.write(json.dumps(record, ensure_ascii=False)+"\n")
            stream.flush()
            records[record["id"]] = record
            print(json.dumps({"completed": len(records), "total": len(samples), "id": record["id"],
                              "error": record["error"]}), flush=True)
    categories = {}
    graded = []
    for row in samples:
        result = dict(records[row["_id"]], correct=grade(row, records[row["_id"]]))
        graded.append(result)
        group = categories.setdefault(row["question_type"], {"count": 0, "correct": 0})
        group["count"] += 1
        group["correct"] += int(result["correct"])
    (args.output/"scored.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False)+"\n" for row in graded))
    summary = {"samples": len(samples), "full_dataset": not bool(args.limit),
               "correct": sum(r["correct"] for r in graded), "errors": sum(bool(r["error"]) for r in graded),
               "truncated": sum(r.get("finish_reason") == "length" for r in graded),
               "categories": categories, "published_score_per_1000": 867,
               "official_accuracy_pass": None, "exact_publisher_harness_equivalence": None}
    summary["accuracy"] = summary["correct"]/len(samples)
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary), flush=True)
    if summary["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
