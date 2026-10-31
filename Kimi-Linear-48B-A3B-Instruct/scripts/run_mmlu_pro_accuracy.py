#!/usr/bin/env python3
"""Run the publisher MMLU-Pro prompt/extractor against a local OpenAI API."""
import argparse
import asyncio
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset
from openai import AsyncOpenAI

MODEL_ROOT = Path(__file__).resolve().parents[1]
PUBLISHER = MODEL_ROOT / "sources" / "mmlu-pro-upstream"
sys.path.insert(0, str(PUBLISHER))
from evaluate_from_apiX import extract_answer, format_example  # noqa: E402


def preprocess(split):
    grouped = defaultdict(list)
    for row in split:
        row = dict(row)
        row["options"] = [opt for opt in row["options"] if opt != "N/A"]
        grouped[row["category"]].append(row)
    return grouped


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True, type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:30148/v1")
    parser.add_argument("--model", default="kimi-linear-48b-a3b")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()
    args.job.mkdir(parents=True, exist_ok=True)
    raw_path = args.job / "responses.jsonl"
    result_dir = args.job / "official-results"
    result_dir.mkdir(exist_ok=True)

    data = load_dataset(
        "parquet",
        data_files={
            "test": str(PUBLISHER / "test.parquet"),
            "validation": str(PUBLISHER / "validation.parquet"),
        },
    )
    test, validation = preprocess(data["test"]), preprocess(data["validation"])
    if sum(map(len, test.values())) != 12032:
        raise RuntimeError("unexpected MMLU-Pro test row count")
    seen = set()
    if raw_path.exists():
        with raw_path.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if row.get("question_id") is not None:
                    seen.add(str(row["question_id"]))

    lock = asyncio.Lock()
    client = AsyncOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        base_url=args.url,
        timeout=3600.0,
        max_retries=0,
    )
    semaphore = asyncio.Semaphore(args.workers)
    total = sum(map(len, test.values()))
    done = len(seen)
    correct = 0
    started = time.monotonic()

    async def evaluate(category, row):
        nonlocal done, correct
        qid = str(row["question_id"])
        if qid in seen:
            return
        prefix = (
            f'The following are multiple choice questions (with answers) about {category}. '
            'Think step by step and then output the answer in the format of '
            '"The answer is (X)" at the end.\n\n'
        )
        for example in validation[category]:
            prefix += format_example(
                example["question"], example["options"], example["cot_content"]
            )
        prompt = prefix + format_example(row["question"], row["options"])
        record = {
            "question_id": row["question_id"],
            "question": row["question"],
            "options": row["options"],
            "answer": row["answer"],
            "category": category,
            "src": row.get("src"),
        }
        async with semaphore:
            try:
                response = await client.chat.completions.create(
                    model=args.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=args.temperature,
                    top_p=0.95,
                    max_tokens=args.max_tokens,
                    extra_body={"top_k": 20},
                )
                text = response.choices[0].message.content or ""
                pred = extract_answer(text)
                record.update(
                    pred=pred,
                    model_outputs=text,
                    finish_reason=response.choices[0].finish_reason,
                    prompt_tokens=response.usage.prompt_tokens if response.usage else None,
                    completion_tokens=response.usage.completion_tokens if response.usage else None,
                    error=None,
                )
            except Exception as e:
                record.update(pred=None, model_outputs="", error=f"{type(e).__name__}: {e}")
        async with lock:
            with raw_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
            seen.add(qid)
            done += 1
            correct += int(record.get("pred") == record["answer"])
            if done % 20 == 0 or done == total:
                elapsed = time.monotonic() - started
                print(
                    json.dumps(
                        {"completed": done, "total": total, "correct": correct,
                         "accuracy_so_far": correct / done, "elapsed_sec": round(elapsed, 1)},
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    try:
        tasks = [evaluate(cat, row) for cat, rows in test.items() for row in rows]
        await asyncio.gather(*tasks)
    finally:
        await client.close()

    records = []
    with raw_path.open(encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    by_category = defaultdict(list)
    for row in records:
        by_category[row["category"]].append(row)
    for category, rows in by_category.items():
        rows.sort(key=lambda r: r["question_id"])
        (result_dir / f"{category}_result.json").write_text(
            json.dumps(rows, ensure_ascii=False), encoding="utf-8"
        )
    valid = [r for r in records if not r.get("error")]
    exact = sum(r.get("pred") == r["answer"] for r in valid)
    summary = {
        "benchmark": "MMLU-Pro",
        "expected_examples": 12032,
        "responses": len(records),
        "api_errors": sum(bool(r.get("error")) for r in records),
        "unparseable_answers": sum(r.get("pred") is None for r in valid),
        "correct": exact,
        "accuracy": exact / len(records) if records else None,
        "paper_reference_percent": 72.7,
        "metric": "all test rows as denominator; publisher extract_answer; failures/unparseable are incorrect",
        "sampling": {"temperature": args.temperature, "top_p": 0.95, "top_k": 20},
        "max_tokens": args.max_tokens,
        "workers": args.workers,
    }
    (args.job / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
