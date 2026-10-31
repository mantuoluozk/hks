#!/usr/bin/env python3
"""GSM8K + MMLU 200+200 public-sample evaluation for Kimi-Linear-48B-A3B-Instruct.

Reuses the exact MiMo-validated data files and scoring rules so the numbers are
protocol-identical to the sibling model's published self-evaluation:

  data   : MiMo-V2.6-Distill-Qwen-9B/data/{gsm8k,mmlu}.jsonl (sha256 pinned)
  GSM8K  : parse final '####  <number>' (last match), numeric compare via Decimal
  MMLU   : parse final 'ANSWER: X' line (X in A-D), exact letter compare
  sampling: temperature=0, top_p=1, greedy, enable_thinking=True, max_tokens=4096

Official note: the Kimi Linear publisher does NOT publish GSM8K or MMLU scores
for this Instruct checkpoint (paper Table 9 only reports MMLU-Pro 72.7, and the
README 51.0 is the 1.4T experimental model). These results are therefore a
self-measured adaptation sanity check, NOT a reproduction of a published baseline.
"""
import argparse
import concurrent.futures
import datetime
import hashlib
import json
import os
import re
import time
import urllib.request
from decimal import Decimal, InvalidOperation
from pathlib import Path

MIMO = Path("/root/private_data/hks/models/MiMo-V2.6-Distill-Qwen-9B")
EXPECTED_SHA = {
    "gsm8k": "03bd22060d201b82f3545a56dc89a01d300a5406d79e1d51529fdedceb698546",
    "mmlu": "84f929be5a22b7f7895961ef19f3842dc3d9a276a42b82c42a6fe95e1395682d",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(text):
    try:
        return Decimal(text.replace(",", "").strip())
    except (InvalidOperation, AttributeError):
        return None


def score(dataset, content, truth):
    text = (content or "").replace("**", "").strip()
    if dataset == "gsm8k":
        matches = re.findall(r"####\s*(-?\d[\d,]*(?:\.\d+)?)", text)
        pred = matches[-1] if matches else None
        return pred, pred is not None and number(pred) == number(truth)
    matches = re.findall(r"(?:^|\n)\s*ANSWER:\s*([ABCD])\s*[.。]?\s*$", text, re.I)
    pred = matches[-1].upper() if matches else None
    return pred, pred == truth


def prompt(dataset, item):
    if dataset == "gsm8k":
        return ("Solve the math problem. Explain briefly, then finish with #### followed by "
                "the final numeric answer.\n\n" + item["question"])
    choices = "\n".join(f"{letter}. {text}" for letter, text in zip("ABCD", item["choices"]))
    return ("Answer this multiple-choice question. Finish with a line exactly like ANSWER: X, "
            "replacing X with A, B, C, or D.\n\n" + item["question"] + "\n" + choices)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--url", default="http://127.0.0.1:30148")
    ap.add_argument("--model", default="kimi-linear-48b-a3b")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=4096)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    protocol = {
        "temperature": 0, "top_p": 1, "max_tokens": args.max_tokens,
        "enable_thinking": True, "few_shot": 0, "workers": args.workers,
        "scoring": "GSM8K final #### decimal; MMLU final ANSWER: A/B/C/D; strip bold; no random fallback",
        "scope": "fixed 200+200 public samples (MiMo-validated); not an official publisher baseline",
        "official_baseline_status": "publisher publishes NO GSM8K/MMLU score for this Instruct checkpoint",
    }

    tasks, sources = [], {}
    for dataset in ("gsm8k", "mmlu"):
        path = MIMO / "data" / f"{dataset}.jsonl"
        actual = digest(path)
        if actual != EXPECTED_SHA[dataset]:
            raise SystemExit(f"{dataset} sha256 mismatch: {actual}")
        items = [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]
        if len(items) != 200:
            raise SystemExit(f"{dataset} expected 200 rows, got {len(items)}")
        sources[dataset] = {"path": str(path), "sha256": actual, "count": len(items)}
        tasks.extend((dataset, i, it) for i, it in enumerate(items))

    manifest = {
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model": args.model, "endpoint": args.url,
        "protocol": protocol, "sources": sources,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def run(task):
        dataset, index, item = task
        text = prompt(dataset, item)
        started = time.monotonic()
        rec = {"dataset": dataset, "index": index,
               "input_sha256": hashlib.sha256(text.encode()).hexdigest(),
               "expected": item["answer"]}
        body = {
            "model": args.model,
            "messages": [{"role": "user", "content": text}],
            "temperature": 0, "top_p": 1, "max_tokens": args.max_tokens,
            "chat_template_kwargs": {"enable_thinking": True},
        }
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            req = urllib.request.Request(args.url + "/v1/chat/completions",
                                         data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            with opener.open(req, timeout=900) as resp:
                raw = json.load(resp)
            choice = raw["choices"][0]
            content = choice["message"].get("content") or ""
            pred, correct = score(dataset, content, item["answer"])
            rec.update(content=content, predicted=pred, correct=correct,
                       truncated=choice.get("finish_reason") == "length",
                       usage=raw.get("usage"), error=None)
        except Exception as exc:  # noqa: BLE001
            rec.update(correct=False, predicted=None, error=repr(exc), truncated=False)
        rec["elapsed_s"] = round(time.monotonic() - started, 2)
        return rec

    records = []
    with (args.output / "predictions.jsonl").open("w", encoding="utf-8") as out:
        with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
            futures = [pool.submit(run, t) for t in tasks]
            for fut in concurrent.futures.as_completed(futures):
                r = fut.result()
                records.append(r)
                out.write(json.dumps(r, ensure_ascii=False) + "\n")
                out.flush()
                if len(records) % 25 == 0:
                    print(f"Completed {len(records)} of {len(tasks)}", flush=True)

    summary = {"scope": protocol["scope"],
               "official_baseline_status": protocol["official_baseline_status"],
               "official_accuracy_pass": None,
               "protocol": protocol, "datasets": {}}
    for dataset in sources:
        rows = [r for r in records if r["dataset"] == dataset]
        correct = sum(bool(r["correct"]) for r in rows)
        summary["datasets"][dataset] = {
            "correct": correct, "total": len(rows),
            "accuracy_percent": round(100 * correct / len(rows), 2),
            "errors": sum(r["error"] is not None for r in rows),
            "truncated": sum(bool(r["truncated"]) for r in rows),
            "unparsed": sum(r["predicted"] is None for r in rows),
        }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    raise SystemExit(1 if any(v["errors"] for v in summary["datasets"].values()) else 0)


if __name__ == "__main__":
    main()
