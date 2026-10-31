#!/usr/bin/env python3
"""GSM8K 200-sample public evaluation for Kimi-Linear-48B-A3B-Instruct.

Reuses the fixed data file shared with the sibling MiMo-V2.6-9B evaluation so the
numbers are protocol-identical to that model's published self-evaluation:

  data    : MiMo-V2.6-Distill-Qwen-9B/data/gsm8k.jsonl (sha256 pinned below)
  scoring : final '#### <number>' parsed and compared as Decimal
            (tolerates '$', thousands separators and bold markup)
  sampling: temperature=0, top_p=1, greedy, enable_thinking=True, max_tokens=4096

Official note: the publisher does NOT publish a GSM8K score for this Instruct
checkpoint, and the figures on the model card belong to the 1.4T experimental
model or to the Base weights. This result is therefore a self-measured
adaptation sanity check, NOT a reproduction of a published baseline.
"""
import argparse
import concurrent.futures
import datetime
import hashlib
import json
import re
import time
import urllib.request
from decimal import Decimal, InvalidOperation
from pathlib import Path

MIMO = Path("/root/private_data/hks/models/MiMo-V2.6-Distill-Qwen-9B")
DATA = MIMO / "data" / "gsm8k.jsonl"
EXPECTED_SHA = "03bd22060d201b82f3545a56dc89a01d300a5406d79e1d51529fdedceb698546"

NUM = r"[-+]?\d[\d,]*(?:\.\d+)?"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(text):
    try:
        return Decimal(str(text).replace(",", "").replace("$", "").strip())
    except (InvalidOperation, AttributeError):
        return None


def parse_gsm8k(content):
    """Return the predicted final numeric string, or None."""
    text = (content or "").replace("**", "").replace("$", "").replace(",", "")
    if "####" in text:
        tail = text.rsplit("####", 1)[1]
        nums = re.findall(NUM, tail)
        if nums:
            return nums[-1].strip()
    m = re.findall(r"(?:final\s+(?:numeric\s+)?answer|answer)\s*(?:is|:)?\s*:?\s*(" + NUM + r")",
                   text, re.I)
    if m:
        return m[-1].strip()
    nums = re.findall(NUM, text)
    return nums[-1].strip() if nums else None


def prompt(item):
    return ("Solve the math problem. Explain briefly, then finish with #### followed by "
            "the final numeric answer.\n\n" + item["question"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--url", default="http://127.0.0.1:30148")
    ap.add_argument("--model", default="kimi-linear-48b-a3b")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=4096)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    actual = digest(DATA)
    if actual != EXPECTED_SHA:
        raise SystemExit(f"gsm8k sha256 mismatch: {actual}")
    items = [json.loads(s) for s in DATA.read_text(encoding="utf-8").splitlines() if s.strip()]
    if len(items) != 200:
        raise SystemExit(f"expected 200 rows, got {len(items)}")

    protocol = {
        "few_shot": 0, "temperature": 0, "top_p": 1, "enable_thinking": True,
        "max_tokens": args.max_tokens, "workers": args.workers,
        "scoring": "final '####' numeric value compared as Decimal",
        "denominator": "all samples; errors / truncation / unparsed counted incorrect",
    }
    manifest = {
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model": args.model, "endpoint": args.url, "benchmark": "GSM8K",
        "dataset": {"path": str(DATA), "sha256": actual, "count": len(items)},
        "protocol": protocol,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def run(item):
        text = prompt(item)
        started = time.monotonic()
        rec = {"question": item["question"], "expected": item["answer"],
               "input_sha256": hashlib.sha256(text.encode()).hexdigest()}
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
            pred = parse_gsm8k(content)
            rec.update(content=content, predicted=pred,
                       correct=pred is not None and number(pred) == number(item["answer"]),
                       truncated=choice.get("finish_reason") == "length",
                       usage=raw.get("usage"), error=None)
        except Exception as exc:  # noqa: BLE001
            rec.update(correct=False, predicted=None, error=repr(exc), truncated=False)
        rec["elapsed_s"] = round(time.monotonic() - started, 2)
        return rec

    records = []
    with (args.output / "predictions.jsonl").open("w", encoding="utf-8") as out:
        with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
            futures = [pool.submit(run, it) for it in items]
            for fut in concurrent.futures.as_completed(futures):
                r = fut.result()
                records.append(r)
                out.write(json.dumps(r, ensure_ascii=False) + "\n")
                out.flush()
                if len(records) % 25 == 0:
                    print(f"Completed {len(records)} of {len(items)}", flush=True)

    correct = sum(1 for r in records if r["correct"])
    summary = {
        "benchmark": "GSM8K",
        "dataset_sha256": actual,
        "dataset_rows_available": len(items),
        "responses": len(records),
        "correct": correct,
        "accuracy_percent": round(100 * correct / len(records), 2),
        "errors": sum(1 for r in records if r["error"] is not None),
        "truncated": sum(1 for r in records if r["truncated"]),
        "unparsed": sum(1 for r in records if r["predicted"] is None),
        "protocol": protocol,
        "publisher_baseline": (
            "the publisher publishes no GSM8K score for this Instruct checkpoint; "
            "this number is a self-measured adaptation check"
        ),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                              encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    raise SystemExit(1 if summary["errors"] else 0)


if __name__ == "__main__":
    main()
