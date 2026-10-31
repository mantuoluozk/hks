#!/usr/bin/env python3
"""GSM8K 200-sample evaluation on the patched vLLM Kimi-Linear service.

Data and protocol identical to the SGLang run so the two engines are comparable:
  data    : MiMo data/gsm8k.jsonl (sha256 pinned)
  scoring : final '####' numeric value, Decimal compare
  sampling: temperature=0, top_p=1, enable_thinking=True, max_tokens=4096
"""
import argparse
import concurrent.futures
import hashlib
import json
import re
import time
import urllib.request
from decimal import Decimal, InvalidOperation
from pathlib import Path

DATA = Path("/root/private_data/hks/models/MiMo-V2.6-Distill-Qwen-9B/data/gsm8k.jsonl")
EXPECTED_SHA = "03bd22060d201b82f3545a56dc89a01d300a5406d79e1d51529fdedceb698546"
NUM = r"[-+]?\d[\d,]*(?:\.\d+)?"


def number(t):
    try:
        return Decimal(str(t).replace(",", "").replace("$", "").strip())
    except (InvalidOperation, AttributeError):
        return None


def parse_gsm8k(content):
    text = (content or "").replace("**", "").replace("$", "").replace(",", "")
    if "####" in text:
        nums = re.findall(NUM, text.rsplit("####", 1)[1])
        if nums:
            return nums[-1].strip()
    m = re.findall(r"(?:final\s+(?:numeric\s+)?answer|answer)\s*(?:is|:)?\s*:?\s*(" + NUM + r")", text, re.I)
    if m:
        return m[-1].strip()
    nums = re.findall(NUM, text)
    return nums[-1].strip() if nums else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--url", default="http://127.0.0.1:30149")
    ap.add_argument("--model", default="kimi-linear-48b-a3b")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=4096)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)

    sha = hashlib.sha256(DATA.read_bytes()).hexdigest()
    if sha != EXPECTED_SHA:
        raise SystemExit(f"data sha mismatch: {sha}")
    items = [json.loads(s) for s in DATA.read_text(encoding="utf-8").splitlines() if s.strip()]
    if len(items) != 200:
        raise SystemExit(f"expected 200, got {len(items)}")

    protocol = {"engine": "vLLM 0.25.1+das185 + vllm-hcu plugin",
                "few_shot": 0, "temperature": 0, "top_p": 1, "enable_thinking": True,
                "max_tokens": a.max_tokens, "workers": a.workers,
                "scoring": "final '####' numeric, Decimal compare",
                "denominator": "all 200; errors/truncation/unparsed counted incorrect"}
    (a.output / "manifest.json").write_text(json.dumps(
        {"model": a.model, "endpoint": a.url, "benchmark": "GSM8K",
         "dataset": {"path": str(DATA), "sha256": sha, "count": len(items)},
         "protocol": protocol}, indent=2), encoding="utf-8")

    def run(it):
        prompt = ("Solve the math problem. Explain briefly, then finish with #### followed by "
                  "the final numeric answer.\n\n" + it["question"])
        t0 = time.time()
        rec = {"question": it["question"], "expected": it["answer"],
               "input_sha256": hashlib.sha256(prompt.encode()).hexdigest()}
        body = {"model": a.model, "messages": [{"role": "user", "content": prompt}],
                "temperature": 0, "top_p": 1, "max_tokens": a.max_tokens,
                "chat_template_kwargs": {"enable_thinking": True}}
        try:
            op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            req = urllib.request.Request(a.url + "/v1/chat/completions",
                                         data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            with op.open(req, timeout=1200) as r:
                raw = json.load(r)
            ch = raw["choices"][0]
            content = ch["message"].get("content") or ""
            pred = parse_gsm8k(content)
            rec.update(content=content, predicted=pred,
                       correct=pred is not None and number(pred) == number(it["answer"]),
                       truncated=ch.get("finish_reason") == "length",
                       usage=raw.get("usage"), error=None)
        except Exception as e:  # noqa: BLE001
            rec.update(correct=False, predicted=None, error=repr(e), truncated=False)
        rec["elapsed_s"] = round(time.time() - t0, 2)
        return rec

    recs = []
    with (a.output / "predictions.jsonl").open("w", encoding="utf-8") as out:
        with concurrent.futures.ThreadPoolExecutor(a.workers) as pool:
            futs = [pool.submit(run, it) for it in items]
            for f in concurrent.futures.as_completed(futs):
                r = f.result()
                recs.append(r)
                out.write(json.dumps(r, ensure_ascii=False) + "\n")
                out.flush()
                if len(recs) % 25 == 0:
                    ok = sum(1 for x in recs if x["correct"])
                    print(f"{len(recs)}/200  acc={100*ok/len(recs):.1f}%", flush=True)

    correct = sum(1 for r in recs if r["correct"])
    summary = {"benchmark": "GSM8K", "engine": protocol["engine"],
               "dataset_sha256": sha, "responses": len(recs), "correct": correct,
               "accuracy_percent": round(100 * correct / len(recs), 2),
               "errors": sum(1 for r in recs if r["error"] is not None),
               "truncated": sum(1 for r in recs if r["truncated"]),
               "unparsed": sum(1 for r in recs if r["predicted"] is None),
               "degenerate_exclamation": sum(1 for r in recs if (r.get("content") or "").strip() and set((r.get("content") or "").strip()) == {"!"}),
               "protocol": protocol}
    (a.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
