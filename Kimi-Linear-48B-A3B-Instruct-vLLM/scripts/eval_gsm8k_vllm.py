#!/usr/bin/env python3
"""GSM8K 200-sample evaluation for the patched vLLM Kimi-Linear service.

Data provenance
---------------
The fixed 200-sample set is the same one used by the sibling MiMo-V2.6-9B
evaluation. The data file is NOT shipped in this repository (the repo does not
carry datasets). Provide it with --data, or rebuild/obtain it as documented in
`docs/gsm8k-200-samples.md`. The expected SHA256 is pinned below so a wrong or
stale file fails loudly instead of silently changing the score.

Protocol
--------
  scoring : final '####' numeric value, Decimal compare
  sampling: temperature=0, top_p=1, enable_thinking=True, max_tokens=4096
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import time
import urllib.request
from decimal import Decimal, InvalidOperation
from pathlib import Path

EXPECTED_SHA = "03bd22060d201b82f3545a56dc89a01d300a5406d79e1d51529fdedceb698546"
DEFAULT_CANDIDATES = [
    os.environ.get("KIMI_GSM8K_PATH", ""),
    "/root/private_data/hks/models/MiMo-V2.6-Distill-Qwen-9B/data/gsm8k.jsonl",
    "./data/gsm8k.jsonl",
]
NUM = r"[-+]?\d[\d,]*(?:\.\d+)?"


def resolve_data(explicit):
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise SystemExit(f"--data not found: {p}")
        return p
    for cand in DEFAULT_CANDIDATES:
        if cand and Path(cand).exists():
            return Path(cand)
    raise SystemExit(
        "GSM8K sample file not found.\n"
        "This repository does not ship datasets. Provide one of:\n"
        "  --data /path/to/gsm8k.jsonl\n"
        "  export KIMI_GSM8K_PATH=/path/to/gsm8k.jsonl\n"
        f"Expected SHA256: {EXPECTED_SHA}\n"
        "See docs/gsm8k-200-samples.md for how the fixed 200-sample set is defined."
    )


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
    ap.add_argument("--data", type=str, default=None, help="path to the fixed 200-sample jsonl")
    ap.add_argument("--url", default="http://127.0.0.1:30149")
    ap.add_argument("--model", default="kimi-linear-48b-a3b")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--allow-sha-mismatch", action="store_true",
                    help="proceed even if the data SHA256 differs (recorded in the summary)")
    a = ap.parse_args()

    data = resolve_data(a.data)
    sha = hashlib.sha256(data.read_bytes()).hexdigest()
    sha_ok = sha == EXPECTED_SHA
    if not sha_ok and not a.allow_sha_mismatch:
        raise SystemExit(
            f"data SHA256 mismatch for {data}\n  expected {EXPECTED_SHA}\n  actual   {sha}\n"
            "Use --allow-sha-mismatch to run anyway (recorded in summary) or supply the right file."
        )

    items = [json.loads(s) for s in data.read_text(encoding="utf-8").splitlines() if s.strip()]
    if len(items) != 200:
        raise SystemExit(f"expected 200 rows in {data}, got {len(items)}")

    a.output.mkdir(parents=True, exist_ok=False)
    protocol = {"engine": "vLLM", "few_shot": 0, "temperature": 0, "top_p": 1,
                "enable_thinking": True, "max_tokens": a.max_tokens, "workers": a.workers,
                "scoring": "final '####' numeric, Decimal compare",
                "denominator": "all 200; errors/truncation/unparsed counted incorrect"}
    (a.output / "manifest.json").write_text(json.dumps({
        "model": a.model, "endpoint": a.url, "benchmark": "GSM8K",
        "dataset": {"path": str(data), "sha256": sha, "sha_matches_pinned": sha_ok,
                    "count": len(items)},
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
               "dataset_sha256": sha, "dataset_sha_matches_pinned": sha_ok,
               "responses": len(recs), "correct": correct,
               "accuracy_percent": round(100 * correct / len(recs), 2),
               "errors": sum(1 for r in recs if r["error"] is not None),
               "truncated": sum(1 for r in recs if r["truncated"]),
               "unparsed": sum(1 for r in recs if r["predicted"] is None),
               "degenerate_exclamation": sum(
                   1 for r in recs
                   if (r.get("content") or "").strip()
                   and set((r.get("content") or "").strip()) == {"!"}),
               "protocol": protocol}
    (a.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                           encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
