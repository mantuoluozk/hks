#!/usr/bin/env python3
"""Performance matrix for the vLLM Kimi-Linear service.

Mirrors the SGLang report matrix so the two engines are directly comparable:
  2 prompt lengths (repeat 32 / 256) x concurrency (1, 2, 4) = 6 points
  each point: 3 warmup requests, then 3*concurrency timed requests
  output budget 128 tokens (EOS may stop earlier)

Records per-request raw data and a per-point summary (mean latency, output tok/s,
TTFT, TPOT) so the two independent runs can be compared for the 20% gate.
"""
import argparse
import concurrent.futures
import json
import time
import urllib.request
from pathlib import Path

BASE_DEFAULT = "http://127.0.0.1:30149"
MODEL = "kimi-linear-48b-a3b"
GEN_TOKENS = 128


def one(prompt, max_tokens, stream=False, base=BASE_DEFAULT, model=MODEL):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "max_tokens": max_tokens, "stream": stream,
            "stream_options": {"include_usage": True} if stream else None}
    body = {k: v for k, v in body.items() if v is not None}
    req = urllib.request.Request(base.rstrip('/') + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    t0 = time.monotonic()
    ttft = None
    if not stream:
        with opener.open(req, timeout=1200) as r:
            raw = json.load(r)
        total = time.monotonic() - t0
        usage = raw.get("usage") or {}
        if not isinstance(usage.get('completion_tokens'),int) or usage['completion_tokens']<=0:
            raise ValueError('Missing/invalid backend completion token usage')
        return {"total": total, "ttft": None,
                "completion_tokens": usage.get("completion_tokens"),
                "finish": raw["choices"][0].get("finish_reason"),"request":body,"response":raw}
    first = None
    usage = {}
    done=False
    events=[]
    with opener.open(req, timeout=1200) as r:
        for line in r:
            s = line.decode("utf-8", "replace").strip()
            if not s.startswith("data: "):
                continue
            payload = s[6:]
            if payload == "[DONE]":
                done=True
                break
            d = json.loads(payload)
            if d.get('error'):raise ValueError(d['error'])
            events.append(d)
            if d.get("usage"):
                usage = d["usage"]
            for ch in d.get("choices") or []:
                delta = (ch.get("delta") or {}).get("content") or (ch.get('delta') or {}).get('reasoning_content') or ''
                if delta and first is None:
                    first = time.monotonic() - t0
    total = time.monotonic() - t0
    if not done or first is None or not isinstance(usage.get('completion_tokens'),int) or usage['completion_tokens']<=0:
        raise ValueError('Incomplete stream, missing content, or missing backend token usage')
    return {"total": total, "ttft": first,
            "completion_tokens": usage.get("completion_tokens"), "finish": None,"request":body,"events":events}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--round-name", default="run")
    ap.add_argument('--base',default=BASE_DEFAULT)
    ap.add_argument('--model',default=MODEL)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)

    points = [(32, 1), (32, 2), (32, 4), (256, 1), (256, 2), (256, 4)]
    results = []
    records=(a.output/'requests.jsonl').open('w',encoding='utf-8')
    for repeat, conc in points:
        prompt = "Summarize the following repeated context and then answer.\n" + ("context token " * repeat) + "\nWhat is 2+3?"
        # warmup: 3 requests
        for _ in range(3):
            r=one(prompt, GEN_TOKENS,base=a.base,model=a.model)
            records.write(json.dumps({'phase':'warmup','prompt_repeat':repeat,'concurrency':conc,**r})+'\n');records.flush()
        n = 3 * conc
        t_start = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(conc) as ex:
            runs = list(ex.map(lambda _: one(prompt, GEN_TOKENS, stream=True,base=a.base,model=a.model), range(n)))
        wall = time.monotonic() - t_start
        for r in runs:records.write(json.dumps({'phase':'measured','prompt_repeat':repeat,'concurrency':conc,**r})+'\n')
        records.flush()
        lat = sorted(r["total"] for r in runs)
        ttf = [r["ttft"] for r in runs if r["ttft"] is not None]
        ct = [r["completion_tokens"] or 0 for r in runs]
        tot_ct = sum(ct)
        mean_tpot = []
        for r in runs:
            if r["ttft"] is not None and r["completion_tokens"] > 1:
                mean_tpot.append((r["total"] - r["ttft"]) / (r["completion_tokens"] - 1))
        point = {
            "prompt_repeat": repeat, "concurrency": conc, "requests": n,
            "wall_s": round(wall, 3),
            "output_tokens_total": tot_ct,
            "output_tok_per_s": round(tot_ct / wall, 3) if wall else None,
            "mean_latency_s": round(sum(lat) / len(lat), 3),
            "p50_latency_s": round(lat[len(lat) // 2], 3),
            "max_latency_s": round(lat[-1], 3),
            "mean_ttft_ms": round(sum(ttf) / len(ttf) * 1000, 2) if ttf else None,
            "mean_tpot_ms": round(sum(mean_tpot) / len(mean_tpot) * 1000, 2) if mean_tpot else None,
        }
        results.append(point)
        print(json.dumps(point, ensure_ascii=False), flush=True)

    (a.output / f"summary-{a.round_name}.json").write_text(
        json.dumps({"engine": "vLLM 0.25.1+das185 + vllm-hcu plugin",
                    "concurrency_matrix": [[p[0], p[1]] for p in points],
                    "base": a.base, "model": a.model,
                    "gen_tokens": GEN_TOKENS, "output_length_policy": "max_tokens budget; EOS allowed", "warmup_per_point": 3,
                    "timed_per_point": "3*concurrency",
                    "points": results}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("wrote", a.output / f"summary-{a.round_name}.json")
    records.close()


if __name__ == "__main__":
    main()
