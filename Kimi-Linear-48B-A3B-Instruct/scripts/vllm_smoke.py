#!/usr/bin/env python3
"""Functional smoke checks against the patched vLLM Kimi-Linear service."""
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = "http://127.0.0.1:30149"
MODEL = "kimi-linear-48b-a3b"


def ask(prompt, max_tokens=256, **kw):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "max_tokens": max_tokens, **kw}
    t0 = time.time()
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(req, timeout=600) as r:
            raw = json.load(r)
        ch = raw["choices"][0]
        return {"ok": True, "content": (ch["message"].get("content") or ""),
                "finish": ch.get("finish_reason"), "sec": round(time.time() - t0, 2),
                "usage": raw.get("usage")}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": repr(e), "sec": round(time.time() - t0, 2)}


checks = [
    ("arithmetic_en", "What is 2+3? Answer with one number only.", 32),
    ("arithmetic_zh", "计算 7+8，只输出阿拉伯数字。", 32),
    ("english_cap", "Reply with the word READY only.", 32),
    ("chinese_gen", "用一句话介绍杭州。", 128),
    ("longer_gen", "请写一段约100字的说明，解释什么是线性注意力。", 300),
]

results = []
for name, prompt, mt in checks:
    r = ask(prompt, mt)
    txt = r.get("content", "")
    # detect the pathological repeated '!' failure mode
    bad = txt.strip() and set(txt.strip()) == {"!"}
    results.append({"name": name, "prompt": prompt, "max_tokens": mt,
                    "content": txt[:200], "finish": r.get("finish"),
                    "ok": r["ok"], "degenerate": bad, "sec": r["sec"]})
    print(f"[{name}] ok={r['ok']} degenerate={bad} finish={r.get('finish')} {r['sec']}s")
    print(f"    -> {txt[:160]!r}")

print("\n=== concurrency consistency (4 identical requests) ===")
with ThreadPoolExecutor(4) as ex:
    outs = list(ex.map(lambda _: ask("What is 12*12? Answer with one number only.", 32), range(4)))
texts = [(o.get("content") or "").strip() for o in outs]
print("  outputs:", texts)
print("  all identical:", len(set(texts)) == 1)
print("  any degenerate:", any(set(t) == {"!"} for t in texts if t))

print("\n=== sequential consistency (same prompt twice) ===")
a = ask("What is 9+10? Answer with one number only.", 32)
b = ask("What is 9+10? Answer with one number only.", 32)
print("  a:", repr((a.get("content") or "")[:80]))
print("  b:", repr((b.get("content") or "")[:80]))
print("  identical:", (a.get("content") or "").strip() == (b.get("content") or "").strip())

summary = {
    "checks": results,
    "concurrency_outputs": texts,
    "concurrency_identical": len(set(texts)) == 1,
    "degenerate_any": any(r["degenerate"] for r in results) or any(set(t) == {"!"} for t in texts if t),
}
open("/root/vllm_smoke.json", "w").write(json.dumps(summary, ensure_ascii=False, indent=2))
print("\nwrote /root/vllm_smoke.json")
