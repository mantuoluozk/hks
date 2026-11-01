#!/usr/bin/env python3
"""Rebuild the fixed 200-sample GSM8K subset from the upstream dataset.

The fixed subset is a deterministic draw from the openai/gsm8k main (test)
split, using seed 20261023, so the same 200 questions are used across models.

The extraction rule mirrors the sibling MiMo-V2.6-9B evaluation:
  * read the `main` split of openai/gsm8k (1319 rows)
  * shuffle with seed 20261023
  * take the first 200
  * keep only the final numeric answer (the text after '####')

Output line format: {"question": "...", "answer": "<numeric string>"}

Expected SHA256 of the resulting file:
    03bd22060d201b82f3545a56dc89a01d300a5406d79e1d51529fdedceb698546

If the printed digest differs, the source revision or the draw rule differs
from the original batch -- do not compare the resulting scores with the
published ones.
"""
import argparse
import hashlib
import json
import random
import re
from pathlib import Path


def final_answer(answer_field: str) -> str:
    """Extract the value after '####' (last occurrence)."""
    text = (answer_field or "").replace(",", "")
    if "####" in text:
        tail = text.rsplit("####", 1)[1]
        m = re.findall(r"[-+]?\d+(?:\.\d+)?", tail.replace(",", ""))
        if m:
            return m[-1]
    m = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    return m[-1] if m else ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True,
                    help="parquet/jsonl of openai/gsm8k main(test); needs `question` and `answer`")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20261023)
    ap.add_argument("--count", type=int, default=200)
    a = ap.parse_args()

    if a.source.suffix == ".parquet":
        import pandas as pd
        rows = pd.read_parquet(a.source).to_dict("records")
    else:
        rows = [json.loads(l) for l in a.source.read_text(encoding="utf-8").splitlines() if l.strip()]

    if len(rows) < a.count:
        raise SystemExit(f"source has only {len(rows)} rows, need {a.count}")

    idx = list(range(len(rows)))
    random.Random(a.seed).shuffle(idx)

    out = []
    for i in idx:
        r = rows[i]
        ans = final_answer(str(r.get("answer", "")))
        if not ans:
            continue
        out.append({"question": str(r["question"]).strip(), "answer": ans})
        if len(out) == a.count:
            break

    if len(out) != a.count:
        raise SystemExit(f"only extracted {len(out)} usable rows, need {a.count}")

    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out), encoding="utf-8")

    digest = hashlib.sha256(a.output.read_bytes()).hexdigest()
    print(f"wrote {len(out)} rows -> {a.output}")
    print(f"sha256: {digest}")
    expected = "03bd22060d201b82f3545a56dc89a01d300a5406d79e1d51529fdedceb698546"
    print("matches pinned value" if digest == expected
          else f"MISMATCH: expected {expected}\n"
               "  -> source revision or draw rule differs from the original batch; "
               "do not compare scores with the published ones")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
