#!/usr/bin/env python3
"""Rescore saved GSM8K/MMLU predictions with a more tolerant final-answer extractor.

The model legitimately answers in forms the original strict regex rejected:
    '#### $178.75'   '#### 3,430'   '#### **42**'   '#### Final Answer:\\n15'
Those are correct answers, not model failures, so parsing them is a scoring fix,
not a leniency that changes the metric's meaning: we still compare the final
numeric value / final option letter to the reference.

Usage: python3 kl_rescore.py <job_dir> [--write]
"""
import json
import re
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path


def number(text):
    try:
        return Decimal(str(text).replace(",", "").replace("$", "").strip())
    except (InvalidOperation, AttributeError):
        return None


NUM = r"[-+]?\d[\d,]*(?:\.\d+)?"


def parse_gsm8k(content):
    """Return (predicted_string, matched_span_kind) or (None, 'none')."""
    text = (content or "").replace("**", "").replace("$", "").replace(",", "")
    if "####" in text:
        tail = text.rsplit("####", 1)[1]
        # take the last number appearing after the marker
        nums = re.findall(NUM, tail)
        if nums:
            return nums[-1].strip(), "hash_marker"
    # fallback: final numeric answer statement
    m = re.findall(r"(?:final\s+(?:numeric\s+)?answer|answer)\s*(?:is|:)?\s*:?\s*(" + NUM + r")",
                   text, re.I)
    if m:
        return m[-1].strip(), "answer_phrase"
    nums = re.findall(NUM, text)
    if nums:
        return nums[-1].strip(), "last_number"
    return None, "none"


def parse_mmlu(content):
    text = (content or "").replace("**", "").replace("*", "")
    m = re.findall(r"ANSWER\s*[:：]\s*\(?([ABCD])\)?", text, re.I)
    if m:
        return m[-1].upper(), "answer_line"
    # fallback: a lone option letter at the very end
    m = re.findall(r"(?:^|\n)\s*\(?([ABCD])\)?\s*[.。]?\s*$", text)
    if m:
        return m[-1].upper(), "trailing_letter"
    return None, "none"


def main():
    job = Path(sys.argv[1])
    write = "--write" in sys.argv
    p = job / "predictions.jsonl"
    recs = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]

    changed = 0
    for r in recs:
        if r["dataset"] == "gsm8k":
            pred, kind = parse_gsm8k(r.get("content"))
            ok = pred is not None and number(pred) == number(r["expected"])
        else:
            pred, kind = parse_mmlu(r.get("content"))
            ok = pred == r["expected"]
        if (pred, ok) != (r.get("predicted"), r.get("correct")):
            changed += 1
        r["predicted"] = pred
        r["correct"] = ok
        r["parse_kind"] = kind

    print(f"records={len(recs)}  changed_by_rescore={changed}")
    summary = {"datasets": {}}
    for ds in ("gsm8k", "mmlu"):
        rows = [r for r in recs if r["dataset"] == ds]
        if not rows:
            continue
        ok = sum(bool(r["correct"]) for r in rows)
        kinds = {}
        for r in rows:
            kinds[r.get("parse_kind")] = kinds.get(r.get("parse_kind"), 0) + 1
        summary["datasets"][ds] = {
            "correct": ok, "total": len(rows),
            "accuracy_percent": round(100 * ok / len(rows), 2),
            "unparsed": sum(r["predicted"] is None for r in rows),
            "truncated": sum(bool(r.get("truncated")) for r in rows),
            "errors": sum(r.get("error") is not None for r in rows),
            "parse_kinds": kinds,
        }
        print(f"\n{ds}: {ok}/{len(rows)} = {100*ok/len(rows):.2f}%")
        print(f"  parse kinds: {kinds}")

    if write:
        with p.open("w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        (job / "summary-rescored.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("\nwritten:", p, "and summary-rescored.json")


if __name__ == "__main__":
    main()
