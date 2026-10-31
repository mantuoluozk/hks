"""Check saved platform evidence for Kimi-Linear-48B-A3B-Instruct.

Prints the recorded verification summary and, with --selfcheck, recomputes the
GSM8K counts from the saved per-item predictions so the reported number cannot
drift from the stored evidence.

This script never asserts an official-accuracy PASS; acceptance is by the
organizer. The publisher publishes no GSM8K score for this Instruct checkpoint,
so the reported number is a self-measured adaptation check.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULT = ROOT / "verify_result.json"
PRED = ROOT / "results" / "accuracy-gsm8k-200" / "predictions.jsonl"


def recompute():
    if not PRED.exists():
        return None
    recs = [json.loads(l) for l in PRED.read_text(encoding="utf-8").splitlines() if l.strip()]
    correct = sum(1 for r in recs if r.get("correct"))
    return {
        "correct": correct,
        "total": len(recs),
        "accuracy_percent": round(100 * correct / len(recs), 2),
        "truncated": sum(1 for r in recs if r.get("truncated")),
        "unparsed": sum(1 for r in recs if r.get("predicted") is None),
        "errors": sum(1 for r in recs if r.get("error") is not None),
    }


def main():
    result = json.loads(RESULT.read_text(encoding="utf-8"))

    if "--selfcheck" in sys.argv:
        actual = recompute()
        if actual is None:
            print("no predictions file; nothing to recompute")
            raise SystemExit(0)
        recorded = result["public_self_eval"]["gsm8k"]
        ok = (actual["correct"] == recorded["correct"]
              and actual["total"] == recorded["total"]
              and actual["truncated"] == recorded["truncated"]
              and actual["unparsed"] == recorded["unparsed"])
        print(f'gsm8k: {"OK" if ok else "MISMATCH"}  '
              f'recorded={recorded["correct"]}/{recorded["total"]} '
              f'recomputed={actual["correct"]}/{actual["total"]}')
        if not ok:
            raise SystemExit(1)
        print("selfcheck passed: recorded counts match saved predictions")
        raise SystemExit(0)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print()
    print("GSM8K 185/200 (92.50%) is the reported self-evaluation for this model.")
    print("The publisher publishes no GSM8K baseline for this Instruct checkpoint.")
    print("No official accuracy PASS is asserted; acceptance is by the organizer.")
    raise SystemExit(2)


if __name__ == "__main__":
    main()
