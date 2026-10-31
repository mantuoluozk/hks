"""Check saved platform evidence for Kimi-Linear-48B-A3B-Instruct.

Prints the recorded verification summary and recomputes the GSM8K/MMLU counts
from the saved per-item predictions so the reported numbers cannot drift from
the stored evidence.

This script never asserts an official-accuracy PASS: the publisher publishes no
GSM8K/MMLU score for this Instruct checkpoint, and the only published figure
(MMLU-Pro full-set EM 72.7%) was not executed in this round.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULT = ROOT / "verify_result.json"
PRED = ROOT / "results" / "accuracy-gsm8k-mmlu-200" / "predictions.jsonl"


def recompute():
    if not PRED.exists():
        return None
    recs = [json.loads(l) for l in PRED.read_text(encoding="utf-8").splitlines() if l.strip()]
    out = {}
    for ds in ("gsm8k", "mmlu"):
        rows = [r for r in recs if r["dataset"] == ds]
        if not rows:
            continue
        correct = sum(1 for r in rows if r.get("correct"))
        out[ds] = {
            "correct": correct,
            "total": len(rows),
            "accuracy_percent": round(100 * correct / len(rows), 2),
            "truncated": sum(1 for r in rows if r.get("truncated")),
            "unparsed": sum(1 for r in rows if r.get("predicted") is None),
            "errors": sum(1 for r in rows if r.get("error") is not None),
        }
    return out


def main():
    result = json.loads(RESULT.read_text(encoding="utf-8"))

    if "--selfcheck" in sys.argv:
        actual = recompute()
        if actual is None:
            print("no predictions file; nothing to recompute")
            raise SystemExit(0)
        checks = []
        for ds, recorded in result.get("public_self_eval", {}).items():
            got = actual.get(ds)
            if not got:
                checks.append((ds, False, "missing"))
                continue
            ok = (got["correct"] == recorded["correct"]
                  and got["total"] == recorded["total"]
                  and got["truncated"] == recorded.get("truncated")
                  and got["unparsed"] == recorded.get("unparsed"))
            checks.append((ds, ok, f'recorded={recorded["correct"]}/{recorded["total"]} '
                                    f'recomputed={got["correct"]}/{got["total"]}'))
        for ds, ok, note in checks:
            print(f'{ds}: {"OK" if ok else "MISMATCH"}  {note}')
        if not all(c[1] for c in checks):
            raise SystemExit(1)
        print("selfcheck passed: recorded counts match saved predictions")
        raise SystemExit(0)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print()
    print("GSM8K 185/200 (92.50%) is the reported accuracy for this model.")
    print("MMLU is incomplete and truncated; it is recorded as evidence only.")
    print("No publisher GSM8K/MMLU baseline exists for this Instruct checkpoint;")
    print("MMLU-Pro full-set (publisher EM 72.7%) was not executed this round.")
    print("No official accuracy PASS is asserted; acceptance is by the organizer.")
    raise SystemExit(2)


if __name__ == "__main__":
    main()
