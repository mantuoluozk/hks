"""Check saved vLLM evidence for Kimi-Linear-48B-A3B-Instruct.

With --selfcheck, recomputes the GSM8K counts and the performance gate from the
stored per-item predictions and the round summaries, so the numbers in the
report cannot drift from the evidence.

Never asserts an official-accuracy PASS; acceptance is by the organizer.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PRED = ROOT / "results" / "accuracy-gsm8k-200" / "predictions.jsonl"
VBASE = ROOT / "results"
SG = ROOT / "results" / "sglang-baseline"


def recompute_gsm8k():
    recs = [json.loads(l) for l in PRED.read_text(encoding="utf-8").splitlines() if l.strip()]
    correct = sum(1 for r in recs if r.get("correct"))
    return {
        "responses": len(recs),
        "correct": correct,
        "accuracy_percent": round(100 * correct / len(recs), 2),
        "errors": sum(1 for r in recs if r.get("error") is not None),
        "truncated": sum(1 for r in recs if r.get("truncated")),
        "unparsed": sum(1 for r in recs if r.get("predicted") is None),
    }


def recompute_gate():
    def load(tag):
        p = VBASE / "performance" / f"summary-{tag}.json"
        return {(x["prompt_repeat"], x["concurrency"]): x
                for x in json.loads(p.read_text(encoding="utf-8"))["points"]}
    r2, r3 = load("roundA"), load("roundB")
    primary = ["output_tok_per_s", "mean_latency_s", "mean_tpot_ms"]
    aux = ["mean_ttft_ms"]
    def mx(metrics):
        worst, where = 0.0, ""
        for k in sorted(r2):
            for m in metrics:
                a, b = r2[k].get(m), r3[k].get(m)
                if a and b:
                    d = abs(b - a) / abs(a) * 100
                    if d > worst:
                        worst, where = d, f"{m}@{k}"
        return round(worst, 2), where
    p, pw = mx(primary)
    a, aw = mx(aux)
    overall = max(p, a)
    return {"throughput_latency_tpot_max_dev_pct": p, "at": pw,
            "ttft_max_dev_pct": a, "ttft_at": aw,
            "overall_max_dev_pct": round(overall, 2), "passed": overall <= 20.0}


def main():
    if "--selfcheck" in sys.argv or len(sys.argv) == 1:
        g = recompute_gsm8k()
        gate = recompute_gate()
        print(f'vLLM GSM8K: {g["correct"]}/{g["responses"]} = {g["accuracy_percent"]}% '
              f'(errors={g["errors"]} truncated={g["truncated"]} unparsed={g["unparsed"]})')
        print(f'performance throughput/latency/TPOT max dev: '
              f'{gate["throughput_latency_tpot_max_dev_pct"]}% ({gate["at"]})')
        print(f'performance TTFT max dev: {gate["ttft_max_dev_pct"]}% ({gate["ttft_at"]})')
        print(f'performance overall max dev: {gate["overall_max_dev_pct"]}% '
              f'-> {"PASS" if gate["passed"] else "FAIL"}')

        ok = True
        if g["correct"] != 183 or g["responses"] != 200:
            print("!! GSM8K mismatch vs report (expected 183/200)")
            ok = False
        if g["errors"] or g["truncated"] or g["unparsed"]:
            print("!! non-zero errors/truncation/unparsed")
            ok = False
        if not gate["passed"]:
            print("!! performance gate failed")
            ok = False

        sg_pred = SG / "predictions-gsm8k.jsonl"
        if sg_pred.exists():
            rows = [json.loads(l) for l in sg_pred.read_text(encoding="utf-8").splitlines() if l.strip()]
            sgc = sum(1 for r in rows if r["correct"])
            print(f'SGLang GSM8K (for the comparison table): {sgc}/{len(rows)} = '
                  f'{round(100*sgc/len(rows),2)}%')
            if sgc != 185:
                print("!! SGLang baseline expected 185")
                ok = False

        print("\nselfcheck passed" if ok else "\n!! SELFCHECK FAILED")
        raise SystemExit(0 if ok else 1)

    # default: informational, never a PASS
    print(json.dumps({
        "model": "moonshotai/Kimi-Linear-48B-A3B-Instruct",
        "framework": "vLLM 0.25.1+das185 + vllm-hcu 0.25.1+dtk2604 (patched KDA)",
        "device": "BW x 4, BF16, TP4",
        "gsm8k": recompute_gsm8k(),
        "performance": recompute_gate(),
        "runtime_fix_required": True,
        "official_accuracy_pass": None,
        "official_acceptance_passed": None,
    }, ensure_ascii=False, indent=2))
    print("\nNo official accuracy PASS is asserted; acceptance is by the organizer.")
    raise SystemExit(2)


if __name__ == "__main__":
    main()
