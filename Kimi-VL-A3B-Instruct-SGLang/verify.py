"""Validate saved raw evidence; pending official acceptance does not fail checks."""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parent
def load(path): return json.loads(path.read_text(encoding='utf-8'))
def finite(x): assert math.isfinite(x) and x>0
def short(target):
    smoke=load(target/'functional/summary.json')
    assert smoke['all_passed'] and smoke['functional_passed']==smoke['functional_total']==8
    for key in ['text','red','blue-zh','ocr','demo','two-images','repeat','stream']:
        row=load(target/'functional'/(key+'.json'))
        assert row['passed'] and row['output'].strip() and row['finish_reason']=='stop'
        finite(row['seconds'])
    summary=load(target/'short/summary.json')
    warm=load(target/'short/warmup.json')
    assert len(warm)==summary['warmup_requests']==16
    for w in warm:
        assert w['output'].strip() and w['done_marker'] and w['usage']['completion_tokens']>0
    assert summary['counted_requests']==32 and len(summary['workloads'])==4
    for cell in summary['workloads']:
        rows=[json.loads(x) for x in (target/'short'/f"c{cell['concurrency']}-o{cell['max_tokens']}"/'requests.jsonl').read_text(encoding='utf-8').splitlines()]
        assert len(rows)==cell['requests']==8
        tokens=0
        for row in rows:
            assert row['output'].strip() and row['done_marker']
            finite(row['seconds']);finite(row['ttft_seconds'])
            n=row['usage']['completion_tokens'];assert 0<n<=cell['max_tokens'];tokens+=n
        assert tokens==cell['completion_tokens']
        assert math.isclose(tokens/cell['wall_seconds'],cell['output_tokens_per_second'],rel_tol=1e-8)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--selfcheck',action='store_true');args=parser.parse_args()
    meta=load(ROOT/'results/release.json')
    screen=ROOT/'results'/meta['screen']
    assert load(screen/'state.json')['stage']=='complete'
    for name in ['eager','decode-graph']:short(screen/name)
    job=ROOT/'results'/meta['formal']
    assert load(job/'verified-summary.json')['complete']
    context=load(ROOT/'results/context-length-proof.json')
    log=(job/'process1/server.log').read_text(encoding='utf-8')
    assert all(line in log for line in context['prefill_log_lines'])
    counts=[int(re.search(r'#new-token: (\d+)',line).group(1)) for line in context['prefill_log_lines']]
    assert len(counts)==10 and sum(counts[5:])==context['client_processor_tokens']==17796
    assert context['client_processor_tokens']+1024<=context['context_length']==24576
    accuracy=job/'process1/ocrbench'
    summary=load(accuracy/'summary.json')
    audit=load(accuracy/'upstream-scoring-audit.json')
    predictions=[json.loads(x) for x in (accuracy/'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    scored=[json.loads(x) for x in (accuracy/'scored.jsonl').read_text(encoding='utf-8').splitlines()]
    assert summary['samples']==len(predictions)==len(scored)==1000 and summary['full_dataset']
    assert len({r['id'] for r in predictions})==1000
    assert summary['errors']==0 and all(not r['error'] and r['output'].strip() for r in predictions)
    assert summary['correct']==sum(int(r['correct']) for r in scored)==audit['upstream_correct']
    assert audit['all_matched'] and not audit['mismatched_ids']
    assert hashlib.sha256((accuracy/'predictions.jsonl').read_bytes()).hexdigest()==audit['predictions_sha256']
    total=0
    for repeat in [1,2]:
        rows=[json.loads(x) for x in (job/'process1'/f'round{repeat}.jsonl').read_text(encoding='utf-8').splitlines()]
        assert len(rows)==6
        expected={(n,c) for n in [4096,16384] for c in [1,8,64]}
        assert {(r['random_input_len'],r['max_concurrency']) for r in rows}==expected
        for r in rows:
            n=max(8,2*r['max_concurrency']);assert r['completed']==n
            assert r['random_output_len']==1024 and r['total_output_tokens']==n*1024
            assert len(r['output_lens'])==n and set(r['output_lens'])=={1024}
            assert len(r['errors'])==n and not any(r['errors'])
            assert len(r['generated_texts'])==n and all(t.strip() for t in r['generated_texts'])
            for t in r['ttfts']:finite(t)
            assert math.isclose(r['output_throughput'],n*1024/r['duration'],rel_tol=1e-6)
            total+=n
    assert total==608
    for f in load(ROOT/'submission-manifest.json')['files']:
        p=ROOT/f['path'];assert p.stat().st_size==f['bytes'],f['path']
        assert hashlib.sha256(p.read_bytes()).hexdigest()==f['sha256'],f['path']
    print(json.dumps({'checks_passed':True,'counted_requests':total,'official_acceptance':None}))
if __name__=='__main__':main()
