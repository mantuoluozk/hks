import argparse,json
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('file');p.add_argument('input',type=int);p.add_argument('output',type=int)
p.add_argument('concurrency',type=int);p.add_argument('requests',type=int)
a=p.parse_args()
rows=[json.loads(line) for line in Path(a.file).read_text().splitlines() if line.strip()]
assert rows,'No benchmark result row'
r=rows[-1]
assert r['random_input_len']==a.input and r['random_output_len']==a.output
assert r['max_concurrency']==a.concurrency
assert r['completed']==a.requests,(r['completed'],a.requests)
assert r['total_output_tokens']==a.requests*a.output,(r['total_output_tokens'],a.requests*a.output)
assert len(r['output_lens'])==a.requests and set(r['output_lens'])=={a.output}
assert len(r['errors'])==a.requests and not any(r['errors'])
assert len(r['generated_texts'])==a.requests and all(t.strip() for t in r['generated_texts']), 'Empty visible output requires diagnosis'
assert len(r['ttfts'])==a.requests and all(t>0 for t in r['ttfts'])
print('Validated completed requests and fixed output token count:',a.file)
