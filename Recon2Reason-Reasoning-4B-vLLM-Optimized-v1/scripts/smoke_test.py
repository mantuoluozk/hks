"""Local diagnostic smoke tests, NOT official OCRBench/MMMU evaluation."""
import base64
import io
import json
import os
import re
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
BASE = f"http://127.0.0.1:{os.environ.get('PORT', '30123')}"
NAME = os.environ.get('SERVED_NAME', 'recon2reason-4b')
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
STAMP = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
OUT = ROOT / 'results' / f'smoke-{STAMP}'
OUT.mkdir(parents=True, exist_ok=False)


def get(path):
    with OPENER.open(BASE + path, timeout=30) as response:
        return response.read().decode()


def make_image(filename, boxes):
    image = Image.new('RGB', (448, 336), 'white')
    draw = ImageDraw.Draw(image)
    for color, coords in boxes:
        draw.rectangle(coords, fill=color)
    path = OUT / filename
    image.save(path)
    return 'data:image/png;base64,' + base64.b64encode(path.read_bytes()).decode()


def request_case(label, messages, expected=None, stream=False):
    body = {'model':NAME, 'messages':messages, 'temperature':0, 'max_tokens':128, 'stream':stream}
    (OUT / f'{label}.request.json').write_text(json.dumps(body, indent=2))
    request = urllib.request.Request(BASE + '/v1/chat/completions', json.dumps(body).encode(),
                                    {'Content-Type':'application/json'})
    start = time.monotonic()
    with OPENER.open(request, timeout=600) as response:
        raw = response.read().decode()
    (OUT / f'{label}.response.txt').write_text(raw)
    elapsed = time.monotonic() - start
    if stream:
        events = [line[6:] for line in raw.splitlines() if line.startswith('data: ')]
        chunks = [json.loads(event) for event in events if event != '[DONE]']
        answer = ''.join(c.get('choices',[{}])[0].get('delta',{}).get('content') or '' for c in chunks if c.get('choices'))
        functional = '[DONE]' in events and bool(answer.strip())
        usage = None
    else:
        data = json.loads(raw)
        answer = data['choices'][0]['message'].get('content') or ''
        functional = bool(answer.strip())
        usage = data.get('usage')
    semantic = None if expected is None else expected(answer)
    result = {'case':label, 'seconds':round(elapsed,3), 'answer':answer,
              'functional_pass':functional, 'diagnostic_semantic_pass':semantic, 'usage':usage}
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


print('health:', get('/health'), flush=True)
models = json.loads(get('/v1/models'))
assert NAME in [m['id'] for m in models['data']], models
(OUT / 'models.json').write_text(json.dumps(models, indent=2))
scene = make_image('left-red-right-blue.png', [('red',(32,90,160,246)), ('blue',(288,90,416,246))])
red = make_image('red.png', [('red',(96,64,352,272))])
blue = make_image('blue.png', [('blue',(96,64,352,272))])


def text(t): return {'type':'text','text':t}
def image(url): return {'type':'image_url','image_url':{'url':url}}


results = []
results.append(request_case('text', [{'role':'user','content':'What is 2 + 3? Answer with one number.'}], lambda a:bool(re.fullmatch(r'\s*5[.!]?\s*',a))))
visual = [{'role':'user','content':[image(scene), text('What color is the rectangle on the left? Answer with one color word in English.')]}]
results.append(request_case('single_image', visual, lambda a:bool(re.search(r'\bred\b',a.lower()))))
multi = [{'role':'user','content':[image(red),image(blue),text('Name the color of the rectangle in the FIRST image, then in the SECOND image. Answer in the form: color, color.')]}]
results.append(request_case('multi_image', multi, lambda a:re.findall(r'\b(?:red|blue)\b',a.lower())[:2]==['red','blue']))
results.append(request_case('multi_turn', visual + [{'role':'assistant','content':results[1]['answer']},
    {'role':'user','content':'Now what color is the rectangle on the right? Answer with one color word in English.'}], lambda a:bool(re.search(r'\bblue\b',a.lower()))))
results.append(request_case('stream_image', visual, lambda a:bool(re.search(r'\bred\b',a.lower())), stream=True))
summary = {'created_at':datetime.now(timezone.utc).isoformat(), 'model':NAME,
           'purpose':'synthetic smoke diagnostics; not official accuracy evaluation',
           'execution_platform':os.environ.get('EXECUTION_PLATFORM','self-owned BW100'),
           'official_acceptance_passed':None,
           'official_scnet_validation':False, 'cases':results,
           'functional_pass':all(r['functional_pass'] for r in results),
           'diagnostic_semantic_pass':all(r['diagnostic_semantic_pass'] is not False for r in results)}
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
print('Artifacts:', OUT, flush=True)
if not summary['functional_pass'] or not summary['diagnostic_semantic_pass']:
    raise SystemExit(1)
