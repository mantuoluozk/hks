"""Record real-audio API checks separately from task accuracy evaluation."""
import base64
import json
import os
import time
from pathlib import Path
from datetime import datetime, timezone
import requests
import soundfile as sf

root = Path(__file__).resolve().parents[1]
out = Path(os.environ.get('SMOKE_OUTPUT_DIR', str(root / 'results' / ('smoke-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')))))
out.mkdir(parents=True)
base = f'http://127.0.0.1:{os.environ.get("PORT", "30128")}'
model = 'qwen3-asr-1.7b'
session = requests.Session();session.trust_env = False
cases = []
for lang in ['en', 'zh']:
    path = root / f'data/examples/asr_{lang}.wav'
    info = sf.info(path)
    data_url = 'data:audio/wav;base64,' + base64.b64encode(path.read_bytes()).decode()
    for mode in ['chat', 'transcription', 'output_stream']:
        t0 = time.perf_counter()
        if mode == 'transcription':
            with path.open('rb') as f:
                response = session.post(base + '/v1/audio/transcriptions',
                    data={'model': model, 'temperature': '0'}, files={'file': (path.name, f, 'audio/wav')}, timeout=300)
            response.raise_for_status()
            body = response.json();text = body.get('text', '')
        else:
            payload = {'model': model, 'messages': [{'role': 'user', 'content': [
                {'type': 'audio_url', 'audio_url': {'url': data_url}}]}],
                'temperature': 0, 'max_tokens': 1024, 'stream': mode == 'output_stream'}
            response = session.post(base + '/v1/chat/completions', json=payload, timeout=300, stream=payload['stream'])
            response.raise_for_status()
            if payload['stream']:
                body = [];pieces = [];done = False
                for line in response.iter_lines():
                    if not line.startswith(b'data: '):continue
                    if line == b'data: [DONE]':done = True;break
                    event = json.loads(line[6:]);body.append(event)
                    if event.get('choices'):pieces.append(event['choices'][0].get('delta', {}).get('content') or '')
                text = ''.join(pieces);assert done, 'SSE stream did not terminate'
            else:
                body = response.json();text = body['choices'][0]['message']['content']
        elapsed = time.perf_counter() - t0
        (out / f'{lang}-{mode}.json').write_text(json.dumps({'response': body, 'text': text}, ensure_ascii=False, indent=2))
        transcript = text.split('<asr_text>', 1)[-1].strip()
        assert transcript, (lang, mode)
        if lang == 'zh':assert any('\u4e00' <= c <= '\u9fff' for c in transcript)
        else:assert any('a' <= c.lower() <= 'z' for c in transcript)
        cases.append({'language_input': lang, 'mode': mode, 'audio_seconds': info.duration,
                      'elapsed_seconds': elapsed, 'rtf': elapsed / info.duration, 'text': text,
                      'transcript': transcript})
summary = {'functional_cases': len(cases), 'cases': cases,
           'cross_interface_exact': {lang: len({c['transcript'] for c in cases if c['language_input'] == lang}) == 1 for lang in ['en', 'zh']},
           'accuracy_evaluated': False, 'official_accuracy_pass': None,
           'stream_note': 'SSE output tokens only; audio input chunk streaming is evaluated separately.'}
(out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
Path(os.environ.get('RUN_RESULTS_DIR', str(root/'results')), 'smoke-latest.txt').write_text(str(out))
print(json.dumps(summary, ensure_ascii=False, indent=2))
