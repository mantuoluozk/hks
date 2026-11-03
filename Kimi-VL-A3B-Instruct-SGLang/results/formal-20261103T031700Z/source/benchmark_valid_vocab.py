"""Run the installed SGLang benchmark with model-valid random text token IDs.

The publisher tokenizer reserves two IDs beyond the checkpoint embedding rows,
and its HF special-ID list does not cover every internal reserved token.
Constrain only synthetic prompt sampling; do not alter weights or tokenizer.
All benchmark traffic, timing and metric computation remain in SGLang.
"""
import json
from pathlib import Path
from sglang.benchmark.datasets import common, image

ROOT = Path(__file__).resolve().parents[1]
config = json.loads((ROOT / 'weights/config.json').read_text())
capacity = config['text_config']['vocab_size']
original_tokens = common.get_available_tokens
original_prompt = common.gen_mm_prompt

def valid_tokens(tokenizer):
    tokens = original_tokens(tokenizer)
    reserved = set(tokenizer.special_tokens.values())
    valid = [token for token in tokens if 0 <= token < capacity and token not in reserved]
    removed = [token for token in tokens if token not in range(capacity)]
    assert valid and removed == [163840, 163841], removed
    assert len(valid) == 163584 and max(valid) == 163583
    return valid

def checked_prompt(tokenizer, image_pad_id, token_num):
    prompt = original_prompt(tokenizer, image_pad_id, token_num)
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    assert ids and all(0 <= token < capacity for token in ids), 'Invalid synthetic input token'
    assert not set(ids).intersection(tokenizer.special_tokens.values()), 'Reserved token in synthetic text'
    return prompt

common.get_available_tokens = valid_tokens
common.get_available_multimodal_text_tokens.cache_clear()
image.gen_mm_prompt = checked_prompt

if __name__ == '__main__':
    import base64
    import dataclasses
    import hashlib
    import os
    from sglang.benchmark import serving
    original_dataset = serving.get_dataset
    def recorded_dataset(args, tokenizer, model_id=None):
        rows = original_dataset(args, tokenizer, model_id)
        dest = Path(os.environ['BENCH_INPUTS_DIR'])
        dest.mkdir(parents=True, exist_ok=False)
        def compact(value):
            if isinstance(value, str) and value.startswith('data:image/'):
                header, content = value.split(',',1)
                blob=base64.b64decode(content)
                name=hashlib.sha256(blob).hexdigest()+'.image'
                if not (dest/name).exists(): (dest/name).write_bytes(blob)
                return {'data_url_header':header,'asset':name,'bytes':len(blob)}
            if isinstance(value,list): return [compact(v) for v in value]
            if isinstance(value,dict): return {k:compact(v) for k,v in value.items()}
            return value
        (dest/'inputs.json').write_text(json.dumps([compact(dataclasses.asdict(r)) for r in rows]))
        return rows
    serving.get_dataset = recorded_dataset
    print('Synthetic text samples base vocabulary only (0..163583); excludes publisher reserved slots and out-of-embedding IDs', flush=True)
    serving.cli_main()
