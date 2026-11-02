"""Run the installed SGLang benchmark with model-valid random text token IDs.

The publisher tokenizer reserves two IDs beyond the checkpoint embedding rows.
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
    valid = [token for token in tokens if 0 <= token < capacity]
    removed = [token for token in tokens if token not in range(capacity)]
    assert valid and removed == [163840, 163841], removed
    return valid

def checked_prompt(tokenizer, image_pad_id, token_num):
    prompt = original_prompt(tokenizer, image_pad_id, token_num)
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    assert ids and all(0 <= token < capacity for token in ids), 'Invalid synthetic input token'
    return prompt

common.get_available_tokens = valid_tokens
common.get_available_multimodal_text_tokens.cache_clear()
image.gen_mm_prompt = checked_prompt

if __name__ == '__main__':
    from sglang.benchmark.serving import cli_main
    print('Synthetic text vocabulary restricted to checkpoint embedding rows:', capacity, flush=True)
    cli_main()
