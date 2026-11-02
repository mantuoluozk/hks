"""Native vLLM adapter for the Trident backbone and LAST-token normalized pooling."""
import json
import os
from pathlib import Path
from vllm import LLM


def load_model():
    settings = json.loads((Path(os.environ['MODEL_PATH']) / 'mmemb_model.json').read_text())
    return LLM(model=os.environ['MODEL_PATH'], runner='pooling', convert='embed',
               hf_overrides={'architectures':['Qwen3VLForConditionalGeneration']},
               pooler_config={'seq_pooling_type':'LAST','use_activation':True},
               dtype='bfloat16', gpu_memory_utilization=0.45, max_model_len=8192, max_num_seqs=8,
               enforce_eager=os.environ.get('TRIDENT_MODE','eager')!='piecewise',
               compilation_config=({'mode':3,'cudagraph_mode':'PIECEWISE','cudagraph_capture_sizes':[1,2,4,8]} if os.environ.get('TRIDENT_MODE')=='piecewise' else None), enable_prefix_caching=False, mm_processor_cache_gb=0,
               mm_processor_kwargs={'min_pixels':settings['image']['min_pixels'],
                                    'max_pixels':settings['image']['max_pixels']})
