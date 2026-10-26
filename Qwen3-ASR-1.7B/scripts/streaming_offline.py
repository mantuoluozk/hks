"""Bind pinned upstream input-chunk streaming methods to native DCU vLLM.

This is incremental PCM input, distinct from SSE output-token streaming.
The official algorithm re-feeds accumulated audio and rolls back five tokens.
It does not promise incremental encoder caching or timestamps.
"""
import json
import os
import time
from pathlib import Path
from datetime import datetime, timezone
import librosa
import numpy as np
import soundfile as sf
from vllm import LLM, SamplingParams
from vllm.transformers_utils.processors.qwen3_asr import Qwen3ASRProcessor
from upstream_streaming import UpstreamStreamingASR


def main():
    root = Path(__file__).resolve().parents[1]
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try:cmd=p.read_bytes().split(b'\0')
        except (OSError,ProcessLookupError):continue
        assert b'vllm.entrypoints.openai.api_server' not in cmd,'Stop online vLLM service before offline PCM validation'
    model_path = os.environ.get('MODEL_PATH', '/data1/model/Qwen3-ASR-1.7B')
    engine = LLM(model=model_path, dtype='bfloat16', gpu_memory_utilization=0.3,
                 max_model_len=8192, max_num_seqs=1, enforce_eager=True,
                 enable_prefix_caching=False, mm_processor_cache_gb=0)
    runner = UpstreamStreamingASR()
    runner.backend = 'vllm'
    runner.model = engine
    runner.processor = Qwen3ASRProcessor.from_pretrained(model_path, fix_mistral_regex=True)
    runner.sampling_params = SamplingParams(temperature=0, max_tokens=1024)
    results = []
    for lang in ['en', 'zh']:
        wav, sr = sf.read(root / f'data/examples/asr_{lang}.wav', dtype='float32')
        if wav.ndim == 2:wav = wav.mean(axis=1)
        if sr != 16000:wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        for step_ms in [1000, 4000]:
            state = runner.init_streaming_state(chunk_size_sec=2.0, unfixed_chunk_num=2, unfixed_token_num=5)
            step = int(step_ms * 16)
            events = [];t0 = time.perf_counter()
            for pos in range(0, len(wav), step):
                runner.streaming_transcribe(np.asarray(wav[pos:pos+step]), state)
                events.append({'audio_samples_seen': min(pos+step, len(wav)),
                               'decode_steps': state.chunk_id, 'language': state.language, 'text': state.text})
            runner.finish_streaming_transcribe(state)
            assert state.text.strip(), (lang, step_ms)
            results.append({'language_input': lang, 'step_ms': step_ms, 'events': events,
                            'audio_seconds': len(wav)/16000, 'elapsed_seconds': time.perf_counter()-t0,
                            'final_language': state.language, 'final_text': state.text})
    out = Path(os.environ.get('STREAMING_OUTPUT_DIR',str(root/'results'/('input-streaming-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')))))
    out.mkdir(parents=True)
    report = {'cases': results, 'completed': len(results), 'official_accuracy_pass': None,
              'upstream_commit': '7c6daf77a2421100f5fb066495372c00129d39ff',
              'algorithm': 'Original five upstream streaming methods, native vLLM model and processor binding.'}
    (out/'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    (Path(os.environ.get('RUN_RESULTS_DIR',str(root/'results')))/'input-streaming-latest.txt').write_text(str(out))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
