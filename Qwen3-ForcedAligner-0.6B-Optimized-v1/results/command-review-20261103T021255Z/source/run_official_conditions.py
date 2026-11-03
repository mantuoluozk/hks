"""Run the supplied 30 alignment conditions using the published Sourcefind implementation."""
import argparse
import csv
import hashlib
import importlib.metadata as metadata
import json
import os
import platform
import time
from pathlib import Path

def main():
    ROOT = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--conditions', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--validate-inputs', action='store_true', help='Only check frozen conditions, helpers and input files')
    a = p.parse_args()
    lock = json.loads((ROOT / 'sources/official-t03t04-20261030.json').read_text())
    assert hashlib.sha256(a.conditions.read_bytes()).hexdigest() == lock['condition_sha256']
    cases = [json.loads(line) for line in a.conditions.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    assert len(cases) == len({c['id'] for c in cases}) == lock['rows'] == 30
    for filename, sha in lock['runtime_helpers'].items():
        assert hashlib.sha256((ROOT / 'scripts' / filename).read_bytes()).hexdigest() == sha, filename
    inputs = []
    for case in cases:
        assert case['model'] == 'forced_aligner' and case['task'] == 'align'
        assert case['seed'] == 42 and case['params'] == {'dtype': 'bf16', 'language': 'English'}
        assert case['evaluation_status'] == 'BOUNDARIES_PENDING'
        asset = case['assets']['audio']
        path = (a.conditions.parent / asset['path']).resolve(strict=True)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == asset['sha256'], case['id']
        assert case['text'].strip()
        inputs.append((case, path))
    print(json.dumps({'input_conditions': len(inputs), 'input_sha256_passed': True}), flush=True)
    if a.validate_inputs:
        raise SystemExit(0)

    # Import the runtime only after validating every required input.
    import librosa
    import numpy as np
    import soundfile as sf
    import torch
    from vllm import LLM
    from smoke_offline import decode
    from upstream_processor import load_processor

    a.output.mkdir(parents=True, exist_ok=False)
    outputs = a.output / 'outputs'
    outputs.mkdir()
    model = Path(os.environ['MODEL_PATH'])
    config = json.loads((model / 'config.json').read_text())
    environment = {'sourcefind_baseline': lock['sourcefind_commit'], 'condition_sha256': lock['condition_sha256'],
                   'python': platform.python_version(), 'model_path': str(model.resolve()),
                   'model_manifest_sha256': hashlib.sha256((ROOT / 'sources/modelscope-files.json').read_bytes()).hexdigest(),
                   'packages': {n: metadata.version(n) for n in ('torch', 'vllm', 'transformers', 'openai', 'librosa')},
                   'devices': [{'name': torch.cuda.get_device_properties(i).name,
                                'arch': torch.cuda.get_device_properties(i).gcnArchName}
                               for i in range(torch.cuda.device_count())],
                   'device_count': torch.cuda.device_count(), 'dtype': 'bfloat16', 'batch_size': 1,
                   'max_model_len': 8192, 'max_num_seqs': 8, 'gpu_memory_utilization': 0.35,
                   'enforce_eager': os.environ.get('ALIGN_EAGER','0')=='1', 'compilation_config': json.loads(os.environ.get('ALIGN_COMPILATION','{}')), 'enable_prefix_caching': False, 'mm_processor_cache_gb': 0,
                   'evaluation': 'word timestamp predictions only; no boundary accuracy or AAS scoring'}
    (a.output / 'environment.json').write_text(json.dumps(environment, ensure_ascii=False, indent=2) + '\n')
    processor = load_processor()
    torch.manual_seed(42)
    llm = LLM(model=str(model), runner='pooling',
              hf_overrides={'architectures': ['Qwen3ASRForcedAlignerForTokenClassification']},
              dtype='bfloat16', gpu_memory_utilization=0.35, max_model_len=8192, max_num_seqs=8,
              enforce_eager=os.environ.get('ALIGN_EAGER','0')=='1', compilation_config=json.loads(os.environ.get('ALIGN_COMPILATION','{}')), enable_prefix_caching=False, mm_processor_cache_gb=0, seed=42)
    rows = []
    with (a.output / '反馈表.csv').open('w', encoding='utf-8-sig', newline='') as feedback:
        writer = csv.DictWriter(feedback, fieldnames=['condition_id', 'model', 'status', 'output_path', 'error', 'parameter_changes'])
        writer.writeheader()
        for case, path in inputs:
            row = {'condition_id': case['id'], 'model': case['model'], 'status': '失败',
                   'output_path': '', 'error': '', 'parameter_changes': '无'}
            try:
                audio, sr = sf.read(path, dtype='float32')
                assert np.isfinite(audio).all() and len(audio) > 0
                if audio.ndim == 2:
                    audio = audio.mean(axis=1)
                if sr != 16000:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                assert len(audio) / 16000 <= 15.01
                words, prompt = processor.encode_timestamp(case['text'], 'English')
                fixture = {'case': {'id': case['id'], 'language': 'English'},
                           'words': words, 'audio_seconds': len(audio) / 16000}
                payload = {'prompt': prompt, 'multi_modal_data': {'audio': (np.asarray(audio, dtype=np.float32), 16000)}}
                start = time.perf_counter()
                output = llm.encode([payload], pooling_task='token_classify', use_tqdm=False)[0]
                record = decode(output, fixture, config, processor, time.perf_counter() - start)
                record.update(timestamp_unit='seconds', evaluation_status='BOUNDARIES_PENDING',
                              audio_sha256=case['assets']['audio']['sha256'], seed=42,
                              condition_sha256=lock['condition_sha256'])
                filename = case['id'] + '.json'
                (outputs / filename).write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
                row.update(status='成功', output_path='outputs/' + filename)
            except Exception as error:
                row['error'] = type(error).__name__ + ': ' + str(error)
            writer.writerow(row)
            feedback.flush()
            rows.append(row)
            summary = {'expected': 30, 'completed': len(rows), 'successful': sum(r['status'] == '成功' for r in rows),
                       'failed': sum(r['status'] == '失败' for r in rows), 'rows': rows,
                       'boundary_accuracy_computed': False, 'all_conditions_processed': len(rows) == 30}
            (a.output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
            print(json.dumps({k: summary[k] for k in ('completed', 'successful', 'failed')}) , flush=True)
    raise SystemExit(0 if summary['successful'] == 30 else 2)


if __name__ == "__main__":
    main()
