"""Single-process AuK stage probe; functional evidence, not HTTP performance."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback

parser = argparse.ArgumentParser()
parser.add_argument('--output', required=True)
parser.add_argument('--seeds', nargs='+', type=int)
parser.add_argument('--seconds', type=float, default=4)
parser.add_argument('--attention-backend', choices=['default','math'], default='default')
parser.add_argument('--compute-dtype', choices=['bfloat16','float32'], default='bfloat16')
parser.add_argument('--conditioning-dtype', choices=['bfloat16','float32'])
args = parser.parse_args()
out = Path(args.output)
out.mkdir(parents=True, exist_ok=True)
ROOT = Path(__file__).resolve().parents[1]
record = {'kind': 'single-process functional stage probe', 'official_accuracy_tested': False,
          'http_performance_measured': False, 'events': [], 'cases': []}

def event(label, **data):
    item = {'event': label, 'time': time.time(), **data}
    record['events'].append(item)
    print(json.dumps(item), flush=True)
    (out/'summary.json').write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')

try:
    import numpy as np
    import torch
    if args.attention_backend == 'math':
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        torch.use_deterministic_algorithms(True)
    import soundfile as sf
    from sglang_omni.models.auk import stages as S
    from sglang_omni.models.auk.reference_encode import AuKConditionEncoder
    from sglang_omni.models.auk.hf_config import make_runtime_config
    from sglang_omni.models.auk.request_builders import AuKPreprocessingContext, set_auk_preprocessing_context, preprocess_auk_payload
    from sglang_omni.proto import OmniRequest, StagePayload
    assert json.loads((ROOT/'results/weights-check-20261028/summary.json').read_text())['main_and_vae_passed']
    assert json.loads((ROOT/'results/conditioner-download-20261028/weights-check.json').read_text())['passed']
    device = torch.device('cuda:0')
    record['device'] = torch.cuda.get_device_name(0)
    record['torch'] = torch.__version__
    event('device_ready', free_bytes=torch.cuda.mem_get_info()[0])
    checkpoint = '/root/private_data/SothisAI/model/Aihub/AuK/main/AuK'
    config = make_runtime_config(checkpoint)
    set_auk_preprocessing_context(AuKPreprocessingContext(config=config))
    event('loading_encoder')
    encoder = AuKConditionEncoder(str(ROOT/'weights/Qwen2.5-Omni-3B'), device=device, dtype=torch.bfloat16)
    event('loading_vae')
    vae = S._load_vae(checkpoint, str(device))
    event('loading_dit')
    flow = S._load_flow(checkpoint, str(device))
    event('models_loaded', free_bytes=torch.cuda.mem_get_info()[0],
          training_modules=sum(m.training for model in [encoder.model,vae,flow] for m in model.modules()))
    cases = [('english', 'Hello, welcome to our speech generation test.',42),
             ('chinese', '你好，欢迎参加语音生成测试。',42)]
    if args.seeds:
        cases = [(f'seed{seed}-{i}', cases[0][1], seed) for i,seed in enumerate(args.seeds)]
    with torch.inference_mode():
        for name, text, seed in cases:
            request = OmniRequest(inputs=text, params={'gen_seconds': args.seconds, 'seed': seed},
                                  metadata={'task': 'tts'})
            payload = StagePayload(request_id=name, request=request, data={})
            event('case_start', name=name)
            started = time.perf_counter()
            payload = preprocess_auk_payload(payload)
            event('seed_propagation', name=name, seed=payload.data.get('seed'))
            payload = S._condition_batch([payload], encoder, vae, flow, device, args.conditioning_dtype or args.compute_dtype)[0]
            tensor = payload.data['conditioning'].float().cpu()
            assert torch.isfinite(tensor).all(), 'Non-finite conditioning'
            event('conditioning_done', name=name, tensor_sha256=hashlib.sha256(tensor.numpy().tobytes()).hexdigest())
            payload = S._sample_batch([payload], flow, device, args.compute_dtype, config.seconds_to_frames(30),
                                      dict(steps=32, cfg_strength=2.0, sway_sampling_coef=-1.0, t_grid=None))[0]
            tensor = payload.data['latent'].float().cpu()
            event('sampling_done', name=name, tensor_sha256=hashlib.sha256(tensor.numpy().tobytes()).hexdigest())
            payload = S._decode_batch([payload], vae, device)[0]
            torch.cuda.synchronize()
            waveform = np.frombuffer(payload.data['audio_waveform'], dtype=np.float32).copy()
            assert waveform.size and np.isfinite(waveform).all()
            assert float(np.std(waveform)) > 1e-6, 'flat output'
            rate = payload.data['sample_rate']
            wav = out/(name+'.wav')
            sf.write(wav, waveform, rate, subtype='FLOAT')
            result = {'name': name, 'text': text, 'seed': seed, 'seconds_requested': args.seconds,
                      'sample_rate': rate, 'samples': waveform.size,
                      'seconds_actual': waveform.size/rate, 'elapsed_seconds': time.perf_counter()-started,
                      'rms': float(np.sqrt(np.mean(waveform**2))), 'peak': float(np.abs(waveform).max()),
                      'waveform_sha256': hashlib.sha256(waveform.tobytes()).hexdigest(),
                      'wav_sha256': hashlib.sha256(wav.read_bytes()).hexdigest(), 'passed': True}
            record['cases'].append(result)
            event('case_done', **result)
    record['passed'] = len(record['cases']) == len(cases)
    event('finished', passed=record['passed'])
except Exception as error:
    record['passed'] = False
    record['error'] = type(error).__name__+': '+str(error)
    record['traceback'] = traceback.format_exc()
    event('failed', error=record['error'])
    raise
