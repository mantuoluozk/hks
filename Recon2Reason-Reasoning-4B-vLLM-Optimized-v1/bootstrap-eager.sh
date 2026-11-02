#!/usr/bin/env bash
set -euo pipefail
if [ -f /opt/recon2reason-vllm/platform/env-scnet.sh ]; then
  source /opt/recon2reason-vllm/platform/env-scnet.sh
else
  source /root/private_data/hks/platform/scnet-notebook/env-scnet.sh
fi
export PYTHONPATH="/opt/qwen3-asr-deps${PYTHONPATH:+:$PYTHONPATH}"
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export MODEL_PATH="${MODEL_PATH:-/root/private_data/SothisAI/model/Aihub/Recon2Reason-Reasoning-4B/main/Recon2Reason-Reasoning-4B}"
mkdir -p results logs
python - <<'PY'
import hashlib,json,os,platform
from pathlib import Path
import torch,vllm,transformers
p=Path(os.environ['MODEL_PATH'])
names=set(json.loads((p/'model.safetensors.index.json').read_text())['weight_map'].values())
names.update(x.name for x in p.iterdir() if x.suffix in {'.json','.jinja','.txt'} and x.name!='configuration.json')
rows=[]
for name in sorted(names):
 f=p/name
 h=hashlib.sha256()
 with f.open('rb') as r:
  for b in iter(lambda:r.read(8*1024*1024),b''): h.update(b)
 rows.append(dict(path=name,bytes=f.stat().st_size,sha256=h.hexdigest()))
reference={}
for line in (p/'SHA256SUMS').read_text().splitlines():
 parts=line.split(maxsplit=1)
 if len(parts)==2: reference[parts[1].lstrip('*').removeprefix('./')]=parts[0]
for r in rows:
 r['reference_match']=r['sha256']==reference[r['path']] if r['path'] in reference else None
assert not any(r['reference_match'] is False for r in rows),rows
assert all(r['reference_match'] is True for r in rows if r['path'].endswith('.safetensors'))
Path('results/preflight.json').write_text(json.dumps(dict(python=platform.python_version(),torch=torch.__version__,vllm=vllm.__version__,transformers=transformers.__version__,device=torch.cuda.get_device_name(0),weights=rows),indent=2))
PY
echo "$$" > results/server.pid
exec python -m vllm.entrypoints.openai.api_server --model "$MODEL_PATH" --served-model-name recon2reason-4b --host 127.0.0.1 --port 30123 --dtype bfloat16 --max-model-len "${MAX_LEN:-8192}" --gpu-memory-utilization "${MEM_FRAC:-0.60}" --max-num-seqs "${MAX_RUNNING_REQUESTS:-8}" --enforce-eager --limit-mm-per-prompt '{"image":2}'
