"""Verify embedded payload and installed isolated packages without network access."""
import hashlib,json,socket
from pathlib import Path
root=Path('/opt/recon2reason-vllm')
rows=json.loads((root/'manifest.json').read_text())
checked=[]
for row in rows:
 paths=[root/row['path']]
 if row['path'].startswith('deps/'):
  paths.append(Path('/opt/qwen3-asr-deps')/row['path'][5:])
 for p in paths:
  assert p.stat().st_size==row['bytes'],p
  assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'],p
  checked.append(str(p))
out=Path('/root/private_data/hks/models/Recon2Reason-Reasoning-4B-vLLM/results/image-check-2116559999150473218.json')
out.write_text(json.dumps(dict(image_page='https://www.scnet.cn/ui/aihub/image/zack233/2116559999150473218',hostname=socket.gethostname(),payload_files=len(rows),total_checks=len(checked),manifest_sha256=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),verified=True),indent=2))
print(out.read_text())
