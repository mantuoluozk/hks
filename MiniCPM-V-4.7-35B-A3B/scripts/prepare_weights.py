"""Create a checked symlink view of existing weights; never copy or download them."""
import argparse
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(__doc__)
p.add_argument('source', type=Path, help='Existing complete publisher/community model directory')
a = p.parse_args()
source = a.source.resolve(strict=True)
manifest = json.loads((root / 'sources/publisher-files.json').read_text())
configs = root / 'sources/publisher-configs'
links = []
for item in manifest['siblings']:
    name = item['rfilename']
    if '/' in name or not name.endswith(('.json', '.jinja', '.safetensors')):
        continue
    target = (configs / name if (configs / name).is_file() else source / name).resolve(strict=True)
    destination = root / 'weights' / name
    if destination.exists() or destination.is_symlink():
        assert destination.is_symlink() and destination.resolve() == target, f'Refusing to replace {destination}'
    links.append((destination, target))
(root / 'weights').mkdir(exist_ok=True)
for destination, target in links:
    if not destination.is_symlink():
        destination.symlink_to(target)
print(f'Prepared {len(links)} links; run check_weights.py before inference')
