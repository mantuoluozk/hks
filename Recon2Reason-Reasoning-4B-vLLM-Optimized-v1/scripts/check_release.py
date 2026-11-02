"""Validate the release file inventory using SHA256; no model execution."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 manifest=ROOT/'sources/release-manifest.json'
 rows=json.loads(manifest.read_text(encoding='utf-8'))['files'];seen=set()
 for row in rows:
  rel=row['path'];assert rel not in seen;seen.add(rel)
  p=(ROOT/rel).resolve();assert p.is_relative_to(ROOT.resolve()) and p.is_file()
  assert p.stat().st_size==row['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'],rel
 assert {'README.md','bootstrap.sh','run.sh','bench.sh','verify.py'}.issubset(seen)
 print('Release file SHA256:',len(rows),'passed')
if __name__=='__main__':main()
