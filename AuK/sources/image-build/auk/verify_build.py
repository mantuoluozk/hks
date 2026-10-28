import hashlib,json,pathlib,subprocess,sys
root=pathlib.Path('/opt/auk')
for name,digest in json.loads((root/'SHA256.json').read_text()).items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
subprocess.run([sys.executable,'-m','pip','install','--no-deps','--no-index','--target',str(root/'site-packages'),*[str(p) for p in sorted((root/'wheels').glob('*.whl'))]],check=True)
