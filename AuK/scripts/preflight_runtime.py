"""Import-only compatibility check; never downloads or loads any model weights."""
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
out = Path(os.environ["RUN_RESULTS_DIR"])
out.mkdir(parents=True, exist_ok=True)
record = {"python": sys.version, "modules": {}, "packages": {},
          "model_loaded": False, "model_accuracy_tested": False,
          "runtime_revision": "428b73225f808f0d93669140983e90839c9aea0e"}
manifest = json.loads((ROOT/"sources/sglang-omni-runtime-files.json").read_text())
compat_path = ROOT/'sources/dtk-compat-files.json'
compat = json.loads(compat_path.read_text())['files'] if compat_path.is_file() else []
patched = {entry['path']: entry['sha256'] for entry in compat}
for entry in manifest["files"]:
    path = ROOT/"runtime/sglang-omni"/entry["path"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == patched.get(entry['path'], entry["sha256"]), entry["path"]
for entry in compat:
    assert hashlib.sha256((ROOT/'runtime/sglang-omni'/entry['path']).read_bytes()).hexdigest() == entry['sha256']
record['compatibility_files_verified'] = len(compat)
record["runtime_source_files_verified"] = len(manifest["files"])
for package in ["torch", "sglang", "transformers", "pydantic", "numpy", "torchaudio"]:
    record["packages"][package] = importlib.metadata.version(package)
for name in ["sglang_omni.config", "sglang_omni.models.auk.config",
             "sglang_omni.models.auk.stages", "sglang_omni.cli"]:
    try:
        module = importlib.import_module(name)
        record["modules"][name] = {"imported": True, "file": module.__file__}
    except Exception as error:
        record["modules"][name] = {"imported": False, "error": type(error).__name__+": "+str(error),
                                   "traceback": traceback.format_exc()}
record["all_imports_passed"] = all(value["imported"] for value in record["modules"].values())
(out/"summary.json").write_text(json.dumps(record, indent=2)+"\n")
print(json.dumps(record, indent=2), flush=True)
if not record["all_imports_passed"]:
    raise SystemExit(2)
