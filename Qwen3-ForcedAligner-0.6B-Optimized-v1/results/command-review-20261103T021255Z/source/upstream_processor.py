"""Load the unmodified pinned text/timestamp processor without the HF model wrapper.

This adapter currently accepts English/Chinese only. Japanese/Korean tokenizers are
not installed or claimed as validated. The original Korean dictionary is retained
so that the upstream constructor can run without changing its implementation.
"""
import ast
import hashlib
import json
import os
from pathlib import Path
from typing import List
import unicodedata

ROOT=Path(__file__).resolve().parents[1]


def load_processor():
    manifest=json.loads((ROOT/'sources/upstream-source.json').read_text())
    for item in manifest['files']:
        path=ROOT/item['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256'],path
    path=ROOT/'sources/upstream/qwen_asr/inference/qwen3_forced_aligner.py'
    tree=ast.parse(path.read_text())
    node=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='Qwen3ForceAlignProcessor')
    namespace={'__file__':str(path),'os':os,'unicodedata':unicodedata,'List':List}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
    return namespace['Qwen3ForceAlignProcessor']()
