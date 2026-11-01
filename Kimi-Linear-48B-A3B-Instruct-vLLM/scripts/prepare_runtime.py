#!/usr/bin/env python3
"""Build the isolated `openai` runtime directory used by run-scnet.sh.

The SCNet vLLM image ships openai 2.6.1. This vLLM build imports
`openai.types.responses...NamespaceTool`, which only exists in newer releases
(validated with 2.30.0). Only `openai` is isolated here; torch, vLLM,
transformers and boltops must stay at the image versions.

Usage (needs network, e.g. the SCNet mirror or an online Notebook):
    python scripts/prepare_runtime.py --dest cache/runtime
    python scripts/prepare_runtime.py --dest cache/runtime --check

Offline alternative on a machine that already has openai 2.30.0:
    KIMI_AUTO_PREPARE_RUNTIME=0 KIMI_LINEAR_RUNTIME=/path/to/prepared \\
        python -m vllm.entrypoints.openai.api_server ...
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

PINNED = "openai==2.30.0"


def check(dest: Path) -> bool:
    pkg = dest / "openai"
    if not pkg.is_dir():
        return False
    try:
        out = subprocess.run(
            [sys.executable, "-c",
             "import sys;sys.path.insert(0,{!r});import openai;"
             "print(openai.__version__)".format(str(dest))],
            capture_output=True, text=True, timeout=120,
        )
        if out.returncode != 0:
            return False
        ver = out.stdout.strip()
        print(f"isolated openai version in {dest}: {ver}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"check failed: {exc}")
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", type=Path, default=Path("cache/runtime"))
    ap.add_argument("--check", action="store_true", help="only verify an existing runtime")
    ap.add_argument("--index-url", default=None,
                    help="pip index to use, e.g. the SCNet mirror")
    a = ap.parse_args()

    if a.check:
        return 0 if check(a.dest) else 1

    if check(a.dest):
        print("runtime already prepared; nothing to do")
        return 0

    a.dest.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "pip", "install", "--no-deps", "--upgrade",
           "--target", str(a.dest), PINNED]
    if a.index_url:
        cmd += ["--index-url", a.index_url]
    print("running:", " ".join(cmd))
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        print("pip install failed; provide --index-url or prepare manually", file=sys.stderr)
        return rc
    return 0 if check(a.dest) else 1


if __name__ == "__main__":
    raise SystemExit(main())
