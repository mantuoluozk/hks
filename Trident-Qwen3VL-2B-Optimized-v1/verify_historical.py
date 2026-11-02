"""Verify original evidence in its original directory."""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).resolve().parent.parent/'Trident-Qwen3VL-2B/verify.py'),run_name='__main__')
