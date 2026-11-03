"""Process-local, exact-source-guarded Kimi projector correction."""
import importlib.abc
import importlib.util
import pathlib
import runpy
import sys

TARGET = 'sglang.srt.models.kimi_vl'
class KimiSource(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != TARGET:
            return None
        original = pathlib.Path(__file__).resolve().parents[1] / 'sources/kimi_vl.patched.py'
        return importlib.util.spec_from_file_location(fullname, original)

sys.meta_path.insert(0, KimiSource())
if __name__ == '__main__':
    runpy.run_module('sglang.launch_server', run_name='__main__')
