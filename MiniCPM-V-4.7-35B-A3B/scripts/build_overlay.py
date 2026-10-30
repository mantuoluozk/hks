"""Reproduce model-local upstream backport, without modifying site-packages."""
import ast, hashlib, json, re
from pathlib import Path
root = Path(__file__).resolve().parents[1]
source = root / "sources/upstream-vllm"
for row in json.loads((root/"sources/upstream-files.json").read_text()):
    assert hashlib.sha256((source/row["file"]).read_bytes()).hexdigest()==row["sha256"], row["file"]
out = root / "python/minicpm_adapter"
out.mkdir(parents=True, exist_ok=True)
base = (source / "minicpmv4_6.py").read_text(encoding="utf-8")
base = re.sub(r"from \.(\w+) import", r"from vllm.model_executor.models.\1 import", base)
base = base.replace("    cached_encode,", "    _cached_encode as cached_encode,")
base = base.replace("    _image_kwargs_from_video,\n", "")
# Port the exact upstream video->image field mapping/helper, absent in DAS 0.25.1.
helper = (source / "minicpmv.py").read_text(encoding="utf-8")
start = helper.index("_VIDEO_TO_IMAGE_KWARGS =")
end = helper.index("def _minicpmv_field_config", start)
base = base.replace("def _minicpmv4_6_field_config", helper[start:end] + "def _minicpmv4_6_field_config", 1)
# The DAS caller passes tok_kwargs as a third argument; these image-only
# routines deliberately leave text tokenization to the inherited text path.
signature = "        mm_kwargs: Mapping[str, object],\n    ) -> Mapping[str, NestedTensors]:"
assert base.count(signature) == 2
base = base.replace(signature, "        mm_kwargs: Mapping[str, object],\n        tok_kwargs: Mapping[str, object] | None = None,\n    ) -> Mapping[str, NestedTensors]:")
context = (source / "processing_context.py").read_text(encoding="utf-8")
context_tree = ast.parse(context)
helpers = []
for node in context_tree.body:
    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "_HF_PROCESSOR_KWARG_SCOPES" for t in node.targets):
        helpers.append(ast.get_source_segment(context, node))
    if isinstance(node, ast.FunctionDef) and node.name in ("_merge_scoped_mm_processor_value", "_resolve_mm_processor_kwargs"):
        helpers.append(ast.get_source_segment(context, node))
assert len(helpers) == 3
base = base.replace("def _minicpmv4_6_field_config", "\n\n".join(helpers) + "\n\ndef _minicpmv4_6_field_config", 1)
bridge = '''    def _merge_and_resolve_mm_processor_kwargs(self, mm_kwargs):
        from typing import get_type_hints
        processor = self.get_hf_processor()
        schema = get_type_hints(processor.valid_processor_kwargs)
        supported = {"text_kwargs": set(schema["text_kwargs"].__annotations__)}
        for modality in ("image", "video"):
            child = getattr(processor, modality + "_processor", None)
            if child is not None:
                scope = "images_kwargs" if modality == "image" else "videos_kwargs"
                supported[scope] = set(child.valid_kwargs.__annotations__)
        return _resolve_mm_processor_kwargs(self.ctx.get_merged_mm_kwargs(mm_kwargs), supported)

'''
base = base.replace("class MiniCPMV4_6ProcessingInfo(MiniCPMVProcessingInfo):\n", "class MiniCPMV4_6ProcessingInfo(MiniCPMVProcessingInfo):\n" + bridge, 1)
model = (source / "minicpmv4_7.py").read_text(encoding="utf-8")
model = model.replace("from .minicpmv import", "from vllm.model_executor.models.minicpmv import")
model = model.replace("from .minicpmv4_6 import", "from .base import")
tests = (source / "test_minicpmv4_7.py").read_text(encoding="utf-8")
tests = tests.replace("vllm.multimodal.processing.context", "minicpm_adapter.base")
tests = tests.replace("vllm.model_executor.models.minicpmv4_6", "minicpm_adapter.base").replace("vllm.model_executor.models.minicpmv4_7", "minicpm_adapter.model")
parser = (source / "parser_minicpmv.py").read_text(encoding="utf-8").replace("vllm.entrypoints.generate.base.protocol", "vllm.entrypoints.openai.engine.protocol")
# DAS 0.25.1 checks token IDs directly instead of consulting reasoning_ended.
parser = parser.replace("    @property\n    def reasoning_ended", "    def is_reasoning_end(self, input_ids):\n        if not self.thinking_enabled:\n            return True\n        return super().is_reasoning_end(input_ids)\n\n    @property\n    def reasoning_ended", 1)
adapter = (source / "minicpmv_reasoning_parser.py").read_text(encoding="utf-8").replace("from vllm.parser.minicpmv import", "from .parser import")
parser_tests = (source/"test_minicpmv_reasoning.py").read_text(encoding="utf-8")
parser_tests = parser_tests.replace("tests.parser.engine.conftest", "minicpm_adapter.parser_helpers").replace("tests.parser.engine.streaming_helpers", "minicpm_adapter.streaming_helpers").replace("vllm.entrypoints.generate.base.protocol", "vllm.entrypoints.openai.engine.protocol").replace("from vllm.parser.minicpmv import", "from minicpm_adapter.parser import")
parser_tests = "from minicpm_adapter import register\nregister()\n" + parser_tests
parser_helpers = (source/"parser_conftest.py").read_text(encoding="utf-8")
streaming_helpers = (source/"streaming_helpers.py").read_text(encoding="utf-8").replace("vllm.entrypoints.generate.base.protocol", "vllm.entrypoints.openai.engine.protocol")
records = []
for name, text in [("base.py", base), ("model.py", model), ("test_upstream.py", tests), ("parser.py", parser), ("reasoning.py", adapter), ("test_parser.py", parser_tests), ("parser_helpers.py", parser_helpers), ("streaming_helpers.py", streaming_helpers)]:
    ast.parse(text)
    (out/name).write_text(text, encoding="utf-8", newline="\n")
    records.append({"file": name, "sha256": hashlib.sha256(text.encode()).hexdigest()})
(root/"sources/overlay-files.json").write_text(json.dumps({"vllm_commit": "c85616efc6f0eaa50ca1f1a09c32368aedffd3d8", "changes": ["model-local import paths", "cached_encode uses DAS _cached_encode identical signature", "copy upstream video field mapping helper", "old protocol import path", "DAS engine no-thinking end hook", "DAS image/video processor tok_kwargs call signature", "parser registry wrapper scoped to minicpmv"], "files": records}, indent=2)+"\n", encoding="utf-8", newline="\n")
print(json.dumps(records))
