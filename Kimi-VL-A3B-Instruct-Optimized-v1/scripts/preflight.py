"""Check the pinned runtime files and native Kimi tokenizer/processor on SCNet."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from download_weights import check, FILES

out = Path(os.environ["RUN_RESULTS_DIR"])
model = Path(os.environ["MODEL_PATH"])
proof = {f["rfilename"]: check(model / f["rfilename"], f) for f in FILES}
index = json.loads((model/"model.safetensors.index.json").read_text())
assert set(index["weight_map"].values()) == {
    f["rfilename"] for f in FILES if f["rfilename"].endswith(".safetensors")}
record = {"publisher_files": proof, "weight_index_valid": True,
          "packages": {name: importlib.metadata.version(name) for name in
                       ["torch", "vllm", "transformers", "tiktoken", "pillow"]}}
(out/"weights-and-environment.json").write_text(json.dumps(record, indent=2)+"\n")

import torch
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer
from vllm.model_executor.models.kimi_vl import KimiVLForConditionalGeneration

tokenizer = AutoTokenizer.from_pretrained(str(model), trust_remote_code=True, local_files_only=True)
processor = AutoProcessor.from_pretrained(str(model), trust_remote_code=True, local_files_only=True)
text = processor.apply_chat_template([{"role": "user", "content": [
    {"type": "image"}, {"type": "text", "text": "Describe this image."}]}],
    add_generation_prompt=True, tokenize=False)
inputs = processor(text=[text], images=[Image.new("RGB", (224, 224), (255, 0, 0))],
                   return_tensors="pt")
record.update({"tokenizer": type(tokenizer).__name__, "processor": type(processor).__name__,
               "native_model_import": True,
               "processor_shapes": {k: list(v.shape) for k, v in inputs.items() if hasattr(v, "shape")},
               "rendered_prompt": text, "device": torch.cuda.get_device_name(0),
               "hip": torch.version.hip,
               "model_inference_verified": False, "official_accuracy_pass": None})
(out/"summary.json").write_text(json.dumps(record, indent=2)+"\n")
print(json.dumps({k: v for k, v in record.items() if k != "publisher_files"}), flush=True)
