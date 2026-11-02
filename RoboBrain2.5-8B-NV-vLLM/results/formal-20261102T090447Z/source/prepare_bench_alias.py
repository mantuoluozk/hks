"""SGLang's image dataset loads its processor from the served model ID.

Resolve the simple service alias locally to avoid a network request for a fake
Hugging Face repository. No model or installed framework files are modified.
"""
import os
from pathlib import Path

root=Path(__file__).resolve().parents[1]
name=os.environ.get('SERVED_NAME','robobrain25-8b-vllm')
assert name and Path(name).name==name and name not in ['.','..']
model=Path(os.environ.get('MODEL_PATH','/root/private_data/SothisAI/model/Aihub/RoboBrain2.5-8B-NV/main/RoboBrain2.5-8B-NV')).resolve(strict=True)
alias=root/name
if alias.exists() or alias.is_symlink():
    assert alias.resolve()==model,(alias,model)
else:
    alias.symlink_to(model,target_is_directory=True)
print('Local benchmark processor alias:',alias,'->',model)
