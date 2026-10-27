"""Launch SGLang's generic Diffusers backend with process-local compatibility fixes."""
import os
import sys
import json
from pathlib import Path
import torch
from transformers import AutoConfig,PretrainedConfig
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

# This SGLang build predates the built-in Qwen3-ASR config in Transformers 5.17.
# Keep the new built-in audio config; it is unrelated to the Qwen3-VL image encoder.
# Do not alter any other registration or modify installed framework files.
original_register=AutoConfig.register
original_standardize_rope=Qwen3VLVisionConfig.standardize_rope_params


def register(model_type,config,exist_ok=False):
    if model_type=='qwen3_asr' and model_type in CONFIG_MAPPING:
        print('Preserving Transformers built-in qwen3_asr registration',flush=True)
        return
    return original_register(model_type,config,exist_ok=exist_ok)


AutoConfig.register=staticmethod(register)
from sglang.cli.main import main
from sglang.multimodal_gen.runtime.pipelines.diffusers_pipeline import DiffusersExecutionStage,DiffusersPipeline
from sglang.multimodal_gen.configs.pipeline_configs.diffusers_generic import DiffusersTI2IPipelineConfig
from sglang.multimodal_gen.configs.sample.diffusers_generic import DiffusersGenericSamplingParams
from sglang.multimodal_gen.registry import register_configs
from sglang.multimodal_gen.runtime.utils.vision import load_image

# SGLang's older generic guard skips standardization when max_position_embeddings
# is absent. Vision configs deliberately lack that text-only field; skipping the
# native method changes their axial default. Restore the unmodified HF method on
# this vision config class only, without writing to the checkpoint or site-packages.
Qwen3VLVisionConfig.standardize_rope_params=original_standardize_rope

register_configs(sampling_param_cls=DiffusersGenericSamplingParams,pipeline_config_cls=DiffusersTI2IPipelineConfig,
                 hf_model_paths=['Qwen/Qwen-Image-2.1-Turbo',os.environ['MODEL_PATH']])

original_load=DiffusersPipeline._load_diffusers_pipeline
original_build=DiffusersExecutionStage._build_pipeline_kwargs
original_postprocess=DiffusersExecutionStage._postprocess_output
original_load_image=DiffusersExecutionStage._load_input_image


def load_pipeline(self,model_path,server_args):
    pipe=original_load(self,model_path,server_args)
    dtypes={name:sorted({str(p.dtype) for p in component.parameters()})
            for name,component in pipe.components.items() if isinstance(component,torch.nn.Module)}
    assert all(d==['torch.bfloat16'] for d in dtypes.values()),dtypes
    assert len(pipe.config.sample_sigmas)==8
    if os.environ.get('HKS_TRACE_DIR'):
        out=Path(os.environ['HKS_TRACE_DIR']);out.mkdir(parents=True,exist_ok=True)
        (out/f'loaded-{os.getpid()}.json').write_text(json.dumps({'pipeline':type(pipe).__name__,
            'parameter_dtypes':dtypes,'sample_sigmas':list(pipe.config.sample_sigmas),
            'component_devices':{name:str(next(c.parameters()).device) for name,c in pipe.components.items() if isinstance(c,torch.nn.Module)},
            'sglang_backend':'diffusers','model_path':str(model_path)},indent=2))
    return pipe


def build_kwargs(self,batch):
    kwargs=original_build(self,batch)
    self._hks_calls=getattr(self,'_hks_calls',0)+1
    if os.environ.get('HKS_TRACE_DIR'):
        out=Path(os.environ['HKS_TRACE_DIR']);out.mkdir(parents=True,exist_ok=True)
        generators=kwargs.get('generator')
        generators=generators if isinstance(generators,list) else [generators]
        images=kwargs.get('image')
        with (out/f'requests-{os.getpid()}.jsonl').open('a') as f:
            f.write(json.dumps({'call_index':self._hks_calls,'prompt':kwargs.get('prompt'),
                'width':kwargs.get('width'),'height':kwargs.get('height'),'seed':batch.seed,
                'generator_devices':[str(g.device) for g in generators if g is not None],
                'true_cfg_scale':kwargs.get('true_cfg_scale'),
                'requested_num_inference_steps':kwargs.get('num_inference_steps'),
                'condition_image_count':len(images) if isinstance(images,list) else int(images is not None)})+'\n')
    if self._hks_calls==1 and os.environ.get('HKS_TRACE_DIR'):
        out=Path(os.environ['HKS_TRACE_DIR']);out.mkdir(parents=True,exist_ok=True)
        (out/f'first-request-{os.getpid()}.json').write_text(json.dumps({
            'width':kwargs.get('width'),'height':kwargs.get('height'),'seed':batch.seed,
            'generator_device':([str(g.device) for g in kwargs['generator']] if isinstance(kwargs['generator'],list) else str(kwargs['generator'].device)),
            'true_cfg_scale':kwargs.get('true_cfg_scale')},indent=2))
        def audit(pipeline,index,timestep,values):
            assert torch.isfinite(values['latents']).all(),index
            with (out/f'first-request-steps-{os.getpid()}.jsonl').open('a') as f:
                f.write(json.dumps({'step':index,'timestep':float(timestep),'latent_shape':list(values['latents'].shape)})+'\n')
            return values
        kwargs['callback_on_step_end']=audit
    return kwargs


def strict_postprocess(self,output):
    if not torch.isfinite(output).all():raise RuntimeError('Non-finite generated image; do not silently repair pixels')
    return original_postprocess(self,output)


def load_images(self,batch):
    # The generic backend keeps only the first path; this model jointly accepts up to four.
    if isinstance(batch.image_path,list):
        if not 1<=len(batch.image_path)<=4:raise ValueError('This model accepts one to four condition images')
        return [load_image(path).convert('RGB') for path in batch.image_path]
    return original_load_image(self,batch)


DiffusersPipeline._load_diffusers_pipeline=load_pipeline
DiffusersExecutionStage._build_pipeline_kwargs=build_kwargs
DiffusersExecutionStage._postprocess_output=strict_postprocess
DiffusersExecutionStage._load_input_image=load_images
if __name__=='__main__':
    sys.argv=['sglang','serve','--model-type','diffusion',*sys.argv[1:]]
    main()
