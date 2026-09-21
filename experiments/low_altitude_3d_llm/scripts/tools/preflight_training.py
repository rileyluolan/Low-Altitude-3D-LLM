#!/usr/bin/env python3
"""Validate exact VLM loading plus forward/backward on a real HUGE training sample."""
from datetime import datetime, timezone
import json
from pathlib import Path
import random

import numpy as np
from omegaconf import OmegaConf
from safetensors import safe_open
import torch
from transformers import AutoConfig, AutoProcessor

from starVLA.dataloader.lerobot_datasets import get_vla_dataset
from starVLA.model.framework.base_framework import baseframework

from experiment_paths import ROOT, load_config
from assemble_base import RAW


def main():
    torch.set_num_threads(4)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    config = load_config(ROOT/'configs/qwen3vl_2b_hugebench_train.yaml')
    model_path = Path(config.framework.qwenvl.base_vlm)
    hf = AutoConfig.from_pretrained(model_path,local_files_only=True)
    assert hf.text_config.rope_scaling['mrope_section'] == [24,20,20]
    AutoProcessor.from_pretrained(model_path,local_files_only=True)
    assert config.trainer.max_train_steps == 16798
    assert config.trainer.num_warmup_steps == 840
    assert 4*config.datasets.vla_data.per_device_batch_size*config.trainer.gradient_accumulation_steps == 512
    assert not config.trainer.freeze_modules
    print('Building the model from the new assembled VLA checkpoint.',flush=True)
    model = baseframework.from_pretrained(config.trainer.pretrained_checkpoint)
    loaded = model.qwen_vl_interface.model.state_dict()
    with safe_open(RAW/'model.safetensors',framework='pt',device='cpu') as handle:
        assert set(handle.keys()) == set(loaded)
        for k in handle.keys():
            assert torch.equal(loaded[k].cpu(),handle.get_tensor(k)),f'Runtime VLM tensor mismatch: {k}'
    print('All 626 VLM tensors match the candidate after full VLA loading.',flush=True)
    model.to(device='cuda',dtype=torch.bfloat16)
    dataset = get_vla_dataset(config.datasets.vla_data,seed=42)
    assert len(dataset) == 1720096,len(dataset)
    batch = [dataset[0]]
    sample = batch[0]
    print('Real sample:',{k:(getattr(v,'shape',None) or type(v).__name__) for k,v in sample.items()},flush=True)
    assert np.asarray(sample['action']).shape[-2:] == (20,4)
    assert len(sample['image']) == 2
    model.train()
    output = model(examples=batch)
    loss = output['action_loss']
    assert torch.isfinite(loss).all()
    loss.backward()
    groups = {k:{'tensors_with_grad':0,'nonzero_grad_tensors':0} for k in ['qwen_vl_interface','project_layers','action_model']}
    for name,p in model.named_parameters():
        group = name.split('.')[0]
        if group in groups and p.grad is not None:
            assert bool(torch.isfinite(p.grad).all()),name
            groups[group]['tensors_with_grad'] += 1
            groups[group]['nonzero_grad_tensors'] += int(bool(torch.count_nonzero(p.grad)))
    assert all(g['nonzero_grad_tensors'] > 0 for g in groups.values()),groups
    print('Forward/backward passed:',float(loss.detach()),groups,flush=True)
    model.zero_grad(set_to_none=True)
    model.eval()
    prediction = model.predict_action(examples=batch)['normalized_actions']
    assert prediction.shape == (1,20,4),prediction.shape
    assert np.isfinite(prediction).all()
    result = {'status':'passed','completed_utc':datetime.now(timezone.utc).isoformat(),
              'checkpoint':str(config.trainer.pretrained_checkpoint),'all_runtime_vlm_tensors_exact':626,
              'real_training_sample':True,'dataset_frames':len(dataset),'loss':float(loss.detach()),
              'gradient_checks':groups,'prediction_shape':list(prediction.shape),
              'prediction_finite':True,'peak_gpu_memory_bytes':torch.cuda.max_memory_allocated(),
              'optimizer_steps_performed':0}
    (ROOT/'runtime').mkdir(parents=True, exist_ok=True)
    (ROOT/'runtime/preflight.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__ == '__main__':
    main()
