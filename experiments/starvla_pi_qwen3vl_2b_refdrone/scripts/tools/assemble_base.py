#!/usr/bin/env python3
"""Create a new VLA initialization; all source checkpoint files are read-only."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import os
from experiment_paths import ROOT, MODEL, BASE, load_config

import torch
from safetensors import safe_open
import yaml

OLD = ROOT.with_name('starvla_pi_qwen3vl_2b')
SOURCE_BASE = Path(os.environ.get('SOURCE_BASE_DIR', OLD / 'artifacts/base/qwen3vl_2b_pi_v3_hugebench_mature_base')).expanduser().resolve()
RAW = Path(os.environ.get('REFDRONE_RAW_DIR', ROOT / 'models/qwen3vl_stage2_raw')).expanduser().resolve()
COMPAT = MODEL
REFERENCE = Path(os.environ.get('REFERENCE_VLM_DIR', OLD / 'models/Qwen3-VL-2B-Instruct')).expanduser().resolve()
OUTPUT = BASE.parent.parent
PREFIX = 'qwen_vl_interface.model.'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(32 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def finite(tensor):
    return all(bool(torch.isfinite(chunk).all()) for chunk in tensor.reshape(-1).split(4_000_000))


def main():
    torch.set_num_threads(4)
    destination = BASE
    if OUTPUT == SOURCE_BASE or COMPAT in (RAW, REFERENCE):
        raise RuntimeError('Output directories must differ from read-only inputs')
    if COMPAT.exists() and any(COMPAT.iterdir()):
        raise RuntimeError('Compatible model directory is not empty; choose a new output directory')
    if destination.exists():
        raise RuntimeError('Destination already exists; refusing to overwrite weights')
    download = json.loads((RAW / 'DOWNLOAD_MANIFEST.json').read_text())
    assert download['status'] == 'complete'
    source = SOURCE_BASE / 'checkpoints/base_pytorch_model.pt'
    source_manifest = json.loads((SOURCE_BASE / 'conversion_manifest.json').read_text())
    base_sha = digest(source)
    assert base_sha == source_manifest['target']['checkpoint_sha256']
    assert base_sha == '87e38d987fd2d1b7fd50b77529717d4fc19111b1c0ca58096f24e933b79b6c4b', 'This run requires the audited mature base'
    candidate_sha = digest(RAW / 'model.safetensors')
    assert candidate_sha == '5fd6eba5da2d7ad97ed6e3c61b7ab6baabddfa9f2c421170cb760bd68e9c0196'
    assert candidate_sha == next(f['sha256'] for f in download['files'] if f['name'] == 'model.safetensors')
    state = torch.load(source, weights_only=True, map_location='cpu', mmap=True)
    assert len(state) == 1154
    current_vlm = {k[len(PREFIX):]: v for k, v in state.items() if k.startswith(PREFIX)}
    ref_config = json.loads((REFERENCE / 'config.json').read_text())
    candidate_config = json.loads((RAW / 'config.json').read_text())
    for part, fields in [('text_config', ['hidden_size','num_hidden_layers','num_attention_heads',
            'num_key_value_heads','head_dim','intermediate_size','vocab_size','max_position_embeddings',
            'rms_norm_eps','hidden_act']), ('vision_config', ['depth','hidden_size','out_hidden_size',
            'patch_size','spatial_merge_size','temporal_patch_size','deepstack_visual_indexes'])]:
        for key in fields:
            assert ref_config[part][key] == candidate_config[part][key], (part,key)
    new_rope = candidate_config['text_config']['rope_parameters'].copy()
    assert new_rope.pop('rope_theta') == ref_config['text_config']['rope_theta']
    assert new_rope == ref_config['text_config']['rope_scaling']
    assert ref_config['tie_word_embeddings'] == candidate_config['tie_word_embeddings'] is True
    for key in ['image_token_id','video_token_id','vision_start_token_id','vision_end_token_id']:
        assert ref_config[key] == candidate_config[key], key
    tokenizer = json.loads((REFERENCE / 'tokenizer.json').read_text())
    new_tokenizer = json.loads((RAW / 'tokenizer.json').read_text())
    assert tokenizer['model']['vocab'] == new_tokenizer['model']['vocab']
    def merges(value):
        return [tuple(x.split(' ')) if isinstance(x, str) else tuple(x) for x in value]
    assert merges(tokenizer['model']['merges']) == merges(new_tokenizer['model']['merges'])
    assert tokenizer['added_tokens'] == new_tokenizer['added_tokens']
    assert json.loads((REFERENCE/'tokenizer_config.json').read_text())['chat_template'] == (RAW/'chat_template.jinja').read_text()
    # Keep the numerically compatible, already validated 4.57 input/config format.
    COMPAT.mkdir(parents=True, exist_ok=True)
    compat_files = []
    for p in REFERENCE.iterdir():
        if p.is_file() and p.suffix in {'.json','.txt'}:
            shutil.copy2(p, COMPAT / p.name)
            compat_files.append({'name':p.name,'sha256':digest(p)})
    (COMPAT / 'model.safetensors').symlink_to(RAW / 'model.safetensors')
    with safe_open(RAW / 'model.safetensors', framework='pt', device='cpu') as handle:
        assert set(handle.keys()) == set(current_vlm)
        candidate = {}
        for key in handle.keys():
            t = handle.get_tensor(key)
            assert t.shape == current_vlm[key].shape, key
            assert finite(t), key
            candidate[key] = t
        embedding = 'model.language_model.embed_tokens.weight'
        assert torch.equal(candidate[embedding], candidate['lm_head.weight']), 'Saved tied embedding/lm_head differ'
        candidate['lm_head.weight'] = candidate[embedding]
        assembled = dict(state)
        for key, tensor in candidate.items():
            assembled[PREFIX+key] = tensor
        for key,tensor in assembled.items():
            if not key.startswith(PREFIX):
                assert finite(tensor), key
                assert torch.equal(tensor,state[key]), key
        destination.parent.mkdir(parents=True,exist_ok=True)
        temporary = destination.with_suffix('.pt.partial')
        torch.save(assembled,temporary)
        temporary.replace(destination)
    reloaded = torch.load(destination,weights_only=True,map_location='cpu',mmap=True)
    assert reloaded.keys() == state.keys()
    changed = 0
    for key,tensor in reloaded.items():
        assert torch.equal(tensor,assembled[key]), key
        if key.startswith(PREFIX):
            changed += int(not torch.equal(tensor,state[key]))
        else:
            assert torch.equal(tensor,state[key]),key
    assert changed > 0
    stats_source = SOURCE_BASE/'dataset_statistics.json'
    shutil.copy2(stats_source,OUTPUT/'dataset_statistics.json')
    from omegaconf import OmegaConf
    config = OmegaConf.to_container(load_config(ROOT/'configs/qwen3vl_2b_hugebench_train.yaml'), resolve=True)
    config['run_id'] = 'qwen3vl_2b_pi_v3_hugebench_refdrone_base'
    config['trainer']['pretrained_checkpoint'] = None
    (OUTPUT/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    manifest = {
        'completed_utc':datetime.now(timezone.utc).isoformat(),
        'identity':'HUGE-pretraining mature VLA base with RefDrone Stage 2 VLM',
        'source_base':{'checkpoint':str(source),'sha256':base_sha,'bytes':source.stat().st_size},
        'source_vlm':{'blob_prefix':download['source_blob_prefix'],'file':str(RAW/'model.safetensors'),
                      'sha256':candidate_sha},
        'target':{'checkpoint':str(destination),'checkpoint_sha256':digest(destination),
                  'checkpoint_size':destination.stat().st_size,'parameters':source_manifest['target']['parameters']},
        'replacement':{'prefix':PREFIX,'vlm_tensors':len(candidate),'changed_vlm_tensors':changed,
                       'preserved_projector_tensors':112,'preserved_action_tensors':416,
                       'all_reloaded_tensors_exact':True,'all_tensors_finite':True,'tied_embeddings_equal':True},
        'compatibility':{'transformers':'4.57.0','raw_saved_version':'5.2.0','reference_files':compat_files,
                         'rope_parameters_match':True,'token_ids_and_chat_template_match':True},
        'dataset_statistics_sha256':digest(stats_source),
        'source_weights_modified':False,
    }
    (OUTPUT/'conversion_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (COMPAT/'COMPATIBILITY.json').write_text(json.dumps(manifest['compatibility'],indent=2)+'\n')
    print(json.dumps(manifest,indent=2),flush=True)


if __name__ == '__main__':
    main()
