#!/usr/bin/env python3
"""Audit the local VLA identity and evidence of post-assembly training on CPU."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(32 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    base = ROOT / 'artifacts/base/qwen3vl_2b_pi_v3_hugebench_mature_base'
    checkpoint = base / 'checkpoints/base_pytorch_model.pt'
    manifest = json.loads((base / 'conversion_manifest.json').read_text())
    actual_sha = digest(checkpoint)
    assert actual_sha == manifest['target']['checkpoint_sha256']
    state = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    assert len(state) == 1154
    shapes = {
        'action_model.state_encoder.layer1.weight': (1024, 4),
        'action_model.action_encoder.layer1.weight': (1024, 4),
        'action_model.action_decoder.layer2.weight': (4, 1024),
        'action_model.action_decoder.layer2.bias': (4,),
    }
    for key, shape in shapes.items():
        assert tuple(state[key].shape) == shape, key
    blocks = sorted({int(k.split('.')[3]) for k in state
                     if k.startswith('action_model.model.transformer_blocks.')})
    assert blocks == list(range(28))
    finite_numel = 0
    for key, tensor in state.items():
        flat = tensor.reshape(-1)
        for chunk in flat.split(4_000_000):
            assert bool(torch.isfinite(chunk).all()), key
        finite_numel += tensor.numel()
    print(f'Checkpoint hash, shapes and {finite_numel:,} finite tensor elements passed.', flush=True)

    projectors = []
    for layer in range(28):
        norm = state[f'project_layers.{layer}.0.weight']
        bias = state[f'project_layers.{layer}.0.bias']
        assert norm.shape == (2048,)
        assert state[f'project_layers.{layer}.1.weight'].shape == (1024, 2048)
        changed = int(torch.count_nonzero(norm != 1))
        nonzero_bias = int(torch.count_nonzero(bias))
        assert changed > 0 and nonzero_bias > 0
        projectors.append({'layer': layer, 'norm_weights_changed_from_one': changed,
                           'norm_bias_nonzero': nonzero_bias})

    original = ROOT / 'models/Qwen3-VL-2B-Instruct/model.safetensors'
    original_sha = digest(original)
    assert original_sha == '7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0'
    comparison = {'tensors': 0, 'changed_tensors': 0, 'elements': 0, 'changed_elements': 0}
    changed_language_layers = set()
    with safe_open(original, framework='pt', device='cpu') as handle:
        for source_key in handle.keys():
            target_key = 'qwen_vl_interface.model.' + source_key
            assert target_key in state, target_key
            expected = handle.get_tensor(source_key)
            actual = state[target_key]
            assert actual.shape == expected.shape, source_key
            changed = 0
            for current, initial in zip(actual.reshape(-1).split(4_000_000),
                                        expected.reshape(-1).split(4_000_000)):
                changed += int(torch.count_nonzero(current != initial.to(current.dtype)))
            comparison['tensors'] += 1
            comparison['changed_tensors'] += int(changed > 0)
            comparison['elements'] += actual.numel()
            comparison['changed_elements'] += changed
            if changed and source_key.startswith('model.language_model.layers.'):
                changed_language_layers.add(int(source_key.split('.')[3]))
    assert changed_language_layers == set(range(28))
    comparison['changed_fraction'] = comparison['changed_elements'] / comparison['elements']
    assert comparison['changed_fraction'] > 0
    assert manifest['transfer']['action_channel_map']['source_indices'] == [0, 1, 2, 5]
    assert manifest['transfer']['initialized_tensor_count'] == 0
    result = {
        'completed_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'passed',
        'checkpoint': str(checkpoint), 'sha256': actual_sha,
        'tensor_count': len(state), 'finite_tensor_elements': finite_numel,
        'framework': 'QwenPI_v3', 'backbone': 'Qwen3-VL-2B-Instruct',
        'language_layers': 28, 'projector_layers': 28, 'action_transformer_layers': 28,
        'state_dim': 4, 'action_dim': 4, 'action_horizon': 20,
        'qwen_original_sha256': original_sha,
        'qwen_training_evidence': comparison,
        'projector_training_evidence': projectors,
        'source_pretraining_record': manifest['source'],
        'channel_map': manifest['transfer']['action_channel_map'],
        'scope': {
            'verified': 'Local integrity, architecture, finite weights, trained projector parameters, and changes to every Qwen language layer versus the official initialization.',
            'recorded_not_independently_log_verified': 'The conversion manifest records 5k projector alignment plus 50k Bridge/RT-1 joint steps. Original OXE logs and the 7D source checkpoint are not present locally; Blob credentials have expired.',
            'performance': 'These checks do not establish downstream policy accuracy.',
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'qwen_comparison': comparison,
                      'trained_projectors': len(projectors), 'report': str(args.output)}), flush=True)


if __name__ == '__main__':
    main()
