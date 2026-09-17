#!/usr/bin/env python3
"""Validate the two four-GPU profiles and the official rendering installation."""
import hashlib
import json
from pathlib import Path
import subprocess

import torch
import yaml
import diff_gaussian_rasterization
import simple_knn._C
import pytorch3d.transforms
import trimesh
import rtree

from experiment_paths import ROOT, WORKSPACE, STARVLA, HUGEBENCH, GAUSSIAN, HUGE_DATA, MODEL, VENV, BASE, config_dict

for profile, global_batch in [('train', 512), ('smoke', 8)]:
    config = config_dict(ROOT / f'configs/qwen3vl_2b_hugebench_{profile}.yaml')
    accelerate = config_dict(ROOT / f'configs/deepspeed_zero2_{profile}.yaml')
    ds = json.loads(Path(accelerate['deepspeed_config']['deepspeed_config_file']).read_text())
    assert accelerate['num_processes'] == 4
    accumulation = config['trainer']['gradient_accumulation_steps']
    micro_batch = config['datasets']['vla_data']['per_device_batch_size']
    assert accumulation == ds['gradient_accumulation_steps']
    assert 4 * micro_batch * accumulation == global_batch
    assert Path(config['trainer']['pretrained_checkpoint']).is_file()
    assert Path(config['framework']['qwenvl']['base_vlm']).is_dir()
    assert Path(config['datasets']['vla_data']['data_root_dir']).is_dir()
    print(f'{profile}: 4 GPUs x {micro_batch} samples x {accumulation} accumulation = {global_batch}')

for file in ['3dgs_renderer.py', 'my_render_traj.py', 'utils/graphics_utils.py']:
    official = HUGEBENCH / 'gaussian_splatting' / file
    installed = GAUSSIAN / file
    assert hashlib.sha256(official.read_bytes()).digest() == hashlib.sha256(installed.read_bytes()).digest(), file
for scene in ['1_office', '2_city', '3_road', '4_lake', 'no1_building', 'no3_door', 'overhead_bridge']:
    for file in ['3dgs_ply/point_cloud_utm50.ply', 'terra_ply/simplified_mesh.obj']:
        assert (HUGE_DATA / 'data_3d' / scene / file).is_file()
mesh = trimesh.creation.box()
locations, _, _ = mesh.ray.intersects_location([[0, 0, 2]], [[0, 0, -1]])
assert len(locations) == 2, 'Collision backend did not detect box intersections'
subprocess.run([str(VENV / 'bin/python'), '-m', 'pip', 'check'], check=True)
assert torch.cuda.device_count() == 4
print('All four GPUs, official renderer files, CUDA extensions, seven scenes and collision backend passed.')
