#!/usr/bin/env python3
"""Run HUGE-Bench with two renderer/policy pairs on four GPUs."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

from experiment_paths import ROOT, WORKSPACE, STARVLA, HUGEBENCH, GAUSSIAN, HUGE_DATA, MODEL, VENV, BASE, config_dict
PYTHON = VENV / 'bin/python'
TOOLS = ROOT / 'scripts/tools'


def checkpoint_path():
    explicit = os.environ.get('CHECKPOINT_PATH')
    if explicit:
        path = Path(explicit).expanduser().resolve()
    else:
        run = ROOT / 'artifacts/checkpoints' / os.environ['RUN_ID']
        path = run / 'final_model/pytorch_model.pt'
        if not path.is_file():
            files = list((run / 'checkpoints').glob('steps_*_pytorch_model.pt'))
            if not files:
                raise FileNotFoundError('No trained checkpoint. Set CHECKPOINT_PATH or use scripts/eval_smoke.sh to check the base model.')
            path = max(files, key=lambda p: int(p.name.split('_')[1]))
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--splits', default=os.environ.get('SPLITS', 'test_seen,test_unseen'))
    parser.add_argument('--out-dir', type=Path)
    parser.add_argument('--rollout-only', action='store_true')
    args = parser.parse_args()
    checkpoint = checkpoint_path()
    gpu_ids = os.environ.get('EVAL_GPUS', '0,1,2,3').split(',')
    shards = int(os.environ.get('NUM_SHARDS', '2'))
    if shards < 1 or len(gpu_ids) < 2 * shards or len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError('Need two distinct GPU IDs per shard; default NUM_SHARDS=2, EVAL_GPUS=0,1,2,3.')
    renderer_gpus, policy_gpus = gpu_ids[:shards], gpu_ids[shards:2 * shards]
    inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'], text=True)
    free = {row.split(',')[0].strip(): int(row.split(',')[1]) for row in inventory.strip().splitlines()}
    for gpu in renderer_gpus + policy_gpus:
        if free.get(gpu, 0) < int(os.environ.get('MIN_FREE_MB', '24000')):
            raise RuntimeError(f'GPU {gpu} does not have enough free memory.')
    seed = int(os.environ.get('SEED', '42'))
    label = checkpoint.stem
    tag = os.environ.get('EVAL_TAG', 'full')
    out_dir = (args.out_dir or ROOT / 'results' / os.environ['RUN_ID'] / label / f'{tag}_seed_{seed}' / 'rollouts').resolve()
    if not out_dir.is_relative_to(WORKSPACE):
        raise ValueError('Evaluation outputs must be inside WORKSPACE_ROOT.')
    run_dir = out_dir.parent
    logs = run_dir / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    settings = {'checkpoint': str(checkpoint), 'checkpoint_bytes': checkpoint.stat().st_size,
                'checkpoint_mtime_ns': checkpoint.stat().st_mtime_ns, 'splits': args.splits,
                'num_trajs_per_split': int(os.environ.get('NUM_TRAJS', '0')),
                'max_steps': int(os.environ.get('MAX_STEPS', '0')),
                'exec_steps': int(os.environ.get('EXEC_STEPS', '10')),
                'seed': seed, 'num_shards': shards, 'renderer_gpus': renderer_gpus,
                'policy_gpus': policy_gpus, 'smooth_overlap': os.environ.get('SMOOTH_OVERLAP', '1') == '1',
                'obstacle_angle_mode': os.environ.get('OBSTACLE_ANGLE_MODE', 'yaw_legacy'),
                'save_video': os.environ.get('SAVE_VIDEO', '0') == '1'}
    if settings['num_trajs_per_split'] < 0 or settings['max_steps'] < 0 or settings['exec_steps'] < 1:
        raise ValueError('NUM_TRAJS/MAX_STEPS must be nonnegative and EXEC_STEPS must be positive.')
    expected_count = 0
    for split in args.splits.split(','):
        if split not in ('test_seen', 'test_unseen'):
            raise ValueError(f'Unknown evaluation split: {split}')
        split_root = HUGE_DATA / 'data_traj' / split
        total = len(list((split_root / 'data').rglob('episode_*.parquet')))
        if not total:
            raise FileNotFoundError(f'No trajectories in {split_root}')
        expected_count += min(total, settings['num_trajs_per_split']) if settings['num_trajs_per_split'] else total
    manifest = run_dir / 'run_config.json'
    if manifest.exists() and json.loads(manifest.read_text()) != settings:
        raise RuntimeError('Existing output has different settings. Choose a new EVAL_TAG or --out-dir.')
    manifest.write_text(json.dumps(settings, indent=2) + '\n')
    render_port = int(os.environ.get('RENDER_PORT_START', '5550'))
    policy_port = int(os.environ.get('POLICY_PORT_START', '6678'))
    ports = [render_port + i for i in range(shards)] + [policy_port + i for i in range(shards)]
    if len(set(ports)) != len(ports):
        raise ValueError('Renderer and policy ports overlap.')
    for port in ports:
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(('127.0.0.1', port))

    processes, handles = [], []
    def start(name, command, gpu=None, cwd=ROOT):
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = '' if gpu is None else gpu
        handle = (logs / f'{name}.log').open('ab', buffering=0)
        handles.append(handle)
        proc = subprocess.Popen([str(x) for x in command], cwd=cwd, env=env,
                                stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        processes.append((name, proc))
        return proc

    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    workers = []
    try:
        for i in range(shards):
            start(f'renderer_{i}', [PYTHON, GAUSSIAN / '3dgs_renderer.py',
                  '--host', '127.0.0.1', '--port', render_port + i, '--ply_template',
                  HUGE_DATA / 'data_3d/{env_id}/3dgs_ply/point_cloud_utm50.ply'],
                  renderer_gpus[i], GAUSSIAN)
            start(f'policy_{i}', [PYTHON, TOOLS / 'serve_policy.py', '--checkpoint', checkpoint,
                  '--base-vlm', MODEL, '--port', policy_port + i,
                  '--seed', seed + i], policy_gpus[i], STARVLA)
        for i in range(shards):
            command = [PYTHON, TOOLS / 'action_infer_starvla.py', '--checkpoint', checkpoint,
                       '--splits', args.splits, '--out-dir', out_dir, '--render-port', render_port + i,
                       '--policy-port', policy_port + i, '--num-trajs', settings['num_trajs_per_split'],
                       '--max-steps', settings['max_steps'], '--exec-steps', settings['exec_steps'],
                       '--num-shards', shards, '--shard-index', i, '--seed', seed + i,
                       '--obstacle-angle-mode', settings['obstacle_angle_mode'], '--skip-existing']
            if not settings['smooth_overlap']:
                command += ['--no-smooth-overlap']
            if settings['save_video']:
                command += ['--save-video']
            workers.append(start(f'rollout_{i}', command))
        print(f'Running {shards} renderer/policy pairs. Logs: {logs}', flush=True)
        while any(proc.poll() is None for proc in workers):
            for name, proc in processes:
                if proc.poll() not in (None, 0):
                    raise RuntimeError(f'{name} failed with exit {proc.returncode}; see {logs / (name + ".log")}')
            time.sleep(1)
        if any(proc.returncode for proc in workers):
            raise RuntimeError(f'Rollout failed; see {logs}')
    finally:
        for _, proc in reversed(processes):
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
        for _, proc in processes:
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        for handle in handles:
            handle.close()
    count = len(list(out_dir.rglob('traj_gt_pred_xyzk.npz')))
    if count != expected_count:
        raise RuntimeError(f'Expected {expected_count} trajectories, found {count}.')
    print(f'Rollout complete: {count} episodes at {out_dir}', flush=True)
    if not args.rollout_only:
        command = [str(ROOT / 'scripts/eval_metrics.sh'), str(out_dir)]
        with (logs / 'metrics.log').open('w') as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
        metrics = json.loads((run_dir / 'metrics.json').read_text())
        # Official CSPL is undefined for a zero-length GT path (e.g. an initial
        # rotation-only smoke segment). Preserve that definition.
        if any(row.get('cr') is None or (row.get('cspl') is None and row['gt_length'] > 1e-12) for row in metrics['episodes']):
            raise RuntimeError(f'Collision metrics are missing; see {logs / "metrics.log"}')
        print('Official metrics:', json.dumps(metrics['overall']), flush=True)
    (run_dir / 'COMPLETE').write_text(f'{count} episodes\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
