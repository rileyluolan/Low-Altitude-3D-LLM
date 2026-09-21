"""LeRobot parquet adapter using HUGE-Bench's official rollout helpers."""
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from experiment_paths import HUGEBENCH
source = HUGEBENCH / "openpi/scripts/action_infer.py"
spec = importlib.util.spec_from_file_location('hugebench_official_rollout', source)
official = importlib.util.module_from_spec(spec)
spec.loader.exec_module(official)
wrap_rad = official.wrap_rad
step_state = official.step_state
convert_state_for_render = official.convert_state_for_render
ActionPlanAverager = official.ActionPlanAverager


def load_tasks(split_root):
    with (split_root / 'meta/tasks.jsonl').open() as stream:
        tasks = [json.loads(line) for line in stream if line.strip()]
    return {int(row['task_index']): row for row in tasks}


def load_episode(path, tasks):
    table = pq.read_table(path, columns=['state', 'env_id', 'task_index', 'episode_index', 'frame_index'])
    rows = table.to_pydict()
    for key in ('env_id', 'task_index', 'episode_index'):
        if len(set(rows[key])) != 1:
            raise ValueError(f'{path}: expected one {key} per episode')
    task = tasks[int(rows['task_index'][0])]
    task_id = str(task['task_id'])
    if task_id not in {'0', 'obstacle', 'hl', 'orbit', 'road', 'building', 'orbit_multi', 'farm'}:
        raise ValueError(f'Unknown HUGE-Bench task ID: {task_id}')
    states = np.asarray(rows['state'], dtype=np.float32)[np.argsort(rows['frame_index'])]
    if states.ndim != 2 or states.shape[1] != 4 or not np.isfinite(states).all():
        raise ValueError(f'Invalid state trajectory in {path}')
    states[:, 3] = [wrap_rad(official.maybe_deg_to_rad(x)) for x in states[:, 3]]
    return states, str(rows['env_id'][0]), str(task['task']), task_id, int(rows['episode_index'][0])


class RenderClient:
    """Adapt the original experiment API to the official JSON/TCP renderer."""
    def __init__(self, host, port):
        deadline = time.monotonic() + 300
        while True:
            try:
                self.client = official.RenderClient(host, port)
                self.client.sock.settimeout(300)
                break
            except ConnectionRefusedError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f'Renderer not ready on {host}:{port}') from None
                time.sleep(0.5)

    def render(self, state, output_path, env_id, task_id, step):
        path = self.client.render(t=step, state_xyzk=state, out_path=str(output_path), env_id=env_id, task_id=task_id)
        with Image.open(path) as image:
            return np.asarray(image.convert('RGB'), dtype=np.uint8).copy()

    def close(self):
        self.client.close()
