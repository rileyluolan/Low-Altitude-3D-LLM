#!/usr/bin/env python3
"""Closed-loop HUGE-Bench rollout through a StarVLA websocket server."""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from tqdm.auto import tqdm


from experiment_paths import ROOT, WORKSPACE, STARVLA, HUGEBENCH, GAUSSIAN, HUGE_DATA, MODEL, VENV, BASE, config_dict
sys.path.insert(0, str(STARVLA))
import hugebench_io as huge  # noqa: E402
from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy  # noqa: E402


def decode_first_image(episode_path: Path, split_root: Path) -> np.ndarray:
    record = pq.read_table(episode_path, columns=["first_image"])["first_image"][0].as_py()
    if isinstance(record, dict) and record.get("bytes") is not None:
        source = io.BytesIO(record["bytes"])
    elif isinstance(record, dict) and record.get("path"):
        source = split_root / record["path"]
    elif isinstance(record, (bytes, bytearray)):
        source = io.BytesIO(record)
    else:
        raise ValueError(f"Cannot decode first_image from {episode_path}")
    with Image.open(source) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def resize_rgb(image: np.ndarray, size: int = 224) -> np.ndarray:
    pil = Image.fromarray(np.asarray(image, dtype=np.uint8))
    return np.asarray(pil.resize((size, size), Image.Resampling.BILINEAR), dtype=np.uint8)


def plot_trajectories(gt: np.ndarray, pred: np.ndarray, output_path: Path) -> None:
    figure = plt.figure(figsize=(8, 6))
    axis = figure.add_subplot(111, projection="3d")
    axis.plot(gt[:, 0], gt[:, 1], gt[:, 2], label="GT")
    axis.plot(pred[:, 0], pred[:, 1], pred[:, 2], label="StarVLA-PI-2B")
    axis.scatter(*gt[0, :3], marker="o", s=35)
    axis.scatter(*gt[-1, :3], marker="x", s=55)
    axis.scatter(*pred[-1, :3], marker="x", s=55)
    axis.set_xlabel("X")
    axis.set_ylabel("Y")
    axis.set_zlabel("Z")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def render_trajectory(renderer, states, env_id, task_id, temp_path, angle_mode):
    frames = []
    for step, state in enumerate(states):
        render_state = huge.convert_state_for_render(state, task_id, angle_mode)
        frames.append(renderer.render(render_state, temp_path, env_id, task_id, step))
    return frames


def save_comparison_video(gt_frames, pred_frames, path: Path, fps: float = 10.0) -> None:
    with imageio.get_writer(path, fps=fps, macro_block_size=2) as writer:
        for gt, pred in zip(gt_frames, pred_frames):
            height = max(gt.shape[0], pred.shape[0])
            def fit(frame):
                width = round(frame.shape[1] * height / frame.shape[0])
                return np.asarray(Image.fromarray(frame).resize((width, height)))
            writer.append_data(np.concatenate([fit(gt), fit(pred)], axis=1))


class StarVLAPolicy:
    def __init__(self, host: str, port: int):
        self.client = WebsocketClientPolicy(host, port)
        metadata = self.client.get_server_metadata()
        contract = metadata.get("runtime_contract", {})
        expected = {
            "image_order": ["episode_first_image", "current_render"],
            "state_input": "raw_env_xyzyaw",
            "state_normalization": "checkpoint_training_transform",
            "action_output": "unnormalized_delta_xyzyaw",
            "state_dim": 4,
            "action_dim": 4,
        }
        mismatches = {key: (contract.get(key), value) for key, value in expected.items() if contract.get(key) != value}
        if mismatches:
            raise ValueError(f"Incompatible StarVLA server contract: {mismatches}; metadata={metadata}")
        self.horizon = int(metadata["action_chunk_size"])
        if self.horizon != 20:
            raise ValueError(f"Expected action horizon 20, got {self.horizon}")

    def infer(self, first_rgb: np.ndarray, current_rgb: np.ndarray, state: np.ndarray, prompt: str) -> np.ndarray:
        example = {
            "image": [resize_rgb(first_rgb), resize_rgb(current_rgb)],
            "lang": str(prompt),
            "state": np.asarray(state, dtype=np.float32).reshape(1, 4),
        }
        response = self.client.predict_action(
            {
                "examples": [example],
                "unnorm_key": "new_embodiment",
                "use_ddim": True,
                "num_ddim_steps": 4,
            }
        )
        if not response.get("ok", False):
            raise RuntimeError(f"StarVLA inference failed: {response.get('error', response)}")
        actions = np.asarray(response["data"]["actions"][0], dtype=np.float32)
        if actions.shape != (self.horizon, 4):
            raise ValueError(f"Expected action shape {(self.horizon, 4)}, got {actions.shape}")
        return actions

    def close(self) -> None:
        self.client.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", type=Path, default=HUGE_DATA / "data_traj")
    parser.add_argument("--splits", default="test_seen,test_unseen")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--task-id", default="auto")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--render-port", type=int, default=5550)
    parser.add_argument("--policy-port", type=int, default=6678)
    parser.add_argument("--num-trajs", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--exec-steps", type=int, default=10)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--smooth-overlap", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--obstacle-angle-mode", choices=["yaw_legacy", "phi"], default="yaw_legacy")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Require num_shards >= 1 and 0 <= shard_index < num_shards")
    splits = [item.strip() for item in args.splits.split(",") if item.strip()]
    policy = StarVLAPolicy(args.host, args.policy_port)
    renderer = huge.RenderClient(args.host, args.render_port)
    try:
        for split in splits:
            split_root = args.data_root / split
            tasks = huge.load_tasks(split_root)
            episode_files = sorted(
                (split_root / "data").glob("**/episode_*.parquet"),
                key=lambda path: int(path.stem.removeprefix("episode_")),
            )
            if args.num_trajs > 0:
                episode_files = episode_files[: args.num_trajs]
            episode_files = episode_files[args.shard_index :: args.num_shards]
            progress = tqdm(episode_files, desc=f"StarVLA rollout {split} shard {args.shard_index}", unit="episode")
            for episode_path in progress:
                gt_states, env_id, prompt, episode_task_id, episode_index = huge.load_episode(episode_path, tasks)
                task_id = episode_task_id if args.task_id.lower() == "auto" else args.task_id
                max_steps = len(gt_states) if args.max_steps <= 0 else min(len(gt_states), args.max_steps)
                if max_steps < 2:
                    continue
                episode_dir = args.out_dir / f"task_{task_id}" / split / env_id / f"episode_{episode_index:04d}"
                trajectory_path = episode_dir / "traj_gt_pred_xyzk.npz"
                if args.skip_existing and (episode_dir / "COMPLETE.json").is_file() and trajectory_path.is_file():
                    continue
                episode_dir.mkdir(parents=True, exist_ok=True)
                temp_path = episode_dir / "_render.png"
                first_rgb = decode_first_image(episode_path, split_root)
                state = gt_states[0].copy()
                state[3] = huge.wrap_rad(state[3])
                predicted_states = [state.copy()]
                plan = huge.ActionPlanAverager()
                step = 0
                while step < max_steps - 1:
                    render_state = huge.convert_state_for_render(state, task_id, args.obstacle_angle_mode)
                    current_rgb = renderer.render(render_state, temp_path, env_id, task_id, step)
                    action_chunk = policy.infer(first_rgb, current_rgb, state, prompt)
                    if not np.isfinite(action_chunk).all():
                        raise ValueError(f'Policy returned non-finite actions for {episode_path}')
                    if args.smooth_overlap:
                        plan.add_chunk(step, action_chunk)
                    execute = min(max(1, args.exec_steps), len(action_chunk), max_steps - 1 - step)
                    for action in action_chunk[:execute]:
                        if args.smooth_overlap:
                            action = plan.get_mean(step)
                        state = huge.step_state(state, action)
                        predicted_states.append(state.copy())
                        step += 1
                        plan.prune_before(step)

                predicted = np.asarray(predicted_states, dtype=np.float32)
                gt = gt_states[: len(predicted)].astype(np.float32)
                temporary_trajectory = episode_dir / "trajectory.partial.npz"
                np.savez_compressed(temporary_trajectory, gt_xyzk=gt, pred_xyzk=predicted)
                temporary_trajectory.replace(trajectory_path)
                plot_trajectories(gt, predicted, episode_dir / "compare_gt_vs_pred_3d.png")
                endpoint_error = float(np.linalg.norm(gt[-1, :3] - predicted[-1, :3]))
                (episode_dir / "instruction.txt").write_text(
                    "\n".join(
                        [
                            f"prompt: {prompt}", f"split: {split}", f"env_id: {env_id}",
                            f"task_id: {task_id}", f"episode_index: {episode_index}",
                            f"checkpoint: {Path(args.checkpoint).resolve()}",
                            f"exec_steps: {args.exec_steps}", f"seed: {args.seed}",
                            f"smooth_overlap: {args.smooth_overlap}",
                            f"obstacle_angle_mode: {args.obstacle_angle_mode}",
                            f"endpoint_error_m: {endpoint_error:.6f}",
                        ]
                    ) + "\n",
                    encoding="utf-8",
                )
                if args.save_video:
                    gt_frames = render_trajectory(renderer, gt, env_id, task_id, temp_path, args.obstacle_angle_mode)
                    pred_frames = render_trajectory(renderer, predicted, env_id, task_id, temp_path, args.obstacle_angle_mode)
                    save_comparison_video(gt_frames, pred_frames, episode_dir / "gt_pred_side_by_side.mp4")
                (episode_dir / "COMPLETE.json").write_text(json.dumps({"states": len(predicted), "video": args.save_video}) + "\n")
                progress.set_postfix(endpoint_m=f"{endpoint_error:.2f}")
    finally:
        policy.close()
        renderer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
