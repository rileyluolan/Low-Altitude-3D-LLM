#!/usr/bin/env python3
"""Serve a StarVLA checkpoint with the HUGE-Bench state/action contract."""

from __future__ import annotations

import argparse
import logging
import random

import numpy as np
import torch

from deployment.model_server.policy_wrapper import PolicyServerWrapper
from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer


class HugeBenchPolicyWrapper(PolicyServerWrapper):
    """Normalize raw UAV state and unnormalize actions with checkpoint stats."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logged_state = False

    @staticmethod
    def _normalize_state(state: np.ndarray, processor) -> np.ndarray:
        state_array = np.array(state, dtype=np.float32, copy=True)
        original_ndim = state_array.ndim
        if original_ndim == 1:
            state_array = state_array[None, :]
        if state_array.ndim != 2:
            raise ValueError(f"Expected state shape (T, D) or (D,), got {state_array.shape}")

        data = {}
        cursor = 0
        for key in processor.state_keys:
            width = processor.state_key_dims.get(key, 1)
            # The first training transform is StateActionToTensor, so replay
            # the pipeline from its NumPy input boundary.
            data[key] = np.ascontiguousarray(state_array[..., cursor : cursor + width])
            cursor += width
        if cursor != state_array.shape[-1]:
            raise ValueError(
                f"State dim mismatch: checkpoint expects {cursor}, request has {state_array.shape[-1]}"
            )
        transformed = processor.transform.apply(data)
        parts = []
        for key in processor.state_keys:
            value = transformed[key]
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            parts.append(np.asarray(value, dtype=np.float32))
        normalized = np.concatenate(parts, axis=-1)
        return normalized[0] if original_ndim == 1 else normalized

    @property
    def metadata(self):
        metadata = dict(super().metadata)
        key = self._default_unnorm_key
        if key is not None:
            processor = self._get_processor(key)
            metadata["framework"] = self._model_cfg["framework"]["name"]
            metadata["runtime_contract"] = {
                "version": 1,
                "image_color_order": "rgb",
                "image_order": ["episode_first_image", "current_render"],
                "state_input": "raw_env_xyzyaw",
                "state_normalization": "checkpoint_training_transform",
                "action_output": "unnormalized_delta_xyzyaw",
                "state_dim": sum(processor.state_key_dims.values()),
                "action_dim": sum(processor.action_key_dims.values()),
            }
        return metadata

    def predict_action(self, examples, unnorm_key=None, **kwargs):
        key = unnorm_key or self._default_unnorm_key
        processor = self._get_processor(key)
        normalized_examples = []
        for example in examples:
            normalized = dict(example)
            if normalized.get("state") is not None:
                raw = np.asarray(normalized["state"], dtype=np.float32)
                normalized["state"] = self._normalize_state(raw, processor)
                if not self._logged_state:
                    logging.info(
                        "HUGE state normalization: raw=%s normalized=%s",
                        raw.reshape(-1, raw.shape[-1])[0],
                        np.asarray(normalized["state"]).reshape(-1, raw.shape[-1])[0],
                    )
                    self._logged_state = True
            normalized_examples.append(normalized)
        return super().predict_action(normalized_examples, unnorm_key=key, **kwargs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--port", type=int, default=6678)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-vlm", required=True)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    wrapper = HugeBenchPolicyWrapper(
        ckpt_path=args.checkpoint,
        device="cuda",
        use_bf16=True,
        config_overrides=[f"framework.qwenvl.base_vlm={args.base_vlm}"],
    )
    server = WebsocketPolicyServer(
        policy=wrapper,
        host="127.0.0.1",
        port=args.port,
        idle_timeout=-1,
        metadata={**wrapper.metadata, "seed": args.seed},
    )
    logging.info("HUGE-Bench StarVLA server ready: %s", wrapper.metadata)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    raise SystemExit(main())
