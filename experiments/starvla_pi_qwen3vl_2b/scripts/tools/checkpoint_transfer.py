"""Tensor transfer rules, shared by checkpoint builders and CPU contract checks."""
import re

BLOCK_RE = re.compile(r"^action_model\.model\.transformer_blocks\.(\d+)\.(.+)$")
CHANNELS = [0, 1, 2, 5]


def layer_map(source_depth, target_depth):
    if source_depth < 2 or target_depth < 2:
        raise ValueError("Need at least two source and target layers")
    return [round(i * (source_depth - 1) / (target_depth - 1)) for i in range(target_depth)]


def transfer_oxe(source, target):
    source_depth = 1 + max(int(m.group(1)) for k in source if (m := BLOCK_RE.match(k)))
    target_depth = 1 + max(int(m.group(1)) for k in target if (m := BLOCK_RE.match(k)))
    if (source_depth, target_depth) != (36, 28):
        raise ValueError(f"Expected 36 -> 28 DiT blocks; got {source_depth} -> {target_depth}")
    mapping = layer_map(source_depth, target_depth)
    transferred, initialized = {}, {}
    for key, tensor in list(target.items()):
        if key.startswith("qwen_vl_interface."):
            initialized[key] = "official_qwen3_vl_2b"
            continue
        if key.startswith("project_layers."):
            initialized[key] = "new_2048_to_1024_projector_seed_42"
            continue
        source_key = key
        if match := BLOCK_RE.match(key):
            source_key = f"action_model.model.transformer_blocks.{mapping[int(match.group(1))]}.{match.group(2)}"
        value = source.get(source_key)
        if value is None or value.shape != tensor.shape:
            raise ValueError(f"Cannot transfer {key} from {source_key}; refusing random action weights")
        target[key] = value.detach().to(dtype=tensor.dtype, device="cpu").contiguous()
        transferred[key] = source_key
    return target, transferred, initialized, mapping, source_depth, target_depth


def transfer_huge(source, target):
    slices = {
        "action_model.action_encoder.layer1.weight": (slice(None), CHANNELS),
        "action_model.state_encoder.layer1.weight": (slice(None), CHANNELS),
        "action_model.action_decoder.layer2.weight": (CHANNELS, slice(None)),
        "action_model.action_decoder.layer2.bias": (CHANNELS,),
    }
    transferred = {}
    if source.keys() != target.keys():
        raise ValueError("Source and target tensor keys differ")
    for key, tensor in list(target.items()):
        value = source[key]
        if key in slices:
            axis = 1 if "encoder" in key else 0
            if value.shape[axis] != 7 or tensor.shape[axis] != 4:
                raise ValueError(f"Expected 7 -> 4 conversion for {key}")
            value = value[slices[key]]
        if value.shape != tensor.shape:
            raise ValueError(f"Unexpected shape difference for {key}: {value.shape} vs {tensor.shape}")
        target[key] = value.detach().to(dtype=tensor.dtype, device="cpu").contiguous()
        transferred[key] = f"{key}[source_indices={CHANNELS}]" if key in slices else key
    return target, transferred
