"""HUGE-Bench data mapping for StarVLA QwenPI_v3.

The source dataset stores two embedded RGB images, a 4D absolute UAV state,
and a pre-computed 4D delta action.  ``action_mode=abs`` in the YAML means
"use the stored action as-is"; it does not mean the physical action is an
absolute pose.
"""

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import (
    StateActionToTensor,
    StateActionTransform,
)


class HugeBenchUAVDataConfig:
    """Two RGB views, 4D state, and 20-step 4D delta-action chunks."""

    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    video_keys = ["video.first_image", "video.current_image"]
    state_keys = ["state.x", "state.y", "state.z", "state.yaw"]
    action_keys = ["action.dx", "action.dy", "action.dz", "action.dyaw"]
    language_keys = ["annotation.human.action.task_description"]

    observation_indices = [0]
    state_indices = [0]
    action_indices = list(range(20))

    def modality_config(self):
        return {
            "video": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.video_keys,
            ),
            "state": ModalityConfig(
                delta_indices=self.state_indices,
                modality_keys=self.state_keys,
            ),
            "action": ModalityConfig(
                delta_indices=self.action_indices,
                modality_keys=self.action_keys,
            ),
            "language": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.language_keys,
            ),
        }

    def transform(self):
        keys = self.state_keys + self.action_keys
        return ComposedModalityTransform(
            transforms=[
                StateActionToTensor(apply_to=keys),
                StateActionTransform(
                    apply_to=self.state_keys,
                    normalization_modes={key: "q99" for key in self.state_keys},
                ),
                StateActionTransform(
                    apply_to=self.action_keys,
                    normalization_modes={key: "q99" for key in self.action_keys},
                ),
            ]
        )


ROBOT_TYPE_CONFIG_MAP = {"hugebench_uav": HugeBenchUAVDataConfig()}
ROBOT_TYPE_TO_EMBODIMENT_TAG = {}
DATASET_NAMED_MIXTURES = {
    "hugebench_train": [("hugebench_train", 1.0, "hugebench_uav")]
}
