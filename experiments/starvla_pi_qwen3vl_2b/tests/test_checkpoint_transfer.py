"""CPU checks for the two weight migrations; no models/downloads/GPU required."""
import unittest
import torch
from checkpoint_transfer import transfer_oxe, transfer_huge


class CheckpointTransferTest(unittest.TestCase):
    def test_oxe_preserves_backbone_projectors_and_maps_action_blocks(self):
        prefix = "action_model.model.transformer_blocks."
        source = {f"{prefix}{i}.weight": torch.full((2, 2), float(i)) for i in range(36)}
        target = {f"{prefix}{i}.weight": torch.zeros(2, 2) for i in range(28)}
        target["qwen_vl_interface.weight"] = torch.full((2,), 123.)
        target["project_layers.0.weight"] = torch.full((2,), 456.)
        result, copied, fresh, mapping, _, _ = transfer_oxe(source, target)
        # Fixed expected selection includes every skipped source block boundary.
        self.assertEqual(mapping, [0, 1, 3, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17,
                                   18, 19, 21, 22, 23, 25, 26, 27, 29, 30, 31, 32, 34, 35])
        self.assertEqual(result[f"{prefix}2.weight"][0, 0].item(), 3.)
        self.assertEqual(result[f"{prefix}27.weight"][0, 0].item(), 35.)
        self.assertTrue(torch.equal(result["qwen_vl_interface.weight"], torch.full((2,), 123.)))
        self.assertTrue(torch.equal(result["project_layers.0.weight"], torch.full((2,), 456.)))
        self.assertEqual(len(copied), 28)
        self.assertEqual(len(fresh), 2)
        del source[f"{prefix}3.weight"]
        with self.assertRaises(ValueError):
            transfer_oxe(source, target)

    def test_huge_channels_and_unchanged_backbone(self):
        source = {
            "action_model.action_encoder.layer1.weight": torch.arange(21.).reshape(3, 7),
            "action_model.state_encoder.layer1.weight": torch.arange(21.).reshape(3, 7) + 100,
            "action_model.action_decoder.layer2.weight": torch.arange(21.).reshape(7, 3),
            "action_model.action_decoder.layer2.bias": torch.arange(7.),
            "qwen_vl_interface.weight": torch.tensor([77., 88.]),
        }
        shapes = [(3, 4), (3, 4), (4, 3), (4,), (2,)]
        target = {key: torch.empty(shape) for key, shape in zip(source, shapes)}
        result, copied = transfer_huge(source, target)
        self.assertEqual(result["action_model.action_encoder.layer1.weight"][0].tolist(), [0., 1., 2., 5.])
        self.assertEqual(result["action_model.state_encoder.layer1.weight"][0].tolist(), [100., 101., 102., 105.])
        self.assertEqual(result["action_model.action_decoder.layer2.weight"][-1].tolist(), [15., 16., 17.])
        self.assertEqual(result["action_model.action_decoder.layer2.bias"].tolist(), [0., 1., 2., 5.])
        self.assertTrue(torch.equal(result["qwen_vl_interface.weight"], source["qwen_vl_interface.weight"]))
        self.assertEqual(len(copied), len(source))
        target["unexpected.weight"] = torch.empty(1)
        with self.assertRaises(ValueError):
            transfer_huge(source, target)


if __name__ == "__main__":
    unittest.main()
