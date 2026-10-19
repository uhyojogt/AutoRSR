import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

from autorsr.inference import collect_images, main, predict_image, preprocess_image, segmentation_state_dict
from autorsr.model import DomainSpecificBatchNorm2d, resnet34_dsbn


class QuadrantModel(torch.nn.Module):
    def forward(self, image, domain_ids=None):
        batch, _, height, width = image.shape
        logits = torch.zeros(batch, 4, height, width, device=image.device)
        for label in range(4):
            row, column = divmod(label, 2)
            logits[:, label, row*height//2:(row+1)*height//2,
                   column*width//2:(column+1)*width//2] = 1
        return logits


class InferenceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_preprocessing_matches_original(self):
        image = np.arange(35*47*3, dtype=np.uint8).reshape(35, 47, 3)
        expected = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (96, 64), interpolation=cv2.INTER_NEAREST)
        expected = expected.astype(np.float32).transpose(2, 0, 1) / 255.0
        tensor = preprocess_image(image, (64, 96))
        self.assertEqual(tuple(tensor.shape), (1, 3, 64, 96))
        np.testing.assert_array_equal(tensor[0].numpy(), expected)

    def test_invalid_network_size(self):
        with self.assertRaises(ValueError):
            preprocess_image(np.zeros((35, 47, 3), np.uint8), (65, 96))

    def test_native_resolution_argmax_export(self):
        image = np.zeros((35, 47, 3), np.uint8)
        result = predict_image(QuadrantModel(), image, "japan", torch.device("cpu"), (64, 96))
        expected = QuadrantModel()(torch.zeros(1, 3, 64, 96)).argmax(1)[0].numpy().astype(np.uint8)
        expected = cv2.resize(expected, (47, 35), interpolation=cv2.INTER_NEAREST)
        np.testing.assert_array_equal(result, expected)
        self.assertEqual(set(np.unique(result)), {0, 1, 2, 3})

    def test_unknown_domain_rejected(self):
        with self.assertRaises(ValueError):
            predict_image(QuadrantModel(), np.zeros((32, 32, 3), np.uint8), "unknown", torch.device("cpu"))

    def test_duplicate_stems_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sample.jpg").touch()
            (root / "sample.png").touch()
            with self.assertRaises(ValueError):
                collect_images(root)

    def test_checkpoint_wrapper_and_prefix(self):
        state = {"module.backbone.conv1.weight": torch.ones(1),
                 "module.boundary_head.bias": torch.zeros(1)}
        result = segmentation_state_dict({"state_dict": state})
        self.assertEqual(list(result), ["conv1.weight"])

    def test_unexpected_auxiliary_parameters_rejected(self):
        with self.assertRaises(ValueError):
            segmentation_state_dict({"backbone.conv1.weight": torch.ones(1), "other.weight": torch.ones(1)})

    def test_empty_checkpoint_rejected(self):
        with self.assertRaises(ValueError):
            segmentation_state_dict({})

    def test_colliding_checkpoint_prefix_rejected(self):
        with self.assertRaises(ValueError):
            segmentation_state_dict({"module.conv1.weight": torch.ones(1), "conv1.weight": torch.ones(1)})

    def test_domain_specific_routing(self):
        layer = DomainSpecificBatchNorm2d(1).eval()
        for index, batch_norm in enumerate(layer.bns):
            batch_norm.running_mean.fill_(index)
        result = layer(torch.full((3, 1, 2, 2), 2.), torch.tensor([0, 1, 2]))
        expected = torch.tensor([2., 1., 0.]) / np.sqrt(1 + 1e-5)
        np.testing.assert_allclose(result[:, 0, 0, 0].detach().numpy(), expected.numpy(), rtol=1e-5)

    def test_segmentation_model_output_shape(self):
        model = resnet34_dsbn(4).eval()
        with torch.no_grad():
            for domain_id in range(3):
                result = model(torch.zeros(1, 3, 64, 96), domain_ids=domain_id)
                self.assertEqual(tuple(result.shape), (1, 4, 64, 96))

    def test_cli_exports_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            images.mkdir()
            self.assertTrue(cv2.imwrite(str(images / "sample.png"), np.zeros((35, 47, 3), np.uint8)))
            checkpoint = root / "test.pth"
            checkpoint.write_bytes(b"synthetic checkpoint placeholder")
            output = root / "outputs"
            arguments = ["--images-dir", str(images), "--checkpoint", str(checkpoint),
                         "--domain", "ningbo", "--device", "cpu", "--size", "64", "96",
                         "--output", str(output), "--save-color", "--save-overlay"]
            with patch("autorsr.inference.load_model", return_value=QuadrantModel()):
                self.assertEqual(main(arguments), 0)
                with self.assertRaises(ValueError):
                    main(arguments)
            for folder in ["masks", "color_masks", "overlays"]:
                saved = cv2.imread(str(output / folder / "sample.png"), cv2.IMREAD_UNCHANGED)
                self.assertEqual(saved.shape[:2], (35, 47))
            metadata = json.loads((output / "run.json").read_text())
            self.assertEqual(metadata["images"], 1)
            self.assertEqual(metadata["domain"], "ningbo")


if __name__ == "__main__":
    unittest.main()
