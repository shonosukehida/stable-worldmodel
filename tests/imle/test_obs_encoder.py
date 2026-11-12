"""s3fmp ConvNet architecture/output/gradient equivalence, CPU only."""

from pathlib import Path
import unittest
import numpy as np
import torch
from stable_worldmodel.imle.obs_encoder import ConvNetObsEncoder


class ConvNetReferenceTests(unittest.TestCase):
    def test_actual_reference_parameters_features_and_gradients(self):
        torch.set_num_threads(1)
        torch.manual_seed(137)
        model = ConvNetObsEncoder()
        fixture = np.load(Path(__file__).with_name('convnet_reference.npz'))
        pixels = torch.tensor(fixture['pixels'], requires_grad=True)
        features = model(pixels)
        np.testing.assert_array_equal(
            features.detach().numpy(), fixture['features']
        )
        features.square().mean().backward()
        np.testing.assert_array_equal(
            pixels.grad.numpy(), fixture['input_grad']
        )
        self.assertEqual(sum(p.numel() for p in model.parameters()), 31904)
        for index, parameter in enumerate(model.parameters()):
            np.testing.assert_array_equal(
                parameter.detach().numpy(), fixture[f'parameter_{index}']
            )
            np.testing.assert_array_equal(
                parameter.grad.numpy(), fixture[f'gradient_{index}']
            )

    def test_resolution_feature_dimension_and_parameter_validation(self):
        for size in ((32, 32), (96, 128), (224, 224)):
            model = ConvNetObsEncoder(feature_dim=16)
            self.assertEqual(model(torch.zeros(2, 3, *size)).shape, (2, 16))
        model = ConvNetObsEncoder(
            feature_dim=8,
            channels=[8, 16],
            kernel_sizes=[5, 3],
            strides=[2, 2],
            paddings=[2, 1],
            pool_size=2,
        )
        self.assertEqual(model(torch.zeros(2, 3, 64, 48)).shape, (2, 8))
        for kwargs in (
            {'channels': []},
            {'channels': [8]},
            {'strides': [0, 2, 2]},
            {'pool_size': 0},
            {'activation': 'unknown'},
        ):
            with self.assertRaises(ValueError):
                ConvNetObsEncoder(**kwargs)
