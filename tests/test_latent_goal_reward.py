"""Exact legacy compatibility and modality weighting, without hardware."""
from types import SimpleNamespace
import unittest

import torch
from stable_worldmodel.reward import (latent_goal_reward, LatentGoalReward,
                                      model_latent_dimensions)


def model(overhead=192, wrist=192, prop=32):
    def mlp(size):
        return SimpleNamespace(net=torch.nn.Sequential(torch.nn.Linear(2, size)))
    return SimpleNamespace(overhead_projector=mlp(overhead), wrist_projector=mlp(wrist), prop_encoder=mlp(prop))


class RewardTests(unittest.TestCase):
    def test_default_and_one_exact_legacy_and_ranking(self):
        generator = torch.Generator().manual_seed(51)
        for dtype in (torch.float32, torch.float64, torch.bfloat16):
            for dimension in (8, 416, 433):
                output = dict(predicted_emb=torch.randn(2, 7, 4, dimension, generator=generator).to(dtype),
                              goal_emb=torch.randn(2, 3, dimension, generator=generator).to(dtype))
                legacy = -((output['predicted_emb'][:, :, -1] - output['goal_emb'][:, -1, None]) ** 2).sum(-1)
                for reward in (latent_goal_reward(output, {}), latent_goal_reward(output, {}, 1.0),
                               LatentGoalReward(object())(output, {})):
                    self.assertTrue(torch.equal(legacy, reward))
                    self.assertTrue(torch.equal(legacy.argmax(1), reward.argmax(1)))

    def test_zero_ignores_proprio_but_not_images(self):
        reward = LatentGoalReward(model(), 0.0)
        output = dict(predicted_emb=torch.zeros(1, 2, 1, 416), goal_emb=torch.zeros(1, 1, 416))
        original = reward(output, {})
        output['predicted_emb'][..., 384:] = 50
        output['goal_emb'][..., 384:] = -20
        self.assertTrue(torch.equal(original, reward(output, {})))
        output['predicted_emb'][..., :384] = 1
        self.assertTrue(torch.equal(reward(output, {}), torch.full((1, 2), -384.)))

    def test_half_and_ranking_switch(self):
        output = dict(predicted_emb=torch.zeros(1, 2, 1, 416), goal_emb=torch.zeros(1, 1, 416))
        output['predicted_emb'][0, 0, 0, 384] = 3  # A image=0, proprio=9
        output['predicted_emb'][0, 1, 0, 0] = 2    # B image=4, proprio=0
        self.assertEqual(LatentGoalReward(model(), 1)(output, {}).argmax().item(), 1)
        self.assertEqual(LatentGoalReward(model(), 0)(output, {}).argmax().item(), 0)
        self.assertTrue(torch.equal(LatentGoalReward(model(), .5)(output, {}), torch.tensor([[-4.5, -4.]])))

    def test_dynamic_boundary_and_shape_validation(self):
        wm = model(3, 4, 2)
        self.assertEqual(model_latent_dimensions(wm), (7, 9))
        reward = LatentGoalReward(wm, .3)
        output = dict(predicted_emb=torch.ones(1, 2, 1, 9), goal_emb=torch.zeros(1, 1, 9))
        torch.testing.assert_close(reward(output, {}), torch.full((1, 2), -7.6))
        output['goal_emb'] = torch.zeros(1, 1, 8)
        with self.assertRaisesRegex(ValueError, 'dimension'):
            reward(output, {})
        with self.assertRaisesRegex(ValueError, 'boundary'):
            LatentGoalReward(object(), 0)
        output['goal_emb'] = torch.zeros(1, 1, 9)
        with self.assertRaisesRegex(ValueError, 'image_dim'):
            latent_goal_reward(output, {}, .5)

    def test_invalid_weight(self):
        for value in (float('nan'), float('inf'), -float('inf'), -.1, True, '0.5', None):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'lambda_prop'):
                latent_goal_reward({}, {}, value)
        # Values above the recommended range are permitted finite penalties.
        self.assertEqual(LatentGoalReward(model(), 2).lambda_prop, 2)


if __name__ == '__main__':
    unittest.main()
