"""CPU IMLE contracts and frozen s3fmp numerical reference (no robot)."""

from pathlib import Path
import unittest
import numpy as np
import torch
from torch import nn
from stable_worldmodel.imle import ActionGenerator, IMLEPolicy, rs_imle_loss


def small_policy():
    encoder = lambda: nn.Sequential(
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 8)
    )
    return IMLEPolicy(
        ActionGenerator(
            8, 16, 8, 8, channels=(8, 16, 32), gripper_channels=(8, 16, 32)
        ),
        encoder(),
        encoder(),
        16,
        8,
        8,
        8,
        n_samples_per_condition=4,
        gripper_loss_weight=0.00625,
    )


class IMLETests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_forward_backward_optimizer(self):
        torch.manual_seed(5)
        p = small_policy()
        batch = dict(
            pixels=torch.randn(2, 8, 3, 8, 8),
            wrist_pixels=torch.randn(2, 8, 3, 8, 8),
            action=torch.randn(2, 16, 8),
        )
        loss = p.compute_loss(batch)
        self.assertTrue(torch.isfinite(loss))
        before = next(p.model.parameters()).detach().clone()
        loss.backward()
        for module in (p.model, p.obs_encoder, p.wrist_obs_encoder):
            self.assertTrue(
                all(
                    x.grad is not None and torch.isfinite(x.grad).all()
                    for x in module.parameters()
                )
            )
        torch.optim.AdamW(
            [
                x
                for m in (p.model, p.obs_encoder, p.wrist_obs_encoder)
                for x in m.parameters()
            ],
            lr=1e-4,
        ).step()
        self.assertFalse(torch.equal(before, next(p.model.parameters())))

    def test_samples_chunks_and_seed(self):
        p = small_policy()
        info = dict(
            pixels=torch.randn(2, 8, 3, 8, 8),
            wrist_pixels=torch.randn(2, 8, 3, 8, 8),
        )
        torch.manual_seed(8)
        first = p.sample_action_sequences(info, 3)
        torch.manual_seed(8)
        torch.testing.assert_close(
            first, p.sample_action_sequences(info, 3), rtol=0, atol=0
        )
        self.assertEqual(first.shape, (2, 3, 16, 8))
        self.assertFalse(torch.equal(first[:, 0], first[:, 1]))
        torch.manual_seed(9)
        pred = p.predict_action(info)
        torch.testing.assert_close(
            pred['action'], pred['action_pred'][:, 7:15]
        )
        with self.assertRaises(ValueError):
            p.sample_action_sequences(info, 0)
        with self.assertRaises(ValueError):
            p.sample_action_sequences(info, 1, denormalize=True)

    def test_rejection_and_reference_edge_semantics(self):
        real = torch.zeros(2, 1, 1)
        fake = torch.tensor(
            [[[[0.01]], [[0.2]], [[0.3]]], [[[0.01]], [[0.02]], [[0.03]]]],
            requires_grad=True,
        )
        loss, logs = rs_imle_loss(real, fake, 0.1, False)
        torch.testing.assert_close(loss, torch.tensor(0.2))
        self.assertEqual(logs['metrics/valid_ratio'].item(), 0.5)
        loss.backward()
        torch.testing.assert_close(
            fake.grad.reshape(2, 3),
            torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]),
        )
        # s3fmp's strict min<global_max mask also drops a single maximum-only candidate.
        value, _ = rs_imle_loss(
            torch.zeros(1, 1, 1), torch.ones(1, 1, 1, 1), 0.0, False
        )
        self.assertEqual(value.item(), 0.0)
        value, _ = rs_imle_loss(
            real, torch.zeros(2, 3, 1, 1, requires_grad=True), 1.0
        )
        self.assertEqual(value.item(), 0.0)
        value.backward()

    def test_frozen_s3fmp_generator_loss_and_gradients(self):
        fixture = np.load(Path(__file__).parent / 's3fmp_reference.npz')
        torch.manual_seed(137)
        model = ActionGenerator(
            8, 16, 8, 8, channels=(8, 16, 32), gripper_channels=(8, 16, 32)
        )
        cond = torch.tensor(fixture['cond'])
        noise = torch.tensor(fixture['noise'])
        target = torch.tensor(fixture['target'])
        predicted = model(cond, noise)
        np.testing.assert_array_equal(
            predicted.detach().numpy(), fixture['prediction']
        )
        joints = model.generate(noise, cond).reshape(2, 2, 16, 7)
        loss, logs = rs_imle_loss(target[..., :7], joints, 0.01, True)
        grip = model.gripper(target[..., :7], cond[::2])
        grip_l1 = (
            torch.nn.functional.l1_loss(
                grip, target[..., 7:], reduction='none'
            )
            .sum((-2, -1))
            .mean()
        )
        total = loss + 0.00625 * grip_l1
        np.testing.assert_array_equal(total.detach().numpy(), fixture['loss'])
        total.backward()
        for name, parameter in model.named_parameters():
            np.testing.assert_array_equal(
                parameter.grad.numpy(), fixture['grad/' + name]
            )


if __name__ == '__main__':
    unittest.main()
