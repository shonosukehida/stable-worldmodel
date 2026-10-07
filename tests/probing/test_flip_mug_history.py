"""Run with python -m unittest discover -s tests/probing."""
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from stable_worldmodel.probing.flip_mug.probe_evaluator import ProbingEvaluator
from stable_worldmodel.probing.flip_mug.plot import plot_one_step_rollout_pca


class WindowModel:
    def __init__(self, history):
        self.predictor = SimpleNamespace(pos_embedding=torch.zeros(1, history, 8))
        self.context_lengths = []

    def encode(self, info):
        return {"emb": info["proprio"], "act_emb": info["action_cartesian"]}

    def predict(self, emb, act):
        self.context_lengths.append(emb.shape[1])
        return emb + act


class SequenceScaler:
    def transform(self, value):
        assert value.ndim == 2
        return value * 2


class HistoryProbeTest(unittest.TestCase):
    def make_probe(self, history=3, count=3):
        probe = ProbingEvaluator.__new__(ProbingEvaluator)
        probe.model = WindowModel(history)
        probe.config = None
        probe.device = "cpu"
        probe.action_key = "action_cartesian"
        probe.transform = {key: lambda frame: frame.float() + 1 for key in ("pixels", "wrist_pixels")}
        probe.process = {"proprio": SequenceScaler(), "action_cartesian": SequenceScaler()}
        probe.val_dataset = None
        probe.dataset = [dict(
            pixels=torch.zeros(history + 1, 3, 2, 2),
            wrist_pixels=torch.ones(history + 1, 3, 2, 2),
            proprio=np.repeat(np.arange(i, i + history + 1)[:, None], 8, axis=1),
            action_cartesian=np.ones((history + 1, 8)),
            label=np.arange(i, i + history + 1),
        ) for i in range(count)]
        return probe

    def test_training_formula_and_alignment(self):
        for history in (1, 3, 5):
            probe = self.make_probe(history)
            result = probe.collect_one_step_rollout_latents(target_keys=("label",))
            self.assertEqual(result["true_z"].shape, (3, 8))
            self.assertEqual(result["pred_z"].shape, (3, 8))
            np.testing.assert_array_equal(result["indices"], [0, 1, 2])
            np.testing.assert_array_equal(result["targets"]["label"], np.arange(3) + history)
            for i, sample in enumerate(probe.dataset):
                info = {"proprio": probe._prepare_proprio_sequence(sample["proprio"]),
                        "action_cartesian": probe._prepare_action_sequence(sample["action_cartesian"], "action_cartesian")}
                out = probe.model.encode(info)
                reference = probe.model.predict(out["emb"][:, :history], out["act_emb"][:, :history])[:, -1]
                np.testing.assert_allclose(result["pred_z"][i], reference[0].numpy(), atol=1e-5)
            self.assertTrue(all(t == history for t in probe.model.context_lengths))
            np.testing.assert_allclose(result["pred_z"], result["true_z"])

    def test_helpers_and_nan(self):
        probe = self.make_probe()
        sample = probe.dataset[0]
        self.assertEqual(probe._prepare_pixel_sequence(sample["pixels"]).shape, (1, 4, 3, 2, 2))
        self.assertEqual(probe._prepare_proprio_sequence(sample["proprio"]).shape, (1, 4, 8))
        action = torch.ones(4, 8)
        action[0, 0] = float("nan")
        prepared = probe._prepare_action_sequence(action, "action_cartesian")
        self.assertEqual(prepared.shape, (1, 4, 8))
        self.assertEqual(prepared[0, 0, 0], 0)
        self.assertTrue(torch.isfinite(prepared).all())
        self.assertEqual(probe._prepare_pixels(sample["pixels"]).shape, (1, 3, 2, 2))

    def test_config_priority_and_short_window(self):
        probe = self.make_probe()
        probe.config = {"history_size": 2, "wm": {"history_size": 4}}
        self.assertEqual(probe._get_history_size(), 2)
        probe.config = {"wm": {"history_size": 2}}
        self.assertEqual(probe._get_history_size(), 2)
        probe.config = None
        probe.dataset[0]["pixels"] = torch.zeros(3, 3, 2, 2)
        with self.assertRaises(AssertionError):
            probe.collect_one_step_rollout_latents()
        probe.model = SimpleNamespace()
        self.assertEqual(probe._get_history_size(), 3)

    def test_single_window_and_plot_alignment(self):
        probe = self.make_probe(count=1)
        self.assertEqual(probe.collect_one_step_rollout_latents()["pred_z"].shape, (1, 8))
        with self.assertRaises(ValueError):
            plot_one_step_rollout_pca({"true_z": np.zeros((4, 8)), "pred_z": np.zeros((3, 8))})
        data = np.random.default_rng(0).normal(size=(5, 8))
        plotted = plot_one_step_rollout_pca({"true_z": data, "pred_z": data + .2}, draw_connections=True)
        np.testing.assert_allclose(plotted["pred_pca"], plotted["pca"].transform(data + .2))


if __name__ == "__main__":
    unittest.main()
