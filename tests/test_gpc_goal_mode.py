"""CPU regression tests for GPC target selection using both cameras and proprio."""
from pathlib import Path
import subprocess
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from stable_worldmodel import policy as policy_module
from stable_worldmodel.policy import GPCPolicy
from stable_worldmodel.reward import LatentGoalReward


class Proposals:
    obs_horizon = 2
    action_horizon = 2
    device = torch.device("cpu")

    def sample_action_sequences(self, info, num_samples, denormalize):
        assert denormalize
        self.candidates = torch.randn(len(info["pixels"]), num_samples, 4, 3)
        # The terminal actions of the evaluated slice are at step 2.
        self.candidates[:, :, 2] = torch.tensor([[0., 0., 0.], [4., 4., 4.], [10., 10., 10.]])
        return self.candidates


class WorldModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.overhead_projector = torch.nn.Linear(1, 1)
        self.wrist_projector = torch.nn.Linear(1, 1)
        self.prop_encoder = torch.nn.Linear(1, 1)

    def encode(self, info):
        self.encoder_keys = set(info)
        return {"emb": torch.stack((info["pixels"][:, :, 0, 0, 0],
                                    info["wrist_pixels"][:, :, 0, 0, 0],
                                    info["proprio"][:, :, 0]), dim=-1)}

    def rollout(self, info, actions):
        self.wm_actions = actions.clone()
        self.rollout_keys = set(info)
        return {"predicted_emb": actions.clone()}


def observations(current=((0, 1, 2), (5, 4, 5)), goal=((5, 5, 4), (1, 0, 0))):
    def history(latest):
        # An unrelated earlier frame detects accidentally using the first or mean latent.
        return np.stack([np.full_like(latest, 20), latest], axis=1)

    def images(values, scale):
        raw = history(values * scale).astype(np.uint8)
        return np.broadcast_to(raw[:, :, None, None, None], (2, 2, 2, 2, 3)).copy()

    output = {}
    for prefix, rows in (("", current), ("goal_", goal)):
        overhead, wrist, prop = np.array(rows, np.float32).T
        output["goal" if prefix else "pixels"] = images(overhead, 2)
        output[prefix + "wrist_pixels"] = images(wrist, 4)
        output[prefix + "proprio"] = history(prop + 1)[:, :, None]
    return output


def build(cls=GPCPolicy, weights=None, **kwargs):
    wm = WorldModel().eval()
    reward = Mock(wraps=LatentGoalReward(wm, **(weights or {})))
    process = {"action_cartesian": Mock(transform=lambda x: x / 2),
               "proprio": Mock(transform=lambda x: x - 1)}
    transform = {"pixels": lambda x: x.float() / 2, "wrist_pixels": lambda x: x.float() / 4}
    return cls(Proposals(), wm, reward, num_candidates=3, process=process, transform=transform, **kwargs)


@pytest.fixture(scope="module")
def previous_policy():
    # Load the pre-goal-mode implementation for exact comparisons of all intermediate values.
    root = Path(__file__).resolve().parents[1]
    revision = "0552164a1c578cc019da8a99668ea3c69d275da4"
    source = subprocess.check_output(["git", "show", f"{revision}:stable_worldmodel/policy.py"], cwd=root)
    namespace = vars(policy_module).copy()
    exec(compile(source, "<pre-goal-mode policy>", "exec"), namespace)
    return namespace["GPCPolicy"]


@pytest.mark.parametrize("mode", [{}, {"goal_mode": "final_goal"}])
@pytest.mark.parametrize("execute_first", [False, True])
def test_exact_previous_inference(previous_policy, mode, execute_first):
    prior = build(previous_policy, execute_first_candidate=execute_first)
    policy = build(execute_first_candidate=execute_first, **mode)
    assert policy.goal_mode == "final_goal"

    def trace(p):
        torch.manual_seed(81)
        chunk = p.get_action_chunk(observations())
        output, info = p.reward_fn.call_args.args
        return (p.diffusion_policy.candidates, p.world_model.wm_actions, output["goal_emb"],
                output["predicted_emb"], p.reward_fn(output, info), p.best_idx,
                p.selected_candidate_index, torch.from_numpy(chunk), torch.get_rng_state())

    for old, new in zip(trace(prior), trace(policy)):
        assert torch.equal(old, new)


@pytest.mark.parametrize("execute_first", [False, True])
def test_current_observation_weights_and_selection(execute_first):
    weights = dict(lambda_overhead=2., lambda_wrist=3., lambda_prop=.5)
    policy = build(goal_mode="current_observation", weights=weights, execute_first_candidate=execute_first)
    info = observations()
    chunk = policy.get_action_chunk(info)
    output, _ = policy.reward_fn.call_args.args
    target = torch.tensor([[[0., 1., 2.]], [[5., 4., 5.]]])
    assert torch.equal(output["goal_emb"], target)
    assert policy.world_model.encoder_keys == {"pixels", "wrist_pixels", "proprio"}
    assert not policy.world_model.rollout_keys.intersection({"goal", "goal_wrist_pixels", "goal_proprio"})

    predictions = output["predicted_emb"]
    assert predictions.shape == (2, 3, 2, target.shape[-1])
    squared = (predictions[:, :, -1] - target) ** 2
    costs = (squared * torch.tensor([2., 3., .5])).sum(-1)
    rewards = policy.reward_fn(output, info)
    assert rewards.shape == (2, 3)
    assert torch.equal(rewards, -costs)
    assert torch.equal(policy.best_idx, costs.argmin(1))
    assert policy.best_idx.tolist() == [0, 2]
    indices = [0, 0] if execute_first else [0, 2]
    assert policy.selected_candidate_index.tolist() == indices
    np.testing.assert_array_equal(chunk, policy.diffusion_policy.candidates[[0, 1], indices, 1:3].numpy())


def test_target_refreshes_per_observation_and_ignores_fixed_goal():
    policy = build(goal_mode="current_observation")

    def trace(info):
        torch.manual_seed(53)
        chunk = policy.get_action_chunk(info)
        output, _ = policy.reward_fn.call_args.args
        return output["goal_emb"], policy.reward_fn(output, info), policy.best_idx, torch.from_numpy(chunk)

    initial = trace(observations())
    changed_goal = trace(observations(goal=((100, 20, -3), (50, 1, 5))))
    for before, after in zip(initial, changed_goal):
        assert torch.equal(before, after)
    latest = ((5, 5, 5), (0, 0, 0))
    info = observations(current=latest)
    # A saved goal is optional for this mode at the policy boundary.
    current = trace({k: v for k, v in info.items() if not k.startswith("goal")})
    assert torch.equal(current[0], torch.tensor(latest, dtype=torch.float32)[:, None])
    assert current[2].tolist() == [2, 0]
    assert not torch.equal(initial[0], current[0])


@pytest.mark.parametrize("mode", [None, "", "current", 1])
def test_invalid_mode(mode):
    with pytest.raises(ValueError, match="goal_mode.*final_goal.*current_observation"):
        build(goal_mode=mode)


def test_current_latent_dimension_must_match_prediction():
    policy = build(goal_mode="current_observation")
    policy.world_model.encode = Mock(return_value={"emb": torch.zeros(2, 2, 1)})
    with pytest.raises(ValueError, match="latent.*matching the WM prediction"):
        policy.get_action_chunk(observations())
